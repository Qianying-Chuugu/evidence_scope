import io
import json
import copy
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pymupdf
from openai import OpenAIError

from agent.harness import AgentExecutionError, AgentHarness, ModelServiceError
from document_management import (
    DocumentConflictError,
    DocumentIndexError,
    DocumentManager,
    DocumentValidationError,
)
from evaluation.audit_citations import audit_claims
from evaluation.metrics import scores
from retrieval.hybrid_search import reciprocal_rank_fusion
from retrieval.index_manager import IndexManager
from tools import evidence_search


def pdf_bytes(text):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text[:1000])
    data = doc.tobytes()
    doc.close()
    return data


def message(content=None, calls=None):
    value = SimpleNamespace(content=content, tool_calls=calls)
    value.model_dump = lambda exclude_none=True: {
        "role": "assistant",
        **({"content": content} if content is not None else {}),
        **({"tool_calls": [{"id": call.id, "type": "function",
                            "function": {"name": call.function.name,
                                         "arguments": call.function.arguments}}
                           for call in calls]} if calls else {}),
    }
    return value


class FusionTests(unittest.TestCase):
    def test_overlapping_chunks_count_as_one_evidence_group(self):
        ranking = [{"chunk_id": "a"}, {"chunk_id": "x"}]
        result = scores(ranking, [{"a", "b"}], k=2)
        self.assertEqual(result["hit_at_2"], 1)
        self.assertEqual(result["evidence_recall_at_2"], 1.0)
        self.assertEqual(result["mrr_at_2"], 1.0)
        self.assertEqual(result["evidence_ndcg_at_2"], 1.0)
        duplicates = scores([{"chunk_id": "a"}, {"chunk_id": "b"}], [{"a", "b"}], k=2)
        self.assertEqual(duplicates["evidence_ndcg_at_2"], 1.0)

    def test_multiple_distinct_evidence_groups(self):
        ranking = [{"chunk_id": "a"}, {"chunk_id": "c"}]
        result = scores(ranking, [{"a", "b"}, {"c"}], k=2)
        self.assertEqual(result["evidence_recall_at_2"], 1.0)
        self.assertAlmostEqual(result["evidence_ndcg_at_2"], (1 + 1 / math.log2(3)) / 2)
        partial = scores(ranking[:1], [{"a", "b"}, {"c"}], k=2)
        self.assertEqual(partial["evidence_recall_at_2"], 0.5)
        self.assertEqual(partial["hit_at_2"], 1)

    def test_weights_deduplication_and_ranks(self):
        def item(name):
            return {"chunk_id": name, "document_id": "d", "page": 1, "text": name}

        results = reciprocal_rank_fusion(
            [item("a"), item("b")], [item("b"), item("c")],
            rrf_k=1, bm25_weight=2, dense_weight=1,
        )
        self.assertEqual([x["chunk_id"] for x in results], ["b", "a", "c"])
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["bm25_rank"], 2)
        self.assertEqual(results[0]["dense_rank"], 1)
        self.assertAlmostEqual(results[0]["rrf_score"], 2 / 3 + 1 / 2)
        self.assertGreater(
            reciprocal_rank_fusion([item("a")], [item("b")], rrf_k=1,
                                   bm25_weight=3)[0]["rrf_score"],
            reciprocal_rank_fusion([item("a")], [item("b")], rrf_k=1,
                                   bm25_weight=1)[0]["rrf_score"],
        )


class CitationAuditTests(unittest.TestCase):
    def test_location_is_checked_separately_from_human_support(self):
        chunks = {"c1": {"source": "sample.pdf", "page": 3, "text": "source passage"}}
        claims = [
            {"claim": "A", "citations": [
                {"source": "sample.pdf", "pdf_page": 3, "chunk_id": "c1"}
            ], "supported": False},
            {"claim": "B", "citations": [
                {"source": "sample.pdf", "pdf_page": 4, "chunk_id": "c1"}
            ], "supported": None},
        ]
        report = audit_claims(
            claims, chunks, {("sample.pdf", 3): "prefix source passage suffix"}
        )
        self.assertEqual(report["location_valid_count"], 1)
        self.assertEqual(report["pdf_verified_citation_count"], 1)
        self.assertEqual(report["human_reviewed_count"], 1)
        self.assertEqual(report["human_supported_count"], 0)
        self.assertEqual(report["claims"][0]["citations"][0]["cited_text"], "source passage")
        self.assertIsNone(report["claims"][1]["citations"][0]["cited_text"])


class HarnessTests(unittest.TestCase):
    def make_harness(self, responses, executor=None, iterations=3):
        client = Mock()
        pending = iter([
            SimpleNamespace(choices=[SimpleNamespace(message=response)])
            for response in responses
        ])
        client.seen_messages = []
        def respond(**kwargs):
            client.seen_messages.append(copy.deepcopy(kwargs["messages"]))
            return next(pending)
        client.chat.completions.create.side_effect = respond
        harness = AgentHarness(
            client, "fake-model", [{"role": "system", "content": "rules"}],
            [{"type": "function"}], executor or (lambda *_: '{"results": []}'),
            max_iterations=iterations,
            catalog_provider=lambda: [{"source": "doc.pdf"}],
        )
        return harness, client

    def test_tool_result_is_returned_to_model_and_each_run_is_fresh(self):
        call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="search_evidence",
                                     arguments='{"query":"risk"}'),
        )
        executor = Mock(return_value='{"results":[{"chunk_id":"c1"}]}')
        harness, client = self.make_harness(
            [message(calls=[call]), message(content="answer"),
             message(content="second answer")], executor,
        )
        self.assertEqual(harness.run("first"), "answer")
        self.assertEqual(harness.run("second"), "second answer")
        executor.assert_called_once_with("search_evidence", {"query": "risk"})
        second_messages = client.seen_messages[1]
        self.assertEqual(second_messages[-1]["role"], "tool")
        self.assertEqual(second_messages[-1]["tool_call_id"], "call-1")
        self.assertIn("c1", second_messages[-1]["content"])
        third_messages = client.seen_messages[2]
        self.assertNotIn("tool", [x["role"] for x in third_messages])

    def test_tool_error_and_iteration_limit(self):
        call = SimpleNamespace(
            id="c", function=SimpleNamespace(name="search_evidence", arguments="{bad"),
        )
        harness, client = self.make_harness(
            [message(calls=[call]), message(content="unsupported")],
        )
        with self.assertRaisesRegex(AgentExecutionError, "检索工具执行失败"):
            harness.run("question")
        tool_message = client.seen_messages[1][-1]
        self.assertIn("error", json.loads(tool_message["content"]))

        repeating, _ = self.make_harness(
            [message(calls=[call])], iterations=1,
        )
        with self.assertRaisesRegex(AgentExecutionError, "最大迭代次数"):
            repeating.run("question")

    def test_model_and_catalog_failures(self):
        harness, client = self.make_harness([])
        client.chat.completions.create.side_effect = OpenAIError("offline")
        with self.assertRaises(ModelServiceError):
            harness.run("question")
        harness.catalog_provider = lambda: (_ for _ in ()).throw(ValueError("bad"))
        with self.assertRaisesRegex(AgentExecutionError, "文档目录读取失败"):
            harness.run("question")


class DocumentAndIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.index = IndexManager(Path(self.temp.name))
        self.index.documents_dir.mkdir()
        self.manager = DocumentManager(self.index, self.index.sync_documents)
        self.text_a = "Alpha evidence " * 12
        self.text_b = "Beta evidence " * 12

    def test_add_modify_delete_invalidates_vectors_and_manifest(self):
        path = self.index.documents_dir / "sample.pdf"
        path.write_bytes(pdf_bytes(self.text_a))
        first = self.index.sync_documents()["documents"]
        doc_id = next(iter(first))
        old_text = self.index.load_all_chunks()[0]["text"]
        self.index.ensure_dense_vectors(
            lambda chunks: np.ones((len(chunks), 3), dtype=np.float32)
        )
        vector_path = self.index._vectors_path(doc_id)
        self.assertTrue(vector_path.exists())

        path.write_bytes(pdf_bytes(self.text_b + " updated"))
        self.index.sync_documents()
        self.assertFalse(vector_path.exists())
        self.assertNotEqual(self.index.load_all_chunks()[0]["text"], old_text)
        self.assertNotIn(old_text, self.index.load_all_chunks()[0]["text"])

        path.unlink()
        self.index.sync_documents()
        self.assertEqual(self.index.list_documents(), {})
        self.assertFalse(self.index._index_dir(doc_id).exists())

    def test_retriever_cache_clears_on_catalog_change(self):
        path = self.index.documents_dir / "sample.pdf"
        path.write_bytes(pdf_bytes(self.text_a))
        with patch.object(evidence_search, "_index_manager", self.index):
            with patch.dict(evidence_search._retriever_cache, {"stale": object()}, clear=True):
                evidence_search.get_document_catalog()
                self.assertNotIn("stale", evidence_search._retriever_cache)
                evidence_search._retriever_cache["stale"] = object()
                evidence_search.get_document_catalog()
                self.assertIn("stale", evidence_search._retriever_cache)
                path.write_bytes(pdf_bytes(self.text_b + " updated"))
                evidence_search.get_document_catalog()
                self.assertNotIn("stale", evidence_search._retriever_cache)

    def test_same_size_and_mtime_change_still_rebuilds_index(self):
        path = self.index.documents_dir / "sample.pdf"
        old_bytes = pdf_bytes(self.text_a)
        self.assertIn(b"MuPDF", old_bytes)
        new_bytes = old_bytes.replace(b"MuPDF", b"ZuPDF")
        self.assertEqual(len(old_bytes), len(new_bytes))
        path.write_bytes(old_bytes)
        first = self.index.sync_documents()
        doc_id = next(iter(first["documents"]))
        old_hash = first["documents"][doc_id]["content_sha256"]
        old_stat = path.stat()
        self.index.ensure_dense_vectors(
            lambda chunks: np.ones((len(chunks), 3), dtype=np.float32)
        )
        vector_path = self.index._vectors_path(doc_id)
        path.write_bytes(new_bytes)
        os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        self.assertEqual(path.stat().st_size, old_stat.st_size)
        self.assertEqual(path.stat().st_mtime_ns, old_stat.st_mtime_ns)
        updated = self.index.sync_documents()
        self.assertNotEqual(updated["documents"][doc_id]["content_sha256"], old_hash)
        self.assertFalse(vector_path.exists())

    def test_upload_validation_duplicates_and_oversize(self):
        with self.assertRaises(DocumentValidationError):
            self.manager.upload("invalid.pdf", io.BytesIO(b"not a pdf"))
        with patch("document_management.MAX_UPLOAD_BYTES", 10):
            with self.assertRaises(DocumentValidationError):
                self.manager.upload("large.pdf", io.BytesIO(b"x" * 11))
        original = pdf_bytes(self.text_a)
        first = self.manager.upload("valid.pdf", io.BytesIO(original))
        self.assertEqual(first["source"], "valid.pdf")
        with self.assertRaises(DocumentConflictError):
            self.manager.upload("valid.pdf", io.BytesIO(pdf_bytes(self.text_b)))
        with self.assertRaises(DocumentConflictError):
            self.manager.upload("copy.pdf", io.BytesIO(original))

    def test_delete_failure_restores_document(self):
        item = self.manager.upload("valid.pdf", io.BytesIO(pdf_bytes(self.text_a)))
        calls = 0
        def failing_sync():
            nonlocal calls
            calls += 1
            if calls > 1:
                raise OSError("index unavailable")
            return self.index.sync_documents()
        self.manager.sync_catalog = failing_sync
        with self.assertRaises(DocumentIndexError):
            self.manager.delete(item["document_id"])
        self.assertTrue((self.index.documents_dir / "valid.pdf").exists())
        self.assertFalse(list(self.index.documents_dir.glob(".deleting-*")))


if __name__ == "__main__":
    unittest.main()
