"""Check cited source locations; a human must judge whether text supports each claim."""

import argparse
import json
from pathlib import Path

import pymupdf
from retrieval.index_manager import IndexManager


def audit_claims(claims, chunks_by_id, page_texts=None):
    rows = []
    for claim in claims:
        citations = []
        for citation in claim["citations"]:
            chunk = chunks_by_id.get(citation["chunk_id"])
            location_valid = (
                chunk is not None
                and chunk["source"] == citation["source"]
                and chunk["page"] == citation["pdf_page"]
            )
            pdf_text_verified = (
                location_valid
                and chunk["text"] in page_texts.get(
                    (citation["source"], citation["pdf_page"]), ""
                )
            ) if page_texts is not None else None
            citations.append({
                **citation,
                "location_valid": location_valid,
                "pdf_text_verified": pdf_text_verified,
                "cited_text": chunk["text"] if location_valid else None,
            })
        supported = claim.get("supported")
        if supported is not None and not isinstance(supported, bool):
            raise ValueError("'supported' must be true, false, or null")
        rows.append({
            "claim": claim["claim"],
            "citations": citations,
            "all_locations_valid": bool(citations) and all(c["location_valid"] for c in citations),
            "human_supported": supported,
            "review_note": claim.get("review_note", ""),
        })
    reviewed = [row for row in rows if row["human_supported"] is not None]
    return {
        "claims": rows,
        "location_valid_count": sum(row["all_locations_valid"] for row in rows),
        "pdf_verified_citation_count": sum(
            citation["pdf_text_verified"] is True
            for row in rows for citation in row["citations"]
        ),
        "claim_count": len(rows),
        "human_reviewed_count": len(reviewed),
        "human_supported_count": sum(row["human_supported"] for row in reviewed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSON file with claims and human support verdicts")
    parser.add_argument("--output", type=Path, help="Where to save the full audit")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data.get("claims"), list):
        raise ValueError("Input requires a claims list")

    manager = IndexManager()
    manager.sync_documents()
    chunks = {}
    page_texts = {}
    requested = {
        (citation["source"], citation["pdf_page"])
        for claim in data["claims"] for citation in claim["citations"]
    }
    for document in manager.list_documents().values():
        source = document["source"]
        pages = [page for name, page in requested if name == source]
        if pages:
            with pymupdf.open(manager.documents_dir / source) as pdf:
                for page in pages:
                    if 1 <= page <= pdf.page_count:
                        page_texts[(source, page)] = pdf[page - 1].get_text(
                            "text", sort=True
                        ).strip()
        for chunk in manager.load_all_chunks([document["document_id"]]):
            chunks[chunk["chunk_id"]] = {
                **chunk,
                "source": document["source"],
            }
    report = audit_claims(data["claims"], chunks, page_texts)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
