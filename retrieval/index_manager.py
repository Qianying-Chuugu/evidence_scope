# 统一管理本地 PDF 的文档清单、Chunk 和 Dense 向量。

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .chunking import chunk_pdf


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MANIFEST_VERSION = 2  #第 2 版索引清单格式

@dataclass(frozen=True)
class IndexSnapshot:
    chunks: list
    vectors: np.ndarray
    documents: dict
# “索引快照”数据容器，用来把一次检索需要的数据打包在一起。
# dataclass 是 Python 标准库提供的“数据类，frozen=True 会阻止对象创建后重新赋值。
# chunks：文档分块列表。vectors：每个 Chunk 对应的 Dense 向量矩阵。documents：文档 ID 与文档信息的对应表。

class IndexManager:
    # 项目中唯一负责读写索引文件的类。
    def __init__(self, data_dir=DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.documents_dir = self.data_dir / "documents"
        self.indexes_dir = self.data_dir / "indexes"
        self.manifest_path = self.indexes_dir / "manifest.json"
    # ---------- 对外功能 ----------
    def sync_documents(self):
        """ 扫描 PDF，只重新处理新增、变化或索引缺失的文档。"""
        # 核心函数***
        pdf_paths, current_documents = self._scan_documents()
        old_manifest = self._load_manifest()
        old_documents = old_manifest["documents"]
        if not current_documents and not old_documents:
            raise FileNotFoundError(f"没有找到 PDF。请把文件放入 {self.documents_dir}")

        # 新旧目录各自取出自己存储的PDF标志，存储在集合中
        current_ids = set(current_documents)
        old_ids = set(old_documents)
        added = current_ids - old_ids # 目前新加入的PDF
        deleted = old_ids - current_ids # 目前新删去的PDF
        # 新旧都存在且发生变化的PDF
        modified = {
            document_id for document_id in current_ids & old_ids
            if self._file_changed(
                current_documents[document_id],
                old_documents[document_id],
            )
        }
        # 新旧都存在且未发生变化的PDF
        unchanged = (current_ids & old_ids) - modified
        # PDF本身没变化，但对应Chunk文件不存在的PDF
        missing_chunks = {
            document_id for document_id in unchanged
            if not self._chunks_path(document_id).exists()
        }

        # 内容变化时旧向量失效，格式升级只重写 Chunk，不重复计算向量。
        content_changed = added | modified | missing_chunks
        rebuild = set(content_changed)
        if old_manifest.get("version") != MANIFEST_VERSION:
            rebuild.update(unchanged) # 如果索引格式变化，就把没有变化的PDF也加入重写chunk
            print("索引数据格式已更新，需要迁移现有分块。")
        self._print_changes(added, modified, deleted, unchanged)
        if not rebuild and not deleted:
            print("没有文档变化，无需重新分块。")
            return old_manifest

        # 开始重写chunk
        new_documents = {}
        for document_id in sorted(current_ids):
            metadata = current_documents[document_id]
            if document_id in rebuild:
                chunks = chunk_pdf(pdf_paths[document_id], document_id)
                self._write_json(self._chunks_path(document_id), chunks)
                indexed_at = datetime.now(timezone.utc).isoformat() #记录本次分块的时间
                if document_id in content_changed:
                    self._vectors_path(document_id).unlink(missing_ok=True) #对应的向量数据也作废
                print(f"已处理：{metadata['source_path']}，生成 {len(chunks)} 个 Chunk")
            else: # 不需要重建的PDF，直接读取数据库
                chunks = self._load_chunks(document_id)
                indexed_at = old_documents[document_id].get("indexed_at")
                print(f"已复用：{metadata['source_path']}，共 {len(chunks)} 个 Chunk")
            new_documents[document_id] = {
                **metadata,
                "status": "ready",
                "chunk_count": len(chunks),
                "indexed_at": indexed_at,
            }
        # 删除已经删除的PDF的数据库数据
        for document_id in sorted(deleted):
            shutil.rmtree(self._index_dir(document_id), ignore_errors=True)
            print(f"已移除索引：{document_id}")
        new_manifest = {
            "version": MANIFEST_VERSION,
            "documents": new_documents,
        }
        # 生成的新manifest索引，存储回数据库
        self._write_json(self.manifest_path, new_manifest)
        return new_manifest

    def list_documents(self):
        """返回 manifest 中的文档信息，不加载 Chunk 和向量。"""
        return self._load_manifest()["documents"]

    def resolve_document_id(self, source):
        """根据文件名或相对路径查找唯一的 document_id。"""
        original_source = source
        source = source.casefold()
        matches = []
        for document_id, document in self.list_documents().items():
            if source in {
                document["source"].casefold(),
                document["source_path"].casefold(),
            }:
                matches.append(document_id)
        if not matches:
            raise KeyError(f"本地知识库中没有文件：{original_source}")
        if len(matches) > 1:
            raise ValueError(
                f"存在多个同名文件，请改用相对路径指定：{original_source}"
            )
        return matches[0]

    def ensure_dense_vectors(self, encode_chunks, document_ids=None):
        """检查向量文件，缺失、损坏或数量错误时重新生成。"""
        manifest = self._load_manifest()
        for document_id in self._select_ids(manifest, document_ids):
            chunks = self._load_chunks(document_id)
            vectors_path = self._vectors_path(document_id)
            if self._valid_vectors(vectors_path, len(chunks)):
                continue
            vectors = encode_chunks(chunks)
            if len(vectors) != len(chunks):
                raise ValueError("向量数量与 Chunk 数量不一致")

            self._write_vectors(vectors_path, vectors)
            print(f"已生成向量：doc_{document_id}，共 {len(vectors)} 条")

    def load_all_chunks(self, document_ids=None):
        """返回全部或指定文档的 Chunk。"""
        manifest = self._load_manifest()
        chunks = []
        for document_id in self._select_ids(manifest, document_ids):
            chunks.extend(self._load_chunks(document_id))
        return chunks

    def load_snapshot(self, document_ids=None):
        """返回检索需要的 Chunk、向量和文档信息。"""
        manifest = self._load_manifest()
        selected_ids = self._select_ids(manifest, document_ids)
        chunks = []
        vector_groups = []
        for document_id in selected_ids:
            document_chunks = self._load_chunks(document_id)
            document_vectors = self._load_vectors(document_id)
            if len(document_chunks) != len(document_vectors):
                raise ValueError(f"Chunk 与向量数量不一致：doc_{document_id}")
            chunks.extend(document_chunks)
            vector_groups.append(document_vectors)
        documents = {
            document_id: manifest["documents"][document_id]
            for document_id in selected_ids
        }
        return IndexSnapshot(
            chunks=chunks,
            vectors=np.concatenate(vector_groups),
            documents=documents,
        )

    # ---------- 内部读写 ----------
    def _scan_documents(self):
        """ 返回两个字典：document_id：PDF 路径(用于程序打开文档)、document_id：元数据(记录文件信息)。 """
        if not self.documents_dir.exists():
            return {}, {}
        paths = sorted(
            self.documents_dir.rglob("*.pdf"),
            key=lambda path: str(path).casefold(),
        )
        pdf_paths = {}
        documents = {}
        for path in paths:
            if not path.is_file():
                continue
            path = path.resolve() # 相对路径转化为绝对路径
            document_id = self._make_document_id(path)
            stat = path.stat() # 读取文件属性
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            pdf_paths[document_id] = path
            documents[document_id] = {
                "document_id": document_id,
                "source": path.name,
                "source_path": path.relative_to(self.data_dir).as_posix(),
                "file_size": stat.st_size,
                "modified_time_ns": stat.st_mtime_ns,
                "content_sha256": digest.hexdigest(),
            }
        return pdf_paths, documents

    def _load_manifest(self):
        if not self.manifest_path.exists():
            return {"version": MANIFEST_VERSION, "documents": {}}

        with self.manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        if not isinstance(manifest, dict) or not isinstance(
            manifest.get("documents"), dict
        ):
            raise ValueError(f"文档清单格式错误：{self.manifest_path}")
        return manifest

    def _load_chunks(self, document_id):
        path = self._chunks_path(document_id)
        if not path.exists():
            raise FileNotFoundError(f"缺少分块文件：{path}")

        with path.open("r", encoding="utf-8") as file:
            chunks = json.load(file)
        if not isinstance(chunks, list):
            raise ValueError(f"分块文件格式错误：{path}")
        return chunks

    def _load_vectors(self, document_id):
        path = self._vectors_path(document_id)
        if not path.exists():
            raise FileNotFoundError(f"缺少向量文件：{path}")
        return np.load(path, allow_pickle=False)

    def _select_ids(self, manifest, document_ids):
        saved_ids = set(manifest["documents"])
        if not saved_ids:
            raise FileNotFoundError("尚未建立文档索引。")

        selected_ids = saved_ids if document_ids is None else set(document_ids)
        unknown_ids = selected_ids - saved_ids
        if unknown_ids:
            raise KeyError(f"清单中不存在文档 ID：{sorted(unknown_ids)}")
        if not selected_ids:
            raise ValueError("至少需要选择一份文档")
        return sorted(selected_ids)

    def _make_document_id(self, path):
        """根据PDF路径生成文档ID"""
        relative_path = path.relative_to(self.data_dir).as_posix().casefold()
        return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:8]

    def _index_dir(self, document_id):
        return self.indexes_dir / f"doc_{document_id}"

    def _chunks_path(self, document_id):
        return self._index_dir(document_id) / "chunks.json"

    def _vectors_path(self, document_id):
        return self._index_dir(document_id) / "dense_vectors.npy"

    @staticmethod
    def _file_changed(current, old):
        fields = ("source_path", "file_size", "modified_time_ns", "content_sha256")
        return any(current[field] != old.get(field) for field in fields)

    @staticmethod
    def _valid_vectors(path, chunk_count):
        if not path.exists():
            return False
        try:
            vectors = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            return False
        return vectors.ndim == 2 and len(vectors) == chunk_count

    @staticmethod
    def _print_changes(added, modified, deleted, unchanged):
        """ 打印目前的文档变化情况 """
        print(
            "文档变化："
            f"新增 {len(added)}，修改 {len(modified)}，"
            f"删除 {len(deleted)}，未变化 {len(unchanged)}"
        )

    @staticmethod
    def _write_json(path, data):
        """ 重写后的chunk数据写入对应数据库json """
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.name}.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        temporary_path.replace(path)

    @staticmethod
    def _write_vectors(path, vectors):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.name}.tmp")
        with temporary_path.open("wb") as file:
            np.save(file, vectors)
        temporary_path.replace(path)


if __name__ == "__main__":
    IndexManager().sync_documents()
