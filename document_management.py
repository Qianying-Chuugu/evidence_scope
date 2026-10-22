"""网页文档管理：校验 PDF，并复用现有索引同步流程。"""

import hashlib
from pathlib import Path
from uuid import uuid4

import pymupdf

from retrieval.index_manager import IndexManager
from tools.evidence_search import get_document_catalog


MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class DocumentValidationError(ValueError):
    pass


class DocumentConflictError(ValueError):
    pass


class DocumentNotFoundError(LookupError):
    pass


class DocumentIndexError(RuntimeError):
    pass


class DocumentManager:
    def __init__(self, index_manager=None, sync_catalog=None):
        self.index_manager = index_manager or IndexManager()
        self.sync_catalog = sync_catalog or get_document_catalog

    @property
    def documents_dir(self):
        return self.index_manager.documents_dir

    def list_documents(self):
        try:
            self.sync_catalog()
        except FileNotFoundError:
            # 新知识库尚无 PDF 时，网页应能正常显示空列表。
            if self.index_manager.list_documents() or any(self.documents_dir.rglob("*.pdf")):
                raise
            return []
        documents = self.index_manager.list_documents().values()
        return sorted(
            (
                {
                    "document_id": doc["document_id"],
                    "source": doc["source"],
                    "source_path": doc["source_path"],
                    "chunk_count": doc["chunk_count"],
                    "file_size": doc["file_size"],
                }
                for doc in documents
            ),
            key=lambda doc: doc["source_path"].casefold(),
        )

    def upload(self, filename, stream):
        name = self._safe_filename(filename)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        if self._name_exists(name):
            raise DocumentConflictError("已有同名 PDF，请先重命名或删除旧文件。")

        temporary = self.documents_dir / f".upload-{uuid4().hex}.tmp"
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("xb") as output:
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise DocumentValidationError("PDF 不能超过 50 MB。")
                    digest.update(chunk)
                    output.write(chunk)
            if size == 0:
                raise DocumentValidationError("PDF 文件为空。")
            self._validate_pdf(temporary)
            if self._content_exists(digest.digest()):
                raise DocumentConflictError("这份 PDF 已在文档库中。")

            target = self.documents_dir / name
            if self._name_exists(name):
                raise DocumentConflictError("已有同名 PDF，请先重命名或删除旧文件。")
            temporary.rename(target)
            try:
                documents = self.list_documents()
            except Exception as error:
                target.unlink(missing_ok=True)
                self._retry_sync()
                raise DocumentIndexError("PDF 已撤回：建立索引失败，请查看服务端日志。") from error
            return next(doc for doc in documents if doc["source_path"] == f"documents/{name}")
        finally:
            temporary.unlink(missing_ok=True)

    def delete(self, document_id):
        documents = self.list_documents()
        document = next((doc for doc in documents if doc["document_id"] == document_id), None)
        if document is None:
            raise DocumentNotFoundError("文档不存在或已经删除。")

        target = (self.index_manager.data_dir / document["source_path"]).resolve()
        if not target.is_relative_to(self.documents_dir.resolve()):
            raise DocumentValidationError("文档路径不在知识库目录内。")
        if not target.is_file():
            raise DocumentNotFoundError("PDF 文件已不存在。")

        temporary = target.with_name(f".deleting-{uuid4().hex}.tmp")
        target.rename(temporary)
        try:
            self.list_documents()
        except Exception as error:
            temporary.rename(target)
            self._retry_sync()
            raise DocumentIndexError("删除未完成：同步索引失败，请查看服务端日志。") from error
        temporary.unlink(missing_ok=True)
        return document

    @staticmethod
    def _safe_filename(filename):
        if not isinstance(filename, str) or not filename or "/" in filename or "\\" in filename:
            raise DocumentValidationError("请选择文件名有效的 PDF。")
        if filename != filename.strip() or filename.rstrip(". ") != filename:
            raise DocumentValidationError("文件名不能以空格或句点结尾。")
        if any(char in filename for char in '<>:"|?*') or any(ord(char) < 32 for char in filename):
            raise DocumentValidationError("PDF 文件名包含不允许的字符。")
        path = Path(filename)
        if path.suffix.casefold() != ".pdf" or not path.stem:
            raise DocumentValidationError("只能上传 .pdf 文件。")
        if path.stem.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
            raise DocumentValidationError("PDF 文件名不可用。")
        return path.stem + ".pdf"

    @staticmethod
    def _validate_pdf(path):
        try:
            # 用字节流打开，避免无效 PDF 在 Windows 上残留文件句柄。
            with pymupdf.open(stream=path.read_bytes(), filetype="pdf") as pdf:
                if pdf.is_encrypted or pdf.page_count == 0:
                    raise DocumentValidationError("PDF 已加密或没有页面。")
                if not any(len(page.get_text("text", sort=True).strip()) >= 100 for page in pdf):
                    raise DocumentValidationError("无法提取足够文字；扫描版 PDF 暂不支持。")
        except DocumentValidationError:
            raise
        except (RuntimeError, ValueError, pymupdf.FileDataError) as error:
            raise DocumentValidationError("文件不是有效的 PDF，或内容无法读取。") from error

    def _name_exists(self, name):
        return any(path.name.casefold() == name.casefold() for path in self.documents_dir.rglob("*.pdf"))

    def _content_exists(self, digest):
        for path in self.documents_dir.rglob("*.pdf"):
            existing = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    existing.update(chunk)
            if existing.digest() == digest:
                return True
        return False

    def _retry_sync(self):
        try:
            self.sync_catalog()
        except Exception:
            pass
