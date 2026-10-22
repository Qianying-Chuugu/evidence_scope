import logging
from io import BytesIO
from functools import lru_cache
from pathlib import Path
from threading import RLock

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from agent.harness import AgentExecutionError, ModelServiceError
from document_management import (
    DocumentConflictError,
    DocumentIndexError,
    DocumentManager,
    DocumentNotFoundError,
    DocumentValidationError,
    MAX_UPLOAD_BYTES,
)
from main import build_agent

app = FastAPI(title="EvidenceScope API")
logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"
document_lock = RLock()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def homepage():
    return FileResponse(STATIC_DIR / "index.html")


@lru_cache(maxsize=1)
def get_agent():
    """首次问答时创建 Agent；健康检查不依赖 API Key。"""
    return build_agent()


@lru_cache(maxsize=1)
def get_document_manager():
    return DocumentManager()

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/documents")
def list_documents():
    with document_lock:
        try:
            return {"documents": get_document_manager().list_documents()}
        except (OSError, ValueError, RuntimeError):
            logger.exception("读取文档目录失败")
            raise HTTPException(status_code=500, detail="读取文档目录失败，请查看服务端日志。")


@app.post("/api/documents", status_code=201)
async def upload_document(request: Request, filename: str = Query(...)):
    if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/pdf":
        raise HTTPException(status_code=415, detail="请上传 PDF 文件。")
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="PDF 不能超过 50 MB。")
        data.extend(chunk)
    with document_lock:
        try:
            document = get_document_manager().upload(filename, BytesIO(data))
        except DocumentConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except DocumentValidationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except (DocumentIndexError, OSError):
            logger.exception("上传 PDF 或建立索引失败")
            raise HTTPException(status_code=500, detail="上传或索引处理失败，请查看服务端日志。")
        return {"document": document}


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    with document_lock:
        try:
            document = get_document_manager().delete(document_id)
        except DocumentNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except DocumentValidationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except (DocumentIndexError, OSError, ValueError):
            logger.exception("删除 PDF 或同步索引失败")
            raise HTTPException(status_code=500, detail="删除或索引同步失败，请查看服务端日志。")
        return {"deleted": document}


@app.post("/api/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="问题不能为空或只包含空格"
        )

    with document_lock:
        try:
            agent = get_agent()
        except RuntimeError as error:
            logger.exception("Agent 初始化失败")
            raise HTTPException(status_code=503, detail="模型服务尚未配置或无法初始化。") from error

        try:
            answer = agent.run(question)
        except ModelServiceError as error:
            logger.exception("Agent 调用模型服务失败")
            raise HTTPException(status_code=502, detail=str(error)) from error
        except AgentExecutionError as error:
            logger.exception("Agent 执行失败")
            raise HTTPException(status_code=500, detail=str(error)) from error

    return {"answer": answer}
