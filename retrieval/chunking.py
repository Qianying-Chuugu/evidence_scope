from pathlib import Path
import pymupdf

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 150

def chunk_pdf(pdf_path, document_id,
                chunk_size=DEFAULT_CHUNK_SIZE,
                overlap=DEFAULT_OVERLAP,):
    """ 把一个PDF文件分块"""
    # 文档地址、文档 ID、分块大小、重叠大小

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0，并且小于 chunk_size")

    pdf_path = Path(pdf_path)
    #路径字符串转换为路径对象
    chunks = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text", sort=True).strip()
            # 字符过少的封面或空白页不进入检索索引。
            if len(text) < 100:
                continue

            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_number = len(chunks) + 1
                chunks.append(
                    {
                        "chunk_id": f"doc_{document_id}_chunk_{chunk_number:04d}",
                        "document_id": document_id,
                        "page": page_number,
                        "text": text[start:end],
                    }
                )

                if end == len(text):
                    break
                start = end - overlap

    return chunks
