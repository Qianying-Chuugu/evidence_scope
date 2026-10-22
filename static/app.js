(() => {
  "use strict";

  const form = document.querySelector("#question-form");
  const question = document.querySelector("#question");
  const answer = document.querySelector("#answer");
  const resultCard = document.querySelector("#result-card");
  const answerState = document.querySelector("#answer-state");
  const sendButton = document.querySelector("#send-button");
  const buttonLabel = document.querySelector("#button-label");
  const spinner = document.querySelector("#button-spinner");
  const characterCount = document.querySelector("#character-count");
  const connectionStatus = document.querySelector("#connection-status");
  const documentCount = document.querySelector("#document-count");
  const documentList = document.querySelector("#document-list");
  const documentsStatus = document.querySelector("#documents-status");
  const uploadForm = document.querySelector("#upload-form");
  const pdfFile = document.querySelector("#pdf-file");
  const uploadButton = document.querySelector("#upload-button");
  let documentBusy = false;

  function showResult(text, state, isError = false) {
    answer.textContent = text;
    answerState.textContent = state;
    answer.classList.toggle("error", isError);
    answer.classList.remove("empty");
  }

  function appendInline(parent, text) {
    const pattern = /\*\*(.+?)\*\*|`([^`]+)`/g;
    let start = 0;
    for (const match of text.matchAll(pattern)) {
      parent.append(document.createTextNode(text.slice(start, match.index)));
      const element = document.createElement(match[1] === undefined ? "code" : "strong");
      element.textContent = match[1] ?? match[2];
      parent.append(element);
      start = match.index + match[0].length;
    }
    parent.append(document.createTextNode(text.slice(start)));
  }

  function showAnswer(text) {
    answer.replaceChildren();
    let list = null;
    let paragraph = null;
    for (const rawLine of text.split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line) {
        list = null;
        paragraph = null;
        continue;
      }

      const heading = line.match(/^(#{1,3})\s+(.+)$/);
      const bullet = line.match(/^[-*]\s+(.+)$/);
      const numbered = line.match(/^\d+[.)]\s+(.+)$/);
      if (heading) {
        list = null;
        paragraph = null;
        const element = document.createElement("h" + Math.min(heading[1].length + 1, 4));
        appendInline(element, heading[2]);
        answer.append(element);
      } else if (bullet || numbered) {
        paragraph = null;
        const tag = bullet ? "ul" : "ol";
        if (!list || list.tagName.toLowerCase() !== tag) {
          list = document.createElement(tag);
          answer.append(list);
        }
        const item = document.createElement("li");
        appendInline(item, (bullet || numbered)[1]);
        list.append(item);
      } else {
        list = null;
        if (!paragraph) {
          paragraph = document.createElement("p");
          answer.append(paragraph);
        } else {
          paragraph.append(document.createElement("br"));
        }
        appendInline(paragraph, line);
      }
    }
    answerState.textContent = "已完成";
    answer.classList.remove("error", "empty");
  }

  function setBusy(busy) {
    question.disabled = busy;
    sendButton.disabled = busy;
    spinner.hidden = !busy;
    buttonLabel.textContent = busy ? "正在研究…" : "开始研究";
    resultCard.setAttribute("aria-busy", String(busy));
  }

  function errorDetail(payload, status) {
    if (typeof payload?.detail === "string") return payload.detail;
    if (Array.isArray(payload?.detail)) return "问题格式不正确，请检查输入。";
    return "请求失败（HTTP " + status + "），请稍后重试。";
  }

  function setDocumentBusy(busy, message = "") {
    documentBusy = busy;
    pdfFile.disabled = busy;
    uploadButton.disabled = busy;
    documentList.querySelectorAll("button").forEach((button) => { button.disabled = busy; });
    documentsStatus.textContent = message;
  }

  function renderDocuments(documents) {
    documentCount.textContent = documents.length + " 份 PDF";
    documentList.replaceChildren();
    if (!documents.length) {
      const empty = document.createElement("li");
      empty.className = "document-empty";
      empty.textContent = "文档库为空，上传 PDF 后即可开始提问。";
      documentList.append(empty);
      return;
    }
    for (const item of documents) {
      const row = document.createElement("li");
      row.className = "document-row";
      const details = document.createElement("div");
      details.className = "document-details";
      const name = document.createElement("strong");
      name.textContent = item.source;
      const meta = document.createElement("span");
      meta.textContent = item.chunk_count + " 个分块 · " + (item.file_size / 1024 / 1024).toFixed(1) + " MB";
      details.append(name, meta);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "document-delete";
      remove.textContent = "删除";
      remove.setAttribute("aria-label", "删除 " + item.source);
      remove.disabled = documentBusy;
      remove.addEventListener("click", () => deleteDocument(item));
      row.append(details, remove);
      documentList.append(row);
    }
  }

  async function loadDocuments() {
    const response = await fetch("/api/documents", { cache: "no-store" });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(errorDetail(payload, response.status));
    if (!Array.isArray(payload.documents)) throw new Error("文档列表格式不正确。");
    renderDocuments(payload.documents);
  }

  async function deleteDocument(item) {
    if (documentBusy || !window.confirm("删除“" + item.source + "”及其索引？")) return;
    setDocumentBusy(true, "正在删除并同步索引…");
    try {
      const response = await fetch("/api/documents/" + encodeURIComponent(item.document_id), { method: "DELETE" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(errorDetail(payload, response.status));
      await loadDocuments();
      documentsStatus.textContent = "已删除 " + item.source + "，索引已同步。";
    } catch (error) {
      documentsStatus.textContent = error.message || "删除失败，请稍后重试。";
    } finally {
      setDocumentBusy(false, documentsStatus.textContent);
    }
  }

  uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = pdfFile.files[0];
    if (!file || documentBusy) return;
    setDocumentBusy(true, "正在上传并建立索引，请稍候…");
    try {
      const response = await fetch("/api/documents?filename=" + encodeURIComponent(file.name), {
        method: "POST",
        headers: { "Content-Type": "application/pdf" },
        body: file,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(errorDetail(payload, response.status));
      await loadDocuments();
      pdfFile.value = "";
      documentsStatus.textContent = "已加入 " + payload.document.source + "，可以开始提问。";
    } catch (error) {
      documentsStatus.textContent = error.message || "上传失败，请稍后重试。";
    } finally {
      setDocumentBusy(false, documentsStatus.textContent);
    }
  });

  async function checkHealth() {
    try {
      const response = await fetch("/api/health", { cache: "no-store" });
      if (!response.ok) throw new Error("health check failed");
      connectionStatus.textContent = "本地服务已连接";
      connectionStatus.classList.remove("offline");
    } catch {
      connectionStatus.textContent = "本地服务未连接";
      connectionStatus.classList.add("offline");
    }
  }

  question.addEventListener("input", () => {
    characterCount.textContent = question.value.length + " / 2000";
  });

  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.ctrlKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = question.value.trim();
    if (!text) {
      showResult("请输入研究问题。", "输入有误", true);
      question.focus();
      return;
    }

    setBusy(true);
    showResult("正在检索文档并整理回答，请稍候…", "处理中");
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(errorDetail(payload, response.status));
      if (typeof payload.answer !== "string") throw new Error("服务返回的数据格式不正确。");
      showAnswer(payload.answer);
    } catch (error) {
      showResult(error.message || "请求失败，请确认本地服务仍在运行。", "请求失败", true);
      checkHealth();
    } finally {
      setBusy(false);
    }
  });

  checkHealth();
  loadDocuments().catch((error) => {
    documentCount.textContent = "读取失败";
    documentsStatus.textContent = error.message || "无法读取文档列表。";
  });
})();
