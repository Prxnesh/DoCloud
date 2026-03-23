const appState = {
  currentDocumentId: "",
  currentFileName: "",
  latestAnalysis: null,
  chatHistory: [],
};

const els = {
  body: document.body,
  healthPill: document.getElementById("health-pill"),
  themeToggle: document.getElementById("theme-toggle"),
  uploadForm: document.getElementById("upload-form"),
  fileInput: document.getElementById("file-input"),
  fileNameDisplay: document.getElementById("file-name-display"),
  candidateLabels: document.getElementById("candidate-labels"),
  uploadFeedback: document.getElementById("upload-feedback"),
  processButton: document.getElementById("process-button"),
  metricFileName: document.getElementById("metric-file-name"),
  metricDocumentId: document.getElementById("metric-document-id"),
  classificationScore: document.getElementById("classification-score"),
  classificationLabel: document.getElementById("classification-label"),
  cvvScore: document.getElementById("cvv-score"),
  cvvBand: document.getElementById("cvv-band"),
  cvvFill: document.getElementById("cvv-fill"),
  scoreRing: document.getElementById("score-ring"),
  summaryText: document.getElementById("summary-text"),
  documentType: document.getElementById("document-type"),
  keywordCount: document.getElementById("keyword-count"),
  entityCount: document.getElementById("entity-count"),
  chunkCount: document.getElementById("chunk-count"),
  keywordBars: document.getElementById("keyword-bars"),
  entityChart: document.getElementById("entity-chart"),
  searchForm: document.getElementById("search-form"),
  searchQuery: document.getElementById("search-query"),
  searchTopK: document.getElementById("search-top-k"),
  searchDocumentId: document.getElementById("search-document-id"),
  searchResults: document.getElementById("search-results"),
  chatForm: document.getElementById("chat-form"),
  chatQuestion: document.getElementById("chat-question"),
  chatTopK: document.getElementById("chat-top-k"),
  chatDocumentId: document.getElementById("chat-document-id"),
  chatLog: document.getElementById("chat-log"),
  chatSubmitButton: document.querySelector("#chat-form button[type='submit']"),
  navLinks: Array.from(document.querySelectorAll(".sidebar-nav .nav-link")),
  dropzone: document.getElementById("dropzone"),
  resultCardTemplate: document.getElementById("result-card-template"),
};

const ENTITY_COLORS = [
  "#0f7d63",
  "#e6902e",
  "#4d8bff",
  "#d55f89",
  "#8a6cff",
  "#00a3a3",
];

function initializeApp() {
  hydrateTheme();
  bindEvents();
  initializeSidebarNavigation();
  setScoreRing(0);
  checkHealth();
}

function hydrateTheme() {
  const savedTheme = localStorage.getItem("cloudinsight-theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const shouldUseDark = savedTheme ? savedTheme === "dark" : prefersDark;
  els.body.classList.toggle("dark", shouldUseDark);
}

function bindEvents() {
  els.themeToggle.addEventListener("click", toggleTheme);
  els.fileInput.addEventListener("change", handleFileSelection);
  els.uploadForm.addEventListener("submit", handleUpload);
  els.searchForm.addEventListener("submit", handleSearch);
  els.chatForm.addEventListener("submit", handleChat);

  ["dragenter", "dragover"].forEach((eventName) => {
    els.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      els.dropzone.classList.add("dragging");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    els.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      els.dropzone.classList.remove("dragging");
    });
  });

  els.dropzone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files;
    if (!file) {
      return;
    }
    els.fileInput.files = event.dataTransfer.files;
    updateSelectedFile(file.name);
  });
}

function initializeSidebarNavigation() {
  if (!els.navLinks.length) {
    return;
  }

  els.navLinks.forEach((link) => {
    link.addEventListener("click", (event) => {
      const href = link.getAttribute("href") || "";
      if (!href.startsWith("#") || href.length <= 1) {
        return;
      }

      const target = document.querySelector(href);
      if (!target) {
        return;
      }

      event.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      setActiveSidebarLink(href);
      window.history.replaceState(null, "", href);
    });
  });

  const sectionSelectors = els.navLinks
    .map((link) => link.getAttribute("href"))
    .filter((href) => href && href.startsWith("#") && href.length > 1);

  const sectionElements = sectionSelectors
    .map((selector) => document.querySelector(selector))
    .filter(Boolean);

  if (sectionElements.length) {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

        if (visible?.target?.id) {
          setActiveSidebarLink(`#${visible.target.id}`);
        }
      },
      {
        root: null,
        rootMargin: "-25% 0px -55% 0px",
        threshold: [0.15, 0.4, 0.7],
      },
    );

    sectionElements.forEach((section) => observer.observe(section));
  }

  if (window.location.hash) {
    setActiveSidebarLink(window.location.hash);
  }
}

function setActiveSidebarLink(targetHash) {
  els.navLinks.forEach((link) => {
    const isActive = link.getAttribute("href") === targetHash;
    link.classList.toggle("active", isActive);
  });
}

function toggleTheme() {
  const isDark = !els.body.classList.contains("dark");
  els.body.classList.toggle("dark", isDark);
  localStorage.setItem("cloudinsight-theme", isDark ? "dark" : "light");
}

async function checkHealth() {
  try {
    const response = await fetch("/api/v1/health");
    if (!response.ok) {
      throw new Error("API health check failed.");
    }

    els.healthPill.classList.add("ready");
    els.healthPill.classList.remove("error");
    els.healthPill.querySelector("span:last-child").textContent = "API ready";
  } catch (error) {
    els.healthPill.classList.remove("ready");
    els.healthPill.classList.add("error");
    els.healthPill.querySelector("span:last-child").textContent = "API unavailable";
  }
}

function handleFileSelection(event) {
  const [file] = event.target.files;
  updateSelectedFile(file ? file.name : "No file selected");
}

function updateSelectedFile(name) {
  els.fileNameDisplay.textContent = name;
}

async function handleUpload(event) {
  event.preventDefault();

  const [file] = els.fileInput.files;
  if (!file) {
    setUploadFeedback("Pick a file before processing.", true);
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  if (els.candidateLabels.value.trim()) {
    formData.append("candidate_labels", els.candidateLabels.value.trim());
  }

  setButtonBusy(els.processButton, true, "Processing...");
  setUploadFeedback("Running the document pipeline...", false);

  try {
    const response = await fetch("/api/v1/documents/process", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Document processing failed.");
    }

    appState.currentDocumentId = payload.document_id;
    appState.currentFileName = payload.file_name;
    appState.latestAnalysis = payload;

    renderAnalysis(payload);
    syncDocumentFilters(payload.document_id);
    setUploadFeedback("Document indexed successfully.", false);
  } catch (error) {
    setUploadFeedback(error.message, true);
  } finally {
    setButtonBusy(els.processButton, false, "Process Document");
  }
}

function renderAnalysis(payload) {
  els.metricFileName.textContent = payload.file_name;
  els.metricDocumentId.textContent = payload.document_id;
  els.summaryText.textContent = payload.summary || "No summary generated.";
  els.documentType.textContent = payload.document_type.toUpperCase();
  els.keywordCount.textContent = String(payload.keywords.length);
  els.entityCount.textContent = String(payload.entities.length);
  els.chunkCount.textContent = String(payload.chunk_count);

  const score = Math.max(0, Math.min(1, payload.classification_score || 0));
  els.classificationScore.textContent = `${Math.round(score * 100)}%`;
  els.classificationLabel.textContent = payload.classification_label || "No label";
  setScoreRing(score);
  renderCVV(payload);

  renderKeywordBars(payload.keywords);
  renderEntityChart(payload.entities);
}

function renderCVV(payload) {
  const classification = Math.max(0, Math.min(1, payload.classification_score || 0));
  const keywordSignal = Math.min(1, (payload.keywords?.length || 0) / 12);
  const entitySignal = Math.min(1, (payload.entities?.length || 0) / 20);
  const cvv = Math.round((classification * 0.65 + keywordSignal * 0.2 + entitySignal * 0.15) * 100);

  els.cvvScore.textContent = String(cvv);
  els.cvvFill.style.width = `${cvv}%`;

  if (cvv >= 75) {
    els.cvvBand.textContent = "High confidence and coverage";
  } else if (cvv >= 45) {
    els.cvvBand.textContent = "Moderate confidence";
  } else {
    els.cvvBand.textContent = "Low confidence, review source quality";
  }
}

function setScoreRing(score) {
  const circumference = 301.59;
  const offset = circumference - circumference * score;
  els.scoreRing.style.strokeDashoffset = `${offset}`;
}

function renderKeywordBars(keywords) {
  els.keywordBars.innerHTML = "";

  if (!keywords.length) {
    els.keywordBars.className = "bar-list empty-state";
    els.keywordBars.textContent = "No keywords were returned for this document.";
    return;
  }

  els.keywordBars.className = "bar-list";
  const maxCount = keywords.length;

  keywords.slice(0, 8).forEach((keyword, index) => {
    const container = document.createElement("article");
    container.className = "bar-item";

    const intensity = Math.max(20, 100 - index * (70 / Math.max(1, maxCount - 1)));
    container.innerHTML = `
      <div class="bar-head">
        <strong>${escapeHtml(keyword)}</strong>
        <span>${Math.round(intensity)}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width: ${intensity}%"></div>
      </div>
    `;
    els.keywordBars.appendChild(container);
  });
}

function renderEntityChart(entities) {
  els.entityChart.innerHTML = "";

  if (!entities.length) {
    els.entityChart.className = "entity-chart empty-state";
    els.entityChart.textContent = "No entities were extracted for this document.";
    return;
  }

  els.entityChart.className = "entity-chart";

  const counts = entities.reduce((acc, entity) => {
    acc[entity.label] = (acc[entity.label] || 0) + 1;
    return acc;
  }, {});
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const maxValue = Math.max(...entries.map(([, count]) => count));

  entries.forEach(([label, count], index) => {
    const pill = document.createElement("article");
    pill.className = "entity-pill";
    const color = ENTITY_COLORS[index % ENTITY_COLORS.length];
    const width = `${Math.max(18, (count / maxValue) * 100)}%`;

    pill.innerHTML = `
      <span class="entity-dot" style="background:${color}"></span>
      <div class="entity-meter">
        <div class="entity-meter-fill" style="width:${width};background:${color}"></div>
      </div>
      <strong>${escapeHtml(label)} · ${count}</strong>
    `;
    els.entityChart.appendChild(pill);
  });
}

async function handleSearch(event) {
  event.preventDefault();

  const query = els.searchQuery.value.trim();
  if (!query) {
    return;
  }

  setSearchLoading(true);

  try {
    const response = await fetch("/api/v1/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: Number(els.searchTopK.value || 5),
        document_id: cleanOptional(els.searchDocumentId.value),
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Search failed.");
    }

    renderSearchResults(payload.results);
  } catch (error) {
    renderSearchError(error.message);
  }
}

function setSearchLoading(isLoading) {
  els.searchResults.className = "results-panel";
  els.searchResults.innerHTML = `<p class="empty-state">${isLoading ? "Searching indexed chunks..." : "Search results will appear here."}</p>`;
}

function renderSearchResults(results) {
  els.searchResults.innerHTML = "";

  if (!results.length) {
    els.searchResults.innerHTML = '<p class="empty-state">No matching passages found.</p>';
    return;
  }

  results.forEach((result) => {
    const node = els.resultCardTemplate.content.firstElementChild.cloneNode(true);
    node.querySelector(".result-id").textContent = result.document_id;
    node.querySelector(".result-score").textContent = `${Math.round(result.score * 100)}% match`;
    node.querySelector(".result-text").textContent = result.text;
    els.searchResults.appendChild(node);
  });
}

function renderSearchError(message) {
  els.searchResults.innerHTML = `<p class="empty-state">${escapeHtml(message)}</p>`;
}

async function handleChat(event) {
  event.preventDefault();

  const question = els.chatQuestion.value.trim();
  if (!question) {
    return;
  }

  appendMessage("user", "You", question);
  appState.chatHistory.push({ role: "user", text: question });
  appState.chatHistory = appState.chatHistory.slice(-8);
  els.chatQuestion.value = "";
  setButtonBusy(els.chatSubmitButton, true, "Thinking...");

  try {
    const response = await fetch("/api/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_k: Number(els.chatTopK.value || 5),
        document_id: cleanOptional(els.chatDocumentId.value) || appState.currentDocumentId || null,
        history: appState.chatHistory,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Chat request failed.");
    }

    appendMessage("assistant", "CloudInsight", payload.answer, payload.results);
    appState.chatHistory.push({ role: "assistant", text: payload.answer });
    appState.chatHistory = appState.chatHistory.slice(-8);
  } catch (error) {
    appendMessage("assistant", "CloudInsight", error.message);
  } finally {
    setButtonBusy(els.chatSubmitButton, false, "Ask Question");
  }
}

function appendMessage(role, speaker, text, sources = []) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  let sourceHtml = "";
  if (sources.length) {
    const compactSources = sources
      .slice(0, 2)
      .map((item) => `${escapeHtml(item.document_id)} · ${Math.round(item.score * 100)}%`)
      .join("<br>");
    sourceHtml = `<div class="message-source">${compactSources}</div>`;
  }

  wrapper.innerHTML = `
    <span class="message-role">${escapeHtml(speaker)}</span>
    <p>${escapeHtml(text)}</p>
    ${sourceHtml}
  `;

  els.chatLog.appendChild(wrapper);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

function syncDocumentFilters(documentId) {
  els.searchDocumentId.value = documentId;
  els.chatDocumentId.value = documentId;
}

function setUploadFeedback(message, isError) {
  els.uploadFeedback.textContent = message;
  els.uploadFeedback.style.color = isError ? "#d65858" : "";
}

function setButtonBusy(button, isBusy, label) {
  button.disabled = isBusy;
  button.textContent = label;
}

function cleanOptional(value) {
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

initializeApp();
