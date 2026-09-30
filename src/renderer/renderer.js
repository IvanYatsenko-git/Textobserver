import { EditorState, StateField, StateEffect } from '@codemirror/state';
import { EditorView, lineNumbers, highlightActiveLineGutter, Decoration } from '@codemirror/view';
import { markdown } from '@codemirror/lang-markdown';
import { indentOnInput } from '@codemirror/language';
import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist/legacy/build/pdf.mjs';
import { buildSegmentDecorations, getThesisColor } from '../utils/overlay.js';
import { segmentText } from '../utils/segmentation.js';

GlobalWorkerOptions.workerSrc = './pdf.worker.mjs';

// ─── Modal ─────────────────────────────────────────────────────────────────────

function showModal(title, placeholder = '') {
  return new Promise((resolve) => {
    const modal = document.getElementById('modalOverlay');
    const modalTitle = document.getElementById('modalTitle');
    const modalInput = document.getElementById('modalInput');
    const modalOk = document.getElementById('modalOk');
    const modalCancel = document.getElementById('modalCancel');

    modalTitle.textContent = title;
    modalInput.placeholder = placeholder;
    modalInput.value = '';
    modal.style.display = 'flex';
    
    modalInput.style.pointerEvents = 'auto';
    modalInput.style.userSelect = 'text';
    modalInput.disabled = false;

    setTimeout(() => modalInput.focus(), 50);

    const cleanup = () => {
      modal.style.display = 'none';
      modalOk.removeEventListener('click', okClick);
      modalCancel.removeEventListener('click', cancelClick);
      modalInput.removeEventListener('keypress', enterKey);
    };
    const okClick = () => { cleanup(); resolve(modalInput.value.trim() || null); };
    const cancelClick = () => { cleanup(); resolve(null); };
    const enterKey = (e) => { if (e.key === 'Enter') okClick(); };

    modalOk.addEventListener('click', okClick);
    modalCancel.addEventListener('click', cancelClick);
    modalInput.addEventListener('keypress', enterKey);
  });
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function markAdjacentSegmentsUnknown(editStart, editEnd, segments) {
  const ids = new Set();
  const BUFFER = 50;
  for (const seg of segments) {
    const overlaps =
      (editStart > seg.startChar && editStart < seg.endChar) ||
      (editEnd > seg.startChar && editEnd < seg.endChar) ||
      (editStart <= seg.startChar && editEnd >= seg.startChar);
    const adjacent =
      Math.abs(editEnd - seg.startChar) < BUFFER ||
      Math.abs(editStart - seg.endChar) < BUFFER;
    if (overlaps || adjacent) ids.add(seg.id);
  }
  return [...ids];
}

function initializeSegmentCoverage(theses) {
  const cov = {};
  for (const t of theses) cov[t.id] = { status: 'unknown', score: 0 };
  return cov;
}

function getWorkingDirFromFilePath(filePath) {
  if (!filePath) return '.';
  const normalized = filePath.replace(/\\/g, '/');
  const lastSlash = normalized.lastIndexOf('/');
  return lastSlash >= 0 ? normalized.substring(0, lastSlash) : '.';
}

function documentKey(filePath) {
  let hash = 2166136261;
  for (const char of filePath.toLowerCase()) {
    hash ^= char.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

function identifySegments(text, filePath) {
  const prefix = documentKey(filePath);
  return segmentText(text).map((segment) => ({ ...segment, id: `${prefix}_${segment.id}` }));
}

function normalizeScore(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;

  // Backend may return 0..1, 0..100, or raw logits.
  if (numeric <= 1) return Math.round(numeric * 100);
  if (numeric <= 100) return Math.round(numeric);

  const sigmoid = 1 / (1 + Math.exp(-numeric));
  return Math.max(0, Math.min(100, Math.round(sigmoid * 100)));
}

function getResultScore(result) {
  if (!result || typeof result !== 'object') return 0;

  const directScore = normalizeScore(result.similarity);
  if (directScore > 0) return directScore;

  const crossScore = normalizeScore(result.cross_score);
  if (crossScore > 0) return crossScore;

  const distance = Number(result.distance);
  if (Number.isFinite(distance)) {
    return Math.max(0, Math.min(100, Math.round((1 - distance) * 100)));
  }

  return 0;
}

// ─── CodeMirror Decorations ────────────────────────────────────────────────────

const setDecorationsEffect = StateEffect.define();

const decorationsField = StateField.define({
  create() { return Decoration.none; },
  update(deco, tr) {
    for (const e of tr.effects) {
      if (e.is(setDecorationsEffect)) {
        const docLength = tr.state.doc.length;
        let safeRangeSet = e.value;
        try {
          safeRangeSet.between(0, docLength, (from, to) => {
            if (from < 0 || to > docLength || from > to) throw new RangeError("Invalid coords");
          });
          return safeRangeSet;
        } catch (err) {
          console.warn("[CodeMirror Shield] Виявлено застарілі координати покриття.");
          return Decoration.none;
        }
      }
    }
    return deco.map(tr.changes);
  },
  provide: (f) => EditorView.decorations.from(f),
});

const setSearchDecorationsEffect = StateEffect.define();

const searchDecorationsField = StateField.define({
  create() { return Decoration.none; },
  update(deco, tr) {
    for (const e of tr.effects) {
      if (e.is(setSearchDecorationsEffect)) {
        const docLength = tr.state.doc.length;
        let safeRangeSet = e.value;
        try {
          safeRangeSet.between(0, docLength, (from, to) => {
            if (from < 0 || to > docLength || from > to) throw new RangeError("Invalid coords");
          });
          return safeRangeSet;
        } catch (err) {
          console.warn("[CodeMirror Shield] Відхилено биті координати пошуку.");
          return Decoration.none;
        }
      }
    }
    return deco.map(tr.changes);
  },
  provide: (f) => EditorView.decorations.from(f),
});

// Безпечний слухач подій редактора
let editRefreshTimer = null;
let isLoadingDocument = false;
const editorUpdateListener = EditorView.updateListener.of((update) => {
  if (update.docChanged) {
    if (isLoadingDocument) return;
    // ВАЖЛИВО: Виклик dispatch під час update суворо заборонений в CM6 і крашить tile-рендерер.
    // Використовуємо setTimeout, щоб очищення відбулося в наступному циклі event loop.
    clearTimeout(editRefreshTimer);
    editRefreshTimer = setTimeout(() => {
      clearSearchHighlights();
      refreshSegmentsAfterEdit();
    }, 120);
  }
});

// ─── App State ─────────────────────────────────────────────────────────────────

let currentFilePath = null;
let currentDirPath = null;
let theses = [];
let coverage = {};
let segments = [];
let editor = null;
let lastSavedContent = '';
let currentFileEditable = true;
let currentDocumentData = null;
let currentViewMode = 'text';
let lastSearchResults = [];
let activeTheses = [];
let pythonIsReady = false;
let isIndexed = false;
let indexInProgress = false;
let indexRequested = false;

// ─── DOM Elements ──────────────────────────────────────────────────────────────

const thesesListEl = document.getElementById('thesesList');
const addThesisBtn = document.getElementById('addThesisBtn');
const resetBtn = document.getElementById('resetBtn');
const pilcrowBtn = document.getElementById('pilcrowBtn');
const debugBtn = document.getElementById('debugBtn');
const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdValueEl = document.getElementById('thresholdValue');

const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const searchCountEl = document.getElementById('searchCount');
const searchResultsEl = document.getElementById('searchResults');
const indexBtn = document.getElementById('indexBtn');
const pythonStatusEl = document.getElementById('pythonStatus');
const documentViewerEl = document.getElementById('documentViewer');
const textViewBtn = document.getElementById('textViewBtn');
const originalViewBtn = document.getElementById('originalViewBtn');
const documentModeLabel = document.getElementById('documentModeLabel');

const statusTextEl = document.getElementById('statusText');
const statusCountEl = document.getElementById('statusCount');

// ─── Settings ──────────────────────────────────────────────────────────────────

let relevanceThreshold = parseInt(localStorage.getItem('relevance-threshold') || '50', 10);
if (thresholdSlider) {
  thresholdSlider.value = relevanceThreshold;
  thresholdSlider.addEventListener('input', () => {
    relevanceThreshold = parseInt(thresholdSlider.value, 10);
    if (thresholdValueEl) thresholdValueEl.textContent = relevanceThreshold;
    localStorage.setItem('relevance-threshold', String(relevanceThreshold));
    
    updateOverlay(); 
    
    if (searchInput && searchInput.value.trim() !== '') {
      performSearch();
    }
  });
}
if (thresholdValueEl) thresholdValueEl.textContent = relevanceThreshold;

// ─── Status bar ────────────────────────────────────────────────────────────────

let statusState = 'готов';
let analyzedSegments = 0;
let inProgressCount = 0;
let debugMode = false;
let pilcrowMode = false;

function refreshSegmentsAfterEdit() {
  if (!editor || !currentFilePath) return;
  const updated = identifySegments(editor.state.doc.toString(), currentFilePath);
  const previousById = new Map(segments.map((segment) => [segment.id, segment]));
  const nextCoverage = {};
  for (const segment of updated) {
    const previous = previousById.get(segment.id);
    const sameText = previous?.text === segment.text;
    nextCoverage[segment.id] = sameText && coverage[segment.id]
      ? { ...coverage[segment.id] }
      : Object.fromEntries(theses.map((thesis) => [thesis.id, { status: 'unknown', score: 0 }]));
  }
  segments = updated;
  coverage = nextCoverage;
  isIndexed = false;
  analyzedSegments = segments.filter((segment) => {
    const statuses = Object.values(coverage[segment.id] || {});
    return statuses.length > 0 && statuses.every((status) => status.status !== 'unknown');
  }).length;
  updateOverlay();
  updateStatusBar();
  if (searchBtn) searchBtn.disabled = true;
  if (indexBtn) indexBtn.disabled = !pythonIsReady;
  showSearchInfo('Документ змінено. Переіндексуйте його для пошуку.');
}

function updateStatusBar() {
  if (statusTextEl) {
    statusTextEl.textContent = statusState;
    statusTextEl.className = 'status-text';
    if (statusState === 'помилка') statusTextEl.classList.add('error');
    else if (statusState.startsWith('аналізую')) statusTextEl.classList.add('syncing');
  }

  if (statusCountEl) {
    statusCountEl.textContent = `${analyzedSegments}/${segments.length} сегментів`;
  }
}

// ─── Python Status ─────────────────────────────────────────────────────────────

function setPythonStatus(ready, text) {
  pythonIsReady = ready;
  if (pythonStatusEl) {
    pythonStatusEl.textContent = ready ? '🟢' : '⏳';
    pythonStatusEl.title = text || (ready ? 'Python готовий' : 'Python завантажується...');
    pythonStatusEl.className = `python-status ${ready ? 'ready' : 'loading'}`;
  }
  if (searchBtn) searchBtn.disabled = !ready || !isIndexed;
  if (indexBtn && indexBtn.disabled !== undefined) indexBtn.disabled = !ready || !currentFilePath;
}

window.api.onPythonReady(() => {
  setPythonStatus(true, 'Python + ChromaDB готові');
  if (currentFilePath && segments.length > 0 && !isIndexed) {
    indexDocument();
  }
});

window.api.onPythonError((msg) => {
  setPythonStatus(false, `Помилка Python: ${msg}`);
  showSearchError(`⚠️ Python недоступний: ${msg}`);
});

// ─── Editor ────────────────────────────────────────────────────────────────────

function initEditor() {
  const editorDiv = document.getElementById('editor');
  if (!editorDiv) return;
  editor = new EditorView({
    state: EditorState.create({
      doc: '',
      extensions: [
        lineNumbers(),
        highlightActiveLineGutter(),
        indentOnInput(),
        markdown(),
        EditorView.lineWrapping,
        decorationsField,
        searchDecorationsField,
        editorUpdateListener, 
      ],
    }),
    parent: editorDiv,
  });
}

// ─── File Loading ──────────────────────────────────────────────────────────────

async function loadFile(filePath) {
  try {
    currentFilePath = filePath;
    currentDirPath = getWorkingDirFromFilePath(filePath);

    const documentData = await window.api.readDocument(filePath);
    currentDocumentData = documentData;
    const rawContent = documentData.content;
    currentFileEditable = documentData.editable;
    // Нормалізація перенесень рядків для уникнення зсувів координат
    // Додано ? для захоплення як \r\n так і одиночних \r
    const content = rawContent.replace(/\r\n?/g, '\n');
    lastSavedContent = content;

    isLoadingDocument = true;
    editor.setState(EditorState.create({
      doc: content,
      extensions: [
        lineNumbers(),
        highlightActiveLineGutter(),
        indentOnInput(),
        markdown(),
        EditorView.lineWrapping,
        decorationsField,
        searchDecorationsField,
        editorUpdateListener, 
      ],
    }));
    isLoadingDocument = false;

    console.log("[DISPATCH] editor state");

    segments = identifySegments(content, filePath);
    console.log(`Сегментовано: ${segments.length} сегментів`);

    renderDocumentViewer(documentData);

    analyzedSegments = 0;
    inProgressCount = 0;
    isIndexed = false;
    statusState = 'готов';

    clearSearchResults();
    await loadThesesAndCoverage();

    analyzedSegments = segments.filter((seg) => {
      const segCov = coverage[seg.id];
      if (!segCov) return false;
      return Object.values(segCov).every((s) => s.status !== 'uncovered');
    }).length;

    updateOverlay();
    updateStatusBar();

    if (pythonIsReady) {
      await indexDocument();
    }

    if (indexBtn) indexBtn.disabled = false;
  } catch (error) {
    console.error('Помилка завантаження файлу:', error);
  }
}

function renderDocumentViewer(documentData) {
  const hasOriginal = documentData.format === 'pdf' || documentData.format === 'docx';
  originalViewBtn.disabled = !hasOriginal;
  originalViewBtn.hidden = !hasOriginal;
  textViewBtn.hidden = !hasOriginal;
  documentModeLabel.textContent = hasOriginal ? 'Перегляд документа' : 'Текстовий режим';
  documentViewerEl.replaceChildren();

  if (documentData.format === 'pdf') {
    renderPdfViewer(documentData.dataBase64);
  } else if (documentData.format === 'docx') {
    renderSafeDocxHtml(documentData.viewerHtml || '<p>Could not display DOCX.</p>');
    mapDocxBlocksToSegments();
  }

  showTextView();
}

function renderSafeDocxHtml(html) {
  const allowedTags = new Set([
    'P', 'BR', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'STRONG', 'B', 'EM', 'I',
    'U', 'S', 'UL', 'OL', 'LI', 'BLOCKQUOTE', 'PRE', 'CODE', 'TABLE', 'THEAD',
    'TBODY', 'TR', 'TH', 'TD', 'A', 'IMG',
  ]);
  const parsed = new DOMParser().parseFromString(html, 'text/html');
  const cleanNode = (parent) => {
    for (const child of [...parent.childNodes]) {
      if (child.nodeType !== Node.ELEMENT_NODE) continue;
      if (!allowedTags.has(child.tagName)) {
        child.replaceWith(document.createTextNode(child.textContent || ''));
        continue;
      }
      const originalHref = child.tagName === 'A' ? child.getAttribute('href') : null;
      const originalSrc = child.tagName === 'IMG' ? child.getAttribute('src') : null;
      for (const attribute of [...child.attributes]) child.removeAttribute(attribute.name);
      if (child.tagName === 'IMG') {
        if (originalSrc && /^data:image\/(png|jpeg|gif|webp);base64,/i.test(originalSrc)) {
          child.setAttribute('src', originalSrc);
          child.setAttribute('alt', '');
        } else {
          child.remove();
          continue;
        }
      } else if (child.tagName === 'A' && originalHref && /^https?:\/\//i.test(originalHref)) {
        child.setAttribute('href', originalHref);
        child.setAttribute('rel', 'noreferrer noopener');
      }
      cleanNode(child);
    }
  };
  cleanNode(parsed.body);
  documentViewerEl.replaceChildren(...[...parsed.body.childNodes]);
}

function normalizedText(text) {
  return String(text || '').replace(/\s+/g, ' ').trim().toLowerCase();
}

function docxSearchTokens(text) {
  return new Set(normalizedText(text).match(/[\p{L}\p{N}]+/gu) || []);
}

function mapDocxBlocksToSegments() {
  if (!documentViewerEl || currentDocumentData?.format !== 'docx' || !segments.length) return;

  const blocks = [...documentViewerEl.children].filter((element) => normalizedText(element.textContent));
  const segmentTokens = segments.map((segment) => docxSearchTokens(segment.text));

  blocks.forEach((block) => {
    const blockText = normalizedText(block.textContent);
    const blockTokens = docxSearchTokens(blockText);
    if (!blockTokens.size) return;

    let bestIndex = 0;
    let bestScore = 0;
    segments.forEach((segment, index) => {
      const segmentText = normalizedText(segment.text);
      const matchingTokens = [...blockTokens].filter((token) => segmentTokens[index].has(token)).length;
      const overlapScore = matchingTokens / blockTokens.size;
      const containsScore = segmentText.includes(blockText) ? 1 : 0;
      const score = (containsScore * 2) + overlapScore;
      if (score > bestScore) {
        bestScore = score;
        bestIndex = index;
      }
    });

    block.dataset.segmentId = segments[bestIndex].id;
  });
}

function clearDocxSearchHighlights() {
  if (!documentViewerEl) return;
  documentViewerEl.querySelectorAll('.docx-search-hit, .docx-search-top').forEach((element) => {
    element.classList.remove('docx-search-hit', 'docx-search-top');
  });
}

function highlightDocxSearchResults(results) {
  if (!documentViewerEl || currentDocumentData?.format !== 'docx' || currentViewMode !== 'original') return;
  clearDocxSearchHighlights();
  const highlighted = new Set();

  results.forEach((result, index) => {
    if (highlighted.has(result.id)) return;
    highlighted.add(result.id);
    const block = documentViewerEl.querySelector(`[data-segment-id="${result.id}"]`);
    if (!block) return;
    block.classList.add(index === 0 ? 'docx-search-top' : 'docx-search-hit');
  });
}

function clearDocxThesisHighlights() {
  if (!documentViewerEl) return;
  documentViewerEl.querySelectorAll('[data-thesis-highlight]').forEach((element) => {
    element.removeAttribute('data-thesis-highlight');
    element.style.removeProperty('background-color');
    element.style.removeProperty('box-shadow');
  });
}

function updateDocxThesisOverlay() {
  if (!documentViewerEl || currentDocumentData?.format !== 'docx' || currentViewMode !== 'original') return;
  clearDocxThesisHighlights();
  if (!activeTheses.length) return;

  const thesisIndexMap = new Map(theses.map((thesis, index) => [thesis.id, index]));
  documentViewerEl.querySelectorAll('[data-segment-id]').forEach((block) => {
    const segmentCoverage = coverage[block.dataset.segmentId];
    if (!segmentCoverage) return;

    const relevant = activeTheses
      .map((thesisId) => ({ thesisId, status: segmentCoverage[thesisId] }))
      .filter(({ status }) => status?.status === 'covered' && Number(status.score) >= relevanceThreshold)
      .sort((left, right) => Number(right.status.score) - Number(left.status.score));

    if (!relevant.length) return;
    const primaryIndex = thesisIndexMap.get(relevant[0].thesisId);
    block.dataset.thesisHighlight = 'true';
    block.style.backgroundColor = getThesisColor(primaryIndex ?? 0);
    block.style.boxShadow = 'inset 4px 0 0 rgba(28, 154, 145, 0.75)';
  });
}

async function renderPdfViewer(dataBase64) {
  if (!dataBase64) {
    documentViewerEl.textContent = 'PDF не містить даних для перегляду.';
    return;
  }

  try {
    const binary = atob(dataBase64);
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    const pdf = await getDocument({ data: bytes, disableWorker: true }).promise;
    documentViewerEl.replaceChildren();

    for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
      const page = await pdf.getPage(pageNumber);
      const viewport = page.getViewport({ scale: 1.25 });
      const pageCanvas = document.createElement('canvas');
      pageCanvas.className = 'pdf-page-canvas';
      pageCanvas.width = Math.ceil(viewport.width);
      pageCanvas.height = Math.ceil(viewport.height);
      pageCanvas.setAttribute('aria-label', `Сторінка PDF ${pageNumber}`);
      documentViewerEl.appendChild(pageCanvas);
      await page.render({ canvasContext: pageCanvas.getContext('2d'), viewport }).promise;
    }
  } catch (error) {
    console.error('Помилка відображення PDF:', error);
    documentViewerEl.textContent = `Не вдалося відобразити PDF: ${error.message}`;
  }
}

function showTextView() {
  currentViewMode = 'text';
  document.getElementById('editor').hidden = false;
  documentViewerEl.hidden = true;
  textViewBtn.classList.add('active');
  originalViewBtn.classList.remove('active');
  updateOverlay();
  highlightSearchResults(lastSearchResults);
}

function showOriginalView() {
  if (!currentDocumentData || originalViewBtn.disabled) return;
  currentViewMode = 'original';
  document.getElementById('editor').hidden = true;
  documentViewerEl.hidden = false;
  textViewBtn.classList.remove('active');
  originalViewBtn.classList.add('active');
  updateOverlay();
  highlightSearchResults(lastSearchResults);
}

textViewBtn.addEventListener('click', showTextView);
originalViewBtn.addEventListener('click', showOriginalView);

// ─── ChromaDB Indexing ─────────────────────────────────────────────────────────

async function indexDocument() {
  if (editor && currentFilePath) refreshSegmentsAfterEdit();
  if (indexInProgress) {
    indexRequested = true;
    return;
  }
  if (!pythonIsReady || !currentFilePath || segments.length === 0) return;
  indexInProgress = true;
  const indexingPath = currentFilePath;
  const chunks = segments.map((seg) => ({ id: seg.id, text: seg.text }));

  try {
    if (indexBtn) {
      indexBtn.disabled = true;
      indexBtn.textContent = '⏳';
    }
    showSearchInfo(`Індексую ${segments.length} сегментів...`);

    const clearResult = await window.api.pythonClear();
    if (clearResult.status !== 'success') throw new Error(clearResult.message || 'Could not clear old index');
    const result = await window.api.pythonIndex(chunks, indexingPath);

    if (result.status === 'success' && currentFilePath === indexingPath) {
      isIndexed = true;
      if (searchBtn) searchBtn.disabled = false;
      showSearchInfo(`✅ Готово: ${result.indexed} сегментів проіндексовано`);
    } else {
      showSearchError(`❌ Помилка індексації: ${result.message}`);
    }
  } catch (err) {
    showSearchError(`❌ ${err.message}`);
  } finally {
    indexInProgress = false;
    if (indexRequested) {
      indexRequested = false;
      queueMicrotask(() => indexDocument());
    }
    if (indexBtn) {
      indexBtn.disabled = !pythonIsReady || !currentFilePath;
      indexBtn.textContent = '⟳ Індекс';
    }
  }
}

// ─── Semantic Search ───────────────────────────────────────────────────────────

async function performSearch() {
  const query = searchInput ? searchInput.value.trim() : '';
  
  if (!query) {
    clearSearchResults();
    return;
  }

  if (!pythonIsReady) return showSearchError('Python ще не готовий...');
  if (!isIndexed) return showSearchError('Спочатку відкрийте файл для індексації.');

  const nResultsVal = searchCountEl ? searchCountEl.value : '10';
  const nResults = nResultsVal === 'all' ? 9999 : parseInt(nResultsVal, 10);

  if (searchBtn) {
    searchBtn.disabled = true;
    searchBtn.textContent = '⏳';
  }
  showSearchInfo('Шукаю...');

  try {
    const result = await window.api.pythonSearch(query, nResults);

    if (result.status === 'error') {
      showSearchError(`❌ ${result.message}`);
      clearSearchHighlights();
      return;
    }

    let results = result.results || [];

    results = results.filter((r) => getResultScore(r) >= relevanceThreshold);
    lastSearchResults = results;

    renderSearchResults(results, query);
    highlightSearchResults(results);

    if (results.length === 0) {
      showSearchInfo(`Нічого не знайдено з порогом ${relevanceThreshold}% та вище.`);
    }
  } catch (err) {
    showSearchError(`❌ ${err.message}`);
  } finally {
    if (searchBtn) {
      searchBtn.disabled = false;
      searchBtn.textContent = '→';
    }
  }
}

// ─── Search Results UI ─────────────────────────────────────────────────────────

function renderSearchResults(results, query) {
  if (!searchResultsEl) return;
  searchResultsEl.innerHTML = '';
  if (results.length === 0) return;

  results.forEach((r, idx) => {
    const div = document.createElement('div');
    div.className = 'search-result-item';
    div.dataset.segmentId = r.id;

    const score = getResultScore(r);
    const scoreClass = score >= 70 ? 'high' : score >= 40 ? 'mid' : 'low';
    const preview = r.text.length > 120 ? r.text.substring(0, 120) + '…' : r.text;

    const header = document.createElement('div');
    header.className = 'result-header';
    const indexLabel = document.createElement('span');
    indexLabel.className = 'result-index';
    indexLabel.textContent = `#${idx + 1}`;
    const scoreLabel = document.createElement('span');
    scoreLabel.className = `result-score ${scoreClass}`;
    scoreLabel.textContent = `${score}%`;
    const idLabel = document.createElement('span');
    idLabel.className = 'result-id';
    idLabel.textContent = String(r.id ?? '');
    header.append(indexLabel, scoreLabel, idLabel);
    const text = document.createElement('div');
    text.className = 'result-text';
    text.textContent = preview;
    div.append(header, text);

    div.addEventListener('click', () => {
      scrollToSegment(r.id);
      document.querySelectorAll('.search-result-item').forEach((el) => el.classList.remove('active'));
      div.classList.add('active');
    });

    searchResultsEl.appendChild(div);
  });
}

function showSearchInfo(msg) {
  if (!searchResultsEl) return;
  const element = document.createElement('div');
  element.className = 'search-info';
  element.textContent = msg;
  searchResultsEl.replaceChildren(element);
}
function showSearchError(msg) {
  if (!searchResultsEl) return;
  const element = document.createElement('div');
  element.className = 'search-error';
  element.textContent = msg;
  searchResultsEl.replaceChildren(element);
}
function clearSearchResults() {
  lastSearchResults = [];
  if (searchResultsEl) searchResultsEl.innerHTML = '';
  clearSearchHighlights();
}

// ─── Search Highlight in Editor ────────────────────────────────────────────────

const SEARCH_HIGHLIGHT_COLOR = 'rgba(255, 200, 0, 0.45)';
const SEARCH_TOP_COLOR = 'rgba(255, 150, 0, 0.55)';

function highlightSearchResults(results) {
  highlightDocxSearchResults(results);
  if (!editor || !segments.length || currentViewMode !== 'text') return;
  const ranges = [];
  const docLength = editor.state.doc.length;

  // ВАЖЛИВО: Захист від дублювання ідентичних координат, що ламає дерево CM6
  const processedSegmentIds = new Set();

  results.forEach((r, idx) => {
    if (processedSegmentIds.has(r.id)) return;
    processedSegmentIds.add(r.id);

    const seg = segments.find((s) => s.id === r.id);
    if (!seg || seg.endChar <= seg.startChar) return;

    let from = seg.startChar;
    let to = seg.endChar;
    if (from < 0) from = 0;
    if (to > docLength) to = docLength;
    if (from >= to) return;

    const color = idx === 0 ? SEARCH_TOP_COLOR : SEARCH_HIGHLIGHT_COLOR;
    const deco = Decoration.mark({
      class: 'search-highlight',
      attributes: { style: `background-color: ${color}; border-radius: 2px;` },
    });
    ranges.push(deco.range(from, to));
  });

  ranges.sort((a, b) => a.from - b.from);
  console.log("[DISPATCH] search highlight");
  editor.dispatch({
    effects: [setSearchDecorationsEffect.of(ranges.length > 0 ? Decoration.set(ranges, true) : Decoration.none)],
  });
}

function clearSearchHighlights() {
  console.log("[DISPATCH] clear search");
  clearDocxSearchHighlights();
  if (editor) editor.dispatch({ effects: [setSearchDecorationsEffect.of(Decoration.none)] });
}

function scrollToSegment(segmentId) {
  if (currentDocumentData?.format === 'docx' && currentViewMode === 'original') {
    showOriginalView();
    const block = documentViewerEl?.querySelector(`[data-segment-id="${segmentId}"]`);
    block?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    return;
  }
  if (!editor) return;
  const seg = segments.find((s) => s.id === segmentId);
  if (!seg) return;

  const docLength = editor.state.doc.length;
  let pos = seg.startChar;
  if (pos < 0 || pos > docLength) return;

  console.log("[DISPATCH] scroll");
  editor.dispatch({ selection: { anchor: pos }, scrollIntoView: true });
  editor.focus();
}

// ─── Theses ────────────────────────────────────────────────────────────────────

async function loadThesesAndCoverage() {
  try {
    const thesesPath = `${currentDirPath}/.textobserver/theses.json`;
    const coveragePath = `${currentDirPath}/.textobserver/coverage.json`;

    try { theses = JSON.parse(await window.api.readFile(thesesPath)); } catch { theses = []; }
    try { coverage = JSON.parse(await window.api.readFile(coveragePath)); } catch { coverage = {}; }

    for (const segment of segments) {
      const legacyId = segment.id.slice(segment.id.indexOf('_') + 1);
      if (!coverage[segment.id] && coverage[legacyId]) coverage[segment.id] = coverage[legacyId];
    }

    for (const seg of segments) {
      if (!coverage[seg.id]) coverage[seg.id] = initializeSegmentCoverage(theses);
    }
    renderThesesList();
  } catch (error) { console.error('Помилка завантаження тез:', error); }
}

async function syncThesis(thesis) {
  if (!pythonIsReady || !isIndexed || !segments.length) return false;

  try {
    const result = await window.api.pythonSearch(thesis.title, segments.length);
    if (result.status !== 'success') return false;

    for (const seg of segments) {
      if (!coverage[seg.id]) coverage[seg.id] = {};
      coverage[seg.id][thesis.id] = { status: 'uncovered', score: 0 };
    }

    result.results.forEach((r) => {
      const score = getResultScore(r);
      if (coverage[r.id]) {
        coverage[r.id][thesis.id] = { status: 'covered', score };
      }
    });

    await saveCoverage();
    updateOverlay();
    return true;
  } catch (err) {
    console.error('Помилка синхронізації тези:', err);
    return false;
  }
}

function renderThesesList() {
  if (!thesesListEl) return;
  thesesListEl.innerHTML = '';
  theses.forEach((thesis, index) => {
    const item = document.createElement('div');
    item.className = 'thesis-item';
    const isChecked = activeTheses.includes(thesis.id);
    item.innerHTML = `
      <input type="checkbox" id="thesis-${index}" class="thesis-checkbox" ${isChecked ? 'checked' : ''}>
      <label for="thesis-${index}"></label>
      <button class="delete-btn" title="Видалити">✕</button>
    `;
    item.querySelector('label').textContent = thesis.title;

    item.querySelector('.thesis-checkbox').addEventListener('change', async (e) => {
      if (e.target.checked) {
        if (!activeTheses.includes(thesis.id)) activeTheses.push(thesis.id);
        await syncThesis(thesis);
      } else {
        activeTheses = activeTheses.filter((id) => id !== thesis.id);
      }
      updateOverlay();
    });

    item.querySelector('.delete-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteThesis(index);
    });

    thesesListEl.appendChild(item);
  });
}

function updateOverlay() {
  updateDocxThesisOverlay();
  if (!editor || segments.length === 0) return;

  console.log("[DISPATCH] overlay");

  const decorations = buildSegmentDecorations(
    segments,
    coverage,
    activeTheses,
    theses,
    {
      showBoundaries: debugMode,
      showPilcrows: pilcrowMode,
      threshold: relevanceThreshold,
    }
  );

  editor.dispatch({
    effects: [setDecorationsEffect.of(decorations)]
  });
}

if (addThesisBtn) {
  addThesisBtn.addEventListener('click', async () => {
    const title = await showModal('Введіть ключові слова або тезу:', 'Назва теми для пошуку');
    if (!title) return;
    const newThesis = { id: Date.now().toString(), title };
    theses.push(newThesis);
    if (!activeTheses.includes(newThesis.id)) activeTheses.push(newThesis.id);
    for (const seg of segments) {
      if (!coverage[seg.id]) coverage[seg.id] = {};
      coverage[seg.id][newThesis.id] = { status: 'uncovered', score: 0 };
    }
    await saveTheses();
    await saveCoverage();
    renderThesesList();
    updateOverlay();
    await syncThesis(newThesis);
  });
}

async function deleteThesis(index) {
  const deleted = theses[index];
  theses.splice(index, 1);
  activeTheses = activeTheses.filter((id) => id !== deleted.id);
  for (const seg of segments) {
    if (coverage[seg.id]) delete coverage[seg.id][deleted.id];
  }
  await saveTheses();
  await saveCoverage();
  renderThesesList();
  updateOverlay();
}

async function saveTheses() {
  try { await window.api.writeFile(`${currentDirPath}/.textobserver/theses.json`, JSON.stringify(theses, null, 2)); }
  catch (err) { console.error(err); }
}

async function saveCoverage() {
  try { await window.api.writeFile(`${currentDirPath}/.textobserver/coverage.json`, JSON.stringify(coverage, null, 2)); }
  catch (err) { console.error(err); }
}

// ─── File Save ─────────────────────────────────────────────────────────────────

function trackEdits() {
  const current = editor.state.doc.toString();
  if (current === lastSavedContent) return;

  let startDiff = 0, endDiff = 0;
  const minLen = Math.min(current.length, lastSavedContent.length);
  for (let i = 0; i < minLen; i++) {
    if (current[i] !== lastSavedContent[i]) { startDiff = i; break; }
  }
  for (let i = minLen - 1; i >= 0; i--) {
    if (current[i] !== lastSavedContent[i]) { endDiff = i; break; }
  }

  const unknownIds = markAdjacentSegmentsUnknown(startDiff, endDiff, segments);
  unknownIds.forEach((id) => {
    if (coverage[id]) {
      Object.keys(coverage[id]).forEach((tid) => { coverage[id][tid].status = 'uncovered'; });
    }
  });
}

async function saveFile() {
  if (!currentFilePath || !editor || !currentFileEditable) {
    if (!currentFileEditable) showSearchInfo('PDF і Word доступні для пошуку, але збереження у вихідний файл не підтримується.');
    return;
  }
  try {
    const content = editor.state.doc.toString();
    refreshSegmentsAfterEdit();
    await window.api.writeFile(currentFilePath, content);
    lastSavedContent = content;
    await saveCoverage();
  } catch (err) { console.error('Помилка збереження:', err); }
}

// ─── Reset ─────────────────────────────────────────────────────────────────────

if (resetBtn) {
  resetBtn.addEventListener('click', async () => {
    if (!currentFilePath) return;
    if (!confirm('Скинути всі результати пошуку по тезах?')) return;

    for (const seg of segments) {
      coverage[seg.id] = {};
      for (const thesis of theses) {
        coverage[seg.id][thesis.id] = { status: 'uncovered', score: 0 };
      }
    }
    analyzedSegments = 0;
    updateStatusBar();
    await saveCoverage();
    updateOverlay();
    alert(`Результати скинуто.`);
  });
}

// ─── Debug / Pilcrow ───────────────────────────────────────────────────────────

if (pilcrowBtn) {
  pilcrowBtn.addEventListener('click', () => {
    pilcrowMode = !pilcrowMode;
    pilcrowBtn.classList.toggle('active', pilcrowMode);
    updateOverlay();
  });
}

if (debugBtn) {
  debugBtn.addEventListener('click', () => {
    debugMode = !debugMode;
    debugBtn.classList.toggle('active', debugMode);
    updateOverlay();
  });
}

// ─── Search Events ─────────────────────────────────────────────────────────────

if (searchBtn) searchBtn.addEventListener('click', performSearch);
if (searchInput) {
  searchInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') performSearch(); });
  
  // Авто-очищення результатів, якщо стерти текст
  searchInput.addEventListener('input', () => {
    if (searchInput.value.trim() === '') {
      clearSearchResults();
    }
  });
}

if (indexBtn) {
  indexBtn.addEventListener('click', async () => {
    if (!currentFilePath) { showSearchError('Спочатку відкрийте файл'); return; }
    isIndexed = false;
    await indexDocument();
  });
}

// ─── File Events ───────────────────────────────────────────────────────────────

window.api.onFileOpen((filePath) => loadFile(filePath));
window.api.onSave(() => saveFile());

document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); saveFile(); }
});

// ─── Init ──────────────────────────────────────────────────────────────────────

function initializeApp() {
  initEditor();
  setPythonStatus(false, 'Python завантажується...');
}

document.addEventListener('DOMContentLoaded', initializeApp);
