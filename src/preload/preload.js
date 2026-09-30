const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  // ─── Файлові операції ─────────────────────────────────────────────────────
  readFile: (filePath) => ipcRenderer.invoke('file:read', filePath),
  readDocument: (filePath) => ipcRenderer.invoke('file:readDocument', filePath),
  writeFile: (filePath, content) => ipcRenderer.invoke('file:write', filePath, content),
  getDir: (filePath) => ipcRenderer.invoke('file:getDir', filePath),

  // ─── Події від main process ───────────────────────────────────────────────
  onFileOpen: (callback) => ipcRenderer.on('file:open', (event, filePath) => callback(filePath)),
  onSave: (callback) => ipcRenderer.on('file:save', () => callback()),
  onPythonReady: (callback) => ipcRenderer.on('python:ready', () => callback()),
  onPythonError: (callback) => ipcRenderer.on('python:error', (event, msg) => callback(msg)),

  // ─── Python Backend ───────────────────────────────────────────────────────
  // Індексувати сегменти в ChromaDB
  pythonIndex: (chunks, sourceFile) => ipcRenderer.invoke('python:index', chunks, sourceFile),
  // Семантичний пошук
  pythonSearch: (query, nResults) => ipcRenderer.invoke('python:search', query, nResults),
  // Очистити колекцію
  pythonClear: () => ipcRenderer.invoke('python:clear'),
  // Перевірити статус
  pythonStatus: () => ipcRenderer.invoke('python:status'),
});
