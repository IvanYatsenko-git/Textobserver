import { app, BrowserWindow, Menu, dialog, ipcMain } from 'electron';
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs/promises';
import { spawn } from 'child_process';
import pdfParse from 'pdf-parse/lib/pdf-parse.js';
import mammoth from 'mammoth';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let mainWindow;
let pythonProcess = null;
let pythonReady = false;
let approvedDocumentPath = null;

function assertTrustedRenderer(event) {
  if (!mainWindow || event.sender !== mainWindow.webContents || event.senderFrame !== event.sender.mainFrame) {
    throw new Error('Untrusted renderer');
  }
}

function resolveApprovedPath(event, filePath, operation) {
  assertTrustedRenderer(event);
  if (typeof filePath !== 'string' || !approvedDocumentPath) {
    throw new Error('No document is open');
  }

  const requested = path.resolve(filePath);
  const documentPath = path.resolve(approvedDocumentPath);
  const documentDir = path.dirname(documentPath);
  const allowedSidecars = new Set([
    path.resolve(documentDir, '.textobserver', 'theses.json'),
    path.resolve(documentDir, '.textobserver', 'coverage.json'),
  ]);
  const allowed = operation === 'write'
    ? requested === documentPath || allowedSidecars.has(requested)
    : requested === documentPath || allowedSidecars.has(requested);

  if (!allowed) throw new Error('Path is outside the active document scope');
  return requested;
}

// ─── Python Backend ───────────────────────────────────────────────────────────

function getPythonBackendPath() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'app.asar.unpacked', 'python_backend');
  }
  // Шлях до папки з Python скриптом відносно main.js
  return path.join(__dirname, '../../python_backend');
}

function startPythonBackend() {
  const backendDir = getPythonBackendPath();
  const scriptPath = path.join(backendDir, 'app_backend.py');

  console.log('[Main] Starting Python backend:', scriptPath);

  // Спробуємо python3 спочатку, потім python
  const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';

  pythonProcess = spawn(pythonCmd, [scriptPath], {
    cwd: backendDir,          // Робоча директорія = папка з backend.py (важливо для chromadb шляху)
    env: { ...process.env, TEXT_OBSERVER_DATA_DIR: app.getPath('userData') },
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  pythonProcess.stdout.setEncoding('utf-8');
  pythonProcess.stderr.setEncoding('utf-8');
  let startupOutput = '';

  // Логуємо stderr з Python (помилки та print() для дебагу)
  pythonProcess.stderr.on('data', (data) => {
    console.log('[Python stderr]:', data.trim());
    // Перший вивід означає що модель завантажена і бекенд готовий
    startupOutput = (startupOutput + data).slice(-256);
    if (!pythonReady && startupOutput.includes('__TEXTOBSERVER_READY__')) {
      pythonReady = true;
      mainWindow?.webContents.send('python:ready');
    }
  });

  pythonProcess.on('error', (err) => {
    console.error('[Main] Failed to start Python:', err.message);
    mainWindow?.webContents.send('python:error', `Не вдалося запустити Python: ${err.message}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`[Main] Python process exited with code ${code}`);
    pythonReady = false;
    pythonProcess = null;
    isProcessingQueue = false;
    if (activePythonTask) {
      activePythonTask.reject(new Error('Python backend stopped'));
      activePythonTask = null;
    }
    while (pythonQueue.length) pythonQueue.shift().reject(new Error('Python backend stopped'));
  });
}

// ─── Python Queue System ──────────────────────────────────────────────────────

const pythonQueue = [];
let isProcessingQueue = false;
let activePythonTask = null;

function processQueue() {
  if (isProcessingQueue || pythonQueue.length === 0) return;
  isProcessingQueue = true;

  const task = pythonQueue.shift();
  activePythonTask = task;
  const request = JSON.stringify({ command: task.command, data: task.data }) + '\n';
  let buffer = '';

  const onData = (chunk) => {
    buffer += chunk;
    // Шукаємо повний JSON рядок
    const newlineIdx = buffer.indexOf('\n');
    if (newlineIdx !== -1) {
      const line = buffer.substring(0, newlineIdx).trim();
      buffer = buffer.substring(newlineIdx + 1);
      pythonProcess.stdout.removeListener('data', onData);
      clearTimeout(timeout);
      
      try {
        task.resolve(JSON.parse(line));
      } catch (e) {
        task.reject(new Error(`Не вдалося розпарсити відповідь Python: ${line}`));
      } finally {
        activePythonTask = null;
        isProcessingQueue = false;
        // Запускаємо наступне завдання у черзі, якщо воно є
        processQueue();
      }
    }
  };

  pythonProcess.stdout.on('data', onData);

  // Таймаут 5 хвилин на випадок якщо Python завис
  const timeout = setTimeout(() => {
    pythonProcess.stdout.removeListener('data', onData);
    task.reject(new Error('Python timeout (5 mins)'));
    activePythonTask = null;
    isProcessingQueue = false;
    pythonProcess.kill();
  }, 300000);

  pythonProcess.stdin.write(request);
}

// Надсилаємо JSON команду Python через чергу і повертаємо Promise з відповіддю
function sendToPython(command, data) {
  return new Promise((resolve, reject) => {
    if (!pythonProcess || !pythonReady) {
      return reject(new Error('Python backend не запущено або ще завантажується'));
    }

    // Додаємо запит у чергу
    pythonQueue.push({ command, data, resolve, reject });
    
    // Спробувати почати обробку (якщо черга була порожня)
    processQueue();
  });
}

// ─── IPC Handlers ─────────────────────────────────────────────────────────────

// Файлові операції
ipcMain.handle('file:read', async (event, filePath) => {
  return await fs.readFile(resolveApprovedPath(event, filePath, 'read'), 'utf-8');
});

ipcMain.handle('file:readDocument', async (event, filePath) => {
  filePath = resolveApprovedPath(event, filePath, 'read');
  const extension = path.extname(filePath).toLowerCase();

  if (extension === '.pdf') {
    const buffer = await fs.readFile(filePath);
    const result = await pdfParse(buffer);
    return {
      content: result.text,
      editable: false,
      format: 'pdf',
      dataBase64: buffer.toString('base64'),
    };
  }

  if (extension === '.docx') {
    const rawText = await mammoth.extractRawText({ path: filePath });
    const htmlResult = await mammoth.convertToHtml({
      path: filePath,
      convertImage: mammoth.images.imgElement((image) => image.read('base64').then((imageBuffer) => ({
        src: `data:${image.contentType};base64,${imageBuffer}`,
      }))),
    });
    return {
      content: rawText.value,
      editable: false,
      format: 'docx',
      viewerHtml: htmlResult.value,
    };
  }

  return { content: await fs.readFile(filePath, 'utf-8'), editable: true, format: 'text' };
});

ipcMain.handle('file:write', async (event, filePath, content) => {
  filePath = resolveApprovedPath(event, filePath, 'write');
  if (typeof content !== 'string') throw new Error('File content must be text');
  const dir = path.dirname(filePath);
  await fs.mkdir(dir, { recursive: true });
  await fs.writeFile(filePath, content, 'utf-8');
});

ipcMain.handle('file:getDir', async (event, filePath) => {
  return path.dirname(resolveApprovedPath(event, filePath, 'read'));
});

// Python: Індексація сегментів
ipcMain.handle('python:index', async (event, chunks, sourceFile) => {
  assertTrustedRenderer(event);
  try {
    const result = await sendToPython('index', { chunks, source_file: sourceFile });
    return result;
  } catch (err) {
    console.error('[Main] python:index error:', err.message);
    return { status: 'error', message: err.message };
  }
});

// Python: Семантичний пошук
ipcMain.handle('python:search', async (event, query, nResults) => {
  assertTrustedRenderer(event);
  try {
    const result = await sendToPython('search', { query, n_results: nResults || 5 });
    return result;
  } catch (err) {
    console.error('[Main] python:search error:', err.message);
    return { status: 'error', message: err.message };
  }
});

// Python: Очистити колекцію для нового файлу
ipcMain.handle('python:clear', async (event) => {
  assertTrustedRenderer(event);
  try {
    const result = await sendToPython('clear', {});
    return result;
  } catch (err) {
    return { status: 'error', message: err.message };
  }
});

// Python: перевірка статусу
ipcMain.handle('python:status', async (event) => {
  assertTrustedRenderer(event);
  return { ready: pythonReady };
});

// ─── Window ───────────────────────────────────────────────────────────────────

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    icon: path.join(__dirname, '../../assets/icon.png'),
    webPreferences: {
      preload: path.join(__dirname, '../preload/preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));

  mainWindow.on('closed', () => {
    mainWindow = null;
    approvedDocumentPath = null;
    // Завершуємо Python при закритті вікна
    if (pythonProcess) {
      pythonProcess.kill();
    }
  });
}

app.whenReady().then(() => {
  createWindow();
  startPythonBackend();
});

app.on('window-all-closed', () => {
  if (pythonProcess) pythonProcess.kill();
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (mainWindow === null) createWindow();
});

// ─── Menu ─────────────────────────────────────────────────────────────────────

const menu = Menu.buildFromTemplate([
  {
    label: 'Файл',
    submenu: [
      {
        label: 'Відкрити',
        accelerator: 'CmdOrCtrl+O',
        click: () => openFile(),
      },
      {
        label: 'Зберегти',
        accelerator: 'CmdOrCtrl+S',
        click: () => mainWindow?.webContents.send('file:save'),
      },
      { type: 'separator' },
      {
        label: 'Вийти',
        accelerator: 'CmdOrCtrl+Q',
        click: () => app.quit(),
      },
    ],
  },
  {
    label: 'Розробник',
    submenu: [
      {
        label: 'DevTools',
        accelerator: 'F12',
        click: () => mainWindow?.webContents.toggleDevTools(),
      },
      {
        label: 'Перезапустити Python',
        click: () => {
          if (pythonProcess) pythonProcess.kill();
          setTimeout(startPythonBackend, 500);
        },
      },
    ],
  },
]);

Menu.setApplicationMenu(menu);

async function openFile() {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Документи', extensions: ['md', 'markdown', 'txt', 'pdf', 'docx'] },
      { name: 'Усі файли', extensions: ['*'] },
    ],
  });

  if (!result.canceled && result.filePaths.length > 0) {
    approvedDocumentPath = path.resolve(result.filePaths[0]);
    mainWindow?.webContents.send('file:open', result.filePaths[0]);
  }
}
