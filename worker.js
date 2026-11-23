// Web Worker: запускает Python-движок (Pyodide) прямо в браузере.
// Все файлы — статические; текст пользователя никуда не отправляется.
//
// Сообщения:  страница -> воркер: {id, op: 'init' | 'edit' | 'synonyms', payload}
//             воркер -> страница: {id, ok, data | error}  и  {progress: 'текст'}

const PYODIDE_VERSION = '314.0.8';
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

const PY_FILES = ['engine.py', 'rules.py', 'synonyms.py', 'context.py', 'webapi.py'];
const WHEELS = {
  base: ['wheels/pymorphy3-2.0.6-py3-none-any.whl', 'wheels/dawg2_python-0.9.0-py3-none-any.whl'],
  ru: ['wheels/pymorphy3_dicts_ru-2.4.417150.4580142-py2.py3-none-any.whl'],
  uk: ['wheels/pymorphy3_dicts_uk-2.4.1.1.1663094765-py2.py3-none-any.whl'],
};
// данные, которые нужны только для одного языка (грузятся при первом тексте на нём)
const DATA = {
  uk: [
    { url: 'data/th_uk_UA.dat', to: '/app/data/th_uk_UA.dat' },
    { url: 'data/cooc_uk.sqlite.gz', to: '/app/data/cooc_uk.sqlite', gunzip: true },
  ],
};

let py = null;
let webapi = null;
let booting = null;
const loadedLangs = new Map();

const say = text => postMessage({ progress: text });

async function get(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Не удалось загрузить ${label || url} (${res.status})`);
  return new Uint8Array(await res.arrayBuffer());
}

// .gz на некоторых серверах приходит уже распакованным — смотрим на сигнатуру, а не на заголовки
async function gunzip(bytes) {
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function boot() {
  booting ||= (async () => {
    say('Загружаю Python (один раз, ~10 МБ)…');
    // модульный воркер: подключаем ES-версию Pyodide через import(), importScripts здесь недоступен
    const { loadPyodide } = await import(PYODIDE_URL + 'pyodide.mjs');
    py = await loadPyodide({ indexURL: PYODIDE_URL });

    say('Загружаю движок…');
    py.FS.mkdirTree('/app/data');
    const sources = await Promise.all(PY_FILES.map(f => get(f)));
    PY_FILES.forEach((f, i) => py.FS.writeFile('/app/' + f, sources[i]));
    for (const w of WHEELS.base) py.unpackArchive(await get(w), 'wheel');
    py.runPython("import sys, importlib; sys.path.insert(0, '/app'); importlib.invalidate_caches()");
    webapi = py.pyimport('webapi');
    say('');
  })();
  return booting;
}

// словари и таблицы нужного языка — при первом обращении, дальше из памяти
function ensureLang(lang) {
  if (!loadedLangs.has(lang)) {
    loadedLangs.set(lang, (async () => {
      const wheels = WHEELS[lang] || [];
      const data = DATA[lang] || [];
      if (wheels.length || data.length) say('Загружаю словари языка (один раз, до ~25 МБ)…');
      const [w, d] = await Promise.all([
        Promise.all(wheels.map(u => get(u))),
        Promise.all(data.map(async x => {
          const bytes = await get(x.url);
          return x.gunzip ? gunzip(bytes) : bytes;
        })),
      ]);
      w.forEach(bytes => py.unpackArchive(bytes, 'wheel'));
      data.forEach((x, i) => py.FS.writeFile(x.to, d[i]));
      py.runPython('importlib.invalidate_caches()');
      say('');
    })().catch(e => { loadedLangs.delete(lang); throw e; }));
  }
  return loadedLangs.get(lang);
}

async function handle({ op, payload }) {
  await boot();
  if (op === 'init') return null;
  const text = payload && typeof payload.text === 'string' ? payload.text : '';
  const lang = webapi.language(text.trim() ? text : 'a');
  await ensureLang(lang);
  say(op === 'edit' ? 'Редактирую…' : '');
  const reply = JSON.parse(webapi[op](JSON.stringify(payload)));
  say('');
  if (!reply.ok) throw Object.assign(new Error(reply.error), { user: true });
  return reply.data;
}

self.onmessage = async ({ data: msg }) => {
  try {
    postMessage({ id: msg.id, ok: true, data: await handle(msg) });
  } catch (e) {
    say('');
    postMessage({ id: msg.id, ok: false, error: e.user ? e.message : `Не удалось запустить редактор: ${e.message || e}` });
  }
};
