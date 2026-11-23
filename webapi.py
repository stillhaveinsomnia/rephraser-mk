"""Мост между браузером и движком: те же запросы и ответы, что у app.py, но без сервера.

Работает внутри Pyodide (Web Worker из worker.js). Вход и выход — JSON-строки:
  {"ok": true, "data": {...}}  или  {"ok": false, "error": "текст для пользователя"}
"""

from __future__ import annotations

import json

import engine

MAX_CHARS = 200_000
COMPLEXITY = ("simple", "medium", "high")
INTERVENTION = ("minimal", "moderate", "deep")
SYNONYMS = ("click", "suggest", "auto")
BAD = "Недопустимый запрос."


def _text(d: dict, key: str = "text", optional: bool = False) -> str | None:
    value = d.get(key)
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise engine.EngineError(BAD)
    if len(value) > MAX_CHARS:
        raise engine.EngineError("Текст слишком длинный: не более 200 000 знаков.")
    return value


def _int(d: dict, key: str, default=None, low: int = 0, high: int = 10**9) -> int:
    value = d.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise engine.EngineError(BAD)
    return value


def _choice(d: dict, key: str, options: tuple[str, ...], default: str) -> str:
    value = d.get(key, default)
    if value not in options:
        raise engine.EngineError(BAD)
    return value


def _reply(fn) -> str:
    try:
        return json.dumps({"ok": True, "data": fn()}, ensure_ascii=False)
    except engine.EngineError as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    except Exception as e:  # непредвиденная ошибка не должна ронять воркер
        return json.dumps({"ok": False, "error": f"Внутренняя ошибка: {type(e).__name__}: {e}"}, ensure_ascii=False)


def language(text: str) -> str:
    """Код языка текста (ru / uk / en) — по нему воркер подгружает нужные словари."""
    return engine.detect_language(text)


def edit(payload: str) -> str:
    def run():
        d = json.loads(payload)
        settings = engine.Settings(
            _int(d, "academic_level", 3, 1, 5),
            _choice(d, "complexity", COMPLEXITY, "medium"),
            _choice(d, "intervention", INTERVENTION, "moderate"),
            bool(d.get("preserve_style", True)),
            _choice(d, "synonyms", SYNONYMS, "click"),
            bool(d.get("expand", False)),
        )
        return engine.edit_text(_text(d), settings, _text(d, "original", True), bool(d.get("fresh", True)))

    return _reply(run)


def synonyms(payload: str) -> str:
    def run():
        d = json.loads(payload)
        text = _text(d)
        start, end = _int(d, "start", None, 0), _int(d, "end", None, 1)
        if end <= start or end > len(text):
            raise engine.EngineError(BAD)
        return engine.synonyms_at(text, start, end)

    return _reply(run)
