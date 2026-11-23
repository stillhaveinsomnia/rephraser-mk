"""Веб-сервер академического редактора (без ИИ и без сетевых запросов). Запуск: python app.py"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import engine

ROOT = Path(__file__).parent
MAX_CHARS = 200_000

app = FastAPI(title="Academic Paraphrasing Engine")


class EditRequest(BaseModel):
    text: str = Field(max_length=MAX_CHARS)
    # исходный авторский текст — чтобы после ручного принятия предложений diff считался от него
    original: str | None = Field(None, max_length=MAX_CHARS)
    academic_level: int = Field(3, ge=1, le=5)
    complexity: Literal["simple", "medium", "high"] = "medium"
    intervention: Literal["minimal", "moderate", "deep"] = "moderate"
    preserve_style: bool = True
    synonyms: Literal["click", "suggest", "auto"] = "click"
    expand: bool = False
    # False — повторный проход после ручной правки: новые автозамены синонимами не делаются
    fresh: bool = True


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/ping")
def ping() -> dict:
    return {"ok": True}


@app.post("/api/edit")
def edit(req: EditRequest) -> dict:
    settings = engine.Settings(req.academic_level, req.complexity, req.intervention, req.preserve_style, req.synonyms, req.expand)
    try:
        return engine.edit_text(req.text, settings, req.original, req.fresh)
    except engine.EngineError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


class SynonymRequest(BaseModel):
    text: str = Field(max_length=MAX_CHARS)
    start: int = Field(ge=0)
    end: int = Field(gt=0)


@app.post("/api/synonyms")
def word_synonyms(req: SynonymRequest) -> dict:
    try:
        return engine.synonyms_at(req.text, req.start, req.end)
    except engine.EngineError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8765")))
