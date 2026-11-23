"""Подбор синонима по контексту на основе корпусной статистики (data/cooc_<lang>.sqlite).

Идея: слово подходит к месту, если в большом корпусе оно действительно встречается
рядом с теми же соседями. Мера — PMI (поточечная взаимная информация) пары
«кандидат — сосед», суммированная по соседним знаменательным словам.
Таблицу строит build_stats.py. Без неё модуль сообщает, что статистики нет.
"""

from __future__ import annotations

import math
import re
import sqlite3
import threading
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data"

RE_CWORD = re.compile(r"[^\W\d_]+(?:['’ʼ-][^\W\d_]+)*", re.UNICODE)
SKIP_POS = {"PREP", "CONJ", "PRCL", "INTJ", "NPRO", "NUMR"}   # служебные слова в контекст не входят
WINDOW = 2          # сколько знаменательных слов слева и справа учитывается
MIN_PAIR = 3        # пара считается подтверждённой, если встретилась не меньше 3 раз

_lock = threading.Lock()


@lru_cache(maxsize=4)
def _db(lang: str):
    path = DATA / f"cooc_{lang}.sqlite"
    if not path.is_file():
        return None
    db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    total = db.execute("SELECT value FROM meta WHERE key='total'").fetchone()[0]
    return db, total


def available(lang: str) -> bool:
    return _db(lang) is not None


@lru_cache(maxsize=200_000)
def freq(lemma: str, lang: str) -> int:
    handle = _db(lang)
    if handle is None:
        return 0
    with _lock:
        row = handle[0].execute("SELECT cnt FROM uni WHERE lemma=?", (lemma,)).fetchone()
    return row[0] if row else 0


@lru_cache(maxsize=500_000)
def pmi(cand: str, neighbour: str, lang: str) -> float:
    """Положительная PMI пары; 0, если пара в корпусе не подтверждена."""
    handle = _db(lang)
    if handle is None:
        return 0.0
    with _lock:
        row = handle[0].execute("SELECT cnt FROM pair WHERE c=? AND n=?", (cand, neighbour)).fetchone()
    if not row or row[0] < MIN_PAIR:
        return 0.0
    fc, fn = freq(cand, lang), freq(neighbour, lang)
    if not fc or not fn:
        return 0.0
    return max(0.0, math.log(row[0] * handle[1] / (fc * fn)))


def score(cand: str, neighbours: list[str], lang: str) -> tuple[float, int]:
    """(сумма PMI по соседям, число соседей, с которыми кандидат реально встречается)."""
    values = [pmi(cand, n, lang) for n in neighbours if n != cand]
    return sum(values), sum(1 for v in values if v > 0)
