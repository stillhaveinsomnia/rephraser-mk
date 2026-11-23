"""Строит таблицу сочетаемости слов из корпусов Лейпцигского университета.

Запуск (один раз, несколько минут):  python build_stats.py
Вход:  data/corpus/*.tar.gz  (например, ukr_wikipedia_2021_1M.tar.gz, ukr_news_2020_1M.tar.gz)
Выход: data/cooc_uk.sqlite — частоты лемм и пар «синоним из словаря — соседнее слово».

Это обычная корпусная статистика (подсчёт совместной встречаемости), не нейросеть.
Архивы не распаковываются на диск: предложения читаются потоком.
"""

from __future__ import annotations

import sqlite3
import sys
import tarfile
import time
from pathlib import Path

import pymorphy3

import synonyms
from context import SKIP_POS, WINDOW, RE_CWORD

LANG = "uk"
ROOT = Path(__file__).parent
CORPUS = ROOT / "data" / "corpus"
OUT = ROOT / "data" / f"cooc_{LANG}.sqlite"
SHIFT = 22
PRUNE_AT = 14_000_000
MIN_PAIR = 3


def main() -> None:
    archives = sorted(CORPUS.glob("*.tar.gz"))
    if not archives:
        sys.exit(f"Нет архивов в {CORPUS}")
    morph = pymorphy3.MorphAnalyzer(lang=LANG)
    full = synonyms._load(LANG)[1]
    vocab = set(full) | {w for ws in full.values() for w in ws}
    print(f"слов в словаре синонимов: {len(vocab)}")

    ids: dict[str, int] = {}
    names: list[str] = []
    in_vocab: list[bool] = []
    forms: dict[str, int] = {}      # словоформа -> id леммы, -1 = служебное слово
    uni: dict[int, int] = {}
    pairs: dict[int, int] = {}
    total = sentences = 0
    t0 = time.time()

    def lemma_id(word: str) -> int:
        p = morph.parse(word)[0]
        if str(p.tag.POS) in SKIP_POS:
            return -1
        lemma = p.normal_form
        i = ids.get(lemma)
        if i is None:
            i = ids[lemma] = len(names)
            names.append(lemma)
            in_vocab.append(lemma in vocab)
        return i

    for archive in archives:
        print(f"читаю {archive.name}")
        with tarfile.open(archive, "r:gz") as tar:
            member = next(m for m in tar if m.name.endswith("-sentences.txt"))
            stream = tar.extractfile(member)
            for raw in stream:
                line = raw.decode("utf-8", "replace").split("\t", 1)[-1].lower()
                seq = []
                for w in RE_CWORD.findall(line):
                    i = forms.get(w)
                    if i is None:
                        i = forms[w] = lemma_id(w)
                    if i >= 0:
                        seq.append(i)
                        uni[i] = uni.get(i, 0) + 1
                total += len(seq)
                n = len(seq)
                for a in range(n):
                    x = seq[a]
                    for b in range(a + 1, min(a + 1 + WINDOW, n)):
                        y = seq[b]
                        if x == y:
                            continue
                        if in_vocab[x]:
                            k = (x << SHIFT) | y
                            pairs[k] = pairs.get(k, 0) + 1
                        if in_vocab[y]:
                            k = (y << SHIFT) | x
                            pairs[k] = pairs.get(k, 0) + 1
                sentences += 1
                if sentences % 100_000 == 0:
                    print(f"  {sentences:>9} предложений, {len(pairs):>9} пар, {time.time() - t0:.0f} с", flush=True)
                if len(pairs) > PRUNE_AT:   # экономия памяти: единичные пары отбрасываются
                    pairs = {k: v for k, v in pairs.items() if v > 1}
                    print(f"  очистка: осталось {len(pairs)} пар", flush=True)

    print(f"записываю {OUT.name}")
    OUT.unlink(missing_ok=True)
    db = sqlite3.connect(OUT)
    db.executescript("""
        CREATE TABLE meta(key TEXT PRIMARY KEY, value INTEGER);
        CREATE TABLE uni(lemma TEXT PRIMARY KEY, cnt INTEGER) WITHOUT ROWID;
        CREATE TABLE pair(c TEXT, n TEXT, cnt INTEGER, PRIMARY KEY(c, n)) WITHOUT ROWID;
    """)
    db.execute("INSERT INTO meta VALUES('total', ?)", (total,))
    db.execute("INSERT INTO meta VALUES('sentences', ?)", (sentences,))
    db.executemany("INSERT INTO uni VALUES(?, ?)", ((names[i], c) for i, c in uni.items() if c >= 2))
    mask = (1 << SHIFT) - 1
    db.executemany("INSERT INTO pair VALUES(?, ?, ?)",
                   ((names[k >> SHIFT], names[k & mask], v) for k, v in pairs.items() if v >= MIN_PAIR))
    db.commit()
    kept = db.execute("SELECT count(*) FROM pair").fetchone()[0]
    db.close()
    print(f"готово: {sentences} предложений, {total} слов, {kept} пар, {OUT.stat().st_size / 1e6:.0f} МБ, "
          f"{time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
