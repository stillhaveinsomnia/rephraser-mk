"""Словарь синонимов: поиск по лемме и постановка синонима в ту же грамматическую форму.

Источник — тезаурусы LibreOffice в формате MyThes (data/th_*.dat). Украинский файл
основан на словарях П. Деркача и С. Караванского (автор сборки — Андрей Рысин,
лицензии GPL/LGPL/MPL). Чтобы подключить другой язык, положите его .dat в data/
и добавьте путь в FILES.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data"
FILES = {"uk": "th_uk_UA.dat", "ru": "th_ru_RU_M_aot_and_v2.dat"}

# стилистические пометы, с которыми синоним не годится для научного текста
BAD_LABEL = re.compile(r"рідше|розм|обл|заст|підсил|рідк|перен|фам|поез|образн|фольк|ірон|пестл|знев|вульг|міф|жарт|лайл|дит|діал|уроч|"
                       r"разг|прост|устар|груб|шутл|ирон|поэт|бран|жарг")
REF_LABEL = re.compile(r"^(?:див\.(?: ще)?|см\.(?: также)?)$")
RE_LABEL = re.compile(r"\(([^)]*)\)")

CASES = {"nomn", "gent", "datv", "accs", "ablt", "loct", "voct", "gen2", "acc2", "loc2"}
GENDERS = {"masc", "femn", "neut"}
VERBAL = {"pres", "past", "futr", "1per", "2per", "3per", "infn", "impr", "perf", "impf", "Impe"}


@lru_cache(maxsize=4)
def _load(lang: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(строгая таблица — только прямые синонимы без помет; полная таблица)."""
    path = DATA / FILES.get(lang, "")
    if not path.is_file():
        return {}, {}
    raw = path.read_bytes()
    encoding = raw.split(b"\n", 1)[0].decode("ascii", "ignore").strip() or "utf-8"
    lines = raw.decode(encoding, "replace").splitlines()[1:]

    plain: dict[str, list[str]] = {}    # без каких-либо помет
    noted: dict[str, list[str]] = {}    # с уточнением значения в скобках
    refs: dict[str, list[str]] = {}     # «див. X»
    head = None
    for line in lines:
        if "|" not in line:
            continue
        if not line.startswith("|") and not line.startswith("("):
            head = line.split("|", 1)[0].strip().lower()
            continue
        if head is None:
            continue
        items = line.split("|")
        # строка может начинаться с пометы значения, общей для всех её синонимов: «(в такт)|барабанити|…»
        line_noted = bool(RE_LABEL.search(items[0])) and not RE_LABEL.sub("", items[0]).strip()
        for item in items:
            labels = RE_LABEL.findall(item)
            word = RE_LABEL.sub("", item).strip().lower()
            if not word or " " in word or word == head or any(BAD_LABEL.search(l) for l in labels):
                continue
            if any(REF_LABEL.match(l.strip()) for l in labels):
                target = refs
            else:
                target = noted if labels or line_noted else plain
            bucket = target.setdefault(head, [])
            if word not in bucket:
                bucket.append(word)

    def build(sources: list[dict[str, list[str]]], loose: bool) -> dict[str, list[str]]:
        table: dict[str, list[str]] = {}

        def add(key: str, value: str) -> None:
            bucket = table.setdefault(key, [])
            if value != key and value not in bucket:
                bucket.append(value)

        for src in sources:
            for h, words in src.items():
                for w in words:
                    add(h, w)
        if loose:
            for h, targets in refs.items():       # «див. X»: сам X и его прямые синонимы
                for t in targets:
                    add(h, t)
                    for w in plain.get(t, []):
                        add(h, w)
            for h, words in list(table.items()):  # обратные связи: если A → B, то и B → A
                for w in words:
                    add(w, h)
        return table

    return build([plain], False), build([plain, noted], True)


def available(lang: str) -> bool:
    return bool(_load(lang)[1])


def _features(tag) -> tuple:
    g = set(tag.grammemes)
    return (frozenset(g & CASES), "plur" in g, frozenset(g & GENDERS), frozenset(g & VERBAL))


def _match_case(src: str, repl: str) -> str:
    return repl[:1].upper() + repl[1:] if src[:1].isupper() else repl


def suggest_pairs(word: str, lang: str, morph, limit: int = 4, strict: bool = False) -> list[tuple[str, str, str]]:
    """Синонимы слова в той же форме: (форма синонима, лемма синонима, лемма исходного слова).

    strict — только прямые синонимы без помет. Вариант отбрасывается, если форма
    исходного слова неоднозначна и синоним в этих формах выглядит по-разному
    («кров» — именительный или винительный: «руда» или «руду»?).
    """
    table = _load(lang)[0 if strict else 1]
    if not table or morph is None:
        return []
    parses = morph.parse(word)[:4]
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for lemma, pos in dict.fromkeys((p.normal_form, p.tag.POS) for p in parses):
        cands = table.get(lemma)
        if not cands:
            continue
        wants = {_features(p.tag) for p in parses if p.normal_form == lemma and p.tag.POS == pos}
        for cand in cands:
            for cp in morph.parse(cand)[:3]:
                if cp.tag.POS != pos or cp.normal_form != cand:
                    continue
                lexeme = cp.lexeme
                forms = {next((f.word for f in lexeme if _features(f.tag) == want), None) for want in wants}
                if len(forms) == 1 and None not in forms:
                    form = _match_case(word, forms.pop())
                    if form.lower() != word.lower() and form not in seen:
                        seen.add(form)
                        out.append((form, cand, lemma))
                break
            if len(out) >= limit:
                return out
        if out:
            break
    return out


def suggest(word: str, lang: str, morph, limit: int = 4, strict: bool = False) -> list[str]:
    return [form for form, _, _ in suggest_pairs(word, lang, morph, limit, strict)]
