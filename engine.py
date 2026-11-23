"""Academic Paraphrasing Engine — академический редактор на правилах, без ИИ.

Никаких языковых моделей и сетевых запросов: только регулярные выражения,
словари правил (rules.py) и офлайн-морфология pymorphy3.

Конвейер:
  1. определить язык; выделить «защищённые» участки (ссылки, URL, DOI, цитаты);
  2. найти научные термины (частотные существительные, аббревиатуры);
  3. автоматически применить правки, разрешённые выбранной степенью вмешательства;
  4. собрать предложения для автора: повторы, длинные предложения, опечатки, регистр;
  5. сверить числа и ссылки с оригиналом, посчитать пословный diff.
"""

from __future__ import annotations

import bisect
import difflib
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import context
import synonyms
from rules import ALPHABET, EXPAND, RULES, STOP, Rule

try:
    import pymorphy3
except ImportError:  # без морфологии работают все правила, кроме повторов и опечаток
    pymorphy3 = None

# ------------------------------------------------------ защищённые элементы

RE_URL = re.compile(r"https?://[^\s<>\"')\]]+|\bdoi:\s*10\.\d{4,9}/[^\s\"<>]+|\b10\.\d{4,9}/[^\s\"<>]+", re.I)
RE_BRACKET = re.compile(r"\[[^\[\]\n]{0,80}\d[^\[\]\n]{0,80}\]|\[\d+\]")
RE_PAREN = re.compile(r"\([^()\n]{0,90}\b(?:1[5-9]|20)\d{2}[a-zа-яіїє]?\b[^()\n]{0,40}\)")
RE_QUOTE = re.compile(r"«[^»\n]{1,600}»|“[^”\n]{1,600}”|„[^“\n]{1,600}“|\"[^\"\n]{1,600}\"")
RE_NUMBER = re.compile(r"\d+(?:[.,]\d+|[  ]\d{3}(?!\d))*")
RE_WORD = re.compile(r"[^\W\d_]+(?:['’ʼ-][^\W\d_]+)*", re.UNICODE)
RE_ANYWORD = re.compile(r"\w+", re.UNICODE)
RE_TOKEN = re.compile(r"\w+|\s+|[^\w\s]", re.UNICODE)
RE_SENT_END = re.compile(r"(?<![\s(][A-ZА-ЯІЇЄҐЁ]\.)(?<=[.!?…])\s+(?=[«\"“(]?[A-ZА-ЯІЇЄҐЁ])")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace(" ", " ")).strip().lower()


def protected_items(text: str) -> dict[str, Counter]:
    urls = Counter(_norm(m.rstrip(".,;")) for m in RE_URL.findall(text))
    no_urls = RE_URL.sub(" ", text)
    return {
        "urls": urls,
        "refs": Counter(_norm(m) for m in RE_BRACKET.findall(no_urls))
        + Counter(_norm(m) for m in RE_PAREN.findall(no_urls)),
        "numbers": Counter(re.sub(r"[\s ]", "", m) for m in RE_NUMBER.findall(no_urls)),
    }


LABELS = {"urls": "URL/DOI", "refs": "ссылки на источники", "numbers": "числа"}


def compare_protected(original: str, edited: str) -> tuple[list[str], list[str]]:
    a, b = protected_items(original), protected_items(edited)
    missing, added = [], []
    for kind, label in LABELS.items():
        lost, new = a[kind] - b[kind], b[kind] - a[kind]
        if lost:
            missing.append(f"{label}: " + ", ".join(sorted(lost.elements())))
        if new:
            added.append(f"{label}: " + ", ".join(sorted(new.elements())))
    return missing, added


def protected_spans(text: str) -> tuple[Spans, Spans]:
    """(жёсткие участки — ссылки и адреса; цитаты — их правит только типографика)."""
    hard = [m.span() for rx in (RE_URL, RE_BRACKET, RE_PAREN) for m in rx.finditer(text)]
    quotes = [m.span() for m in RE_QUOTE.finditer(text)]
    return Spans(hard), Spans(quotes)


class Spans:
    """Набор интервалов с проверкой пересечения за O(log n)."""

    def __init__(self, spans=()):
        self.items = sorted(spans)
        self.starts = [a for a, _ in self.items]
        self.max_end, top = [], -1
        for _, b in self.items:
            top = max(top, b)
            self.max_end.append(top)

    def hit(self, s: int, e: int) -> bool:
        i = bisect.bisect_left(self.starts, e)
        return i > 0 and self.max_end[i - 1] > s


class Taken:
    """Растущий набор непересекающихся интервалов."""

    def __init__(self):
        self.items: list[tuple[int, int]] = []

    def add(self, s: int, e: int) -> bool:
        i = bisect.bisect_left(self.items, (s, e))
        if (i > 0 and self.items[i - 1][1] > s) or (i < len(self.items) and self.items[i][0] < e):
            return False
        self.items.insert(i, (s, e))
        return True


# ------------------------------------------------------------- язык и леммы

def detect_language(text: str) -> str:
    low = text.lower()
    cyr = len(re.findall(r"[а-яёіїєґ]", low))
    lat = len(re.findall(r"[a-z]", low))
    if lat > cyr:
        return "en"
    uk = len(re.findall(r"[іїєґ]", low))
    ru = len(re.findall(r"[ыэъё]", low))
    return "uk" if uk > ru else "ru"


@lru_cache(maxsize=2)
def _morph(lang: str):
    if pymorphy3 is None or lang not in ("ru", "uk"):
        return None
    try:
        return pymorphy3.MorphAnalyzer(lang=lang)
    except Exception:
        return None


@lru_cache(maxsize=50_000)
def _parse(word: str, lang: str) -> tuple[str, str, bool]:
    """(лемма, часть речи, есть ли слово в словаре)."""
    m = _morph(lang)
    if m is None:
        w = word.lower()
        if lang == "en":
            w = re.sub(r"(?<=\w{3})(?:ing|ed|es|s)$", "", w)
        return w, "", True
    p = m.parse(word)[0]
    return p.normal_form, str(p.tag.POS or ""), m.word_is_known(word)


def _spell_candidates(word: str, lang: str) -> list[str]:
    m, abc = _morph(lang), ALPHABET.get(lang)
    if m is None or not abc:
        return []
    w, seen = word.lower(), []
    splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    # пропущенная буква — самая частая опечатка, поэтому вставки проверяются первыми
    cands = [a + c + b for a, b in splits for c in abc]
    cands += [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    cands += [a + c + b[1:] for a, b in splits if b for c in abc]
    cands += [a + b[1:] for a, b in splits if b]
    for c in cands:
        if c != w and c not in seen and len(c) > 2 and m.word_is_known(c):
            seen.append(c)
            if len(seen) == 3:
                break
    return seen


# -------------------------------------------------------------- типографика

def _typo_rules(lang: str) -> list[Rule]:
    t = "Типографика"
    rules = [
        (r"(?<=\S) {2,}(?=\S)", " ", "Лишние пробелы", 0),
        (r"(?<=[^\W\d_])[ \t]+(?=[,;:!?»)\.])", "", "Пробел перед знаком препинания", 0),
        (r"(?<=[^\W\d_])([,;:!?])(?=[^\W\d_])", r"\1 ", "Нет пробела после знака препинания", 0),
        (r"(?<=[а-яёіїєґa-z]{2})\.(?=[А-ЯЁІЇЄҐA-Z][а-яёіїєґa-z])", ". ", "Нет пробела после точки", 0),
        (r"(?<=[)\]»])\.(?=[А-ЯЁІЇЄҐA-Z][а-яёіїєґa-z])", ". ", "Нет пробела после точки", 0),
        (r",{2,}", ",", "Двойная запятая", 0),
    ]
    if lang in ("ru", "uk"):
        rules += [
            (r"(?<=[^\W\d_]{2}) - (?=[^\W\d_]|[«\"„“])", " — ", "Дефис вместо тире", 0),
            (r"\"(?=\S)([^\"\n]{1,300}?)(?<=\S)\"", r"«\1»", "Кавычки-«ёлочки»", 0),
            (r"(?<![\w-])([^\W\d_]+)\s+\1(?![\w-])", r"\1", "Слово повторено дважды подряд", re.I),
        ]
    return [Rule(re.compile(p, flags | re.U), (r,), "typo", m or t) for p, r, m, flags in rules]


# ------------------------------------------------------------------- правки

@dataclass
class Fix:
    start: int
    end: int
    kind: str
    message: str
    replacements: list[str]
    auto: bool = False


@dataclass
class Settings:
    academic_level: int = 3
    complexity: str = "medium"
    intervention: str = "moderate"
    preserve_style: bool = True
    synonyms: str = "click"   # click — только по клику; suggest — предлагать по тексту; auto — заменять
    expand: bool = False      # развёрнутые обороты вместо коротких союзов


AUTO_KINDS = {
    "minimal": {"typo", "spell"},
    "moderate": {"typo", "spell", "error", "filler"},
    "deep": {"typo", "spell", "error", "filler", "register"},
}
LONG_SENTENCE = {"simple": 25, "medium": 35, "high": 50}
SENT_START = ".!?…\n"


def _match_case(src: str, repl: str) -> str:
    if src[:1].isupper() and repl[:1].islower():
        return repl[:1].upper() + repl[1:]
    return repl


def _removal(text: str, s: int, e: int) -> tuple[int, int, str]:
    """Границы удаления фразы вместе с обособляющими её запятыми."""
    ls = s
    while ls > 0 and text[ls - 1] in " \t":
        ls -= 1
    rs = e
    while rs < len(text) and text[rs] in " \t":
        rs += 1
    left_comma = ls > 0 and text[ls - 1] == ","
    right_comma = rs < len(text) and text[rs] == ","
    if ls == 0 or text[ls - 1] in SENT_START:  # начало предложения: следующая буква станет заглавной
        end = rs + 1 if right_comma else rs
        while end < len(text) and text[end] in " \t":
            end += 1
        if end < len(text) and text[end].isalpha():
            return s, end + 1, text[end].upper()
        return s, end, ""
    if left_comma and right_comma:
        return ls - 1, rs + 1, ""
    if left_comma and rs < len(text) and text[rs] in ".!?;:)":
        return ls - 1, e, ""
    if right_comma:
        return ls, e, ""
    return s, rs, ""


def _collect(text: str, rules: list[Rule], settings: Settings, auto_kinds: set[str]) -> list[Fix]:
    hard, quotes = protected_spans(text)
    found: list[Fix] = []
    for rule in rules:
        if rule.level > settings.academic_level:
            continue
        for m in rule.pattern.finditer(text):
            s, e = m.span()
            if s == e or hard.hit(s, e) or (rule.kind != "typo" and quotes.hit(s, e)):
                continue
            repls = []
            for r in rule.repls:
                if r == "" and rule.kind != "typo":
                    s, e, rep = _removal(text, m.start(), m.end())
                    repls.append(rep)
                else:
                    rep = m.expand(r)
                    repls.append(rep if rule.kind == "typo" else _match_case(m.group(), rep))
            auto = rule.kind in auto_kinds and len(repls) >= 1 and (rule.kind == "register" or len(repls) == 1)
            if rule.kind == "register" and settings.preserve_style:
                auto = False
            found.append(Fix(s, e, rule.kind, rule.msg, repls, auto))
    # при пересечении побеждает более длинное совпадение
    found.sort(key=lambda f: (-(f.end - f.start), f.start))
    taken = Taken()
    return sorted((f for f in found if taken.add(f.start, f.end)), key=lambda f: f.start)


def apply_auto(text: str, lang: str, settings: Settings) -> tuple[str, list[dict]]:
    rules = _typo_rules(lang) + RULES[lang]
    auto_kinds = AUTO_KINDS[settings.intervention]
    applied: list[dict] = []
    for _ in range(5):
        fixes = [f for f in _collect(text, rules, settings, auto_kinds) if f.auto]
        if not fixes:
            break
        for f in reversed(fixes):
            before, after = text[f.start:f.end], f.replacements[0]
            if before == after:
                continue
            applied.append({"kind": f.kind, "message": f.message, "before": before, "after": after})
            text = text[:f.start] + after + text[f.end:]
    return text, applied


# ------------------------------------------------------------------- анализ

def sentences(text: str) -> list[tuple[int, int]]:
    spans = []
    for line in re.finditer(r"[^\n]+", text):
        pos = line.start()
        for m in RE_SENT_END.finditer(line.group()):
            spans.append((pos, line.start() + m.start()))
            pos = line.start() + m.end()
        spans.append((pos, line.end()))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def find_terms(text: str, lang: str) -> tuple[list[str], set[str]]:
    """Термины: частотные существительные, аббревиатуры, латиница в кириллическом тексте."""
    words = RE_WORD.findall(RE_URL.sub(" ", text))
    threshold = max(3, len(words) // 150)
    freq: Counter = Counter()
    for w in words:
        if len(w) < 4:
            continue
        lemma, pos, _ = _parse(w.lower(), lang)
        if pos in ("NOUN", "") and lemma not in STOP[lang]:
            freq[lemma] += 1
    lemmas = {l for l, n in freq.items() if n >= threshold}
    shown = [l for l, n in freq.most_common(15) if n >= threshold]
    abbr = Counter(w for w in words if len(w) >= 2 and w.isupper())
    shown += [w for w, _ in abbr.most_common(8)]
    if lang != "en":
        latin = Counter(w for w in words if len(w) >= 3 and re.fullmatch(r"[A-Za-z-]+", w))
        shown += [w for w, _ in latin.most_common(8) if w not in shown]
    return shown, lemmas


def analyse(text: str, lang: str, settings: Settings, term_lemmas: set[str]) -> list[Fix]:
    hard, quotes = protected_spans(text)
    blocked = Spans(hard.items + quotes.items)
    out: list[Fix] = []
    limit = LONG_SENTENCE[settings.complexity]
    word_freq = Counter(w.lower() for w in RE_WORD.findall(text))
    clause = {"ru": r"\bкотор\w+|\bчто\b", "uk": r"\bяк(?:ий|а|е|і|ого|ої|ому|им|ою|их|ими)\b|\bщо\b",
              "en": r"\bwhich\b|\bthat\b"}[lang]

    for s, e in sentences(text):
        sent = text[s:e]
        words = [(m.start() + s, m.end() + s, m.group()) for m in RE_WORD.finditer(sent)]
        if not words:
            continue
        first = words[0]
        if len(words) > limit:
            out.append(Fix(first[0], first[1], "long",
                           f"Предложение из {len(words)} слов (порог — {limit}): подумайте, не разделить ли его", []))
        elif len(re.findall(clause, sent, re.I)) >= 4:
            out.append(Fix(first[0], first[1], "long",
                           "Нагромождение придаточных в одном предложении — его трудно читать", []))

        seen: dict[str, str] = {}
        for ws, we, w in words:
            if blocked.hit(ws, we):
                continue
            low = w.lower()
            lemma, pos, known = _parse(low, lang)
            # возможная опечатка: слова нет в словаре, оно встречается один раз и не похоже на имя
            if (not known and len(w) >= 5 and w.islower() and word_freq[low] == 1
                    and "-" not in w and lang in ALPHABET
                    and re.fullmatch(rf"[{ALPHABET[lang]}’ʼ]+", low)):
                out.append(Fix(ws, we, "typo?", "Слова нет в словаре — проверьте написание (возможно, это термин)",
                               [_match_case(w, c) for c in _spell_candidates(low, lang)]))
                continue
            if len(w) < 4 or lemma in STOP[lang] or lemma in term_lemmas:
                continue
            if _morph(lang) and pos not in ("NOUN", "ADJF", "ADJS", "VERB", "INFN", "ADVB", "PRTF", "GRND"):
                continue
            if lemma in seen:
                syns = synonyms.suggest(w, lang, _morph(lang))
                hint = " — синонимы из словаря, проверьте оттенок значения" if syns else ""
                out.append(Fix(ws, we, "repeat", f"Повтор: «{seen[lemma]}» уже есть в этом предложении{hint}", syns))
            else:
                seen[lemma] = w

    for line in re.finditer(r"[^\n]+", text):
        para = line.group()
        for a, b, name in (("(", ")", "скобки"), ("«", "»", "кавычки")):
            if para.count(a) != para.count(b):
                idx = line.start() + (para.rfind(a) if para.count(a) > para.count(b) else para.find(b))
                out.append(Fix(idx, idx + 1, "pair", f"Непарные {name} в абзаце", []))
    return out


# ----------------------------------------------------------------- синонимы

SYN_POS = {"NOUN", "ADJF", "VERB", "ADVB"}
# Пороги автозамены по степени вмешательства:
# (сколько соседних слов должны подтверждать синоним, какую долю оценки исходного слова он должен набрать)
AUTO_SYN = {"minimal": (2, 1.0), "moderate": (1, 0.7), "deep": (1, 0.35)}
MIN_FREQ = 30   # синоним должен быть употребительным: не реже 30 раз в корпусе


@lru_cache(maxsize=50_000)
def _pairs(word: str, lang: str, strict: bool = False) -> tuple[tuple[str, str, str], ...]:
    return tuple(synonyms.suggest_pairs(word, lang, _morph(lang), limit=10, strict=strict))


def _synonyms(word: str, lang: str) -> tuple[str, ...]:
    return tuple(form for form, _, _ in _pairs(word, lang))


def _content(text: str, s: int, e: int, lang: str) -> list[tuple[int, int, str, str, str]]:
    """Знаменательные слова отрезка: (начало, конец, слово, лемма, часть речи)."""
    out = []
    for m in RE_WORD.finditer(text, s, e):
        lemma, pos, _ = _parse(m.group().lower(), lang)
        if pos not in context.SKIP_POS:
            out.append((m.start(), m.end(), m.group(), lemma, pos))
    return out


def _neighbours(content: list, k: int) -> list[str]:
    w = context.WINDOW
    return [c[3] for c in content[max(0, k - w):k] + content[k + 1:k + 1 + w]]


def rank(word: str, neighbours: list[str], lang: str, strict: bool = False) -> tuple[list[tuple[str, float, int, int]], float]:
    """Синонимы слова, упорядоченные по соответствию контексту.

    Возвращает ([(форма, оценка, число подтверждающих соседей, частота в корпусе)], оценка исходного слова).
    Без корпусной статистики порядок словарный, оценки нулевые.
    """
    pairs = _pairs(word.lower(), lang, strict)
    if not pairs:
        return [], 0.0
    if not context.available(lang):
        return [(_match_case(word, form), 0.0, 0, 0) for form, _, _ in pairs], 0.0
    original, _ = context.score(pairs[0][2], neighbours, lang)
    ranked = []
    for form, cand, _ in pairs:
        sc, support = context.score(cand, neighbours, lang)
        ranked.append((_match_case(word, form), sc, support, context.freq(cand, lang)))
    ranked.sort(key=lambda r: (r[2] == 0, -r[1]))
    return ranked, original


def synonym_targets(text: str, lang: str, term_lemmas: set[str], strict: bool = False) -> list[tuple]:
    """Слова, которые можно заменить синонимом: (начало, конец, слово, варианты по убыванию уместности, оценка слова).

    Не трогаются: термины, имена и слова с заглавной буквы, цитаты, ссылки, служебные слова.
    """
    morph = _morph(lang)
    if morph is None or not synonyms.available(lang):
        return []
    hard, quotes = protected_spans(text)
    blocked = Spans(hard.items + quotes.items)
    out = []
    for s, e in sentences(text):
        content = _content(text, s, e, lang)
        for k, (ws, we, w, lemma, pos) in enumerate(content):
            if len(w) < 4 or not w.islower() or "-" in w or blocked.hit(ws, we):
                continue
            if pos not in SYN_POS or lemma in term_lemmas or not morph.word_is_known(w):
                continue
            if any(p.normal_form in STOP[lang] for p in morph.parse(w)[:4]):
                continue
            ranked, original = rank(w, _neighbours(content, k), lang, strict)
            if ranked:
                out.append((ws, we, w, ranked, original))
    return out


def auto_synonyms(text: str, lang: str, settings: Settings, term_lemmas: set[str]) -> tuple[str, list[dict]]:
    """Автозамена: синоним подставляется, только если корпус подтверждает его рядом с соседними словами."""
    stats = context.available(lang)
    need, ratio = AUTO_SYN[settings.intervention]
    applied = []
    # без статистики — только прямые синонимы без помет
    for s, e, w, ranked, original in reversed(synonym_targets(text, lang, term_lemmas, strict=not stats)):
        form, sc, support, fr = ranked[0]
        if stats and not (support >= need and fr >= MIN_FREQ and sc >= ratio * original):
            continue
        text = text[:s] + form + text[e:]
        applied.append({"kind": "synonym", "message": "Автозамена синонимом", "before": w, "after": form})
    return text, applied


def expand_phrases(text: str, lang: str) -> tuple[str, list[dict]]:
    """Развёрнутые обороты вместо коротких союзов (текст становится немного длиннее)."""
    hard, quotes = protected_spans(text)
    found = []
    for rule in EXPAND.get(lang, []):
        for m in rule.pattern.finditer(text):
            if not hard.hit(*m.span()) and not quotes.hit(*m.span()):
                found.append((m.start(), m.end(), _match_case(m.group(), rule.repls[0])))
    taken, applied = Taken(), []
    for s, e, repl in sorted(found, reverse=True):
        if taken.add(s, e):
            applied.append({"kind": "expand", "message": "Развёрнутый оборот", "before": text[s:e], "after": repl})
            text = text[:s] + repl + text[e:]
    return text, applied


def swapped_words(original: str, edited: str, lang: str) -> list[Fix]:
    """Слова, уже заменённые синонимом (автоматически или автором), с вариантами и возвратом."""
    la, lb = original.splitlines(keepends=True), edited.splitlines(keepends=True)
    if len(la) != len(lb) or not synonyms.available(lang):
        return []
    out, pos = [], 0
    for x, y in zip(la, lb):
        if x != y:
            a = [(m.group(), m.start()) for m in RE_TOKEN.finditer(x)]
            b = [(m.group(), m.start()) for m in RE_TOKEN.finditer(y)]
            sm = difflib.SequenceMatcher(None, [t.lower() for t, _ in a], [t.lower() for t, _ in b], autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag != "replace" or i2 - i1 != j2 - j1:
                    continue
                for (old, _), (new, at) in zip(a[i1:i2], b[j1:j2]):
                    if not RE_WORD.fullmatch(old) or not RE_WORD.fullmatch(new) or old.lower() == new.lower():
                        continue
                    alts = _synonyms(old.lower(), lang)
                    if new.lower() not in alts:
                        continue
                    others = [_match_case(new, c) for c in alts if c != new.lower()][:4]
                    out.append(Fix(pos + at, pos + at + len(new), "swapped",
                                   f"Заменено синонимом: было «{old}»", others + [old]))
        pos += len(y)
    return out


# --------------------------------------------------------------------- diff

def word_diff(original: str, edited: str) -> tuple[list[list[str]], int, int]:
    """Пословный diff. Правки не пересекают границы строк, поэтому сравнение идёт построчно."""
    la, lb = original.splitlines(keepends=True), edited.splitlines(keepends=True)
    if len(la) == len(lb):
        blocks = [(x, y, x == y) for x, y in zip(la, lb)]
    else:  # строки добавлены или удалены вручную — сначала сопоставляем строки целиком
        sm = difflib.SequenceMatcher(None, la, lb, autojunk=False)
        blocks = [("".join(la[i1:i2]), "".join(lb[j1:j2]), tag == "equal") for tag, i1, i2, j1, j2 in sm.get_opcodes()]
    ops, kept, total = [], 0, 0
    for x, y, same in blocks:
        o, k, t = _diff_block(x, y, same)
        ops += o
        kept += k
        total += t
    return ops, kept, total


def _diff_block(original: str, edited: str, same: bool) -> tuple[list[list[str]], int, int]:
    if same:
        n = len(RE_ANYWORD.findall(edited))
        return ([["eq", edited]] if edited else []), n, n
    a, b = RE_TOKEN.findall(original), RE_TOKEN.findall(edited)
    sm = difflib.SequenceMatcher(None, [t.lower() for t in a], [t.lower() for t in b], autojunk=False)
    ops: list[list[str]] = []
    kept = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            ops.append(["eq", "".join(b[j1:j2])])
            kept += sum(1 for t in b[j1:j2] if RE_ANYWORD.fullmatch(t))
        else:
            if i2 > i1:
                ops.append(["del", "".join(a[i1:i2])])
            if j2 > j1:
                ops.append(["ins", "".join(b[j1:j2])])
    total = sum(1 for t in b if RE_ANYWORD.fullmatch(t))
    return ops, kept, total


# ------------------------------------------------------------- точка входа

class EngineError(Exception):
    """Ошибка, текст которой можно показать пользователю."""


LANG_NAMES = {"ru": "русский", "uk": "украинский", "en": "английский"}


def edit_text(text: str, settings: Settings, original: str | None = None, fresh: bool = True) -> dict:
    if settings.intervention not in AUTO_KINDS or settings.complexity not in LONG_SENTENCE \
            or not 1 <= settings.academic_level <= 5 or settings.synonyms not in ("click", "suggest", "auto"):
        raise EngineError("Недопустимые настройки.")
    if not RE_ANYWORD.search(text):
        raise EngineError("Текст пуст.")
    original = text if original is None else original

    lang = detect_language(text)
    edited, applied = apply_auto(text, lang, settings)
    terms, term_lemmas = find_terms(edited, lang)

    # автозамена синонимами — только при первом проходе, иначе замены пошли бы по кругу
    if settings.synonyms == "auto" and fresh:
        edited, more = auto_synonyms(edited, lang, settings, term_lemmas)
        applied += more
    if settings.expand and fresh:
        edited, more = expand_phrases(edited, lang)
        applied += more

    pending = [f for f in _collect(edited, RULES[lang], settings, set()) if not f.auto]
    taken = Taken()
    for f in pending:
        taken.add(f.start, f.end)
    swapped = swapped_words(original, edited, lang)
    extra = swapped + analyse(edited, lang, settings, term_lemmas)
    if settings.synonyms == "suggest":
        note = ("Есть синонимы: первыми идут подходящие к соседним словам" if context.available(lang)
                else "Есть синонимы в словаре")
        extra += [Fix(s, e, "synonym", note, [r[0] for r in ranked[:5]])
                  for s, e, _, ranked, _ in synonym_targets(edited, lang, term_lemmas)]
    for f in extra:
        if taken.add(f.start, f.end):
            pending.append(f)
    pending.sort(key=lambda f: f.start)

    ops, kept, total = word_diff(original, edited)
    missing, added = compare_protected(original, edited)

    return {
        "edited": edited,
        "language": LANG_NAMES[lang],
        "diff": ops,
        "stats": {
            "words_original": len(RE_ANYWORD.findall(original)),
            "words_edited": total,
            "words_kept": kept,
            "changed_percent": round(100 * (total - kept) / total, 1) if total else 0.0,
            "auto_fixes": len(applied),
            "suggestions": sum(1 for f in pending if f.kind != "swapped"),
            "swapped": len(swapped),
        },
        "checks": {"numbers_and_refs_ok": not missing and not added, "missing": missing, "added": added,
                   "morphology": _morph(lang) is not None, "synonyms": synonyms.available(lang),
                   "context": context.available(lang)},
        "terms": terms,
        "applied": applied[:300],
        "suggestions": [
            {"start": f.start, "end": f.end, "kind": f.kind, "message": f.message,
             "replacements": f.replacements, "text": edited[f.start:f.end]}
            for f in pending
        ],
    }


def synonyms_at(text: str, start: int, end: int) -> dict:
    """Синонимы для слова text[start:end] — по запросу автора (клик по слову)."""
    word = text[start:end]
    if not RE_WORD.fullmatch(word):
        raise EngineError("Выберите одно слово.")
    lang = detect_language(text)
    info = {"word": word, "synonyms": [], "fits": [], "protected": False,
            "available": synonyms.available(lang), "context": context.available(lang)}
    hard, quotes = protected_spans(text)
    if hard.hit(start, end) or quotes.hit(start, end):
        return info | {"protected": True}
    neighbours: list[str] = []
    for s, e in sentences(text):
        if s <= start < e:
            content = _content(text, s, e, lang)
            k = next((i for i, c in enumerate(content) if c[0] == start), None)
            if k is not None:
                neighbours = _neighbours(content, k)
            break
    ranked, _ = rank(word, neighbours, lang)
    return info | {"synonyms": [r[0] for r in ranked[:8]], "fits": [r[2] > 0 for r in ranked[:8]]}
