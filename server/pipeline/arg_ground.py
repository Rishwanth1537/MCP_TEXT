"""
Argument-level grounding: verify each (argument, value) of a planned tool call against the conversation that preceded it.

Verdicts per argument
    GROUNDED     the value (normalised) is supported by the context
    UNGROUNDED   a checkable value that the context does not support        -> the only verdict that can block a call
    UNCHECKABLE  booleans, empty values, and structures this module does not interpret (never block)

Matching is type-aware and deterministic:
    numbers   digits with separators/decimals normalised, spelled-out numbers up to 99 and simple multiples ("twelve",
              "forty five", "a hundred"), percentages, k/m suffixes
    strings   case-/space-/punctuation-insensitive containment, token-set containment, and fuzzy token match
              (difflib ratio >= FUZZY) to absorb typos, plural forms and minor re-spellings; light stemming (running/run)
    typed normalisers: ISO dates and timestamps against written dates and clock times ("2 pm" = T14:00), ISO-4217 currency
    codes against currency names (JPY = Japanese yen)
This replaces exact string comparison (as used by simple argument-grounding checks) with normalised matching, which is what
keeps the false-block rate low on real conversations where values are paraphrased or reformatted.

Limitations (documented, not hidden): no world knowledge ("MSFT" for "Microsoft" is UNGROUNDED), no unit conversion,
no multi-hop arithmetic ("500 miles" derived from two numbers in the context), English only.
"""
from __future__ import annotations

import calendar
import functools
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

FUZZY = 0.88

_ONES = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_NUM_TOKEN = re.compile(r"(?<![\w.])(?:(?<![\w])(-)(?=\d))?(\d[\d,]*(?:\.\d+)?)\s*(k|m|million|thousand|%)?", re.I)
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


@dataclass
class ArgVerdict:
    argument: str
    value: Any
    verdict: str          # GROUNDED | UNGROUNDED | UNCHECKABLE
    reason: str = ""
    score: float = 1.0    # match strength in [0, 1]; strings use the best fuzzy ratio, numbers and typed values are 0 or 1


def _norm_text(s: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", str(s).lower())).strip()


_ORDINALS = {
    "first": 1.0, "second": 2.0, "third": 3.0, "fourth": 4.0, "fifth": 5.0,
    "sixth": 6.0, "seventh": 7.0, "eighth": 8.0, "ninth": 9.0, "tenth": 10.0,
    "eleventh": 11.0, "twelfth": 12.0, "thirteenth": 13.0, "fourteenth": 14.0,
    "fifteenth": 15.0, "sixteenth": 16.0, "seventeenth": 17.0, "eighteenth": 18.0,
    "nineteenth": 19.0, "twentieth": 20.0
}

_FRACTIONS = {
    "half": 0.5, "a half": 0.5, "one half": 0.5,
    "quarter": 0.25, "a quarter": 0.25, "one quarter": 0.25,
    "three quarters": 0.75, "three quarter": 0.75,
    "third": 1.0 / 3.0, "a third": 1.0 / 3.0, "one third": 1.0 / 3.0,
    "two thirds": 2.0 / 3.0
}


def _spelled_numbers(text: str) -> set:
    toks = _norm_text(text).replace("-", " ").split()
    out = set()
    num_words = set(_ONES.keys()) | set(_TENS.keys()) | {"hundred", "thousand", "million", "billion", "and"}

    # 1. Single and compound spelled numbers (>99)
    i = 0
    while i < len(toks):
        if toks[i] in num_words and toks[i] != "and":
            block = []
            while i < len(toks) and toks[i] in num_words:
                block.append(toks[i])
                i += 1
            cur, tot = 0, 0
            for w in block:
                if w in _ONES:
                    cur += _ONES[w]
                elif w in _TENS:
                    cur += _TENS[w]
                elif w == "hundred":
                    cur = (cur if cur != 0 else 1) * 100
                elif w in ("thousand", "million", "billion"):
                    mult = {"thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}[w]
                    cur = (cur if cur != 0 else 1) * mult
                    tot += cur
                    cur = 0
            tot += cur
            out.add(float(tot))
        else:
            if toks[i] in _ORDINALS:
                out.add(_ORDINALS[toks[i]])
            if toks[i] in _FRACTIONS:
                out.add(_FRACTIONS[toks[i]])
            i += 1

    t_lower = text.lower()
    for frac_phrase, frac_val in _FRACTIONS.items():
        if frac_phrase in t_lower:
            out.add(frac_val)

    return out


def _unit_conversions(text: str) -> set:
    vals = set()
    t = text.lower()

    # Distance: miles <-> km
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:miles?|mi)\b", t):
        v = float(m.group(1))
        km = v * 1.609344
        vals |= {km, round(km, 1), round(km, 2), round(km, 3)}
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:kilometers?|kilometres?|km)\b", t):
        v = float(m.group(1))
        mi = v / 1.609344
        vals |= {mi, round(mi, 1), round(mi, 2), round(mi, 3)}

    # Mass: lbs <-> kg
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:pounds?|lbs?)\b", t):
        v = float(m.group(1))
        kg = v * 0.45359237
        vals |= {kg, round(kg, 1), round(kg, 2), round(kg, 3)}
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:kilograms?|kilos?|kg)\b", t):
        v = float(m.group(1))
        lb = v / 0.45359237
        vals |= {lb, round(lb, 1), round(lb, 2), round(lb, 3)}

    # Temperature: C <-> F
    for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*(?:degrees?\s+)?(?:celsius|c)\b", t):
        v = float(m.group(1))
        f = v * 9.0 / 5.0 + 32.0
        vals |= {f, round(f, 1), round(f, 2)}
    for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*(?:degrees?\s+)?(?:fahrenheit|f)\b", t):
        v = float(m.group(1))
        c = (v - 32.0) * 5.0 / 9.0
        vals |= {c, round(c, 1), round(c, 2)}

    # Currency symbols: $, €, £, ¥
    for m in re.finditer(r"[\$€£¥](\d[\d,]*(?:\.\d+)?)", t):
        v = float(m.group(1).replace(",", ""))
        vals.add(v)

    return vals


def numbers_in(text: str) -> set:
    vals = set()
    for m in _NUM_TOKEN.finditer(text):
        try:
            v = float(m.group(2).replace(",", ""))
        except ValueError:
            continue
        suf = (m.group(3) or "").lower()
        mine = {v}
        if suf in ("k", "thousand"):
            mine.add(v * 1e3)
        elif suf in ("m", "million"):
            mine.add(v * 1e6)
        if suf == "%":
            mine.add(v / 100.0)
        if m.group(1):                       # explicit minus sign: also admit the negative of THIS number (latitudes, deltas, temperatures)
            mine |= {-x for x in mine if x > 0}
        vals |= mine
    return vals | _spelled_numbers(text) | _unit_conversions(text)


def _fuzzy_score(value_norm: str, context_norm: str) -> float:
    """1.0 if the value is contained (or its tokens are all present); otherwise the best difflib ratio over context windows."""
    if not value_norm or value_norm in context_norm:
        return 1.0
    vt, ct = value_norm.split(), context_norm.split()
    if set(vt) <= set(ct):
        return 1.0
    n, best = len(vt), 0.0
    sm = SequenceMatcher(None, "", value_norm)          # seq2 = the value; seq1 = each window (difflib caches seq2 statistics)
    for w in range(max(1, n - 1), n + 2):
        for i in range(0, max(1, len(ct) - w + 1)):
            sm.set_seq1(" ".join(ct[i:i + w]))
            if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best:   # upper bounds: cannot beat the best so far
                continue
            best = max(best, sm.ratio())
            if best >= 1.0:
                return 1.0
    return best


_CURRENCY = {
    "usd": ["us dollar", "us dollars", "united states dollar", "dollar", "dollars", "usd", "bucks"],
    "eur": ["euro", "euros", "eur"], "gbp": ["pound", "pounds", "british pound", "british pounds", "sterling", "gbp"],
    "jpy": ["yen", "japanese yen", "jpy"], "inr": ["rupee", "rupees", "indian rupee", "indian rupees", "inr"],
    "cad": ["canadian dollar", "canadian dollars", "cad"], "aud": ["australian dollar", "australian dollars", "aud"],
    "chf": ["swiss franc", "swiss francs", "franc", "francs", "chf"], "cny": ["yuan", "renminbi", "chinese yuan", "cny", "rmb"],
    "mxn": ["mexican peso", "mexican pesos", "peso", "pesos", "mxn"], "krw": ["won", "korean won", "south korean won", "krw"],
    "brl": ["brazilian real", "brazilian reais", "real", "reais", "brl"], "rub": ["ruble", "rubles", "rouble", "russian ruble", "rub"],
    "sek": ["swedish krona", "krona", "sek"], "nok": ["norwegian krone", "krone", "nok"], "dkk": ["danish krone", "dkk"],
    "nzd": ["new zealand dollar", "nzd", "kiwi dollar"], "sgd": ["singapore dollar", "sgd"], "hkd": ["hong kong dollar", "hkd"],
    "zar": ["south african rand", "rand", "zar"], "try": ["turkish lira", "lira", "try"], "aed": ["dirham", "dirhams", "aed"],
}
_SUFFIXES = ("ing", "ed", "es", "s")


def _stem(w: str) -> str:
    for suf in _SUFFIXES:
        if len(w) > len(suf) + 2 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]
    return w


def _stems(text_norm: str) -> set:
    return {_stem(t) for t in text_norm.split()}


_ISO = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?\s*(?:Z|[+-]\d{2}:?\d{2})?\s*$")
_YM = re.compile(r"^\s*(\d{4})-(\d{2})\s*$")
_QUARTER = re.compile(r"^\s*(\d{4})-Q([1-4])\s*$", re.I)
_YEAR = re.compile(r"(?<![\d-])((?:19|20)\d{2})(?![\d-])")
_YM_ISO = re.compile(r"(?<![\d-])((?:19|20)\d{2})[-/](0[1-9]|1[0-2])(?![\d-])")
_CTX_TIME = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)|\b(\d{1,2}):(\d{2})\b", re.I)
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
_MONTH_PAT = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\.?,?\s+(?:of\s+)?((?:19|20)\d{2})\b",
    re.I
)
_MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12
}


@functools.lru_cache(maxsize=4096)
def _months_years(text: str) -> Tuple[frozenset, frozenset]:
    t = text.lower()
    ym = set()
    for m in _MONTH_PAT.finditer(t):
        mn_str = m.group(1).lower().rstrip('.')
        if mn_str in _MONTH_MAP:
            ym.add((int(m.group(2)), _MONTH_MAP[mn_str]))
    for m in _YM_ISO.finditer(t):
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            ym.add((y, mo))
    years = {int(y) for y in _YEAR.findall(t)}
    return frozenset(ym), frozenset(years)


def _context_times(context: str) -> set:
    out = set()
    for m in _CTX_TIME.finditer(context):
        if m.group(3):
            h, mi = int(m.group(1)) % 12, int(m.group(2) or 0)
            if m.group(3).lower().startswith("p"):
                h += 12
            out.add((h, mi))
        else:
            out.add((int(m.group(4)), int(m.group(5))))
    for m in re.finditer(r"T(\d{2}):(\d{2})", context):
        out.add((int(m.group(1)), int(m.group(2))))
    return out


def _date_in(context: str, y: int, mo: int, d: int) -> bool:
    c = context.lower()
    forms = [f"{y:04d}-{mo:02d}-{d:02d}", f"{mo}/{d}/{y}", f"{mo:02d}/{d:02d}/{y}", f"{d}/{mo}/{y}"]
    if any(f in c for f in forms):
        return True
    mn = _MONTHS[mo - 1]
    pat = rf"({mn}|{mn[:3]})\.?\s+{d}(st|nd|rd|th)?,?\s+{y}|{d}(st|nd|rd|th)?\s+(of\s+)?({mn}|{mn[:3]})\.?,?\s+{y}"
    return re.search(pat, c) is not None


def _check_iso(argument: str, value: str, context: str):
    v = value.strip()
    c = context.lower()

    # 0. Verbatim containment in context
    if v.lower() in c:
        return ArgVerdict(argument, value, "GROUNDED", "")

    # 1. Year-Month (e.g. 2023-07)
    ym_m = _YM.match(v)
    if ym_m:
        y, mo = int(ym_m.group(1)), int(ym_m.group(2))
        ym, _ = _months_years(context)
        forms = [f"{y:04d}-{mo:02d}", f"{mo:02d}/{y}", f"{mo}/{y}"]
        if (y, mo) in ym or any(f in c for f in forms):
            return ArgVerdict(argument, value, "GROUNDED", "")
        return ArgVerdict(argument, value, "UNGROUNDED", f"month {y:04d}-{mo:02d} not found in context", 0.0)

    # 2. Quarter (e.g. 2020-Q1)
    q_m = _QUARTER.match(v)
    if q_m:
        y, q = int(q_m.group(1)), int(q_m.group(2))
        q_words = ['first', 'second', 'third', 'fourth']
        pats = [f'{y}-q{q}', f'q{q}-{y}', f'q{q} {y}', f'q{q}, {y}', f'{q_words[q-1]} quarter of {y}', f'{q_words[q-1]} quarter {y}']
        if any(p in c for p in pats):
            return ArgVerdict(argument, value, "GROUNDED", "")
        return ArgVerdict(argument, value, "UNGROUNDED", f"quarter {y}-Q{q} not found in context", 0.0)

    # 3. Full ISO date / timestamp
    m = _ISO.match(v)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not _date_in(context, y, mo, d):
        ym, _ = _months_years(context)
        last_d = calendar.monthrange(y, mo)[1] if 1 <= mo <= 12 else 0
        if (y, mo) in ym and d in (1, last_d):
            pass
        else:
            return ArgVerdict(argument, value, "UNGROUNDED", f"date {y:04d}-{mo:02d}-{d:02d} not found in context", 0.0)
    if m.group(4) is not None:
        t = (int(m.group(4)), int(m.group(5)))
        if t not in ((0, 0), (23, 59)) and t not in _context_times(context):
            return ArgVerdict(argument, value, "UNGROUNDED", f"time {t[0]:02d}:{t[1]:02d} not found in context", 0.0)
    return ArgVerdict(argument, value, "GROUNDED", "")


def _check_currency(argument: str, value: str, context: str):
    key = value.strip().lower()
    if key not in _CURRENCY:
        return None
    ctx = _norm_text(context)
    names = _CURRENCY[key]
    if any(re.search(rf"(?<!\w){re.escape(n)}(?!\w)", ctx) for n in names) or key in ctx.split():
        return ArgVerdict(argument, value, "GROUNDED", "")
    return ArgVerdict(argument, value, "UNGROUNDED", f"currency {value} not mentioned in context", 0.0)


def check_value(argument: str, value: Any, context: str) -> ArgVerdict:
    if isinstance(value, bool) or value is None:
        return ArgVerdict(argument, value, "UNCHECKABLE", "boolean or null value")
    if isinstance(value, (int, float)):
        ok = float(value) in numbers_in(context) or any(abs(float(value) - x) < 1e-9 for x in numbers_in(context))
        return ArgVerdict(argument, value, "GROUNDED" if ok else "UNGROUNDED", "" if ok else f"number {value} not found in context", 1.0 if ok else 0.0)
    if isinstance(value, str):
        if not value.strip():
            return ArgVerdict(argument, value, "UNCHECKABLE", "empty string")
        # a purely numeric string is a number
        if re.fullmatch(r"\s*[\d,]+(?:\.\d+)?\s*", value):
            return check_value(argument, float(value.replace(",", "")), context)
        for special in (_check_iso, _check_currency):
            v = special(argument, value, context)
            if v is not None:
                return v
        vn, cn = _norm_text(value), _norm_text(context)
        sc = 1.0 if (bool(vn) and _stems(vn) <= _stems(cn)) else _fuzzy_score(vn, cn)
        ok = sc >= FUZZY
        return ArgVerdict(argument, value, "GROUNDED" if ok else "UNGROUNDED", "" if ok else f"'{value}' not found in context", round(sc, 4))
    if isinstance(value, (list, tuple)):
        verdicts = [check_value(argument, v, context) for v in value]
        bad = [v for v in verdicts if v.verdict == "UNGROUNDED"]
        if bad:
            return ArgVerdict(argument, value, "UNGROUNDED", bad[0].reason, min(v.score for v in bad))
        return ArgVerdict(argument, value, "GROUNDED" if verdicts else "UNCHECKABLE")
    return ArgVerdict(argument, value, "UNCHECKABLE", f"unsupported type {type(value).__name__}")


def check_arguments(arguments: Dict[str, Any], context: str) -> List[ArgVerdict]:
    return [check_value(k, v, context) for k, v in arguments.items()]


def call_is_grounded(arguments: Dict[str, Any], context: str, schema: Dict[str, Any] | None = None, policy: str = "strict") -> bool:
    """
    policy "strict":        every checkable argument must be grounded.
    policy "required_only": only arguments the tool schema marks as required can block a call; an ungrounded OPTIONAL argument
                            (usually a model-chosen default such as a size or a category) is reported but does not block.
                            Needs `schema` = {"required": [...], "optional": [...]}; falls back to "strict" without it.
    """
    verdicts = check_arguments(arguments, context)
    if policy == "required_only" and schema:
        optional = set(schema.get("optional", []))
        return not any(v.verdict == "UNGROUNDED" and v.argument not in optional for v in verdicts)
    return not any(v.verdict == "UNGROUNDED" for v in verdicts)


def call_score(arguments: Dict[str, Any], context: str) -> float:
    """Call-level grounding score in [0, 1]: the weakest match among the checkable arguments (1.0 if none are checkable)."""
    scores = [v.score for v in check_arguments(arguments, context) if v.verdict != "UNCHECKABLE"]
    return min(scores) if scores else 1.0


def conformal_threshold(gold_scores, alpha: float) -> float:
    """
    Split-conformal threshold for 'block iff call_score < tau'.  With n calibration scores from genuine calls
    (exchangeable with deployment calls), choosing tau as the k-th smallest score with k = floor(alpha * (n + 1))
    gives P(a new genuine call is blocked) <= alpha (finite-sample, distribution-free).  Returns 0.0 (never block)
    when alpha * (n + 1) < 1 or when more than alpha of genuine calls already score 0, i.e. when the requested false-block
    rate cannot be met by this stage; callers should report that instead of pretending otherwise.
    """
    import math
    xs = sorted(gold_scores)
    k = math.floor(alpha * (len(xs) + 1))
    if k < 1:
        return 0.0
    return xs[k - 1]
