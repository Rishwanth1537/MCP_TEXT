"""
Provenance-aware argument grounding (E1, registered as Deviation 21 in review/PREREGISTRATION.md).

The context checker in arg_ground.py grounds a value against one block of text. On real agent traffic most values do not come from
the user's words: identifiers come from earlier tool outputs, date ranges are derived from a month, page sizes are tool usage, and
search queries are rewritten. This module keeps the same deterministic matchers but changes the EVIDENCE and adds typed rules:

    evidence      a list of (source, text) items, source in {"user", "tool"} (agent-produced text is not evidence by default:
                  a value the agent invented and repeated would otherwise ground itself)
    provenance    every GROUNDED leaf records the first source that grounded it (user before tool)
    configs       "P1" tool outputs as evidence; "P2" + leaf-by-leaf nested values + date-range expansion;
                  "P3" + argument-class rules (control parameters, generated programs, free-text search arguments)

Nothing here is fitted; every rule is fixed in the pre-registration.
"""
from __future__ import annotations

import calendar
import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from server.pipeline.arg_ground import (
    _ISO, _YM, _QUARTER, _months_years, _MONTHS, ArgVerdict, _norm_text, _stem, check_value, numbers_in,
)
from server.pipeline.schema_validator import validate_schema_arg

FUZZY_TOOL_CHARS = 100_000
LONG_STRING_TOKENS = 40
CONTROL_ARGS = {"limit", "max_results", "maxresults", "per_page", "perpage", "page_size", "pagesize", "page", "offset", "top_k",
                "topk", "n", "count", "size", "num_results", "numresults"}
PROGRAM_ARGS = {"code", "script", "sql", "query_sql", "pipeline", "filter", "expression", "regex", "command", "cmd"}
FREE_TEXT_ARGS = {"query", "q", "search", "search_query", "search_term", "keywords", "keyword", "text", "question", "prompt", "topic"}
FREE_TEXT_MIN = 0.60
STOP = set("""the and for with from that this what which who whom whose when where why how are was were has have had not but
all any can will would about into over under than then them they their there these those you your""".split())
CONFIGS = ("P1", "P2", "P3")


@dataclass
class LeafVerdict:
    path: str
    value: Any
    verdict: str            # GROUNDED | UNGROUNDED | UNCHECKABLE
    source: str = ""        # user | tool | rule | "" (for UNGROUNDED / UNCHECKABLE)
    reason: str = ""
    score: float = 1.0


@dataclass
class CallVerdict:
    blocked: bool
    leaves: List[LeafVerdict] = field(default_factory=list)
    score: float = 1.0


class Evidence:
    """Evidence items in priority order. Pre-computes normalised text once per call."""

    def __init__(self, items: Sequence[Tuple[str, str]]):
        self.items = [(s, t or "") for s, t in items if t]
        self.by_source: Dict[str, str] = {}
        for s, t in self.items:
            self.by_source[s] = (self.by_source.get(s, "") + "\n" + t) if s in self.by_source else t
        self.sources = [s for s in ("user", "tool", "agent") if s in self.by_source] + \
            [s for s in self.by_source if s not in ("user", "tool", "agent")]
        self.all_text = "\n".join(self.by_source[s] for s in self.sources)
        self._stems = None

    def text_for_fuzzy(self, source: str) -> str:
        t = self.by_source[source]
        return t if source == "user" or len(t) <= FUZZY_TOOL_CHARS else t[-FUZZY_TOOL_CHARS:]

    def stems(self) -> set:
        if self._stems is None:
            self._stems = {_stem(w) for w in _norm_text(self.all_text).split()}
        return self._stems


# ---------------------------------------------------------------- date-range expansion (P2)
_YM_ISO = re.compile(r"\b(\d{4})-(\d{2})\b(?!-\d)")
def _date_range_grounded(value: str, text: str) -> bool:
    v = value.strip()
    c = text.lower()

    # 1. Year-Month
    ym_m = _YM.match(v)
    if ym_m:
        y, mo = int(ym_m.group(1)), int(ym_m.group(2))
        ym, _ = _months_years(text)
        forms = [f"{y:04d}-{mo:02d}", f"{mo:02d}/{y}", f"{mo}/{y}"]
        return (y, mo) in ym or any(f in c for f in forms)

    # 2. Quarter
    q_m = _QUARTER.match(v)
    if q_m:
        y, q = int(q_m.group(1)), int(q_m.group(2))
        q_words = ['first', 'second', 'third', 'fourth']
        pats = [f'{y}-q{q}', f'q{q}-{y}', f'q{q} {y}', f'q{q}, {y}', f'{q_words[q-1]} quarter of {y}', f'{q_words[q-1]} quarter {y}']
        if any(p in c for p in pats):
            return True
        # Cross-source trajectory match: year is mentioned in evidence, and quarter phrase is in evidence
        _, years = _months_years(text)
        if y in years and (f"q{q}" in c or f"{q_words[q-1]} quarter" in c):
            return True
        return False

    # 3. Full ISO date
    m = _ISO.match(v)
    if not m:
        return False
    if m.group(4) is not None:
        hm = (int(m.group(4)), int(m.group(5)), int(m.group(6) or 0))
        if hm not in ((0, 0, 0), (23, 59, 0), (23, 59, 59)):
            return False
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not 1 <= mo <= 12:
        return False
    last = calendar.monthrange(y, mo)[1]
    ym, years = _months_years(text)
    if (y, mo) in ym:
        if d in (1, last):
            return True
        if d == 7 and ("first week" in c or "1st week" in c):
            return True
        if d == 15 and ("first half" in c or "1st half" in c):
            return True
        if d == 16 and ("second half" in c or "2nd half" in c):
            return True
    return y in years and (mo, d) in ((1, 1), (12, 31))


_HOST_DATE_PAT = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _host_date_grounded(value: str, ev: Evidence) -> bool:
    """Resolve relative date phrases in user prompt against host-supplied current date/time evidence."""
    host_text = ev.by_source.get("host", "")
    if not host_text:
        return False
    hm = _HOST_DATE_PAT.search(host_text)
    if not hm:
        return False
    try:
        host_dt = date(int(hm.group(1)), int(hm.group(2)), int(hm.group(3)))
    except ValueError:
        return False

    user_l = ev.by_source.get("user", "").lower()
    v = value.strip()

    # Check Year-Month (e.g. 2025-07)
    ym_m = _YM.match(v)
    if ym_m:
        y, mo = int(ym_m.group(1)), int(ym_m.group(2))
        if (y, mo) == (host_dt.year, host_dt.month) and any(p in user_l for p in ("this month", "current month", "currently", "recent")):
            return True
        prev_y, prev_mo = (host_dt.year - 1, 12) if host_dt.month == 1 else (host_dt.year, host_dt.month - 1)
        if (y, mo) == (prev_y, prev_mo) and "last month" in user_l:
            return True
        return False

    # Check ISO date / timestamp
    m = _ISO.match(v)
    if not m:
        return False
    if m.group(4) is not None:
        t_hm = (int(m.group(4)), int(m.group(5)), int(m.group(6) or 0))
        if t_hm not in ((0, 0, 0), (23, 59, 0), (23, 59, 59)):
            return False
    try:
        val_dt = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return False

    if val_dt == host_dt and any(p in user_l for p in ("today", "tonight", "this morning", "currently", "right now")):
        return True
    if val_dt == host_dt - timedelta(days=1) and "yesterday" in user_l:
        return True
    if val_dt == host_dt + timedelta(days=1) and "tomorrow" in user_l:
        return True
    if (host_dt - timedelta(days=14) <= val_dt <= host_dt + timedelta(days=7)) and any(
        p in user_l for p in ("this week", "past week", "last week", "past 7 days", "last 7 days", "recent", "lately", "other day")
    ):
        return True
    if (val_dt.year, val_dt.month) == (host_dt.year, host_dt.month):
        last_d = calendar.monthrange(host_dt.year, host_dt.month)[1]
        if val_dt.day in (1, last_d, host_dt.day) and any(p in user_l for p in ("this month", "current month")):
            return True
    return False


# ---------------------------------------------------------------- argument-class rules (P3)
def _key(path: str) -> str:
    leaf = path.split(".")[-1]
    return re.sub(r"\[\d+\]$", "", leaf).lower()


def _free_text_grounded(value: str, ev: Evidence) -> bool:
    words = [w for w in _norm_text(value).split() if len(w) >= 3 and w not in STOP]
    if not words:
        return False
    stems = ev.stems()
    hit = sum(_stem(w) in stems for w in words)
    return hit / len(words) >= FREE_TEXT_MIN


def _as_json(value: str):
    v = value.strip()
    if not v or v[0] not in "[{":
        return None
    try:
        out = json.loads(v)
    except ValueError:
        return None
    return out if isinstance(out, (dict, list)) else None


# ---------------------------------------------------------------- leaf checking
_CACHE: Dict[tuple, Any] = {}


def _cached_check(path: str, value: Any, text: str):
    """check_value memoised on (value, text); the verdict does not depend on the argument name, so evaluating several
    configurations over the same evidence does not repeat the slow fuzzy scan. Results are identical to check_value."""
    if isinstance(value, (list, dict)):
        return check_value(path, value, text)
    key = (type(value).__name__, value, len(text), hash(text))
    hit = _CACHE.get(key)
    if hit is None:
        if len(_CACHE) > 200_000:
            _CACHE.clear()
        if isinstance(value, str) and len(_norm_text(value).split()) > LONG_STRING_TOKENS:
            # long strings (code, pasted data): no fuzzy window scan (minutes per value on long evidence); grounded only if the
            # normalised value or its stem set is contained in the evidence (implementation note to Deviation 21)
            ok = _cheap_full_match(path, value, text)
            hit = ArgVerdict(path, value, "GROUNDED" if ok else "UNGROUNDED", "" if ok else "long string not contained", 1.0 if ok else 0.0)
        else:
            hit = check_value(path, value, text)
        _CACHE[key] = hit
    return hit.__class__(path, value, hit.verdict, hit.reason, hit.score)


def _cheap_full_match(path: str, value: Any, text: str) -> bool:
    """Non-fuzzy checks over the whole text: numbers, ISO dates, containment and stem-set containment."""
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)) or (isinstance(value, str) and re.fullmatch(r"\s*[\d,]+(?:\.\d+)?\s*", value)):
        num = float(str(value).replace(",", ""))
        return num in numbers_in(text)
    if not isinstance(value, str):
        return False
    if _ISO.match(value):
        return check_value(path, value, text).verdict == "GROUNDED"
    key = (len(text), hash(text))
    if key not in _NORM:
        if len(_NORM) > 64:
            _NORM.clear()
        tn = _norm_text(text)
        _NORM[key] = (tn, {_stem(w) for w in tn.split()})
    tn, ts = _NORM[key]
    vn = _norm_text(value)
    return bool(vn) and (vn in tn or {_stem(w) for w in vn.split()} <= ts)


_NORM: Dict[tuple, tuple] = {}


def _check_leaf(path: str, value: Any, ev: Evidence, config: str,
                schema: Optional[Dict[str, Any]] = None, tool_name: Optional[str] = None) -> LeafVerdict:
    key = _key(path)
    if config == "P3":
        # Phase B2a: Validate enum, flag, control, and buffer arguments against schema
        # Note: if parameter is explicitly contradicted by prompt or outside bounds, schema returns UNGROUNDED.
        user_ctx = ev.by_source.get("user", "")
        schema_res = validate_schema_arg(tool_name, path, value, user_ctx, explicit_schema=schema)
        if schema_res is not None:
            verdict, reason = schema_res
            return LeafVerdict(path, value, verdict, "schema" if verdict == "GROUNDED" else "", reason, 1.0 if verdict == "GROUNDED" else 0.0)
        if key in CONTROL_ARGS and isinstance(value, int) and not isinstance(value, bool):
            return LeafVerdict(path, value, "UNCHECKABLE", "rule", "control parameter", 1.0)
        if key in PROGRAM_ARGS and isinstance(value, str):
            return LeafVerdict(path, value, "UNCHECKABLE", "rule", "generated program", 1.0)
    first_reason = ""
    best_score = 0.0
    for src in ev.sources:
        text = ev.by_source[src]
        v = _cached_check(path, value, ev.text_for_fuzzy(src))
        # long tool outputs: the fuzzy path saw only the most recent FUZZY_TOOL_CHARS; exact, number and date checks see everything
        if v.verdict == "UNGROUNDED" and len(text) > len(ev.text_for_fuzzy(src)) and _cheap_full_match(path, value, text):
            v = v.__class__(path, value, "GROUNDED", "", 1.0)
        if v.verdict == "GROUNDED":
            return LeafVerdict(path, value, "GROUNDED", src, "", 1.0)
        if v.verdict == "UNCHECKABLE":
            return LeafVerdict(path, value, "UNCHECKABLE", "", v.reason, 1.0)
        best_score = max(best_score, v.score)
        first_reason = first_reason or v.reason
    if config in ("P2", "P3") and isinstance(value, str) and _date_range_grounded(value, ev.all_text):
        return LeafVerdict(path, value, "GROUNDED", "rule", "date-range expansion", 1.0)
    if config == "P3" and isinstance(value, str) and _host_date_grounded(value, ev):
        return LeafVerdict(path, value, "GROUNDED", "host", "relative date resolved against host clock", 1.0)
    if config == "P3" and key in FREE_TEXT_ARGS and isinstance(value, str) and _free_text_grounded(value, ev):
        return LeafVerdict(path, value, "GROUNDED", "rule", "free-text search argument", 1.0)
    return LeafVerdict(path, value, "UNGROUNDED", "", first_reason, best_score)


def _walk(path: str, value: Any, ev: Evidence, config: str,
          schema: Optional[Dict[str, Any]] = None, tool_name: Optional[str] = None) -> Iterable[LeafVerdict]:
    nested = config in ("P2", "P3")
    if nested and isinstance(value, str) and config == "P3" and _key(path) not in PROGRAM_ARGS:
        parsed = _as_json(value)
        if parsed is not None:
            value = parsed
    if isinstance(value, dict):
        if not nested:
            yield LeafVerdict(path, value, "UNCHECKABLE", "", "unsupported type dict")
            return
        for k, v in value.items():
            yield from _walk(f"{path}.{k}", v, ev, config, schema=schema, tool_name=tool_name)
        return
    if isinstance(value, (list, tuple)):
        if nested:
            for i, v in enumerate(value):
                yield from _walk(f"{path}[{i}]", v, ev, config, schema=schema, tool_name=tool_name)
            return
        if any(isinstance(v, (dict, list, tuple)) for v in value):
            yield LeafVerdict(path, value, "UNCHECKABLE", "", "nested list")
            return
        for i, v in enumerate(value):
            yield _check_leaf(f"{path}[{i}]", v, ev, config, schema=schema, tool_name=tool_name)
        return
    yield _check_leaf(path, value, ev, config, schema=schema, tool_name=tool_name)


def check_call(arguments: Dict[str, Any], evidence: Sequence[Tuple[str, str]], config: str = "P3",
               include_agent: bool = False, schema: Optional[Dict[str, Any]] = None,
               tool_name: Optional[str] = None) -> CallVerdict:
    """Strict policy: a call is blocked when any checkable leaf is UNGROUNDED."""
    if config not in CONFIGS:
        raise ValueError(f"config must be one of {CONFIGS}")
    items = [(s, t) for s, t in evidence if include_agent or s != "agent"]
    ev = Evidence(items)
    leaves = [lv for k, v in (arguments or {}).items() for lv in _walk(k, v, ev, config, schema=schema, tool_name=tool_name)]
    call_sc = min((lv.score for lv in leaves if lv.verdict != "UNCHECKABLE"), default=1.0)
    return CallVerdict(blocked=any(lv.verdict == "UNGROUNDED" for lv in leaves), leaves=leaves, score=call_sc)


# ---------------------------------------------------------------- evidence builders for the stored benchmark formats
_AH_LINE = re.compile(r"^\[(tool call|tool result|assistant)\] ?", re.M)


def evidence_from_agenthallu_premise(premise: str) -> List[Tuple[str, str]]:
    """Split a load_agenthallu.py premise into user / tool / agent items. Text before the first tagged line is the user task
    (it may be cut at the front when the premise was truncated to its last 6,000 characters)."""
    parts = _AH_LINE.split(premise)
    items = [("user", parts[0].replace("User task:\n", "", 1).strip())]
    for tag, text in zip(parts[1::2], parts[2::2]):
        items.append(("tool" if tag == "tool result" else "agent", text.strip()))
    return items


_DOJO = re.compile(r"(?m)^\[(system|user|assistant|tool)\]")


def evidence_from_agentdojo_premise(premise: str) -> List[Tuple[str, str]]:
    """System and user messages are 'user' (what the deployer and the user said); tool messages are 'tool'; assistant is 'agent'."""
    parts = _DOJO.split(premise)
    items = [("user", parts[0])] if parts[0].strip() else []
    for role, text in zip(parts[1::2], parts[2::2]):
        items.append(("user" if role in ("system", "user") else "tool" if role == "tool" else "agent", text))
    return items
