"""
Schema validator for B2: validates enum, flag, control, and buffer arguments
against tool JSON Schema definitions (enum lists, numeric bounds, defaults, patterns).

Values strictly inside schema constraints (without contradicting an explicit user choice)
resolve to GROUNDED (ALLOW instead of ASK/BLOCK). Out-of-bounds or invalid enum values
resolve to UNGROUNDED (BLOCK).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_SCHEMA_FILE = Path(__file__).resolve().parent.parent / "data" / "tool_schemas.json"
_REGISTRY: Dict[str, Any] = {}


def _load_registry() -> Dict[str, Any]:
    global _REGISTRY
    if not _REGISTRY and _SCHEMA_FILE.exists():
        try:
            _REGISTRY = json.loads(_SCHEMA_FILE.read_text(encoding="utf-8"))
        except Exception:
            _REGISTRY = {"tools": {}, "generic_control_properties": {}}
    return _REGISTRY


def _extract_keys(arg_path: str) -> Tuple[str, str]:
    """Return (top_key, leaf_key) from a path like 'filter.overallStatus[0]'."""
    no_idx = re.sub(r"\[\d+\]$", "", arg_path)
    top_key = re.sub(r"\[\d+\]$", "", no_idx.split(".")[0])
    leaf_key = no_idx.split(".")[-1]
    return top_key, leaf_key


def _enum_contradiction(value: str, enum_vals: list, context: str) -> Optional[str]:
    """
    Return the conflicting enum item if context explicitly mentions another item from enum_vals
    as a standalone token and does NOT mention `value`.
    """
    if not context:
        return None
    ctx_l = context.lower()
    val_l = str(value).lower()
    if re.search(rf"(?<![a-zA-Z0-9_]){re.escape(val_l)}(?![a-zA-Z0-9_])", ctx_l):
        return None
    for candidate in enum_vals:
        cand_l = str(candidate).lower()
        if cand_l == val_l:
            continue
        if len(cand_l) >= 3 and re.search(rf"(?<![a-zA-Z0-9_]){re.escape(cand_l)}(?![a-zA-Z0-9_])", ctx_l):
            return str(candidate)
    return None


def validate_schema_arg(
    tool_name: Optional[str],
    arg_path: str,
    value: Any,
    context: str = "",
    explicit_schema: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[str, str]]:
    """
    Validate an argument leaf against explicit schema, tool schema registry, or generic control bounds.
    Returns:
      - ("GROUNDED", reason) if valid inside schema bounds/enum
      - ("UNGROUNDED", reason) if violating schema bounds/enum or contradicting user enum choice
      - None if no schema constraint applies to this argument
    """
    reg = _load_registry()
    top_key, leaf_key = _extract_keys(arg_path)

    prop: Optional[Dict[str, Any]] = None
    if explicit_schema and isinstance(explicit_schema.get("properties"), dict):
        props = explicit_schema["properties"]
        prop = props.get(leaf_key) or props.get(top_key)

    if prop is None and tool_name:
        t_props = reg.get("tools", {}).get(tool_name, {}).get("properties", {})
        prop = t_props.get(leaf_key) or t_props.get(top_key)

    if prop is None and tool_name:
        g_props = reg.get("generic_control_properties", {})
        prop = g_props.get(leaf_key) or g_props.get(leaf_key.lower()) or g_props.get(top_key)

    if prop is None:
        return None

    declared_type = prop.get("type")

    # 1. Integer / numeric bounds or default check
    if declared_type in ("integer", "number"):
        if isinstance(value, bool):
            return ("UNGROUNDED", f"argument '{arg_path}' expected {declared_type}, got bool")
        num_val: Optional[float] = None
        if isinstance(value, int):
            num_val = float(value)
        elif isinstance(value, float) and declared_type == "number":
            num_val = value
        elif isinstance(value, float) and declared_type == "integer" and value.is_integer():
            num_val = value
        elif isinstance(value, str) and re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", value):
            parsed = float(value.strip())
            if declared_type == "number" or parsed.is_integer():
                num_val = parsed

        if num_val is None:
            return ("UNGROUNDED", f"argument '{arg_path}' expected {declared_type}, got {value!r}")

        minimum = prop.get("minimum")
        maximum = prop.get("maximum")
        if minimum is not None and num_val < minimum:
            return ("UNGROUNDED", f"argument '{arg_path}' value {num_val} below minimum {minimum}")
        if maximum is not None and num_val > maximum:
            return ("UNGROUNDED", f"argument '{arg_path}' value {num_val} exceeds maximum {maximum}")
        if minimum is not None or maximum is not None or prop.get("default") == num_val:
            return ("GROUNDED", "valid within schema bounds")
        return None

    # 2. CSV Enum check (e.g. Slack channel_types: "public_channel, private_channel")
    if declared_type == "csv_enum":
        if not isinstance(value, str) or not value.strip():
            return ("UNGROUNDED", f"argument '{arg_path}' expected non-empty csv_enum string")
        allowed = set(prop.get("enum", []))
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts or any(p not in allowed for p in parts):
            return ("UNGROUNDED", f"argument '{arg_path}' has disallowed token in {value!r}")
        return ("GROUNDED", "valid within schema csv_enum")

    # 3. Single Enum check
    enum_vals = prop.get("enum")
    if enum_vals is not None:
        if value not in enum_vals:
            return ("UNGROUNDED", f"argument '{arg_path}' value {value!r} not in enum {enum_vals}")
        if isinstance(value, str):
            conflict = _enum_contradiction(value, enum_vals, context)
            if conflict is not None:
                return ("UNGROUNDED", f"enum '{value}' contradicts '{conflict}' mentioned in context")
        return ("GROUNDED", "valid within schema enum")

    # 4. Pattern check
    pattern = prop.get("pattern")
    if pattern and isinstance(value, str):
        if not re.fullmatch(pattern, value):
            return ("UNGROUNDED", f"argument '{arg_path}' value {value!r} fails pattern {pattern}")
        return ("GROUNDED", "valid against schema pattern")

    return None
