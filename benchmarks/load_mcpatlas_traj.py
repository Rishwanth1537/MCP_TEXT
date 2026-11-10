"""Build benchmarks/data/mcpatlas/cases_traj.json: MCP-Atlas reference calls WITH their trajectory evidence (Deviation 21, E1).

Unlike load_mcpatlas.py (premise = task prompt only), every case keeps the evidence that existed before the call, tagged by source:
    user          the task PROMPT
    tool_results  outputs of all earlier tool calls of the same trajectory (in order, with the tool name)
    agent         earlier assistant text and earlier tool-call arguments (agent-produced; NOT counted as grounding evidence in E1,
                  kept only for a sensitivity row)
Every reference call is clean (label True). Tasks are split into a design half and a held-out half by sha256(TASK) parity, so the
held-out half can be run once after the method is frozen.

Reads parquet/JSON only; never executes anything from the data.
Usage: python -I -X utf8 benchmarks/load_mcpatlas_traj.py <MCP-Atlas.parquet>
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_text(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content)
    return str(content)


def half_of(task):
    return "design" if int(hashlib.sha256(task.encode()).hexdigest(), 16) % 2 == 0 else "holdout"


def build(parquet_path):
    df = pd.read_parquet(parquet_path)
    cases, roles = [], {}
    for _, row in df.iterrows():
        traj = json.loads(row["TRAJECTORY"])
        names = {}
        tool_results, agent = [], []
        k = 0
        for msg in traj:
            role = msg.get("role")
            roles[role] = roles.get(role, 0) + 1
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    fn = tc["function"]
                    args = json.loads(fn["arguments"]) if fn.get("arguments") else {}
                    cases.append({
                        "id": f"mcpatlas-{row['TASK']}-{k}", "task": row["TASK"], "idx": k, "half": half_of(row["TASK"]),
                        "user": row["PROMPT"], "tool_results": list(tool_results), "agent": list(agent),
                        "call": {"name": fn["name"], "arguments": args}, "label": True,
                    })
                    names[tc.get("id")] = fn["name"]
                    k += 1
                # the assistant text and the call arguments of THIS message become agent evidence only for later calls
                if msg.get("content"):
                    agent.append(_text(msg["content"]))
                for tc in msg.get("tool_calls") or []:
                    agent.append(f"{tc['function']['name']} {tc['function'].get('arguments') or ''}")
            elif role == "tool":
                tool_results.append({"tool": names.get(msg.get("tool_call_id"), "?"), "text": _text(msg.get("content"))})
    return cases, roles


def main():
    cases, roles = build(Path(sys.argv[1]))
    out = Path(__file__).parent / "data" / "mcpatlas" / "cases_traj.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
    halves = {h: sum(c["half"] == h for c in cases) for h in ("design", "holdout")}
    tasks = {h: len({c["task"] for c in cases if c["half"] == h}) for h in halves}
    print(f"wrote {len(cases)} cases to {out}; calls per half {halves}; tasks per half {tasks}; message roles {roles}")


if __name__ == "__main__":
    main()
