from __future__ import annotations

import json

from . import AuditResult


def render_terminal_report(result: AuditResult) -> str:
    lines = []
    lines.append("=== OLMo 3 Filter Audit ===")
    lines.append(f"Input: {result.input_name} ({result.char_count} chars, {result.word_count} words)")
    lines.append("")
    lines.append(f"{'Stage':30} {'Result':8} Details")
    lines.append(f"{'-'*30} {'-'*8} {'-'*30}")
    for f in result.filters:
        detail = ""
        if f.details:
            detail = f.details
        elif f.value is not None:
            detail = str(f.value)
        lines.append(f"{f.name:30} {f.result:8} {detail}")

    passed, failed, _ = result.counts()
    lines.append("")
    lines.append(f"Overall: {result.overall} ({failed} failed, {passed} passed)")
    return "\n".join(lines)


def render_json_report(result: AuditResult) -> str:
    return json.dumps(result.to_json(), indent=2)
