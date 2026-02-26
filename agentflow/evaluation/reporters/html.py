"""
HTML reporter for evaluation results.

Generates interactive HTML reports for evaluation results.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

from agentflow.evaluation.reporters._utils import (
    case_display_name,
    case_status_info,
    format_timestamp,
    format_tool_calls,
)
from agentflow.evaluation.reporters.base import BaseReporter


if TYPE_CHECKING:
    from agentflow.evaluation.eval_result import EvalCaseResult, EvalReport


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --color-pass: #22c55e;
            --color-fail: #ef4444;
            --color-warn: #f59e0b;
            --color-bg: #f8fafc;
            --color-card: #ffffff;
            --color-border: #e2e8f0;
            --color-text: #1e293b;
            --color-muted: #64748b;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                         Oxygen, Ubuntu, sans-serif;
            background-color: var(--color-bg);
            color: var(--color-text);
            line-height: 1.6;
            padding: 2rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            margin-bottom: 2rem;
        }}

        h1 {{
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }}

        .timestamp {{
            color: var(--color-muted);
            font-size: 0.875rem;
        }}

        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .stat-card {{
            background: var(--color-card);
            border: 1px solid var(--color-border);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }}

        .stat-value {{
            font-size: 2rem;
            font-weight: 700;
        }}

        .stat-label {{
            color: var(--color-muted);
            font-size: 0.875rem;
        }}

        .stat-pass .stat-value {{ color: var(--color-pass); }}
        .stat-fail .stat-value {{ color: var(--color-fail); }}
        .stat-warn .stat-value {{ color: var(--color-warn); }}

        .progress-bar {{
            background: var(--color-border);
            border-radius: 999px;
            height: 8px;
            margin-top: 0.5rem;
            overflow: hidden;
        }}

        .progress-fill {{
            background: var(--color-pass);
            height: 100%;
            transition: width 0.3s ease;
        }}

        section {{
            margin-bottom: 2rem;
        }}

        h2 {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
        }}

        .case-list {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}

        .case-item {{
            background: var(--color-card);
            border: 1px solid var(--color-border);
            border-radius: 8px;
            padding: 1rem;
            cursor: pointer;
            transition: box-shadow 0.2s;
        }}

        .case-item:hover {{
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }}

        .case-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .case-status {{
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: white;
            font-size: 0.75rem;
        }}

        .case-status.pass {{ background: var(--color-pass); }}
        .case-status.fail {{ background: var(--color-fail); }}
        .case-status.error {{ background: var(--color-warn); }}

        .case-name {{
            flex: 1;
            font-weight: 500;
        }}

        .case-duration {{
            color: var(--color-muted);
            font-size: 0.875rem;
        }}

        .case-details {{
            display: none;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid var(--color-border);
        }}

        .case-item.expanded .case-details {{
            display: block;
        }}

        .criterion-list {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}

        .criterion-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem;
            background: var(--color-bg);
            border-radius: 4px;
        }}

        .criterion-icon {{
            font-size: 0.875rem;
        }}

        .criterion-name {{
            flex: 1;
        }}

        .criterion-score {{
            font-family: monospace;
        }}

        .error-message {{
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-radius: 4px;
            padding: 0.75rem;
            color: #991b1b;
            margin-top: 0.5rem;
            font-family: monospace;
            font-size: 0.875rem;
        }}

        /* --- Rich execution data sections --- */
        .detail-section {{
            margin-top: 0.75rem;
        }}

        .detail-section summary {{
            cursor: pointer;
            font-weight: 600;
            font-size: 0.875rem;
            color: var(--color-text);
            padding: 0.25rem 0;
            user-select: none;
        }}

        .response-box {{
            background: #f1f5f9;
            border: 1px solid var(--color-border);
            border-radius: 4px;
            padding: 0.75rem;
            margin-top: 0.5rem;
            white-space: pre-wrap;
            word-break: break-word;
            font-family: monospace;
            font-size: 0.8rem;
            max-height: 200px;
            overflow-y: auto;
        }}

        .tool-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 0.5rem;
            font-size: 0.8rem;
        }}

        .tool-table th, .tool-table td {{
            border: 1px solid var(--color-border);
            padding: 0.5rem;
            text-align: left;
        }}

        .tool-table th {{
            background: #f1f5f9;
            font-weight: 600;
        }}

        .tool-table td {{
            font-family: monospace;
            word-break: break-all;
        }}

        .trajectory-timeline {{
            margin-top: 0.5rem;
            padding-left: 1rem;
            border-left: 2px solid var(--color-border);
        }}

        .traj-step {{
            padding: 0.25rem 0 0.25rem 0.75rem;
            font-size: 0.8rem;
            position: relative;
        }}

        .traj-step::before {{
            content: '';
            position: absolute;
            left: -0.45rem;
            top: 0.55rem;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--color-muted);
        }}

        .traj-step.node::before {{ background: #6366f1; }}
        .traj-step.tool::before {{ background: #f59e0b; }}

        .traj-label {{ font-weight: 600; }}
        .traj-detail {{ color: var(--color-muted); margin-left: 0.25rem; }}

        .node-box {{
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 4px;
            padding: 0.5rem 0.75rem;
            margin-top: 0.35rem;
            font-size: 0.8rem;
        }}

        .node-box-title {{
            font-weight: 600;
            color: #1e40af;
        }}

        .node-box p {{
            margin: 0.25rem 0 0 0;
            font-family: monospace;
            white-space: pre-wrap;
            word-break: break-word;
            color: var(--color-text);
        }}

        .criterion-reason {{
            color: var(--color-muted);
            font-size: 0.8rem;
            margin-left: 1.5rem;
            margin-top: 0.1rem;
        }}

        .filter-bar {{
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}

        .filter-btn {{
            padding: 0.5rem 1rem;
            border: 1px solid var(--color-border);
            background: var(--color-card);
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.875rem;
        }}

        .filter-btn.active {{
            background: var(--color-text);
            color: white;
        }}

        footer {{
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--color-border);
            color: var(--color-muted);
            font-size: 0.875rem;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 {title}</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>

        <section class="summary">
            <div class="stat-card">
                <div class="stat-value">{total_cases}</div>
                <div class="stat-label">Total Cases</div>
            </div>
            <div class="stat-card stat-pass">
                <div class="stat-value">{passed_cases}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat-card stat-fail">
                <div class="stat-value">{failed_cases}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card stat-warn">
                <div class="stat-value">{error_cases}</div>
                <div class="stat-label">Errors</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{pass_rate_pct}%</div>
                <div class="stat-label">Pass Rate</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {pass_rate_pct}%"></div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{duration}s</div>
                <div class="stat-label">Duration</div>
            </div>
        </section>

        <section>
            <h2>Test Cases</h2>
            <div class="filter-bar">
                <button class="filter-btn active" data-filter="all">All</button>
                <button class="filter-btn" data-filter="pass">Passed</button>
                <button class="filter-btn" data-filter="fail">Failed</button>
                <button class="filter-btn" data-filter="error">Errors</button>
            </div>
            <div class="case-list">
{case_items}
            </div>
        </section>

        <footer>
            <p>Generated by Agentflow Evaluation Framework</p>
        </footer>
    </div>

    <script>
        // Toggle case details
        document.querySelectorAll('.case-item').forEach(item => {{
            item.addEventListener('click', () => {{
                item.classList.toggle('expanded');
            }});
        }});

        // Filter buttons
        document.querySelectorAll('.filter-btn').forEach(btn => {{
            btn.addEventListener('click', () => {{
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                const filter = btn.dataset.filter;
                document.querySelectorAll('.case-item').forEach(item => {{
                    if (filter === 'all') {{
                        item.style.display = '';
                    }} else {{
                        item.style.display = item.dataset.status === filter ? '' : 'none';
                    }}
                }});
            }});
        }});
    </script>
</body>
</html>
"""


CASE_ITEM_TEMPLATE = """
            <div class="case-item" data-status="{status}">
                <div class="case-header">
                    <div class="case-status {status}">{icon}</div>
                    <span class="case-name">{name}</span>
                    <span class="case-duration">{duration}s</span>
                </div>
                <div class="case-details">
{metadata_section}
{response_section}
{tool_section}
{trajectory_section}
{node_visits_section}
{node_section}
{messages_section}
{turn_results_section}
                    <div class="criterion-list">
{criteria_items}
                    </div>
{error_section}
                </div>
            </div>
"""


class HTMLReporter(BaseReporter):
    """Generate interactive HTML reports for evaluation results.

    Creates a self-contained HTML file with styled output and
    interactive features for exploring test results.

    Example:
        ```python
        reporter = HTMLReporter()
        reporter.save(report, "results/report.html")
        ```
    """

    def __init__(
        self,
        include_details: bool = True,
        include_actual_response: bool = True,
        include_tool_call_details: bool = True,
        include_node_responses: bool = True,
        include_trajectory: bool = True,
    ):
        """Initialize the HTML reporter.

        Args:
            include_details: Whether to include criterion details.
            include_actual_response: Show agent response text.
            include_tool_call_details: Show tool calls with args/results.
            include_node_responses: Show per-node intermediate data.
            include_trajectory: Show execution trajectory timeline.
        """
        self.include_details = include_details
        self.include_actual_response = include_actual_response
        self.include_tool_call_details = include_tool_call_details
        self.include_node_responses = include_node_responses
        self.include_trajectory = include_trajectory

    # --- BaseReporter interface ---

    def generate(self, report: EvalReport, output_dir: str | None = None) -> str | None:
        """Generate an HTML report.

        If *output_dir* is provided the file is written there; otherwise
        the report HTML is returned as a string.
        """
        if output_dir is None:
            return self.to_html(report)
        path = str(Path(output_dir) / "report.html")
        self.save(report, path)
        return path

    # --- Existing public API (preserved for backward compat) ---

    def to_html(self, report: EvalReport) -> str:
        """Convert report to HTML string."""
        title = report.eval_set_name or report.eval_set_id
        timestamp = format_timestamp(report.timestamp)

        case_items = [self._render_case(r) for r in report.results]

        return HTML_TEMPLATE.format(
            title=html.escape(title),
            timestamp=timestamp,
            total_cases=report.summary.total_cases,
            passed_cases=report.summary.passed_cases,
            failed_cases=report.summary.failed_cases,
            error_cases=report.summary.error_cases,
            pass_rate_pct=f"{report.summary.pass_rate * 100:.0f}",
            duration=f"{report.summary.total_duration_seconds:.2f}",
            case_items="\n".join(case_items),
        )

    def _render_case(self, result: EvalCaseResult) -> str:  # noqa: PLR0912, PLR0915
        """Render a single case item with rich execution data."""
        status, icon = case_status_info(result)
        name = html.escape(case_display_name(result))
        duration = f"{result.duration_seconds:.2f}"

        # --- Agent actual response (no truncation) ---
        response_section = ""
        if self.include_actual_response and getattr(result, "actual_response", None):
            escaped = html.escape(result.actual_response)
            response_section = (
                '                    <details class="detail-section" open>'
                "<summary>Agent Response</summary>"
                f'<div class="response-box">{escaped}</div>'
                "</details>"
            )

        # --- Tool calls table (includes call_id, no truncation) ---
        tool_section = ""
        if self.include_tool_call_details and getattr(result, "actual_tool_calls", None):
            tools = format_tool_calls(result.actual_tool_calls)
            if tools:
                rows = []
                for t in tools:
                    rows.append(
                        "<tr>"
                        f"<td>{html.escape(t['name'])}</td>"
                        f"<td>{html.escape(t.get('call_id', ''))}</td>"
                        f"<td>{html.escape(t['args'])}</td>"
                        f"<td>{html.escape(t['result'])}</td>"
                        "</tr>"
                    )
                tool_section = (
                    '                    <details class="detail-section">'
                    f"<summary>Tool Calls ({len(tools)})</summary>"
                    '<table class="tool-table"><thead><tr>'
                    "<th>Tool</th><th>Call ID</th><th>Arguments</th><th>Result</th>"
                    "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table></details>"
                )

        # --- Trajectory timeline (full step details) ---
        trajectory_section = ""
        if self.include_trajectory and getattr(result, "actual_trajectory", None):
            steps = []
            for step in result.actual_trajectory:
                if isinstance(step, dict):
                    stype = step.get("step_type", step.get("type", "node")).lower()
                    sname = html.escape(
                        str(step.get("name", step.get("node", step.get("tool", ""))))
                    )
                    sargs = step.get("args", {})
                    smeta = step.get("metadata", {})
                    stimestamp = step.get("timestamp")
                elif hasattr(step, "step_type"):
                    stype = (
                        step.step_type.value
                        if hasattr(step.step_type, "value")
                        else str(step.step_type)
                    )
                    sname = html.escape(str(step.name))
                    sargs = step.args if hasattr(step, "args") else {}
                    smeta = step.metadata if hasattr(step, "metadata") else {}
                    stimestamp = step.timestamp if hasattr(step, "timestamp") else None
                else:
                    stype = "node"
                    sname = html.escape(str(step))
                    sargs = {}
                    smeta = {}
                    stimestamp = None
                css_cls = "tool" if stype == "tool" else "node"
                detail_parts = [f'<span class="traj-detail">{sname}</span>']
                if sargs:
                    import json as _json

                    try:
                        args_str = _json.dumps(sargs, default=str, ensure_ascii=False)
                    except (TypeError, ValueError):
                        args_str = str(sargs)
                    _muted = "color:var(--color-muted);font-size:0.75rem;"
                    detail_parts.append(
                        f'<span class="traj-detail" style="{_muted}">'
                        f" args: {html.escape(args_str)}</span>"
                    )
                if smeta:
                    _muted = "color:var(--color-muted);font-size:0.75rem;"
                    detail_parts.append(
                        f'<span class="traj-detail" style="{_muted}">'
                        f" meta: {html.escape(str(smeta))}</span>"
                    )
                if stimestamp:
                    _muted = "color:var(--color-muted);font-size:0.75rem;"
                    detail_parts.append(
                        f'<span class="traj-detail" style="{_muted}">' f" @{stimestamp}</span>"
                    )
                steps.append(
                    f'<div class="traj-step {css_cls}">'
                    f'<span class="traj-label">[{stype.upper()}]</span>'
                    + "".join(detail_parts)
                    + "</div>"
                )
            if steps:
                trajectory_section = (
                    '                    <details class="detail-section">'
                    f"<summary>Execution Trajectory ({len(steps)} steps)</summary>"
                    '<div class="trajectory-timeline">' + "\n".join(steps) + "</div></details>"
                )

        # --- Node responses (full fields, correct key: response_text) ---
        node_section = ""
        if self.include_node_responses and getattr(result, "node_responses", None):
            boxes = []
            for nr in result.node_responses:
                if isinstance(nr, dict):
                    nname = html.escape(str(nr.get("node_name", "?")))
                    nout = html.escape(str(nr.get("response_text", "")))
                    nr_tools = nr.get("tool_call_names", [])
                    nr_final = nr.get("is_final", False)
                    nr_has_tools = nr.get("has_tool_calls", False)
                    nr_timestamp = nr.get("timestamp", 0)
                    nr_input_msgs = nr.get("input_messages", [])
                else:
                    nname = html.escape(str(nr))
                    nout = ""
                    nr_tools = []
                    nr_final = False
                    nr_has_tools = False
                    nr_timestamp = 0
                    nr_input_msgs = []
                final_badge = (
                    ' <span style="color:var(--color-pass);font-weight:600;">[FINAL]</span>'
                    if nr_final
                    else ""
                )
                _ms = "color:var(--color-muted);font-size:0.75rem;"
                tools_info = (
                    f'<br/><span style="{_ms}">'
                    f'tools: {html.escape(", ".join(nr_tools))}'
                    '</span>'
                    if nr_tools
                    else ""
                )
                tool_calls_flag = (
                    f'<br/><span style="{_ms}">' "has_tool_calls: True</span>"
                    if nr_has_tools
                    else ""
                )
                ts_info = (
                    f'<br/><span style="{_ms}">' f"timestamp: {nr_timestamp}</span>"
                    if nr_timestamp
                    else ""
                )
                input_info = ""
                if nr_input_msgs:
                    input_info = (
                        f'<br/><span style="{_ms}">'
                        f"input_messages: {len(nr_input_msgs)} messages"
                        "</span>"
                    )
                boxes.append(
                    f'<div class="node-box">'
                    f'<span class="node-box-title">{nname}{final_badge}</span>'
                    f"<p>{nout}</p>"
                    f"{tools_info}{tool_calls_flag}{ts_info}{input_info}"
                    f"</div>"
                )
            if boxes:
                node_section = (
                    '                    <details class="detail-section">'
                    f"<summary>Node Responses ({len(boxes)})</summary>"
                    + "\n".join(boxes)
                    + "</details>"
                )

        # --- Criteria (full details — no truncation) ---
        criteria_items: list[str] = []
        if self.include_details:
            for cr in result.criterion_results:
                cr_icon = "✓" if cr.passed else "✗"
                reason_html = ""
                if getattr(cr, "reason", None):
                    reason_html = f'<div class="criterion-reason">{html.escape(cr.reason)}</div>'
                # Render ALL details as a key-value list
                details_html = ""
                if cr.details:
                    detail_items = []
                    for dk, dv in cr.details.items():
                        if dk == "reason":
                            continue  # Already shown above
                        dv_str = html.escape(str(dv))
                        detail_items.append(
                            f'<span style="color:var(--color-muted);font-size:0.75rem;">'
                            f"<strong>{html.escape(dk)}</strong>: {dv_str}</span>"
                        )
                    if detail_items:
                        details_html = (
                            '<div style="margin-left:1.5rem;margin-top:0.25rem;">'
                            + "<br/>".join(detail_items)
                            + "</div>"
                        )
                error_html = ""
                if cr.error:
                    error_html = (
                        f'<div class="criterion-reason" style="color:#991b1b;">'
                        f"Error: {html.escape(cr.error)}</div>"
                    )
                criteria_items.append(
                    f'                        <div class="criterion-item">'
                    f'<span class="criterion-icon">{cr_icon}</span>'
                    f'<span class="criterion-name">{html.escape(cr.criterion)}</span>'
                    f'<span class="criterion-score">{cr.score:.2f} / {cr.threshold}</span>'
                    f"</div>{reason_html}{error_html}{details_html}"
                )

        # --- Error ---
        error_section = ""
        if result.error:
            error_section = (
                f'                    <div class="error-message">{html.escape(result.error)}</div>'
            )

        # --- Metadata ---
        metadata_section = ""
        if getattr(result, "metadata", None):
            meta_items = []
            for mk, mv in result.metadata.items():
                meta_items.append(
                    f'<span style="color:var(--color-muted);font-size:0.8rem;">'
                    f"<strong>{html.escape(str(mk))}</strong>: {html.escape(str(mv))}</span>"
                )
            if meta_items:
                metadata_section = (
                    '                    <details class="detail-section">'
                    "<summary>Metadata</summary>"
                    '<div style="padding:0.5rem;">' + "<br/>".join(meta_items) + "</div></details>"
                )

        # --- Node visits ---
        node_visits_section = ""
        if getattr(result, "node_visits", None):
            nv_html = " → ".join(html.escape(nv) for nv in result.node_visits)
            node_visits_section = (
                '                    <details class="detail-section">'
                f"<summary>Node Visits ({len(result.node_visits)})</summary>"
                '<div style="padding:0.5rem;font-family:monospace;'
                f'font-size:0.8rem;">{nv_html}</div>'
                "</details>"
            )

        # --- Messages ---
        messages_section = ""
        if getattr(result, "messages", None):
            msg_items = []
            for msg in result.messages:
                role = msg.get("role", "?") if isinstance(msg, dict) else "?"
                content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                role_color = (
                    "#6366f1"
                    if role == "user"
                    else "#22c55e"
                    if role == "assistant"
                    else "var(--color-muted)"
                )
                msg_items.append(
                    f'<div style="padding:0.25rem 0;font-size:0.8rem;">'
                    f'<span style="color:{role_color};font-weight:600;">'
                    f"[{html.escape(role)}]</span> "
                    f"{html.escape(content)}</div>"
                )
            if msg_items:
                messages_section = (
                    '                    <details class="detail-section">'
                    f"<summary>Messages ({len(result.messages)})</summary>"
                    '<div style="padding:0.5rem;">' + "\n".join(msg_items) + "</div></details>"
                )

        # --- Turn results (multi-turn per-turn data) ---
        turn_results_section = ""
        if getattr(result, "turn_results", None):
            turn_items = []
            for tr in result.turn_results:
                tidx = tr.get("turn_index", "?")
                user_input = html.escape(str(tr.get("user_input", "")))
                agent_resp = html.escape(str(tr.get("agent_response", "")))
                turn_tcs = tr.get("tool_calls", [])
                turn_nv = tr.get("node_visits", [])
                _ts = "color:var(--color-muted);font-size:0.75rem;"
                tc_info = (
                    f'<br/><span style="{_ts}">' f"tool_calls: {len(turn_tcs)}</span>"
                    if turn_tcs
                    else ""
                )
                nv_info = (
                    f'<br/><span style="{_ts}">'
                    f'nodes: {html.escape(" → ".join(turn_nv))}'
                    '</span>'
                    if turn_nv
                    else ""
                )
                turn_items.append(
                    f'<div class="node-box">'
                    f'<span class="node-box-title">Turn {tidx}</span>'
                    f"<p><strong>User:</strong> {user_input}</p>"
                    f"<p><strong>Agent:</strong> {agent_resp}</p>"
                    f"{tc_info}{nv_info}"
                    f"</div>"
                )
            if turn_items:
                turn_results_section = (
                    '                    <details class="detail-section">'
                    f"<summary>Turn-by-Turn Results ({len(result.turn_results)} turns)</summary>"
                    + "\n".join(turn_items)
                    + "</details>"
                )

        return CASE_ITEM_TEMPLATE.format(
            status=status,
            icon=icon,
            name=name,
            duration=duration,
            metadata_section=metadata_section,
            response_section=response_section,
            tool_section=tool_section,
            trajectory_section=trajectory_section,
            node_visits_section=node_visits_section,
            node_section=node_section,
            messages_section=messages_section,
            turn_results_section=turn_results_section,
            criteria_items="\n".join(criteria_items),
            error_section=error_section,
        )

    def save(self, report: EvalReport, path: str) -> None:
        """Save report to HTML file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            f.write(self.to_html(report))

    @classmethod
    def quick_save(cls, report: EvalReport, path: str) -> None:
        """Convenience method to quickly save a report."""
        reporter = cls()
        reporter.save(report, path)
