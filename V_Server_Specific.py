import json
import os
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st
import requests
from requests.exceptions import RequestException

try:
    import plotly.graph_objs as go

    PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    go = None
    PLOTLY_AVAILABLE = False

try:
    from src.agent_orch.agents.base_agent import BaseAgent
except ModuleNotFoundError:
    import sys

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from src.agent_orch.agents.base_agent import BaseAgent

class DashboardPublisherAgent(BaseAgent):
    def __init__(self, name: str = "dashboard_publisher"):
        super().__init__(name)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect outputs from upstream agents and persist them for the dashboard.
        """
        dashboard_data: Dict[str, Any] = {}

        def to_records(payload) -> List[Dict[str, Any]]:
            if isinstance(payload, pd.DataFrame):
                return payload.to_dict(orient="records")
            if isinstance(payload, list):
                records: List[Dict[str, Any]] = []
                for item in payload:
                    sub = to_records(item)
                    if sub:
                        records.extend(sub)
                return records
            if isinstance(payload, dict):
                return [payload]
            return []

        def normalise_forecasts(raw: Any) -> Dict[str, Dict[str, List[List[Dict[str, Any]]]]]:
            normalised: Dict[str, Dict[str, List[List[Dict[str, Any]]]]] = {}

            def push(server_id: str, model_name: str, records: List[Dict[str, Any]]):
                if not records:
                    return
                server_bucket = normalised.setdefault(server_id, {})
                server_bucket.setdefault(model_name, []).append(records)

            def process_entry(entry_key: str, entry_value: Any):
                if isinstance(entry_value, dict):
                    for sub_key, sub_value in entry_value.items():
                        records = to_records(sub_value)
                        if not records:
                            continue
                        server_id = str(records[0].get("server_id", entry_key))
                        model_label = str(records[0].get("model") or sub_key or entry_key)
                        push(server_id, model_label, records)
                else:
                    records = to_records(entry_value)
                    if records:
                        server_id = str(records[0].get("server_id", entry_key))
                        model_label = str(records[0].get("model") or entry_key)
                        push(server_id, model_label, records)

            if isinstance(raw, dict):
                for key, value in raw.items():
                    process_entry(str(key), value)
            elif isinstance(raw, list):
                for idx, value in enumerate(raw):
                    process_entry(f"series_{idx}", value)

            return normalised

        dashboard_data["forecasts"] = normalise_forecasts(state.get("forecasts", {}))
        dashboard_data["recommendation"] = state.get("recommendation", {})
        dashboard_data["cost_impact"] = state.get("cost_impact", {})
        dashboard_data["metadata"] = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "resource_type": state.get("resource_type", ""),
            "server_id": state.get("server_id", ""),
        }
        data_dir = Path(__file__).resolve().parents[3] / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        dashboard_path = data_dir / "dashboard_data.json"
        with dashboard_path.open("w", encoding="utf-8") as fp:
            json.dump(dashboard_data, fp, indent=4, default=str)

        state["dashboard"] = f"Dashboard data published to {dashboard_path}"
        return state

def launch_dashboard():
    """Entry point for Streamlit UI. Separated to avoid running during imports."""
    DEFAULT_API_BASE = "http://localhost:8000"

    st.set_page_config(page_title="AI Rightsizing & Cost Dashboard", layout="wide")

    def default_column_label(raw: str) -> str:
        label = raw.replace("_", " ").title()
        return label.replace("Id", "ID")

    def humanise_dataframe(
        df: pd.DataFrame,
        column_map: Dict[str, str],
        column_order: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        if column_order:
            ordered = [col for col in column_order if col in df.columns]
            df = df[ordered + [col for col in df.columns if col not in ordered]]
        rename_map = {col: column_map.get(col, default_column_label(col)) for col in df.columns}
        return df.rename(columns=rename_map)

    # TABLE_STYLE = """
    # <style>
    # .dark-table-caption {
    #     font-family: 'Inter', sans-serif;
    #     font-size: 0.95rem;
    #     font-weight: 600;
    #     color: #e2e8f0;
    #     margin-bottom: 0.5rem;
    #     text-transform: uppercase;
    #     letter-spacing: 0.08em;
    # }
    # .dark-table-wrapper {
    #     border-radius: 14px;
    #     border: 1px solid rgba(148, 163, 184, 0.25);
    #     overflow: hidden;
    #     box-shadow: 0 18px 36px rgba(2, 6, 23, 0.55);
    #     background: linear-gradient(180deg, rgba(15, 23, 42, 0.9), rgba(2, 6, 23, 0.95));
    # }
    # .dark-table {
    #     width: 100%;
    #     border-collapse: collapse;
    #     font-family: 'Inter', sans-serif;
    # }
    # .dark-table thead th {
    #     padding: 0.85rem 1rem;
    #     text-align: left;
    #     font-size: 0.78rem;
    #     letter-spacing: 0.1em;
    #     font-weight: 700;
    #     text-transform: uppercase;
    #     color: #93c5fd;
    #     background: rgba(30, 41, 59, 0.9);
    #     border-bottom: 1px solid rgba(148, 163, 184, 0.35);
    # }
    # .dark-table tbody td {
    #     padding: 0.9rem 1rem;
    #     font-size: 0.96rem;
    #     color: #f8fafc;
    #     border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    # }
    # .dark-table tbody tr:nth-child(odd) {
    #     background: rgba(17, 24, 39, 0.92);
    # }
    # .dark-table tbody tr:nth-child(even) {
    #     background: rgba(11, 17, 32, 0.92);
    # }
    # .dark-table tbody tr:hover {
    #     background: rgba(59, 130, 246, 0.22);
    #     transition: background 0.2s ease;
    # }
    # .dark-table td.highlight-cell {
    #     background: linear-gradient(135deg, rgba(59, 130, 246, 0.85), rgba(37, 99, 235, 0.85)) !important;
    #     color: #fef9c3 !important;
    #     font-weight: 600;
    # }
    # </style>
    # """
    TABLE_STYLE = """
    <style>
    /* ============================
    Professional Dark Table Theme
    - Accessible contrast (AA+)
    - Design tokens via CSS variables
    - Smooth but subtle interactions
    - Reduced-motion + prefers-color-scheme support
    - Compact/comfortable density toggles
    ============================ */

    :root {
    /* Color tokens */
    --surface-0: #0b1220;         /* page background */
    --surface-1: #0f172a;         /* card background */
    --surface-2: #111827;         /* row odd */
    --surface-3: #0c1424;         /* row even */
    --surface-4: #1f2937;         /* header bg */

    --text-0: #f8fafc;            /* primary text */
    --text-1: #c7d2fe;            /* header text accent */
    --muted:  #94a3b8;            /* borders/subtle text */

    --brand-0: #3b82f6;           /* primary accent */
    --brand-1: #2563eb;           /* accent darker */
    --brand-2: #60a5fa;           /* accent lighter */

    --focus:  #22d3ee;            /* focus ring (cyan) */
    --warn:   #f59e0b;
    --good:   #22c55e;
    --bad:    #ef4444;

    /* Elevation & effects */
    --ring: 0 0 0 1px rgba(148, 163, 184, 0.18);
    --shadow-lg: 0 18px 36px rgba(2, 6, 23, 0.55), 0 6px 16px rgba(2, 6, 23, 0.35);
    --blur: saturate(135%) blur(8px);

    /* Radii & spacing */
    --radius-lg: 14px;
    --radius-sm: 10px;
    --px: 1rem;                   /* base cell padding X */
    --py: 0.85rem;                /* base cell padding Y */
    --px-compact: 0.75rem;
    --py-compact: 0.6rem;

    /* Typography */
    --font-sans: 'Inter', ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji', 'Segoe UI Emoji';
    --fs-body: 0.96rem;
    --fs-head: 0.78rem;
    --ls-head: 0.08em;
    }

    @media (prefers-color-scheme: dark) {
    :root {
        /* (kept for future overrides if needed) */
    }
    }

    /* Caption */
    .dark-table-caption {
    font-family: var(--font-sans);
    font-size: 0.92rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.65rem;
    text-transform: uppercase;
    letter-spacing: var(--ls-head);
    opacity: 0.92;
    }

    /* Wrapper */
    .dark-table-wrapper {
    border-radius: var(--radius-lg);
    border: 1px solid rgba(148, 163, 184, 0.20);
    overflow: auto; /* enables horizontal scroll if needed */
    background:
        linear-gradient(180deg, rgba(15, 23, 42, 0.88), rgba(2, 6, 23, 0.95)),
        radial-gradient(1200px 300px at 10% -10%, rgba(59,130,246,0.12), transparent 60%),
        radial-gradient(900px 240px at 90% -20%, rgba(2,132,199,0.10), transparent 60%);
    box-shadow: var(--shadow-lg);
    backdrop-filter: var(--blur);
    }

    /* Table */
    .dark-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-sans);
    font-size: var(--fs-body);
    min-width: 640px; /* keep structure on tiny screens, wrapper scrolls */
    }

    /* Header */
    .dark-table thead th {
    position: sticky;
    top: 0;
    z-index: 1;
    padding: 0.85rem 1rem;
    text-align: left;
    font-size: var(--fs-head);
    letter-spacing: 0.10em;
    font-weight: 700;
    text-transform: uppercase;
    color: var(--text-1);
    background:
        linear-gradient(180deg, rgba(31, 41, 59, 0.95), rgba(31, 41, 59, 0.85));
    border-bottom: 1px solid rgba(148, 163, 184, 0.35);
    white-space: nowrap;
    }

    /* Optional: sort affordance (just add .is-sortable on TH) */
    .dark-table thead th.is-sortable {
    cursor: pointer;
    }
    .dark-table thead th.is-sortable:hover {
    color: var(--brand-2);
    }

    /* Body cells */
    .dark-table tbody td {
    padding: var(--py) var(--px);
    font-size: var(--fs-body);
    color: var(--text-0);
    border-bottom: 1px solid rgba(148, 163, 184, 0.14);
    vertical-align: middle;
    }

    /* Density controls: add .is-compact to .dark-table */
    .dark-table.is-compact tbody td,
    .dark-table.is-compact thead th {
    padding: var(--py-compact) var(--px-compact);
    }

    /* Zebra stripes */
    .dark-table tbody tr:nth-child(odd) {
    background: rgba(17, 24, 39, 0.92);
    }
    .dark-table tbody tr:nth-child(even) {
    background: rgba(12, 20, 36, 0.92);
    }

    /* Row hover & active states */
    .dark-table tbody tr {
    transition: background 140ms ease, transform 140ms ease;
    }
    .dark-table tbody tr:hover {
    background:
        linear-gradient(180deg, rgba(59,130,246,0.18), rgba(37,99,235,0.16));
    }
    .dark-table tbody tr:is(.is-active, .selected) {
    background:
        linear-gradient(180deg, rgba(59,130,246,0.28), rgba(37,99,235,0.26));
    box-shadow: inset 0 0 0 1px rgba(96, 165, 250, 0.35);
    }

    /* Highlighted cell */
    .dark-table td.highlight-cell {
    background:
        linear-gradient(135deg, rgba(59,130,246,0.88), rgba(37,99,235,0.88)) !important;
    color: #fef9c3 !important;
    font-weight: 650;
    border-radius: var(--radius-sm);
    }

    /* Numeric alignment helper */
    .dark-table td.is-numeric,
    .dark-table th.is-numeric {
    text-align: right;
    font-variant-numeric: tabular-nums;
    }

    /* Status chips (optional utility classes) */
    .dark-table .chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4em;
    padding: 0.18rem 0.5rem;
    border-radius: 999px;
    font-size: 0.82rem;
    line-height: 1;
    border: 1px solid rgba(148,163,184,0.22);
    background: rgba(148,163,184,0.10);
    }
    .dark-table .chip--good { border-color: rgba(34,197,94,0.35); background: rgba(34,197,94,0.12); color: #bbf7d0; }
    .dark-table .chip--warn { border-color: rgba(245,158,11,0.35); background: rgba(245,158,11,0.12); color: #fde68a; }
    .dark-table .chip--bad  { border-color: rgba(239,68,68,0.35);  background: rgba(239,68,68,0.12);  color: #fecaca; }

    /* Links inside cells */
    .dark-table a {
    color: var(--brand-2);
    text-decoration: none;
    border-bottom: 1px dashed rgba(96,165,250,0.35);
    }
    .dark-table a:hover {
    color: #dbeafe;
    border-bottom-style: solid;
    }

    /* Focus styles for keyboard navigation */
    .dark-table :is(th, td):focus-within {
    outline: 2px solid var(--focus);
    outline-offset: -2px;
    border-radius: 6px;
    }

    /* Footer (if used) */
    .dark-table tfoot td {
    padding: 0.85rem 1rem;
    color: var(--muted);
    background: rgba(17, 24, 39, 0.95);
    }

    /* Subtle row separators on dense data */
    .dark-table tbody tr + tr td {
    box-shadow: inset 0 1px 0 rgba(148, 163, 184, 0.06);
    }

    /* Reduce motion for sensitive users */
    @media (prefers-reduced-motion: reduce) {
    .dark-table tbody tr {
        transition: none;
    }
    }

    /* Optional: elevate header on scroll (visual hint) */
    .dark-table-wrapper:has(thead th) {
    scroll-padding-top: 0.5rem;
    }
    .dark-table-wrapper:has(thead th)::after {
    content: "";
    position: sticky;
    top: 0;
    height: 0.01px; /* creates a stacking context for sticky header */
    display: block;
    }

    /* Caption spacing when placed above wrapper */
    .dark-table-caption + .dark-table-wrapper {
    margin-top: 0.25rem;
    }
    </style>
    """


    def ensure_table_css_loaded():
        state_key = "_dark_table_css_loaded"
        if not st.session_state.get(state_key, False):
            st.markdown(TABLE_STYLE, unsafe_allow_html=True)
            st.session_state[state_key] = True

    def render_styled_table(
        df: pd.DataFrame,
        *,
        percent_cols: Optional[Dict[str, int]] = None,
        currency_cols: Optional[Dict[str, int]] = None,
        highlight_cols: Optional[List[str]] = None,
        caption: Optional[str] = None,
    ) -> None:
        if df.empty:
            st.info("No data available.")
            return

        ensure_table_css_loaded()

        df = df.copy().reset_index(drop=True)

        percent_cols = percent_cols or {}
        currency_cols = currency_cols or {}
        highlight_set = set(highlight_cols or [])

        def _format_percent(value: Any, decimals: int) -> str:
            if pd.isna(value):
                return "—"
            try:
                return f"{float(value):.{decimals}f}%"
            except (TypeError, ValueError):
                text = str(value).strip()
                return text or "—"

        def _format_currency(value: Any, decimals: int) -> str:
            if pd.isna(value):
                return "—"
            try:
                return f"${float(value):,.{decimals}f}"
            except (TypeError, ValueError):
                text = str(value).strip()
                return text or "—"

        for col, decimals in percent_cols.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda val: _format_percent(val, decimals))

        for col, decimals in currency_cols.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda val: _format_currency(val, decimals))

        df = df.fillna("—")

        header_html = "".join(f"<th>{escape(str(column))}</th>" for column in df.columns)
        body_rows: List[str] = []
        for _, row in df.iterrows():
            cells: List[str] = []
            for column, value in row.items():
                cell_class = " class='highlight-cell'" if column in highlight_set else ""
                cells.append(f"<td{cell_class}>{escape(str(value))}</td>")
            body_rows.append(f"<tr>{''.join(cells)}</tr>")
        table_html = (
            "<div class='dark-table-wrapper'>"
            "<table class='dark-table'>"
            f"<thead><tr>{header_html}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody>"
            "</table>"
            "</div>"
        )
        if caption:
            st.markdown(f"<p class='dark-table-caption'>{escape(str(caption))}</p>", unsafe_allow_html=True)
        st.markdown(table_html, unsafe_allow_html=True)

    def render_forecast_plots(
        section_title: str,
        trace_payload: Optional[List[Dict[str, Any]]],
        *,
        x_label: str = "Date",
        y_label: str = "Usage",
    ) -> bool:
        if not PLOTLY_AVAILABLE:
            st.info(
                "Install Plotly (`pip install plotly`) to view interactive forecast charts."
            )
            return False

        if not trace_payload:
            return False

        figure = go.Figure()
        added = False
        for trace in trace_payload:
            if not isinstance(trace, dict):
                continue
            x_values = trace.get("x") or trace.get("ds")
            y_values = trace.get("y") or trace.get("yhat")
            if x_values is None or y_values is None:
                continue
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode=trace.get("mode", "lines"),
                    name=trace.get("name")
                    or trace.get("model")
                    or trace.get("label")
                    or "Series",
                    line=trace.get("line"),
                )
            )
            added = True

        if not added:
            return False

        figure.update_layout(
            title=section_title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_dark",
            margin=dict(l=0, r=0, t=48, b=0),
            height=420,
        )
        st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
        return True

    def api_get(url: str, *, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Dict[str, Any]:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def api_post(url: str, payload: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    @st.cache_data(ttl=120)
    def fetch_server_options(api_base_url: str) -> List[Dict[str, Any]]:
        """Return list of {vm_id, server_name, label} dicts."""
        data = api_get(f"{api_base_url.rstrip('/')}/servers", timeout=15)
        servers_raw = data.get("servers", [])
        result = []
        for item in servers_raw:
            if isinstance(item, dict):
                vm_id = str(item.get("vm_id", ""))
                name = item.get("server_name") or vm_id
                result.append({"vm_id": vm_id, "server_name": name, "label": f"{name} ({vm_id})"})
            else:
                result.append({"vm_id": str(item), "server_name": str(item), "label": str(item)})
        return result

    @st.cache_data(ttl=120)
    def fetch_resource_options(api_base_url: str, server_id: str) -> List[str]:
        data = api_get(
            f"{api_base_url.rstrip('/')}/servers/{server_id}/resources",
            timeout=15,
        )
        return data.get("resources", [])

    def fetch_state_snapshot(api_base_url: str, server_id: str, resource_type: str) -> Optional[Dict[str, Any]]:
        try:
            return api_get(
                f"{api_base_url.rstrip('/')}/state",
                params={"server_id": server_id, "resource_type": resource_type},
                timeout=20,
            )
        except (RequestException, ValueError):
            return None

    def display_cost_summary(cost_data: Dict[str, Any]):
        if not cost_data:
            st.info("Cost analysis not available.")
            return

        df_cost = pd.DataFrame([cost_data])
        numeric_cols = [
            "total_size",
            "unit_cost",
            "scale_percent",
            "base_cost",
            "new_cost",
            "percentage_change",
            "savings",
            "additional_cost",
        ]
        for col in numeric_cols:
            if col in df_cost.columns:
                df_cost[col] = pd.to_numeric(df_cost[col], errors="coerce")

        if "scale_action" in df_cost.columns:
            df_cost["scale_action"] = (
                df_cost["scale_action"].astype(str).str.replace("_", " ").str.title()
            )

        if "timestamp" in df_cost.columns:
            df_cost["timestamp"] = pd.to_datetime(df_cost["timestamp"], errors="coerce").dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        if {"base_cost", "new_cost"}.issubset(df_cost.columns):
            base_cost_val = df_cost.at[0, "base_cost"]
            projected_cost_val = df_cost.at[0, "new_cost"]
            if pd.notna(base_cost_val) and pd.notna(projected_cost_val):
                df_cost["monthly_cost_delta"] = projected_cost_val - base_cost_val
                df_cost["monthly_cost_delta"] = df_cost["monthly_cost_delta"].round(2)

        df_cost = df_cost.round(2)

        cost_column_labels = {
            "timestamp": "Evaluated At",
            "resource_type": "Resource Type",
            "total_size": "Current Capacity Units",
            "unit_cost": "Unit Cost ($/Hr)",
            "scale_action": "Recommended Action",
            "scale_percent": "Scale Percentage (%)",
            "base_cost": "Baseline Monthly Cost ($)",
            "new_cost": "Projected Monthly Cost ($)",
            "savings": "Estimated Monthly Savings ($)",
            "additional_cost": "Estimated Monthly Increase ($)",
            "percentage_change": "Cost Change (%)",
            "monthly_cost_delta": "Monthly Cost Delta ($)",
        }
        cost_column_order = [
            "timestamp",
            "resource_type",
            "scale_action",
            "scale_percent",
            "total_size",
            "unit_cost",
            "base_cost",
            "new_cost",
            "monthly_cost_delta",
            "savings",
            "additional_cost",
            "percentage_change",
        ]
        df_cost = humanise_dataframe(df_cost, cost_column_labels, cost_column_order)
        render_styled_table(
            df_cost,
            percent_cols={
                "Scale Percentage (%)": 0,
                "Cost Change (%)": 2,
            },
            currency_cols={
                "Unit Cost ($/Hr)": 4,
                "Baseline Monthly Cost ($)": 2,
                "Projected Monthly Cost ($)": 2,
                "Monthly Cost Delta ($)": 2,
                "Estimated Monthly Savings ($)": 2,
                "Estimated Monthly Increase ($)": 2,
            },
            highlight_cols=[
                "Recommended Action",
                "Monthly Cost Delta ($)",
            ],
        )

        base_cost = cost_data.get("base_cost") or cost_data.get("current_cost")
        projected_cost = cost_data.get("new_cost") or cost_data.get("projected_cost")
        if base_cost not in (None, 0) and projected_cost is not None:
            diff = projected_cost - base_cost
            pct_change = (diff / base_cost) * 100 if base_cost else 0
            trend = "decrease" if diff < 0 else "increase"
            color = "green" if diff < 0 else "red"
            st.markdown(
                f"**Projected Cost Change:** "
                f"<span style='color:{color};font-size:18px;'>"
                f"{pct_change:.2f}% {trend} (${abs(diff):,.2f})</span>",
                unsafe_allow_html=True,
            )

    session_defaults = {
        "pipeline_result": None,
        "approval_result": None,
        "state_snapshot": None,
        "selected_server": "",
        "selected_resource": "",
        "active_server": "",
        "active_resource": "",
        "api_base_url": DEFAULT_API_BASE,
        # CR workflow state
        "cr_created": False,
        "cr_sys_id": None,
        "cr_number": None,
        "cr_approved": False,
        "cr_cancelled": False,
        "cr_approval_message": "",
        "implement_result": None,
    }
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)

    st.title("AI-Based Rightsizing & Cost Optimization Dashboard")
    st.markdown("Use this dashboard to run the forecasting pipeline, review recommendations, and submit scaling approval.")

    st.session_state.setdefault("api_base_url", DEFAULT_API_BASE)
    env_api_override = os.getenv("DASHBOARD_API_BASE_URL", "").strip()
    if env_api_override:
        st.session_state["api_base_url"] = env_api_override

    api_base_url = st.session_state["api_base_url"]
    # st.caption("Ensure `main.py` is running (FastAPI server).")

    resource_options: List[str] = []

    api_base_clean = api_base_url.strip()
    server_data: List[Dict[str, Any]] = []
    if api_base_clean:
        try:
            server_data = fetch_server_options(api_base_clean)
        except Exception as exc:
            st.warning(f"Could not load server list: {exc}")

# ----------------------------
# Build label → vm_id mapping
# ----------------------------
    server_label_to_id = {
        item["label"]: item["vm_id"] for item in server_data
    }
    server_labels = [
       item["label"] for item in server_data
    ]

# ----------------------------
# MULTI-SERVER SELECT (REPLACES selectbox)
# ----------------------------
    col_server, col_resource = st.columns(2)

    selected_servers = col_server.multiselect(
    "Select Servers",
    options=server_labels,
    key="active_servers"
    )

# resolve selected labels → vm_ids
    resolved_server_ids = [
    server_label_to_id[label]
    for label in selected_servers
    if label in server_label_to_id
    ]

    # ============================
    # BUILD BULK JOB LIST
    # ============================

    
# ----------------------------
# MULTI-RESOURCE SELECT
# ----------------------------
    if api_base_clean and resolved_server_ids:
        try:
        # fetch resources for first selected server (or modify backend for multi-server support)
            resource_options = fetch_resource_options(
                api_base_clean,
                resolved_server_ids[0]
            )
        except Exception as exc:
            st.warning(f"Could not load resources: {exc}")
            resource_options = []
    else:
        resource_options = []

    resource_choices = resource_options

    selected_resources = col_resource.multiselect(
    "Select Resource Types",
    options=resource_choices,
    key="active_resources"
    )
    resolved_resources = selected_resources
    # ============================
    # BUILD BULK JOB LIST
    # ============================

    jobs = [
        {
            "server_id": server_id,
            "resource_type": resource
        }
        for server_id in resolved_server_ids
        for resource in selected_resources
    ]

    use_bulk_pipeline = len(jobs) > 1

    stored_servers = st.session_state.get("selected_servers", [])
    stored_resources = st.session_state.get("selected_resources", [])
    
    if st.session_state.get("pipeline_result") and (
        resolved_server_ids != stored_servers
        or selected_resources != stored_resources
    ):
        st.session_state["pipeline_result"] = None
        st.session_state["approval_result"] = None
        st.session_state["state_snapshot"] = None
        st.session_state["selected_server"] = ""
        st.session_state["selected_resource"] = ""
        # Reset CR workflow
        st.session_state["cr_created"] = False
        st.session_state["cr_sys_id"] = None
        st.session_state["cr_number"] = None
        st.session_state["cr_approved"] = False
        st.session_state["cr_cancelled"] = False
        st.session_state["cr_approval_message"] = ""
        st.session_state["implement_result"] = None

    col_run, col_refresh = st.columns([2, 1])
    run_disabled = not (
        api_base_clean and
        resolved_server_ids and
        selected_resources
    )
    with col_run:
        if st.button(
            "Run Pipeline",
            type="primary",
            use_container_width=True,
            disabled=run_disabled
        ):

            with st.spinner("Running LangGraph pipeline..."):
                try:

                    all_results = []

                    for job in jobs:

                        payload = {
                            "server_ids": [job["server_id"]],
                            "resource_type": job["resource_type"]
                        }

                        # AUTO ROUTE PIPELINE
                        if use_bulk_pipeline:
                            url = f"{api_base_clean.rstrip('/')}/pipeline/bulk"
                        else:
                            url = f"{api_base_clean.rstrip('/')}/pipeline"
                    
                        result = api_post(url, payload)
                        all_results.append(result)
                    # STORE CLEAN STATE
                    st.session_state["pipeline_result"] = all_results
                    st.session_state["selected_servers"] = selected_servers
                    st.session_state["selected_resources"] = selected_resources
                    st.session_state["jobs"] = jobs

                    st.success("Pipeline completed successfully")

                except RequestException as exc:
                    detail_message = ""

                    if getattr(exc, "response", None) is not None:
                        try:
                            payload = exc.response.json()
                            if isinstance(payload, dict):
                                detail_message = payload.get("detail") or payload.get("message") or ""
                            else:
                                detail_message = str(payload)
                        except Exception:
                            detail_message = exc.response.text

                    st.error(f"Pipeline call failed: {exc}")

                    if detail_message:
                        st.error(detail_message)
    with col_refresh:
        if st.button(
            "Refresh State",
            use_container_width=True,
            disabled=not (
                api_base_clean
                and st.session_state["selected_server"]
                and st.session_state["selected_resource"]
            ),
        ):
            st.session_state["state_snapshot"] = fetch_state_snapshot(
                api_base_clean,
                st.session_state["selected_server"],
                st.session_state["selected_resource"],
            )
            st.info("State refreshed from backend.")

    pipeline_result = st.session_state["pipeline_result"]
    state_snapshot = st.session_state["state_snapshot"]
    approval_result = st.session_state["approval_result"]

    if not pipeline_result:
        st.info("Run the pipeline to view recommendations, anomalies, and cost analysis.")
        pipeline_result = None

    st.markdown("### Recommendation Overview")
    recommendations = []

    for item in pipeline_result or []:
        if isinstance(item, dict):
            rec = item.get("recommendation")
            if rec:
                recommendations.append(rec)

    # choose first recommendation (or you can merge later)
    recommendation = recommendations[0] if recommendations else {}
    if recommendation:
        rec_df = pd.DataFrame(recommendation)
        for col in ["avg_usage", "max_usage", "forecasted_max", "scale_percent"]:
            if col in rec_df.columns:
                rec_df[col] = pd.to_numeric(rec_df[col], errors="coerce")
        for col in ["avg_usage", "max_usage", "forecasted_max"]:
            if col in rec_df.columns:
                rec_df[col] = rec_df[col].round(2)
        if "scale_percent" in rec_df.columns:
            rec_df["scale_percent"] = rec_df["scale_percent"].round().astype("Int64")
        if "decision" in rec_df.columns:
            rec_df["decision"] = rec_df["decision"].astype(str).str.replace("_", " ").str.title()
        recommendation_column_labels = {
            "vpu_id": "Resource ID",
            "avg_usage": "Average Usage (%)",
            "max_usage": "Peak Usage (%)",
            "forecasted_max": "Forecasted Peak (%)",
            "decision": "Recommended Action",
            "scale_percent": "Scale Percentage (%)",
            "total_size": "Current Capacity Units",
        }
        recommendation_column_order = [
            "vpu_id",
            "avg_usage",
            "max_usage",
            "forecasted_max",
            "decision",
            "scale_percent",
            "total_size",
        ]
        rec_df = humanise_dataframe(rec_df, recommendation_column_labels, recommendation_column_order)
        render_styled_table(
            rec_df,
            percent_cols={
                "Average Usage (%)": 2,
                "Peak Usage (%)": 2,
                "Forecasted Peak (%)": 2,
                "Scale Percentage (%)": 0,
            },
            highlight_cols=["Recommended Action"],
        )
        decision = recommendation.get("decision", "").replace("_", " ").title()
        st.markdown(f"**Decision:** {decision}")
    else:
        st.warning("No recommendation available. Check upstream data or rerun the pipeline.")

    anomaly_count = sum(
        item.get("anomaly_count", 0)
        for item in pipeline_result
        if isinstance(item, dict)
    )
    if anomaly_count:
        st.warning(f"{anomaly_count} anomalies detected for this workload. Review before approving scaling.")

    if state_snapshot and state_snapshot.get("state", {}).get("anomalies"):
        with st.expander("View anomaly details"):
            anomaly_df = pd.DataFrame(state_snapshot["state"]["anomalies"])
            for col in ["live_usage", "forecasted_usage", "deviation_percent"]:
                if col in anomaly_df.columns:
                    anomaly_df[col] = pd.to_numeric(anomaly_df[col], errors="coerce").round(2)
            anomaly_columns = {
                "date": "Date",
                "live_usage": "Observed Usage (%)",
                "forecasted_usage": "Forecasted Usage (%)",
                "deviation_percent": "Deviation (%)",
            }
            anomaly_order = ["date", "live_usage", "forecasted_usage", "deviation_percent"]
            anomaly_df = humanise_dataframe(anomaly_df, anomaly_columns, anomaly_order)
            render_styled_table(
                anomaly_df,
                percent_cols={
                    "Observed Usage (%)": 2,
                    "Forecasted Usage (%)": 2,
                    "Deviation (%)": 2,
                },
                highlight_cols=["Deviation (%)"],
            )

    st.markdown("### Manual Approval")
    default_scale_percent = recommendation.get("scale_percent", 0) if recommendation else 0
    decision_type = recommendation.get("decision") if recommendation else None

    if decision_type == "no_change":
        st.info("Recommendation is `no_change`. Submit to log manual acknowledgement or rerun the pipeline after new data.")

    with st.form("approval_form"):
        approval_choice = st.radio(
            "Proceed with scaling?",
            options=["Yes", "No"],
            horizontal=True,
            index=0 if decision_type and decision_type != "no_change" else 1,
        )

        scale_percent_input = st.number_input(
            "Scale Percent",
            min_value=0,
            max_value=100,
            value=int(default_scale_percent or 0),
            step=5,
            disabled=decision_type == "no_change",
        )

        approval_submitted = st.form_submit_button(
            "Submit Approval",
        )

        if approval_submitted:
            try:
                approval_payload = {
                    "server_id": st.session_state["selected_server"],
                    "resource_type": st.session_state["selected_resource"],
                    "approval": "yes" if approval_choice == "Yes" else "no",
                    "scale_percent": int(scale_percent_input),
                    "run_id": pipeline_result.get("run_id"),
                }
                approval_resp = api_post(
                    f"{api_base_clean.rstrip('/')}/approve_scaling/",
                    approval_payload,
                )
                st.session_state["approval_result"] = approval_resp
                st.session_state["state_snapshot"] = fetch_state_snapshot(
                    api_base_clean,
                    st.session_state["selected_server"],
                    st.session_state["selected_resource"],
                )
                st.success("Approval decision posted successfully.")
                # Capture CR info returned by the post-approval workflow
                cr_flag = approval_resp.get("cr_created", False)
                cr_num = approval_resp.get("change_number")
                # Fallback: extract CR number from feedback if flag wasn't propagated
                if not cr_flag and not cr_num:
                    import re
                    fb = approval_resp.get("feedback") or ""
                    m = re.search(r"Change Request (CHG\d+) created", fb)
                    if m:
                        cr_flag = True
                        cr_num = m.group(1)
                st.session_state["cr_created"] = cr_flag
                st.session_state["cr_sys_id"] = approval_resp.get("change_sys_id")
                st.session_state["cr_number"] = cr_num
                st.session_state["cr_approved"] = False
                st.session_state["cr_cancelled"] = False
                st.session_state["cr_approval_message"] = ""
                st.session_state["implement_result"] = None
            except RequestException as exc:
                detail = ""
                if getattr(exc, "response", None) is not None:
                    try:
                        payload = exc.response.json()
                        detail = payload.get("detail") if isinstance(payload, dict) else str(payload)
                    except Exception:
                        detail = exc.response.text
                message = f"Approval call failed: {exc}"
                if detail:
                    message += f"\nDetails: {detail}"
                st.error(message)
    jobs = [
        {
            "server_id": server_id,
            "resource_type": resource
        }
        for server_id in resolved_server_ids
        for resource in selected_resources
    ]

    use_bulk_pipeline = len(jobs) > 1

    approval_result = st.session_state["approval_result"]
    if approval_result:
        st.markdown("### Approval Outcome")
        col_left, col_right = st.columns(2)
        col_left.metric("Manual Proceed", str(approval_result.get("manual_proceed")))
        col_right.metric("State Status", approval_result.get("state_status"))
        if approval_result.get("feedback"):
            st.info(approval_result["feedback"])

        st.markdown("#### Updated Cost Impact")
        for item in pipeline_result:
            display_cost_summary(item.get("cost_impact") or {})

    # ============================================================
    # Change Request Workflow (Check Approval → Implement)
    # ============================================================
    if approval_result and approval_result.get("manual_proceed"):
        cr_created = st.session_state.get("cr_created", False)
        cr_number = st.session_state.get("cr_number")
        cr_approved = st.session_state.get("cr_approved", False)
        cr_cancelled = st.session_state.get("cr_cancelled", False)
        implement_result = st.session_state.get("implement_result")

        if cr_created and cr_number:
            st.markdown("---")
            st.markdown("### Change Request Workflow")
            st.info(f"Change Request **{cr_number}** has been created and submitted for approval.")

            step_check_col, step_impl_col = st.columns(2)

            # ---- Check Approval ----
            with step_check_col:
                if cr_cancelled:
                    st.error("CR was cancelled/closed.")
                elif cr_approved:
                    st.success("CR is **Approved** ✔")
                else:
                    if st.session_state.get("cr_approval_message"):
                        st.warning(st.session_state["cr_approval_message"])
                    if st.button("Check Approval Status", use_container_width=True):
                        with st.spinner("Checking CR approval in ServiceNow..."):
                            try:
                                check_resp = requests.get(
                                    f"{api_base_clean.rstrip('/')}/check_cr_approval",
                                    params={
                                        "server_id": st.session_state["selected_server"],
                                        "resource_type": st.session_state["selected_resource"],
                                    },
                                    timeout=30,
                                )
                                check_resp.raise_for_status()
                                check_data = check_resp.json()

                                st.session_state["cr_approved"] = check_data.get("cr_approved", False)
                                st.session_state["cr_cancelled"] = check_data.get("cr_cancelled", False)
                                st.session_state["cr_approval_message"] = check_data.get("cr_approval_message", "")

                                if check_data.get("cr_approved"):
                                    st.success("CR is **Approved**! You can now implement.")
                                elif check_data.get("cr_cancelled"):
                                    st.error("CR was cancelled/closed.")
                                else:
                                    st.warning(check_data.get("cr_approval_message", "Pending approval..."))
                            except RequestException as exc:
                                detail = ""
                                if getattr(exc, "response", None) is not None:
                                    try:
                                        payload = exc.response.json()
                                        detail = payload.get("detail") if isinstance(payload, dict) else str(payload)
                                    except Exception:
                                        detail = exc.response.text
                                st.error(f"Approval check failed: {exc}" + (f"\nDetails: {detail}" if detail else ""))

            # ---- Implement (only visible after approval) ----
            with step_impl_col:
                if implement_result:
                    if implement_result.get("scaling_executed"):
                        st.success("Resize completed & CR closed!")
                    else:
                        st.error("Resize failed. See details below.")
                elif cr_approved:
                    if st.button("Implement Resize", type="primary", use_container_width=True):
                        with st.spinner("Implementing resize & closing CR..."):
                            try:
                                impl_resp = api_post(
                                    f"{api_base_clean.rstrip('/')}/implement_resize",
                                    {
                                        "server_id": st.session_state["selected_server"],
                                        "resource_type": st.session_state["selected_resource"],
                                    },
                                )
                                st.session_state["implement_result"] = impl_resp
                                if impl_resp.get("scaling_executed"):
                                    st.success("VM resize completed and Change Request closed!")
                                else:
                                    st.error(f"Resize issue: {impl_resp.get('feedback', 'Unknown error')}")
                            except RequestException as exc:
                                detail = ""
                                if getattr(exc, "response", None) is not None:
                                    try:
                                        payload = exc.response.json()
                                        detail = payload.get("detail") if isinstance(payload, dict) else str(payload)
                                    except Exception:
                                        detail = exc.response.text
                                st.error(f"Implement call failed: {exc}" + (f"\nDetails: {detail}" if detail else ""))

            # Show implementation results if available
            implement_result = st.session_state.get("implement_result")
            if implement_result:
                st.markdown("### Implementation Result")
                if implement_result.get("feedback"):
                    st.info(implement_result["feedback"])
                if implement_result.get("scaling_result"):
                    st.markdown("#### Scaling Result Details")
                    st.json(implement_result["scaling_result"])

    st.markdown("### Cost Impact Preview")
    display_cost_summary(pipeline_result.get("cost_impact") or {})

    st.markdown("### Forecast Visualisations")

    def _extract_trace_list(payload: Any) -> Optional[List[Dict[str, Any]]]:
        if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
            return payload
        if isinstance(payload, dict):
            for key in ("traces", "data"):
                candidate = payload.get(key)
                if isinstance(candidate, list) and all(isinstance(item, dict) for item in candidate):
                    return candidate
        return None

    plot_container = pipeline_result.get("plots") or {}
    phase1_payload = None
    phase2_payload = None

    if isinstance(plot_container, dict):
        phase1_payload = _extract_trace_list(
            plot_container.get("phase1")
            or plot_container.get("phase_1")
            or plot_container.get("phaseOne")
        )
        phase2_payload = _extract_trace_list(
            plot_container.get("phase2")
            or plot_container.get("phase_2")
            or plot_container.get("phaseTwo")
        )
        if phase1_payload is None and phase2_payload is None and _extract_trace_list(plot_container):
            phase1_payload = _extract_trace_list(plot_container)
    elif isinstance(plot_container, list):
        phase1_payload = _extract_trace_list(plot_container)

    phase1_payload = phase1_payload or _extract_trace_list(pipeline_result.get("plot_phase1"))
    phase2_payload = phase2_payload or _extract_trace_list(pipeline_result.get("plot_phase2"))

    charts_drawn = False
    if render_forecast_plots(
        f"{resolved_server or 'Server'} – {resolved_resource or 'Resource'} (Phase 1)",
        phase1_payload,
        y_label="Usage / Forecast",
    ):
        charts_drawn = True
    if render_forecast_plots(
        f"{resolved_server or 'Server'} – {resolved_resource or 'Resource'} (Phase 2)",
        phase2_payload,
        y_label="Projected Usage",
    ):
        charts_drawn = True

    if not charts_drawn:
        st.info(
            "Forecast plot data is not available from the pipeline response. "
            "Provide Plotly-compatible trace data in `pipeline_result['plots']` to visualise forecasts."
        )

    if state_snapshot:
        with st.expander("Debug: Raw state snapshot"):
            st.json(state_snapshot)

    st.markdown("---")
    st.caption("Powered by ForecastingAgent, RightsizingAgent, CostAnalyzerAgent, NotificationAgent, and DashboardPublisherAgent.")


if __name__ == "__main__":
    launch_dashboard()
