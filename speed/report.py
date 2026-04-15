"""Self-contained HTML quality report generation for SPEED pipeline runs."""

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional


def _render_histogram(values: List[float], title: str, threshold: Optional[float] = None) -> str:
    """Render a histogram as a base64-encoded PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(values, bins=30, color="#4a90d9", edgecolor="#2c5f8a", alpha=0.85)
    if threshold is not None:
        ax.axvline(threshold, color="#d94a4a", linestyle="--", linewidth=1.5, label=f"Limit: {threshold:.2f}")
        ax.axvline(2 * threshold, color="#d9944a", linestyle=":", linewidth=1.5, label=f"Relaxed: {2 * threshold:.2f}")
        ax.legend(fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Value", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def generate_report(
    quality_metrics: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    processing_time: Optional[float] = None,
) -> str:
    """
    Generate a self-contained HTML quality report.

    Parameters
    ----------
    quality_metrics : list of dict
        Quality metrics per window/file. Each dict should have keys:
        filename, oha, thv, chv, bcr, quality_rating.
    config : dict, optional
        Full pipeline config to display in the report.
    output_path : str, optional
        Path to write the HTML report. If None, returns HTML string only.
    processing_time : float, optional
        Total processing time in seconds.

    Returns
    -------
    str
        The generated HTML string.
    """
    if not quality_metrics:
        html = "<html><body><h1>SPEED Quality Report</h1><p>No quality metrics available.</p></body></html>"
        if output_path:
            Path(output_path).write_text(html)
        return html

    # Count ratings
    ratings = [m.get("quality_rating", "unknown") for m in quality_metrics]
    n_good = ratings.count("good")
    n_ok = ratings.count("ok")
    n_bad = ratings.count("bad")
    n_unknown = len(ratings) - n_good - n_ok - n_bad
    n_total = len(quality_metrics)

    # Unique files
    filenames = list({m.get("filename", "unknown") for m in quality_metrics})

    # Extract metric arrays (skip None values)
    oha_vals = [m["oha"] for m in quality_metrics if m.get("oha") is not None]
    thv_vals = [m["thv"] for m in quality_metrics if m.get("thv") is not None]
    chv_vals = [m["chv"] for m in quality_metrics if m.get("chv") is not None]
    bcr_vals = [m["bcr"] for m in quality_metrics if m.get("bcr") is not None]

    # Render histograms
    histograms_html = ""
    if oha_vals:
        for label, vals, limit in [
            ("OHA (Overall High Amplitude)", oha_vals, 0.8),
            ("THV (Temporal High Variance)", thv_vals, 0.5),
            ("CHV (Channel High Variance)", chv_vals, 0.5),
            ("BCR (Bad Channel Ratio)", bcr_vals, 0.8),
        ]:
            if vals:
                img_b64 = _render_histogram(vals, label, limit)
                histograms_html += f'<img src="data:image/png;base64,{img_b64}" alt="{label}">\n'

    # Per-file table rows
    table_rows = ""
    for m in quality_metrics:
        rating = m.get("quality_rating", "unknown")
        rating_class = {"good": "good", "ok": "ok", "bad": "bad"}.get(rating, "")
        fname = Path(m.get("filename", "unknown")).name
        oha = f'{m["oha"]:.4f}' if m.get("oha") is not None else "-"
        thv = f'{m["thv"]:.4f}' if m.get("thv") is not None else "-"
        chv = f'{m["chv"]:.4f}' if m.get("chv") is not None else "-"
        bcr = f'{m["bcr"]:.4f}' if m.get("bcr") is not None else "-"
        w_start = m.get("window_start_time", "-")
        w_end = m.get("window_end_time", "-")
        table_rows += f"""<tr>
            <td>{fname}</td>
            <td class="{rating_class}">{rating}</td>
            <td>{oha}</td><td>{thv}</td><td>{chv}</td><td>{bcr}</td>
            <td>{w_start}</td><td>{w_end}</td>
        </tr>\n"""

    # Config section
    config_html = ""
    if config:
        import json
        config_str = json.dumps(config, indent=2, default=str)
        config_html = f"""
        <details>
            <summary>Pipeline Configuration</summary>
            <pre>{config_str}</pre>
        </details>
        """

    # Processing time
    time_str = f"{processing_time:.1f}s" if processing_time else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SPEED Quality Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; color: #333; }}
    h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
    h2 {{ color: #34495e; margin-top: 30px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px; margin: 20px 0; }}
    .stat {{ background: white; padding: 15px; border-radius: 8px; text-align: center;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .stat .value {{ font-size: 2em; font-weight: bold; }}
    .stat .label {{ font-size: 0.85em; color: #666; margin-top: 5px; }}
    .stat.good .value {{ color: #27ae60; }}
    .stat.ok .value {{ color: #f39c12; }}
    .stat.bad .value {{ color: #e74c3c; }}
    .histograms {{ display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; }}
    .histograms img {{ max-width: 48%; border-radius: 4px; background: white; padding: 5px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px;
             overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 10px 0; }}
    th {{ background: #34495e; color: white; padding: 10px 12px; text-align: left;
          cursor: pointer; user-select: none; font-size: 0.9em; }}
    th:hover {{ background: #2c3e50; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 0.88em; }}
    tr:hover {{ background: #f8f9fa; }}
    .good {{ color: #27ae60; font-weight: bold; }}
    .ok {{ color: #f39c12; font-weight: bold; }}
    .bad {{ color: #e74c3c; font-weight: bold; }}
    details {{ margin: 20px 0; background: white; padding: 15px; border-radius: 8px; }}
    summary {{ cursor: pointer; font-weight: bold; color: #2c3e50; }}
    pre {{ background: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto;
           font-size: 0.85em; }}
    .footer {{ text-align: center; color: #999; font-size: 0.8em; margin-top: 30px; }}
</style>
</head>
<body>
<h1>SPEED Quality Report</h1>

<div class="summary">
    <div class="stat"><div class="value">{len(filenames)}</div><div class="label">Files</div></div>
    <div class="stat"><div class="value">{n_total}</div><div class="label">Windows</div></div>
    <div class="stat good"><div class="value">{n_good}</div><div class="label">Good</div></div>
    <div class="stat ok"><div class="value">{n_ok}</div><div class="label">OK</div></div>
    <div class="stat bad"><div class="value">{n_bad}</div><div class="label">Bad</div></div>
    <div class="stat"><div class="value">{time_str}</div><div class="label">Time</div></div>
</div>

<h2>Quality Metric Distributions</h2>
<div class="histograms">
{histograms_html}
</div>

<h2>Per-Window Quality</h2>
<table id="metricsTable">
<thead>
<tr>
    <th onclick="sortTable(0)">Filename</th>
    <th onclick="sortTable(1)">Rating</th>
    <th onclick="sortTable(2)">OHA</th>
    <th onclick="sortTable(3)">THV</th>
    <th onclick="sortTable(4)">CHV</th>
    <th onclick="sortTable(5)">BCR</th>
    <th onclick="sortTable(6)">Start</th>
    <th onclick="sortTable(7)">End</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>

{config_html}

<div class="footer">Generated by SPEED</div>

<script>
let sortDir = {{}};
function sortTable(col) {{
    const table = document.getElementById("metricsTable");
    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.rows);
    const dir = sortDir[col] = !(sortDir[col] || false);
    rows.sort((a, b) => {{
        let va = a.cells[col].textContent.trim();
        let vb = b.cells[col].textContent.trim();
        let na = parseFloat(va), nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) return dir ? na - nb : nb - na;
        return dir ? va.localeCompare(vb) : vb.localeCompare(va);
    }});
    rows.forEach(r => tbody.appendChild(r));
}}
</script>
</body>
</html>"""

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)

    return html
