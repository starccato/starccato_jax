import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

CACHE_FILENAME = "snr_vs_logbf_cache_v2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot logBF_sig_alt vs SNR metrics for one-detector runs."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Ignore cached metrics and rebuild aggregated data.",
    )
    return parser.parse_args()


def safe_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_metric_files(out_dir: Path) -> List[Path]:
    return sorted(out_dir.glob("*/metrics.csv"))


def load_cache(cache_path: Path, metrics_files: List[Path]) -> Optional[List[Dict]]:
    if not cache_path.exists():
        return None

    try:
        cached = json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        return None

    cached_files: Dict[str, float] = cached.get("files", {})
    current_files = {str(path): path.stat().st_mtime for path in metrics_files}

    if cached_files.keys() != current_files.keys():
        return None

    for path_str, mtime in current_files.items():
        cached_mtime = cached_files.get(path_str)
        if cached_mtime is None or cached_mtime != mtime:
            return None

    return cached.get("rows")


def save_cache(cache_path: Path, metrics_files: List[Path], rows: List[Dict]) -> None:
    payload = {
        "files": {str(path): path.stat().st_mtime for path in metrics_files},
        "rows": rows,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload))


def load_fixed_csv(path: Path) -> pd.DataFrame:
    """
    Load a metrics.csv file where the header may include an extra target_snr column.
    The data rows do not include that field, so we drop it from the header to keep
    the columns aligned with the values.
    """
    with path.open("r") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if len(lines) < 2:
        raise ValueError(f"{path} is missing data rows")

    headers = lines[0].split(",")
    values = lines[1].split(",")

    if "target_snr" in headers:
        idx = headers.index("target_snr")
        del headers[idx]

    if len(headers) != len(values):
        raise ValueError(
            f"Header/value length mismatch for {path}: {len(headers)} headers vs {len(values)} values"
        )

    df = pd.DataFrame([values], columns=headers)
    return df


def parse_metrics_file(metrics_file: Path) -> List[Dict]:
    rows: List[Dict] = []

    try:
        df = load_fixed_csv(metrics_file)
    except Exception as exc:  # pragma: no cover - defensive guardrail for bad files
        print(f"Skipping {metrics_file}: {exc}")
        return rows

    for record in df.to_dict(orient="records"):
        snr_excess = safe_float(record.get("snr_excess"))
        snr_mf_sig = safe_float(record.get("snr_mf_sig"))
        log_bf = safe_float(record.get("logBF_sig_alt"))

        if log_bf is None:
            lnz_signal = safe_float(record.get("lnz_signal"))
            lnz_noise = safe_float(record.get("lnz_noise"))
            lnz_glitch = safe_float(record.get("lnz_glitch"))

            if lnz_signal is None:
                continue

            alternatives = [value for value in (lnz_noise, lnz_glitch) if value is not None]
            if not alternatives:
                continue

            log_bf = lnz_signal - max(alternatives)

        if log_bf is None:
            continue

        if snr_excess is None and snr_mf_sig is None:
            continue

        rows.append(
            {
                "inject": (record.get("inject") or "unknown").strip(),
                "snr_excess": snr_excess,
                "snr_mf_sig": snr_mf_sig,
                "log_bf": log_bf,
                "source": str(metrics_file),
            }
        )

    return rows


def load_metrics(metrics_files: List[Path]) -> List[Dict]:
    metrics: List[Dict] = []
    for metrics_file in metrics_files:
        metrics.extend(parse_metrics_file(metrics_file))
    return metrics


def plot_single_metric(
    rows: List[Dict],
    counts: Counter,
    x_field: str,
    y_field: str,
    xlabel: str,
    ylabel: str,
    outfile: Path,
    color_map: Dict[str, str],
) -> bool:
    fig, ax = plt.subplots()
    used_labels = set()
    has_points = False

    for row in rows:
        x_value = row.get(x_field)
        y_value = row.get(y_field)
        if x_value is None or y_value is None:
            continue

        has_points = True
        inject_type = row["inject"]
        color = color_map.get(inject_type, "tab:gray")
        label = None
        if inject_type not in used_labels:
            used_labels.add(inject_type)
            label = f"{inject_type} ({counts.get(inject_type, 0)})"

        ax.scatter(x_value, y_value, color=color, label=label, alpha=0.7)

    if not has_points:
        plt.close(fig)
        return False

    counts_label = ", ".join(f"{inj}: {counts.get(inj, 0)}" for inj in sorted(counts))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs {xlabel} ({counts_label})")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Trigger Type")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return True


def plot_metrics(rows: List[Dict], counts: Counter) -> None:
    if not rows:
        raise RuntimeError("No metrics available to plot.")

    color_map = {"signal": "tab:blue", "glitch": "tab:orange", "noise": "tab:green"}
    work_dir = Path(__file__).resolve().parent

    y_field = "log_bf"
    y_label = "logBF_sig_alt (signal vs max(glitch, noise))"
    configs = [
        ("snr_mf_sig", "SNR (matched filter signal)", work_dir / "snr_mf_sig_vs_logbf_sig_alt.png"),
        ("snr_excess", "SNR (excess)", work_dir / "snr_excess_vs_logbf_sig_alt.png"),
    ]

    produced_any = False
    for x_field, xlabel, outfile in configs:
        plotted = plot_single_metric(
            rows, counts, x_field, y_field, xlabel, y_label, outfile, color_map
        )
        if plotted:
            produced_any = True
            print(f"Saved plot to {outfile}")
        else:
            print(f"Skipping {x_field} plot: no data available.")

    if not produced_any:
        raise RuntimeError("No valid data to plot for any SNR metric.")


def main() -> None:
    args = parse_args()
    work_dir = Path(__file__).resolve().parent
    out_dir = work_dir / "out"
    cache_path = work_dir / "cache" / CACHE_FILENAME

    metrics_files = find_metric_files(out_dir)
    if not metrics_files:
        raise RuntimeError(f"No metrics.csv files found under {out_dir}")

    if args.clean and cache_path.exists():
        cache_path.unlink()

    rows = None if args.clean else load_cache(cache_path, metrics_files)
    if rows is None:
        rows = load_metrics(metrics_files)
        save_cache(cache_path, metrics_files, rows)

    print(f"Loaded {len(rows)} metric entries from {len(metrics_files)} files.")
    counts = Counter(row["inject"] for row in rows)
    for inject_type, count in sorted(counts.items()):
        print(f"  {inject_type}: {count}")
    plot_metrics(rows, counts)


if __name__ == "__main__":
    main()
