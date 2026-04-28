"""Aggregate per-model evaluation results and select the winning model.

Reads all *_metrics.json files from the results directory, computes a
weighted composite score, and prints a recommendation. Optionally logs
the comparison report to MLflow.

Scoring (all metrics normalised to [0, 1], higher composite = better):
    0.5 * quality_from_FID  +  0.3 * SSIM  +  0.2 * speed_from_latency

    FID   normalised as  (1 - FID/max_FID)   — lower FID → higher score
    SSIM  normalised as  SSIM/max_SSIM       — higher SSIM → higher score
    latency normalised as (1 - lat/max_lat)  — lower latency → higher score

Usage:
    python evaluation/compare.py --results-dir results/

    # With MLflow:
    python evaluation/compare.py \
        --results-dir results/ \
        --mlflow-tracking-uri http://localhost:5000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

WEIGHT_FID     = 0.5
WEIGHT_SSIM    = 0.3
WEIGHT_LATENCY = 0.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_normalise(values: list[float], higher_is_better: bool) -> list[float]:
    """Normalise a list of values to [0, 1]. Missing (-1) values map to 0."""
    valid = [v for v in values if v >= 0]
    if not valid:
        return [0.0] * len(values)

    vmin, vmax = min(valid), max(valid)
    span = vmax - vmin if vmax != vmin else 1.0

    normalised = []
    for v in values:
        if v < 0:                      # metric was unavailable
            normalised.append(0.0)
        elif higher_is_better:
            normalised.append((v - vmin) / span)
        else:                          # lower is better → invert
            normalised.append(1.0 - (v - vmin) / span)
    return normalised


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    metric_files = sorted(results_dir.glob("*_metrics.json"))

    if not metric_files:
        log.error("No *_metrics.json files found in %s", results_dir)
        return

    records: list[dict] = []
    for f in metric_files:
        data = json.loads(f.read_text())
        records.append(data)
        log.info("Loaded: %s  FID=%.4f  SSIM=%.4f  latency=%.0f ms",
                 data.get("model", f.stem),
                 data.get("fid", -1),
                 data.get("ssim", -1),
                 data.get("latency_ms_mean", -1))

    # Extract raw metric vectors
    fids      = [r.get("fid",              -1.0) for r in records]
    ssims     = [r.get("ssim",             -1.0) for r in records]
    latencies = [r.get("latency_ms_mean",  -1.0) for r in records]

    # Normalise
    fid_norm  = _safe_normalise(fids,      higher_is_better=False)
    ssim_norm = _safe_normalise(ssims,     higher_is_better=True)
    lat_norm  = _safe_normalise(latencies, higher_is_better=False)

    # Composite score
    for i, rec in enumerate(records):
        rec["score_fid_contrib"]     = round(WEIGHT_FID     * fid_norm[i],  4)
        rec["score_ssim_contrib"]    = round(WEIGHT_SSIM    * ssim_norm[i], 4)
        rec["score_latency_contrib"] = round(WEIGHT_LATENCY * lat_norm[i],  4)
        rec["composite_score"]       = round(
            rec["score_fid_contrib"] + rec["score_ssim_contrib"] + rec["score_latency_contrib"], 4
        )

    # Sort by composite score descending
    records.sort(key=lambda r: r["composite_score"], reverse=True)
    winner = records[0]

    # Print table
    header = f"{'Model':<20} {'FID':>8} {'SSIM':>8} {'Lat(ms)':>10} {'Score':>8}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in records:
        marker = " <-- WINNER" if r is winner else ""
        print(
            f"{r['model']:<20} "
            f"{r.get('fid', -1):>8.3f} "
            f"{r.get('ssim', -1):>8.4f} "
            f"{r.get('latency_ms_mean', -1):>10.1f} "
            f"{r['composite_score']:>8.4f}"
            f"{marker}"
        )
    print(sep)
    print(f"\nRecommended model: {winner['model']}  (score={winner['composite_score']})")
    print(f"  FID={winner.get('fid', -1):.4f}  "
          f"SSIM={winner.get('ssim', -1):.4f}  "
          f"latency_mean={winner.get('latency_ms_mean', -1):.0f} ms\n")

    # Save consolidated report
    report = {
        "weights": {
            "fid":     WEIGHT_FID,
            "ssim":    WEIGHT_SSIM,
            "latency": WEIGHT_LATENCY,
        },
        "models": records,
        "winner": winner["model"],
    }
    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    log.info("Comparison report saved to %s", report_path)

    # MLflow logging
    use_mlflow = MLFLOW_AVAILABLE and bool(args.mlflow_tracking_uri)
    if use_mlflow:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment("pixelsense-model-comparison")
        with mlflow.start_run(run_name="model-comparison"):
            for rec in records:
                prefix = rec["model"]
                mlflow.log_metrics({
                    f"{prefix}/fid":             rec.get("fid", -1),
                    f"{prefix}/ssim":            rec.get("ssim", -1),
                    f"{prefix}/latency_ms_mean": rec.get("latency_ms_mean", -1),
                    f"{prefix}/composite_score": rec["composite_score"],
                })
            mlflow.log_param("winner", winner["model"])
            mlflow.log_artifact(str(report_path))
        log.info("MLflow comparison run logged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PixelSense model evaluation results")
    parser.add_argument("--results-dir", default="results/",
                        help="Directory containing *_metrics.json files")
    parser.add_argument("--output",      default="results/comparison_report.json")
    parser.add_argument("--mlflow-tracking-uri", default="")
    compare(parser.parse_args())
