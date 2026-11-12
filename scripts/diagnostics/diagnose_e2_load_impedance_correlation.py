"""Diagnose load versus intrinsic-impedance magnitude in the E2 simulation.

Diagnostic only: this module does not import, instantiate, train, or evaluate a
neural network.  It reads ground-truth simulation targets and reports whether
signed relative intrinsic-impedance magnitude is related to simulated load.
It cannot establish whether N7-B boundary measurements identify that target.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import kendalltau, pearsonr, spearmanr


SCRIPT_PATH = Path(__file__).resolve()
SIM_DIR = SCRIPT_PATH.parent
DEFAULT_DATASET = (SIM_DIR / "E2_continuous_20k_20261111T150919" /
                   "e2_continuous_load_20k.pkl")
SENSOR_COUNT = 19
EXPECTED_SAMPLES = 20_000
LOAD_BINS = (
    ("0.01-0.25", 0.01, 0.25, False),
    ("0.25-0.50", 0.25, 0.50, False),
    ("0.50-0.75", 0.50, 0.75, False),
    ("0.75-1.00", 0.75, 1.00, True),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return jsonable(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows: writer.writerow(jsonable(row))


def safe_corr(x, y, method: str):
    x = np.asarray(x, float); y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]; y = y[valid]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    function = {"pearson": pearsonr, "spearman": spearmanr,
                "kendall": kendalltau}[method]
    try:
        return float(function(x, y).statistic)
    except (ValueError, FloatingPointError):
        return np.nan


def direction_label(correlation: float) -> str:
    if not np.isfinite(correlation): return "undetermined"
    if correlation > 0: return "impedance magnitude increases with load"
    if correlation < 0: return "impedance magnitude decreases with load"
    return "no monotonic direction"


def strength_label(correlation: float) -> str:
    """Documented descriptive bands; not hypothesis-test thresholds."""
    if not np.isfinite(correlation): return "undetermined"
    magnitude = abs(correlation)
    if magnitude >= 0.80: return "strong"
    if magnitude >= 0.50: return "moderate"
    return "weak"


def summarize_values(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: np.nan for key in ("minimum", "median", "maximum", "std")}
    return {"minimum": float(values.min()), "median": float(np.median(values)),
            "maximum": float(values.max()), "std": float(values.std())}


def validate_and_load(dataset_path: Path):
    dataset_path = dataset_path.resolve()
    if not dataset_path.is_file(): raise FileNotFoundError(dataset_path)
    validation_path = dataset_path.parent / "validation.json"
    provenance_path = dataset_path.parent / "source_provenance_manifest.json"
    if not validation_path.is_file(): raise FileNotFoundError(validation_path)
    if not provenance_path.is_file(): raise FileNotFoundError(provenance_path)
    archived_validation = json.loads(validation_path.read_text(encoding="utf-8-sig"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
    actual_hash = sha256(dataset_path)
    if archived_validation.get("dataset_sha256") != actual_hash:
        raise ValueError("Dataset SHA256 does not match validation.json")
    with dataset_path.open("rb") as handle: data = pickle.load(handle)
    required = ("Z_unit", "Z_unit_0", "load_kg", "y_cls")
    missing = [key for key in required if key not in data]
    if missing: raise KeyError(f"Required fields absent: {missing}")
    z = np.asarray(data["Z_unit"], np.complex128)
    z0 = np.asarray(data["Z_unit_0"], np.complex128)
    load = np.asarray(data["load_kg"], float)
    binary = np.asarray(data["y_cls"], np.uint8)
    if z.ndim != 2 or z.shape[1] != SENSOR_COUNT:
        raise ValueError(f"Z_unit must be (N,19), got {z.shape}")
    n = len(z)
    if z0.shape != (SENSOR_COUNT,) or load.shape != (n, SENSOR_COUNT) or binary.shape != (n, SENSOR_COUNT):
        raise ValueError(f"Shape mismatch: Z0={z0.shape}, load={load.shape}, y={binary.shape}")
    if not all(np.isfinite(a).all() for a in (z, z0, load, binary)):
        raise ValueError("Required arrays contain NaN or Inf")
    if np.any(np.abs(z0) == 0): raise ValueError("Z_unit_0 contains zero magnitude")
    if np.any(load < 0) or not np.array_equal(binary, (load > 0).astype(np.uint8)):
        raise ValueError("load_kg/y_cls are inconsistent")
    sensor_order = data.get("calibration_metadata", {}).get("sensor_order")
    if sensor_order is None: sensor_order = [f"S{i}" for i in range(1, 20)]
    if list(sensor_order) != [f"S{i}" for i in range(1, 20)]:
        raise ValueError(f"Unexpected sensor order: {sensor_order}")
    delta = (np.abs(z) - np.abs(z0)[None, :]) / np.abs(z0)[None, :]
    pressed_count = binary.sum(axis=1).astype(int)
    validation = {
        "status": "PASS", "dataset_path": dataset_path, "dataset_sha256": actual_hash,
        "dataset_type": type(data).__name__, "available_keys": sorted(data),
        "required_fields": {"Z_unit": {"shape": z.shape, "dtype": str(z.dtype), "unit": "ohm, complex"},
            "Z_unit_0": {"shape": z0.shape, "dtype": str(z0.dtype), "unit": "ohm, complex"},
            "load_kg": {"shape": load.shape, "dtype": str(load.dtype), "unit": "kg"},
            "y_cls": {"shape": binary.shape, "dtype": str(binary.dtype), "unit": "binary"}},
        "intrinsic_magnitude_and_phase_available": True,
        "sample_identifiers": "No explicit field; deterministic zero-based array indices are used",
        "sensor_identifiers": list(sensor_order),
        "sample_metadata_available": False,
        "finite_nonzero_unloaded_reference": True,
        "signed_delta_shape": delta.shape,
        "archived_validation_status": archived_validation.get("status"),
        "load_impedance_mapping_note": provenance.get("load_interpolation", {}).get("limitation"),
    }
    return data, z, z0, load, binary, delta, pressed_count, validation, provenance


def subset_masks(load, binary, pressed_count):
    finite = np.isfinite(load)
    return {
        "all_valid": finite,
        "pressed_only": finite & (binary == 1),
        "pressed_in_1_to_4_sensor_samples": finite & (binary == 1) &
            ((pressed_count >= 1) & (pressed_count <= 4))[:, None],
    }


def correlation_rows(load, delta, binary, pressed_count, sensors):
    rows = []
    masks = subset_masks(load, binary, pressed_count)
    for sensor_index, sensor in enumerate(sensors):
        for subset, full_mask in masks.items():
            mask = full_mask[:, sensor_index]
            x = load[mask, sensor_index]; y = delta[mask, sensor_index]
            stats = summarize_values(y)
            pearson = safe_corr(x, y, "pearson")
            spearman = safe_corr(x, y, "spearman")
            mean_constant = float(np.mean(y)) if len(y) else np.nan
            median_constant = float(np.median(y)) if len(y) else np.nan
            rows.append({"sensor": sensor, "sensor_index": sensor_index,
                "subset": subset, "n_observations": int(len(y)),
                "pearson_r": pearson, "spearman_rho": spearman,
                "direction": direction_label(spearman), "strength_band": strength_label(spearman),
                "delta_min": stats["minimum"], "delta_median": stats["median"],
                "delta_max": stats["maximum"], "delta_std": stats["std"],
                "mean_constant_prediction": mean_constant,
                "mean_constant_mae": float(np.mean(np.abs(y - mean_constant))) if len(y) else np.nan,
                "mean_constant_rmse": float(np.sqrt(np.mean((y - mean_constant) ** 2))) if len(y) else np.nan,
                "median_constant_prediction": median_constant,
                "median_constant_mae": float(np.mean(np.abs(y - median_constant))) if len(y) else np.nan})
    return rows


def bin_mask(values, low, high, include_high):
    return (values >= low) & ((values <= high) if include_high else (values < high))


def binned_rows(load, delta, sensors):
    rows = []
    for j, sensor in enumerate(sensors):
        groups = [("unloaded", load[:, j] == 0)]
        groups += [(name, bin_mask(load[:, j], low, high, inclusive))
                   for name, low, high, inclusive in LOAD_BINS]
        for name, mask in groups:
            values = delta[mask, j]; values = values[np.isfinite(values)]
            if len(values):
                p5, q1, median, q3, p95 = np.percentile(values, [5, 25, 50, 75, 95])
                row = {"sensor": sensor, "load_group": name, "count": len(values),
                    "median_delta": median, "q1_delta": q1, "q3_delta": q3,
                    "iqr_delta": q3-q1, "p05_delta": p5, "p95_delta": p95,
                    "mean_delta": values.mean(), "std_delta": values.std()}
            else:
                row = {"sensor": sensor, "load_group": name, "count": 0,
                    **{key: np.nan for key in ("median_delta", "q1_delta", "q3_delta",
                    "iqr_delta", "p05_delta", "p95_delta", "mean_delta", "std_delta")}}
            rows.append(row)
    return rows


def unique_valid_pair_codes(loads, minimum_difference, maximum_pairs, rng):
    """Return exact valid pairs when small, otherwise deterministic unique sampling."""
    order = np.argsort(loads); sorted_loads = loads[order]; n = len(loads)
    starts = np.searchsorted(sorted_loads, sorted_loads + minimum_difference, side="left")
    total_valid = int(np.maximum(n - starts, 0).sum())
    if total_valid <= maximum_pairs:
        left, right = [], []
        for i, start in enumerate(starts):
            if start < n:
                left.extend([order[i]] * (n - start)); right.extend(order[start:].tolist())
        return np.asarray(left, int), np.asarray(right, int), total_valid, "exact"
    codes = set(); attempts = 0; limit = maximum_pairs * 100
    while len(codes) < maximum_pairs and attempts < limit:
        batch = min(4 * (maximum_pairs - len(codes)), 1_000_000)
        a = rng.integers(0, n, batch); b = rng.integers(0, n, batch)
        lo = np.minimum(a, b); hi = np.maximum(a, b)
        valid = (lo != hi) & (np.abs(loads[lo] - loads[hi]) >= minimum_difference)
        codes.update((lo[valid] * n + hi[valid]).tolist()); attempts += batch
    selected = np.fromiter(sorted(codes)[:maximum_pairs], dtype=np.int64)
    return selected // n, selected % n, total_valid, "deterministic_unique_sampling"


def ranking_rows(load, delta, binary, sensors, args):
    rows = []; windows = {}
    for j, sensor in enumerate(sensors):
        ids = np.flatnonzero(binary[:, j] == 1)
        sensor_rng = np.random.default_rng(args.seed + 1009 * (j + 1))
        shuffled = ids.copy(); sensor_rng.shuffle(shuffled)
        cut = max(3, len(shuffled) // 2)
        direction_ids = shuffled[:cut]; evaluation_ids = shuffled[cut:]
        direction_rho = safe_corr(load[direction_ids, j], delta[direction_ids, j], "spearman")
        direction = 0 if not np.isfinite(direction_rho) or direction_rho == 0 else int(np.sign(direction_rho))
        x = load[evaluation_ids, j]; y = delta[evaluation_ids, j]
        left, right, possible, method = unique_valid_pair_codes(
            x, args.minimum_pair_load_difference_kg, args.maximum_pairs_per_sensor, sensor_rng)
        if len(left) and direction:
            product = direction * (x[right] - x[left]) * (y[right] - y[left])
            accuracy = float(np.mean(np.where(product > 0, 1.0, np.where(product == 0, 0.5, 0.0))))
        else: accuracy = np.nan
        rows.append({"sensor": sensor, "pressed_observations": len(ids),
            "direction_estimation_observations": len(direction_ids),
            "ranking_evaluation_observations": len(evaluation_ids),
            "direction_spearman_rho": direction_rho, "expected_direction_sign": direction,
            "expected_direction": direction_label(direction_rho),
            "evaluation_spearman_rho": safe_corr(x, y, "spearman"),
            "evaluation_kendall_tau": safe_corr(x, y, "kendall"),
            "minimum_load_difference_kg": args.minimum_pair_load_difference_kg,
            "total_possible_valid_pairs": possible, "evaluated_pairs": len(left),
            "pair_selection_method": method, "pairwise_ranking_accuracy": accuracy})
        sensor_windows = {}
        for center in (0.5, 1.0):
            mask = (binary[:, j] == 1) & (np.abs(load[:, j] - center) <= args.load_window_half_width_kg)
            values = delta[mask, j]
            sensor_windows[f"{center:.1f}_kg_window"] = {
                "bounds_kg": [max(0, center-args.load_window_half_width_kg),
                              min(1, center+args.load_window_half_width_kg)],
                "count": int(mask.sum()), "median_delta": float(np.median(values)) if len(values) else np.nan,
                "q1_delta": float(np.percentile(values, 25)) if len(values) else np.nan,
                "q3_delta": float(np.percentile(values, 75)) if len(values) else np.nan}
        windows[sensor] = sensor_windows
    return rows, windows


def pressed_count_rows(load, delta, binary, pressed_count, sensors):
    rows = []
    for count in range(1, 5):
        for j, sensor in enumerate(sensors):
            mask = (pressed_count == count) & (binary[:, j] == 1)
            rows.append({"sensor": sensor, "pressed_sensor_count": count,
                "observations": int(mask.sum()),
                "pearson_r": safe_corr(load[mask, j], delta[mask, j], "pearson"),
                "spearman_rho": safe_corr(load[mask, j], delta[mask, j], "spearman"),
                "kendall_tau": safe_corr(load[mask, j], delta[mask, j], "kendall"),
                "median_delta": float(np.median(delta[mask, j])) if mask.any() else np.nan,
                "std_delta": float(np.std(delta[mask, j])) if mask.any() else np.nan})
    return rows


def co_press_diagnostics(load, delta, binary, sensors):
    output = {}
    for j, sensor in enumerate(sensors):
        rows = []
        for partner in range(SENSOR_COUNT):
            if partner == j: continue
            base = binary[:, j] == 1
            absent = base & (binary[:, partner] == 0); present = base & (binary[:, partner] == 1)
            rows.append({"partner": sensors[partner], "absent_n": int(absent.sum()),
                "present_n": int(present.sum()),
                "absent_spearman": safe_corr(load[absent, j], delta[absent, j], "spearman"),
                "present_spearman": safe_corr(load[present, j], delta[present, j], "spearman")})
        output[sensor] = rows
    return output


def save_figure(fig, directory: Path, stem: str):
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_figures(load, delta, binary, correlation, binned, ranking, count_rows,
                 sensors, directory: Path):
    directory.mkdir(parents=False, exist_ok=False)
    pressed_corr = {r["sensor"]: r["spearman_rho"] for r in correlation if r["subset"] == "pressed_only"}
    fig, ax = plt.subplots(figsize=(10, 4.8)); values = [pressed_corr[s] for s in sensors]
    ax.bar(sensors, values, color=["#2878b5" if v >= 0 else "#c4473a" for v in values])
    ax.axhline(0, color="black", lw=.8); ax.set_ylabel("Spearman ρ (dimensionless)")
    ax.set_xlabel("Physical sensor"); ax.set_title("Pressed-sample load–impedance monotonicity")
    ax.tick_params(axis="x", rotation=45); fig.tight_layout()
    save_figure(fig, directory, "01_per_sensor_spearman")

    fig, axes = plt.subplots(5, 4, figsize=(13, 15), sharex=True)
    for j, ax in enumerate(axes.flat):
        if j >= 19: ax.axis("off"); continue
        mask = binary[:, j] == 1
        ax.hexbin(load[mask, j], delta[mask, j], gridsize=35, mincnt=1, cmap="viridis")
        ax.set_title(sensors[j]); ax.set_xlabel("Load (kg)"); ax.set_ylabel("Signed Δ|Z|/|Z₀|")
    fig.suptitle("Simulated intrinsic magnitude response", y=1.002); fig.tight_layout()
    save_figure(fig, directory, "02_load_vs_relative_impedance_panels")

    lookup = {(r["sensor"], r["load_group"]): r for r in binned}; names = [x[0] for x in LOAD_BINS]
    centers = np.arange(len(names)); fig, axes = plt.subplots(5, 4, figsize=(13, 15), sharex=True)
    for j, ax in enumerate(axes.flat):
        if j >= 19: ax.axis("off"); continue
        rows = [lookup[(sensors[j], name)] for name in names]
        med = np.array([r["median_delta"] for r in rows]); low = np.array([r["p05_delta"] for r in rows]); high = np.array([r["p95_delta"] for r in rows])
        ax.errorbar(centers, med, yerr=[med-low, high-med], marker="o", capsize=3)
        ax.set_title(sensors[j]); ax.set_xticks(centers, names, rotation=35); ax.set_ylabel("Signed Δ|Z|/|Z₀|")
        ax.set_xlabel("Load interval (kg)")
    fig.suptitle("Load-binned median and 5th–95th percentiles", y=1.002); fig.tight_layout()
    save_figure(fig, directory, "03_load_binned_impedance")

    rank = {r["sensor"]: r["pairwise_ranking_accuracy"] for r in ranking}
    fig, ax = plt.subplots(figsize=(10, 4.8)); ax.bar(sensors, [rank[s] for s in sensors], color="#5b8c5a")
    ax.axhline(.5, color="black", ls="--", lw=.8, label="chance/tie reference")
    ax.set_ylim(0, 1); ax.set_ylabel("Pairwise ranking accuracy"); ax.set_xlabel("Physical sensor")
    ax.tick_params(axis="x", rotation=45); ax.legend(); fig.tight_layout()
    save_figure(fig, directory, "04_pairwise_ranking_accuracy")

    matrix = np.full((4, 19), np.nan)
    for row in count_rows: matrix[row["pressed_sensor_count"]-1, int(row["sensor"][1:])-1] = row["spearman_rho"]
    fig, ax = plt.subplots(figsize=(11, 3.8)); image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(19), sensors, rotation=45); ax.set_yticks(range(4), ["1", "2", "3", "4"])
    ax.set_xlabel("Physical sensor"); ax.set_ylabel("Simultaneously pressed sensors")
    ax.set_title("Spearman ρ by loading configuration"); fig.colorbar(image, ax=ax, label="Spearman ρ")
    fig.tight_layout(); save_figure(fig, directory, "05_correlation_by_pressed_count")


def build_summary(correlation, ranking, count_rows, windows, provenance, exclusions):
    pressed = [r for r in correlation if r["subset"] == "pressed_only"]
    bands = {band: [r["sensor"] for r in pressed if r["strength_band"] == band]
             for band in ("strong", "moderate", "weak", "undetermined")}
    directions = Counter(r["direction"] for r in pressed)
    stability = {}
    for sensor in [f"S{i}" for i in range(1, 20)]:
        values = np.array([r["spearman_rho"] for r in count_rows if r["sensor"] == sensor], float)
        finite = values[np.isfinite(values)]
        stability[sensor] = {"rho_range": float(np.ptp(finite)) if len(finite) else np.nan,
                             "same_sign": bool(len(finite) and (np.all(finite >= 0) or np.all(finite <= 0)))}
    return {
        "scope": {"question_answered": "association between simulated load and intrinsic |Z| change",
                  "question_not_answered": "inferability of intrinsic |Z| change from N7-B boundary measurements"},
        "strength_bands": {"definition": "descriptive |Spearman rho| bands: strong >=0.80, moderate >=0.50, weak <0.50; not significance thresholds",
                           "sensors": bands},
        "directions": dict(directions), "ranking_statistics": ranking,
        "half_and_one_kg_windows": windows,
        "pressed_count_stability": stability,
        "information_sufficiency": {
            "target_available": True, "target_shape": [EXPECTED_SAMPLES, SENSOR_COUNT],
            "target_definition": "signed (|Z|-|Z0|)/|Z0|",
            "load_relationship_assessable": True,
            "N7B_inferability_assessable_by_this_script": False,
            "warning": "High target correlation does not imply low boundary-to-target prediction error."},
        "load_curve_transfer_assumption": provenance.get("load_interpolation", {}).get("limitation"),
        "exclusions": exclusions,
    }


def report_markdown(summary):
    bands = summary["strength_bands"]["sensors"]
    def listing(name): return ", ".join(bands[name]) if bands[name] else "none"
    stable = summary["pressed_count_stability"]
    stable_count = sum(v["same_sign"] for v in stable.values())
    return f"""# E2 load–intrinsic impedance magnitude diagnostic

## Scope

This analysis measures association between simulated physical load and the signed relative intrinsic magnitude target. It does **not** test whether N7-B boundary measurements can predict that target.

## Monotonic relationships

The descriptive bands are documented as: strong `|Spearman ρ| >= 0.80`, moderate `>= 0.50`, and weak `< 0.50`. They are summaries, not statistical-significance thresholds.

- Strong: {listing('strong')}
- Moderate: {listing('moderate')}
- Weak: {listing('weak')}
- Undetermined: {listing('undetermined')}

Direction counts: `{json.dumps(summary['directions'], ensure_ascii=False)}`.

## 0.5 kg and 1.0 kg neighborhoods

The analysis uses configured windows around 0.5 kg and 1.0 kg; it does not assume exact values occur. Counts, medians, and quartiles for every sensor are stored in `diagnostic_summary.json`. Distinguishability should be judged from these distributions and their overlap, not only their medians.

## Loading-pattern sensitivity

Across the 1-, 2-, 3-, and 4-pressed configurations, {stable_count}/19 sensors retain one correlation sign wherever the statistic is defined. Detailed correlations are in `pressed_count_statistics.csv`. These associations across patterns are not causal evidence.

## Suitability of magnitude-only regression

The proposed 19-dimensional signed relative intrinsic-magnitude target is directly available and finite in this simulation. A follow-up predictive experiment is scientifically reasonable only for sensors showing sufficiently stable, informative relationships for the intended question. This diagnostic cannot establish observability from N7-B, and high load–impedance correlation does not guarantee low neural-network regression error.

## Important limitation

{summary.get('load_curve_transfer_assumption')}
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=SIM_DIR)
    parser.add_argument("--minimum-pair-load-difference-kg", type=float, default=0.25)
    parser.add_argument("--maximum-pairs-per-sensor", type=int, default=200_000)
    parser.add_argument("--load-window-half-width-kg", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0 < args.minimum_pair_load_difference_kg <= 1): raise ValueError("Invalid pair load difference")
    if args.maximum_pairs_per_sensor < 1: raise ValueError("maximum pairs must be positive")
    if not (0 < args.load_window_half_width_kg <= .25): raise ValueError("Invalid load window")
    (data, z, z0, load, binary, delta, pressed_count, validation,
     provenance) = validate_and_load(args.dataset)
    del data, z, z0
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output = args.output_root.resolve() / f"E2_load_impedance_diagnostic_{stamp}"
    output.mkdir(parents=False, exist_ok=False)
    sensors = validation["sensor_identifiers"]
    assigned_positive = np.zeros_like(load, bool)
    for _, low, high, inclusive in LOAD_BINS:
        assigned_positive |= bin_mask(load, low, high, inclusive)
    positive = load > 0
    exclusions = {"nonfinite_delta": int((~np.isfinite(delta)).sum()),
                  "positive_load_entries_outside_bins": int((positive & ~assigned_positive).sum()),
                  "unloaded_entries": int((load == 0).sum())}
    config = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "script": SCRIPT_PATH, "script_sha256": sha256(SCRIPT_PATH),
        "dataset": args.dataset.resolve(), "dataset_sha256": validation["dataset_sha256"],
        "check_only": args.check_only, "random_seed": args.seed,
        "minimum_pair_load_difference_kg": args.minimum_pair_load_difference_kg,
        "maximum_pairs_per_sensor": args.maximum_pairs_per_sensor,
        "load_window_half_width_kg": args.load_window_half_width_kg,
        "load_bin_convention": "[low, high), except final [0.75,1.00] includes 1.00",
        "correlation_strength_bands": {"strong_abs_rho_min": .8, "moderate_abs_rho_min": .5,
                                       "purpose": "descriptive, not significance thresholds"},
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}}
    write_json(output / "data_validation.json", validation)
    write_json(output / "run_config.json", config)
    if args.check_only:
        print(f"PASS check-only; no full diagnostic calculations performed: {output}")
        return
    correlations = correlation_rows(load, delta, binary, pressed_count, sensors)
    binned = binned_rows(load, delta, sensors)
    ranking, windows = ranking_rows(load, delta, binary, sensors, args)
    count_statistics = pressed_count_rows(load, delta, binary, pressed_count, sensors)
    co_press = co_press_diagnostics(load, delta, binary, sensors)
    summary = build_summary(correlations, ranking, count_statistics, windows, provenance, exclusions)
    summary["co_pressed_sensor_diagnostics"] = co_press
    write_csv(output / "per_sensor_correlations.csv", correlations)
    write_csv(output / "load_binned_impedance_statistics.csv", binned)
    write_csv(output / "pairwise_ranking_statistics.csv", ranking)
    write_csv(output / "pressed_count_statistics.csv", count_statistics)
    write_json(output / "diagnostic_summary.json", summary)
    (output / "diagnostic_report.md").write_text(report_markdown(summary), encoding="utf-8")
    make_figures(load, delta, binary, correlations, binned, ranking,
                 count_statistics, sensors, output / "figures")
    print(f"Diagnostic complete: {output}")


if __name__ == "__main__":
    main()
