"""Generate the fixed-E2, 20,000-sample continuous-load simulation dataset.

This script is intended to be runnable directly with Spyder F5.  It reuses the
validated topology, grounded-nodal forward operator, and shared normalized
load-response curve implementation from
``generate_simulation_curve_nodal_forward_50k.py`` without modifying that
historical source.

Important limitation
--------------------
RC_opt_results.pkl is paired with experimental labels at only 0.5 and 1.0 kg.
It does not identify a reliable continuous response by itself.  The continuous
relationship used here is therefore the project's existing PCHIP interpolation
of normalized R/C responses measured on five sensors at 5 kHz.  The median
("shared") response is transferred to the E2 100-kHz unloaded G/B reference.
That frequency transfer is a documented simulation assumption, not a measured
19-sensor continuous-load calibration at 100 kHz.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
SIM_DIR = SCRIPT_PATH.parent
PROJECT_DIR = SIM_DIR.parent
E2_DIR = (PROJECT_DIR / "Simulation_Experiment" /
          "experiment_I5_S2_E1_E2_E3" / "E2_weak_S2_prior" /
          "init_30pct")
E2_BASELINE = E2_DIR / "baseline_sensor_parameters.csv"
E2_SUMMARY = E2_DIR / "fit_summary.json"
E2_CHECKPOINT = E2_DIR / "checkpoint.pt"
CURVE_FILE = PROJECT_DIR / "Data" / "Nano_Sensors_five.xlsx"
RC_RESULTS = PROJECT_DIR / "Data" / "RC_opt_results.pkl"
MEASUREMENT_DIR = PROJECT_DIR / "Data" / "Load_label"
FORWARD_SOURCE = SIM_DIR / "generate_simulation_curve_nodal_forward_50k.py"

FREQUENCY_HZ = 100_000.0
N_LOADED = 19_975
N_UNLOADED = 25
N_TOTAL = N_LOADED + N_UNLOADED
N_SENSORS = 19
N_PAIRS = 91


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False,
                               allow_nan=False), encoding="utf-8")


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required source file is missing: {path}")


def import_forward_source():
    require_file(FORWARD_SOURCE)
    spec = importlib.util.spec_from_file_location("validated_curve_forward", FORWARD_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(str(FORWARD_SOURCE))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_e2_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load E2 G+jB at 100 kHz in exact physical order S1..S19."""
    require_file(E2_BASELINE)
    require_file(E2_SUMMARY)
    with E2_BASELINE.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_columns = {"sensor", "G_S", "B_S", "Z_abs_ohm"}
    if not rows or not expected_columns.issubset(rows[0]):
        raise ValueError(f"Unexpected E2 columns; required {sorted(expected_columns)}")
    expected_sensors = [f"S{i}" for i in range(1, 20)]
    sensors = [row["sensor"].strip() for row in rows]
    if sensors != expected_sensors:
        raise ValueError(f"E2 sensor order is not S1..S19: {sensors}")
    g = np.array([float(row["G_S"]) for row in rows], dtype=np.float64)
    b = np.array([float(row["B_S"]) for row in rows], dtype=np.float64)
    z_abs_csv = np.array([float(row["Z_abs_ohm"]) for row in rows])
    if g.shape != (19,) or b.shape != (19,) or not np.isfinite([g, b]).all():
        raise ValueError("E2 reference must contain 19 finite G/B values")
    if np.any(g <= 0) or np.any(b <= 0):
        raise ValueError("E2 intrinsic conductance and susceptance must be positive")
    z0 = 1.0 / (g + 1j * b)
    if not np.allclose(np.abs(z0), z_abs_csv, rtol=2e-12, atol=1e-6):
        raise ValueError("E2 Z_abs_ohm does not equal abs(1/(G+jB))")
    summary = json.loads(E2_SUMMARY.read_text(encoding="utf-8-sig"))
    if summary.get("scenario") != "E2_weak_S2_prior":
        raise ValueError("Unexpected E2 scenario")
    if not np.isclose(float(summary.get("fraction", np.nan)), 0.30):
        raise ValueError("E2 result is not the requested 30% initialization")
    if not np.isclose(float(summary.get("lambda", np.nan)), 1e-4):
        raise ValueError("Unexpected E2 regularization lambda")
    return g, b, z0, summary


def source_manifest(seed: int, args, curve_model: dict) -> dict:
    sources = [E2_BASELINE, E2_SUMMARY, E2_CHECKPOINT, CURVE_FILE,
               RC_RESULTS, FORWARD_SOURCE]
    measurement_files = sorted(MEASUREMENT_DIR.glob("*_Measurement*.txt"))
    sources.extend(measurement_files)
    records = []
    for path in sources:
        require_file(path)
        records.append({"path": str(path.resolve()), "sha256": sha256(path),
                        "bytes": path.stat().st_size})
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator_script": str(SCRIPT_PATH),
        "generator_script_sha256": sha256(SCRIPT_PATH),
        "random_seed": seed,
        "sources": records,
        "calibration": {
            "scenario": "E2_weak_S2_prior", "initialization_fraction": 0.30,
            "frequency_hz": FREQUENCY_HZ,
            "definition": "Y_unit_0 = G_S + j*B_S; Z_unit_0 = 1/Y_unit_0",
            "sensor_order": [f"S{i}" for i in range(1, 20)],
            "historical_R0_C0_fallback_used": False,
        },
        "load_interpolation": {
            "method": curve_model.get("method"),
            "shared_method": curve_model.get("shared_method"),
            "curve_name_for_all_19_sensors": "shared",
            "raw_curve_frequency_hz": curve_model.get("frequency_hz"),
            "simulation_frequency_hz": FREQUENCY_HZ,
            "normalization": "R(load)/R(0) and C(load)/C(0), shared across sensors",
            "interpolator": "PCHIP in log response, no extrapolation",
            "limitation": ("Normalized five-sensor response measured at 5 kHz is "
                           "transferred to E2 baselines at 100 kHz. RC_opt_results "
                           "has paired labels only at 0.5 and 1.0 kg and therefore "
                           "does not independently support continuous interpolation."),
            "load_range_kg": [args.min_load_kg, args.max_load_kg],
        },
    }


def independent_nodal_forward(z_sensors: np.ndarray, edges, pairs) -> np.ndarray:
    """NumPy cross-check of the grounded nodal solve for a small batch."""
    def incidence(items):
        a = np.zeros((14, len(items)), dtype=np.float64)
        for k, (u, v) in enumerate(items):
            a[u - 1, k] = 1.0
            a[v - 1, k] = -1.0
        return a
    a = incidence(edges)[:-1]
    c = incidence(pairs)[:-1]
    result = np.empty((len(z_sensors), len(pairs)), dtype=np.complex128)
    for i, z in enumerate(z_sensors):
        lap = (a * (1.0 / z)[None, :]) @ a.T
        voltages = np.linalg.solve(lap, c)
        result[i] = np.sum(c * voltages, axis=0)
    return result


def validate(dataset: dict, forward, curve_model: dict, r0, c0,
             args, output_dir: Path) -> dict:
    z_all = dataset["Z_all"]
    z_unit = dataset["Z_unit"]
    z0 = dataset["Z_unit_0"]
    z_all0 = dataset["Z_all_0"]
    loads = dataset["load_kg"]
    labels = dataset["y_cls"]
    edges = np.asarray(dataset["physical_edges"])
    pairs = np.asarray(dataset["original_pairs"])
    checks = {}
    checks["shapes"] = (z_all.shape == (N_TOTAL, N_PAIRS) and
                        z_unit.shape == (N_TOTAL, N_SENSORS) and
                        z0.shape == (N_SENSORS,) and z_all0.shape == (N_PAIRS,) and
                        loads.shape == labels.shape == (N_TOTAL, N_SENSORS) and
                        edges.shape == (N_SENSORS, 2) and pairs.shape == (N_PAIRS, 2))
    checks["finite_complex_impedances"] = bool(np.isfinite(z_all).all() and
                                                np.isfinite(z_unit).all())
    unpressed = labels == 0
    pressed = labels == 1
    checks["unpressed_zero_load"] = bool(np.all(loads[unpressed] == 0))
    z0_grid = np.broadcast_to(z0, z_unit.shape)
    checks["unpressed_e2_impedance"] = bool(np.array_equal(z_unit[unpressed],
                                                            z0_grid[unpressed]))
    checks["pressed_positive_load"] = bool(np.all(loads[pressed] > 0))
    checks["classification_matches_load"] = bool(np.array_equal(labels,
                                                                  (loads > 0).astype(np.uint8)))
    rr, cc, zz = forward.sensor_states(loads * 1000.0, r0, c0, curve_model,
                                       ["shared"] * 19, FREQUENCY_HZ)
    del rr, cc
    checks["load_impedance_correspondence"] = bool(np.allclose(
        zz, z_unit, rtol=2e-13, atol=1e-9))
    checks["canonical_pair_order"] = bool(np.array_equal(
        pairs, np.asarray(forward.ORIGINAL_PAIRS, dtype=np.int16)))
    cross_indices = np.array([0, N_LOADED // 2, N_LOADED - 1, N_TOTAL - 1])
    independent = independent_nodal_forward(z_unit[cross_indices], edges, pairs)
    checks["independent_grounded_nodal_crosscheck"] = bool(np.allclose(
        independent, z_all[cross_indices], rtol=2e-10, atol=1e-7))
    checks["all_unloaded_reproduce_Z_all_0"] = bool(np.allclose(
        z_all[-N_UNLOADED:], z_all0[None, :], rtol=2e-12, atol=1e-8))
    counts = labels.sum(axis=1).astype(int)
    checks["loaded_count_range_1_to_19"] = bool(np.all((counts[:N_LOADED] >= 1) &
                                                        (counts[:N_LOADED] <= 19)))
    checks["exactly_25_unloaded"] = bool(np.count_nonzero(counts == 0) == 25 and
                                          np.all(counts[-25:] == 0))
    checks["positive_passive_intrinsic_admittance"] = bool(np.all(
        (1.0 / z_unit).real > 0) and np.all((1.0 / z_unit).imag > 0))
    checks["load_range"] = bool(np.all(loads[pressed] >= args.min_load_kg) and
                                 np.all(loads[pressed] <= args.max_load_kg))
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise AssertionError("Validation failed: " + ", ".join(failed))
    loaded_values = loads[pressed]
    report = {
        "status": "PASS", "checks": checks,
        "array_shapes": {k: list(np.asarray(dataset[k]).shape) for k in
                         ("Z_all", "Z_unit", "Z_unit_0", "Z_all_0", "load_kg",
                          "y_cls", "physical_edges", "original_pairs")},
        "sample_counts": {"total": N_TOTAL, "loaded": N_LOADED,
                          "unloaded": N_UNLOADED},
        "pressed_sensor_count_distribution": dict(sorted(Counter(
            counts[:N_LOADED].tolist()).items())),
        "continuous_load_kg": {"minimum": float(loaded_values.min()),
                               "maximum": float(loaded_values.max()),
                               "mean": float(loaded_values.mean()),
                               "std": float(loaded_values.std()),
                               "unique_values": int(np.unique(loaded_values).size)},
        "forward_crosscheck_sample_indices": cross_indices.tolist(),
        "output_directory": str(output_dir),
    }
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--min-load-kg", type=float, default=0.010)
    parser.add_argument("--max-load-kg", type=float, default=1.000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, default=SIM_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 < args.min_load_kg <= args.max_load_kg <= 1.0):
        raise ValueError("Load range must be positive and within calibrated 0..1 kg")
    forward = import_forward_source()
    g0, b0, z0, e2_summary = load_e2_reference()
    require_file(CURVE_FILE)
    curve_model = forward.load_curves(CURVE_FILE)
    shared_grid = np.asarray(curve_model["sensors"]["shared"]["load_g"], float)
    if args.max_load_kg * 1000 > shared_grid[-1]:
        raise ValueError("Requested loads exceed the shared curve; extrapolation is forbidden")

    rng = np.random.default_rng(args.seed)
    load_g = forward.sample_loads(rng, N_LOADED, N_UNLOADED, 1, 19,
                                  "continuous", args.min_load_kg * 1000,
                                  args.max_load_kg * 1000, shared_grid)
    load_kg = load_g / 1000.0
    y_cls = (load_kg > 0).astype(np.uint8)
    omega = 2.0 * np.pi * FREQUENCY_HZ
    r0 = 1.0 / g0
    c0 = b0 / omega
    r_unit, c_unit, z_unit = forward.sensor_states(
        load_g, r0, c0, curve_model, ["shared"] * 19, FREQUENCY_HZ)
    # Preserve the calibrated complex reference bit-for-bit for every
    # unpressed element; the R/C round-trip can otherwise differ by a few ulps.
    zero_mask = load_g == 0
    z0_grid = np.broadcast_to(z0, z_unit.shape)
    z_unit[zero_mask] = z0_grid[zero_mask]
    y_scale = float(np.median(np.abs(g0 + 1j * b0)))
    device = ("cuda" if forward.torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not forward.torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    print(f"Generating {N_TOTAL} samples ({N_LOADED} loaded + {N_UNLOADED} unloaded) on {device}")
    z_all = forward.nodal_forward_batch(z_unit, y_scale=y_scale,
                                        batch_size=args.batch_size, device=device)
    z_all0 = forward.nodal_forward_batch(z0[None, :], y_scale=y_scale,
                                         batch_size=1, device=device)[0]

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = args.output_root.resolve() / f"E2_continuous_20k_{stamp}"
    output_dir.mkdir(parents=False, exist_ok=False)
    manifest = source_manifest(args.seed, args, curve_model)
    dataset = {
        "Z_all": z_all, "Z_unit": z_unit, "Z_unit_0": z0, "Z_all_0": z_all0,
        "load_kg": load_kg, "y_cls": y_cls,
        "physical_edges": np.asarray(forward.PHYSICAL_EDGES, dtype=np.int16),
        "original_pairs": np.asarray(forward.ORIGINAL_PAIRS, dtype=np.int16),
        "frequency": FREQUENCY_HZ, "random_seed": args.seed,
        "R_unit": r_unit, "C_unit": c_unit, "G_unit_0": g0, "B_unit_0": b0,
        "generator": "E2_30pct_shared_continuous_curve_grounded_nodal_91pairs",
        "n_total": N_TOTAL, "n_loaded": N_LOADED, "zero_repeats": N_UNLOADED,
        "load_interpolation_method": manifest["load_interpolation"],
        "calibration_metadata": {**manifest["calibration"], "fit_summary": e2_summary},
        "source_provenance": manifest,
    }
    validation = validate(dataset, forward, curve_model, r0, c0, args, output_dir)
    dataset_path = output_dir / "e2_continuous_load_20k.pkl"
    with dataset_path.open("wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest["outputs"] = {"dataset": str(dataset_path),
                           "dataset_sha256": sha256(dataset_path)}
    manifest_path = output_dir / "source_provenance_manifest.json"
    validation_path = output_dir / "validation.json"
    write_json(manifest_path, manifest)
    validation["dataset_sha256"] = manifest["outputs"]["dataset_sha256"]
    validation["manifest_sha256"] = sha256(manifest_path)
    write_json(validation_path, validation)
    print(f"Saved dataset:   {dataset_path}")
    print(f"Saved validation:{validation_path}")
    print(f"Saved provenance:{manifest_path}")
    print("Validation: PASS")


if __name__ == "__main__":
    main()
