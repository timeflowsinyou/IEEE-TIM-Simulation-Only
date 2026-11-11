"""Train N7-A/N7-B binary-pressure and continuous-load models on E2 simulation.

Run directly with Spyder F5 or from a terminal.  No real measurement data is
opened by this script.  Both topologies use identical sample splits, model
architecture, hyperparameters, and simulated validation-based early stopping.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from n7k11_dual_task_inference import (
    N7K11DualHeadCNN, N7_TOPOLOGIES, normalize_features, predict_from_checkpoint,
    raw_features, unordered_pair_indices,
)


SCRIPT_PATH = Path(__file__).resolve()
SIM_DIR = SCRIPT_PATH.parent
DEFAULT_DATASET = (SIM_DIR / "E2_continuous_20k_20261111T150919" /
                   "e2_continuous_load_20k.pkl")
INFERENCE_SOURCE = SIM_DIR / "n7k11_dual_task_inference.py"


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


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False,
                               allow_nan=False), encoding="utf-8")


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_and_validate_dataset(path: Path):
    path = path.resolve()
    if not path.is_file(): raise FileNotFoundError(path)
    validation_path = path.parent / "validation.json"
    provenance_path = path.parent / "source_provenance_manifest.json"
    for required in (validation_path, provenance_path):
        if not required.is_file(): raise FileNotFoundError(required)
    validation = json.loads(validation_path.read_text(encoding="utf-8-sig"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
    if validation.get("status") != "PASS" or not all(validation.get("checks", {}).values()):
        raise ValueError("Dataset validation report is not an all-PASS report")
    actual_hash = sha256(path)
    if actual_hash != validation.get("dataset_sha256"):
        raise ValueError("Dataset SHA256 differs from validation report")
    with path.open("rb") as handle: data = pickle.load(handle)
    required = ("Z_all", "Z_all_0", "Z_unit", "load_kg", "y_cls",
                "physical_edges", "original_pairs")
    missing = [key for key in required if key not in data]
    if missing: raise KeyError(f"Dataset missing {missing}")
    z = np.asarray(data["Z_all"], np.complex128)
    reference = np.asarray(data["Z_all_0"], np.complex128)
    loads = np.asarray(data["load_kg"], np.float32)
    labels = np.asarray(data["y_cls"], np.uint8)
    z_unit = np.asarray(data["Z_unit"], np.complex128)
    pairs = np.asarray(data["original_pairs"], int)
    if z.shape != (20000, 91) or reference.shape != (91,) or z_unit.shape != (20000, 19):
        raise ValueError("Unexpected impedance shapes")
    if loads.shape != labels.shape != (20000, 19): raise ValueError("Unexpected targets")
    if not all(np.isfinite(array).all() for array in (z, reference, z_unit, loads)):
        raise ValueError("Nonfinite dataset")
    if not np.array_equal(labels, (loads > 0).astype(np.uint8)):
        raise ValueError("Binary labels do not match positive loads")
    # This checks numerical pairing, not empirical physical validity. The latter
    # remains limited by the documented 5-kHz normalized-curve transfer.
    if not validation["checks"].get("load_impedance_correspondence", False):
        raise ValueError("Generator did not validate load/impedance correspondence")
    return data, z, reference, loads, labels, pairs, validation, provenance, actual_hash


def exact_duplicate_groups(z: np.ndarray) -> list[np.ndarray]:
    groups = defaultdict(list)
    contiguous = np.ascontiguousarray(z)
    for i, row in enumerate(contiguous):
        groups[hashlib.sha256(row.view(np.uint8)).digest()].append(i)
    return [np.asarray(v, dtype=np.int64) for v in groups.values() if len(v) > 1]


def make_splits(labels: np.ndarray, seed: int):
    """70/15/15 stratified by count; exact duplicate groups stay in training."""
    rng = np.random.default_rng(seed)
    counts = labels.sum(axis=1).astype(int)
    train, validation, test = [], [], []
    # Count-zero rows are the one known exact duplicate group. Keeping them
    # together prevents leakage of the identical fixed baseline.
    train.extend(np.flatnonzero(counts == 0).tolist())
    for count in range(1, 20):
        ids = np.flatnonzero(counts == count)
        rng.shuffle(ids)
        n_test = int(round(0.15 * len(ids)))
        n_validation = int(round(0.15 * len(ids)))
        test.extend(ids[:n_test]); validation.extend(ids[n_test:n_test + n_validation])
        train.extend(ids[n_test + n_validation:])
    train = np.asarray(sorted(train), np.int64)
    validation = np.asarray(sorted(validation), np.int64)
    test = np.asarray(sorted(test), np.int64)
    if len(np.unique(np.concatenate([train, validation, test]))) != len(labels):
        raise AssertionError("Splits do not partition all sample IDs")
    return train, validation, test


def fit_normalizer(raw: np.ndarray, train_ids: np.ndarray):
    values = raw[train_ids].reshape(-1, 3).astype(np.float64)
    mean = values.mean(axis=0)
    std = np.maximum(values.std(axis=0), 1e-8)
    return mean, std


def combined_loss(logits, predicted_load, labels, loads, pos_weight,
                  regression_weight: float, zero_weight: float):
    classification = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weight)
    pressed = labels > 0.5
    unpressed = ~pressed
    pressed_loss = nn.functional.smooth_l1_loss(predicted_load[pressed], loads[pressed])
    zero_loss = predicted_load[unpressed].abs().mean()
    regression = pressed_loss + zero_weight * zero_loss
    return classification + regression_weight * regression, classification, pressed_loss, zero_loss


@torch.no_grad()
def run_prediction(model, x, ids, device, batch_size):
    probabilities, loads = [], []
    for start in range(0, len(ids), batch_size):
        xb = torch.from_numpy(x[ids[start:start + batch_size]]).to(device)
        logits, load = model(xb)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        loads.append(load.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(loads)


def f1_values(truth, predicted):
    truth = np.asarray(truth, bool); predicted = np.asarray(predicted, bool)
    tp = np.sum(truth & predicted, axis=0)
    fp = np.sum(~truth & predicted, axis=0)
    fn = np.sum(truth & ~predicted, axis=0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    micro = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)
    return float(micro), float(f1.mean()), precision, recall, f1


def choose_threshold(truth, probability):
    rows = []
    for threshold in np.linspace(0.05, 0.95, 91):
        micro, macro, _, recall, _ = f1_values(truth, probability >= threshold)
        exact = float(np.all(truth == (probability >= threshold), axis=1).mean())
        rows.append((macro, micro, exact, float(recall.min()), float(threshold)))
    return max(rows)[-1], rows


def detailed_metrics(truth_cls, truth_load, probability, raw_load, threshold):
    predicted_cls = probability >= threshold
    predicted_load = raw_load * predicted_cls
    micro, macro, precision, recall, f1 = f1_values(truth_cls, predicted_cls)
    pressed = truth_cls.astype(bool); unpressed = ~pressed
    errors = predicted_load - truth_load
    pressed_error = errors[pressed]
    overall = {
        "binary": {"micro_f1": micro, "macro_f1": macro,
                   "exact_match_accuracy": float(np.all(predicted_cls == truth_cls, axis=1).mean()),
                   "minimum_per_sensor_pressed_recall": float(recall.min()),
                   "threshold": float(threshold)},
        "regression": {
            "pressed_mae_kg": float(np.abs(pressed_error).mean()),
            "pressed_rmse_kg": float(np.sqrt(np.mean(pressed_error ** 2))),
            "pressed_bias_kg": float(pressed_error.mean()),
            "false_positive_gated_load_unpressed_mean_kg": float(predicted_load[unpressed].mean()),
            "raw_predicted_load_unpressed_mean_kg": float(raw_load[unpressed].mean()),
        },
    }
    per_sensor = []
    for sensor in range(19):
        mask = pressed[:, sensor]
        err = errors[mask, sensor]
        per_sensor.append({"sensor": f"S{sensor + 1}", "precision": precision[sensor],
                           "recall": recall[sensor], "f1": f1[sensor],
                           "pressed_mae_kg": float(np.abs(err).mean()),
                           "pressed_rmse_kg": float(np.sqrt(np.mean(err ** 2))),
                           "pressed_bias_kg": float(err.mean()),
                           "pressed_support": int(mask.sum())})
    ranges = []
    for low, high in ((0.01, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.000001)):
        mask = pressed & (truth_load >= low) & (truth_load < high)
        err = errors[mask]
        ranges.append({"load_min_kg": low, "load_max_kg": min(high, 1.0),
                       "support": int(mask.sum()), "mae_kg": float(np.abs(err).mean()),
                       "rmse_kg": float(np.sqrt(np.mean(err ** 2))),
                       "bias_kg": float(err.mean())})
    by_count = []
    counts = truth_cls.sum(axis=1).astype(int)
    for count in range(1, 20):
        sample_mask = counts == count
        t = truth_cls[sample_mask]; p = predicted_cls[sample_mask]
        mic, mac, _, rec, _ = f1_values(t, p)
        pmask = t.astype(bool)
        err = (predicted_load[sample_mask] - truth_load[sample_mask])[pmask]
        by_count.append({"pressed_sensor_count": count, "samples": int(sample_mask.sum()),
                         "micro_f1": mic, "macro_f1": mac,
                         "minimum_sensor_recall": float(rec.min()),
                         "exact_match_accuracy": float(np.all(t == p, axis=1).mean()),
                         "pressed_mae_kg": float(np.abs(err).mean()),
                         "pressed_rmse_kg": float(np.sqrt(np.mean(err ** 2)))})
    diagnostics = {"load_ranges": ranges, "by_pressed_sensor_count": by_count}
    return overall, per_sensor, diagnostics, predicted_cls.astype(np.uint8), predicted_load


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: raise ValueError("No rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows([jsonable(row) for row in rows])


def train_one(topology, x, labels, loads, train_ids, validation_ids, test_ids,
              reference, pair_indices, dataset_path, dataset_hash, output_dir, args, device):
    seed_all(args.seed)
    model = N7K11DualHeadCNN(args.max_load_kg).to(device)
    train_labels = labels[train_ids].astype(np.float32)
    positive = train_labels.sum(axis=0)
    negative = len(train_labels) - positive
    pos_weight = torch.tensor(negative / np.maximum(positive, 1), dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x[train_ids]),
                         torch.from_numpy(labels[train_ids].astype(np.float32)),
                         torch.from_numpy(loads[train_ids].astype(np.float32))),
                         batch_size=args.batch_size, shuffle=True,
                         generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x[validation_ids]),
                             torch.from_numpy(labels[validation_ids].astype(np.float32)),
                             torch.from_numpy(loads[validation_ids].astype(np.float32))),
                             batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=args.lr_patience, min_lr=args.min_lr)
    best_state = None; best_loss = np.inf; best_epoch = 0; stale = 0; history = []
    for epoch in range(1, args.epochs + 1):
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        for name, batches in (("train", loader), ("validation", val_loader)):
            model.train(name == "train"); totals = np.zeros(4); seen = 0
            with torch.set_grad_enabled(name == "train"):
                for xb, cb, lb in batches:
                    xb=xb.to(device); cb=cb.to(device); lb=lb.to(device)
                    if name == "train": optimizer.zero_grad(set_to_none=True)
                    logits, predicted_load = model(xb)
                    losses = combined_loss(logits, predicted_load, cb, lb, pos_weight,
                                           args.regression_weight, args.zero_weight)
                    if name == "train":
                        losses[0].backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                    totals += np.array([float(v.detach().cpu()) for v in losses]) * len(xb)
                    seen += len(xb)
            for key, value in zip(("total", "classification", "pressed_regression", "zero_regression"), totals / seen):
                row[f"{name}_{key}_loss"] = value
        scheduler.step(row["validation_total_loss"]); history.append(row)
        if row["validation_total_loss"] < best_loss - args.min_delta:
            best_loss = row["validation_total_loss"]; best_epoch = epoch; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{topology} epoch {epoch}: train={row['train_total_loss']:.5f} "
                  f"val={row['validation_total_loss']:.5f}", flush=True)
        if epoch >= args.min_epochs and stale >= args.patience: break
    if best_state is None: raise RuntimeError("No finite best model")
    model.load_state_dict(best_state); model.eval()
    val_probability, val_load = run_prediction(model, x, validation_ids, device, args.batch_size)
    threshold, threshold_rows = choose_threshold(labels[validation_ids], val_probability)
    test_probability, test_load = run_prediction(model, x, test_ids, device, args.batch_size)
    overall, per_sensor, diagnostics, test_binary, test_gated_load = detailed_metrics(
        labels[test_ids], loads[test_ids], test_probability, test_load, threshold)
    checkpoint_path = output_dir / f"{topology}_best_checkpoint.pt"
    checkpoint = {
        "format_version": 1, "topology": topology, "model_state_dict": best_state,
        "model_config": {"class": "N7K11DualHeadCNN", "max_load_kg": args.max_load_kg,
                         "backbone": "Residual CNN 3->64->128->256 with GroupNorm and global pooling",
                         "heads": "two independent 256->64->19 heads; sigmoid-bounded load"},
        "selected_pairs": N7_TOPOLOGIES[topology], "pair_indices": pair_indices,
        "Z_all_0": reference, "feature_definition": ["abs(Z)/abs(Z0)",
        "sin(angle(Z/Z0))", "cos(angle(Z/Z0))"],
        "feature_mean": args.feature_mean[topology], "feature_std": args.feature_std[topology],
        "output_definition": {"classification": "19 pressed probabilities/binary states",
                              "regression": "19 continuous loads", "load_unit": "kg"},
        "classification_threshold": threshold, "random_seed": args.seed,
        "split_seed": args.split_seed, "train_ids": train_ids,
        "validation_ids": validation_ids, "test_ids": test_ids,
        "dataset_path": str(dataset_path), "dataset_sha256": dataset_hash,
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "loss": {"classification": "BCEWithLogits with training-only per-sensor pos_weight",
                 "pressed_regression": "SmoothL1 over true-pressed entries",
                 "unpressed_regression": "mean absolute raw load over true-unpressed entries",
                 "regression_weight": args.regression_weight, "zero_weight": args.zero_weight},
        "hyperparameters": {k: getattr(args, k) for k in ("epochs", "min_epochs", "patience",
             "batch_size", "lr", "min_lr", "lr_patience", "weight_decay", "min_delta")},
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": torch.__version__},
    }
    torch.save(checkpoint, checkpoint_path)
    reloaded = predict_from_checkpoint(checkpoint_path, np.asarray(args.z_for_reload)[test_ids[:64]],
                                       device=str(device), batch_size=64)
    max_reload_difference = float(np.max(np.abs(
        reloaded["pressed_probability"] - test_probability[:64])))
    if max_reload_difference > 2e-6:
        raise AssertionError(f"Reloaded checkpoint prediction mismatch: {max_reload_difference}")
    overall["checkpoint_reload_max_probability_difference"] = max_reload_difference
    overall["best_epoch"] = best_epoch
    write_csv(output_dir / f"{topology}_training_history.csv", history)
    write_csv(output_dir / f"{topology}_per_sensor_metrics.csv", per_sensor)
    dump_json(output_dir / f"{topology}_load_regression_diagnostics.json", diagnostics)
    np.savez_compressed(output_dir / f"{topology}_test_predictions.npz", sample_ids=test_ids,
        pressed_probability=test_probability.astype(np.float32), pressed_binary=test_binary,
        raw_load_kg=test_load.astype(np.float32), load_kg=test_gated_load.astype(np.float32),
        truth_binary=labels[test_ids], truth_load_kg=loads[test_ids])
    return overall, checkpoint_path, threshold_rows


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output-root", type=Path, default=SIM_DIR)
    p.add_argument("--seed", type=int, default=43); p.add_argument("--split-seed", type=int, default=24)
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--min-epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=15); p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--lr-patience", type=int, default=6); p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--regression-weight", type=float, default=2.0)
    p.add_argument("--zero-weight", type=float, default=0.10)
    p.add_argument("--max-load-kg", type=float, default=1.0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--check-only", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA unavailable")
    (data, z, reference, loads, labels, pairs, validation, provenance,
     dataset_hash) = load_and_validate_dataset(args.dataset)
    train_ids, validation_ids, test_ids = make_splits(labels, args.split_seed)
    duplicates = exact_duplicate_groups(z)
    for group in duplicates:
        memberships = sum(bool(np.intersect1d(group, ids).size)
                          for ids in (train_ids, validation_ids, test_ids))
        if memberships != 1: raise AssertionError("Exact duplicate group crosses splits")
    raw_by_topology = {}; x_by_topology = {}; pair_indices = {}
    args.feature_mean = {}; args.feature_std = {}; args.z_for_reload = z
    for name, selected in N7_TOPOLOGIES.items():
        pair_indices[name] = unordered_pair_indices(pairs, selected)
        raw = raw_features(z, reference, pair_indices[name])
        mean, std = fit_normalizer(raw, train_ids)
        raw_by_topology[name] = raw
        x_by_topology[name] = normalize_features(raw, mean, std)
        args.feature_mean[name] = mean; args.feature_std[name] = std
        if x_by_topology[name].shape != (20000, 11, 3): raise AssertionError("Feature shape")
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = args.output_root.resolve() / f"E2_N7K11_binary_continuous_{stamp}"
    output_dir.mkdir(parents=False, exist_ok=False)
    np.savez_compressed(output_dir / "split_ids.npz", train_ids=train_ids,
                        validation_ids=validation_ids, test_ids=test_ids)
    shutil.copy2(INFERENCE_SOURCE, output_dir / INFERENCE_SOURCE.name)
    audit = {
        "input_shape_by_topology": {name: list(x.shape) for name, x in x_by_topology.items()},
        "pair_indices": pair_indices, "split_sizes": {"train": len(train_ids),
        "validation": len(validation_ids), "test": len(test_ids)},
        "duplicate_groups": {"count": len(duplicates),
         "sizes": sorted([len(group) for group in duplicates], reverse=True),
         "policy": "each exact duplicate group confined to one split; fixed unloaded group kept in training"},
        "simulation_only": True, "real_measurements_or_labels_accessed": False,
        "intrinsic_Z_used_as_model_input": False,
        "intrinsic_Z_role": "dataset consistency diagnostics only",
        "load_impedance_validation_scope": ("Generator consistency/PCHIP reproduction, not independent "
                                             "empirical validation of 5-kHz to 100-kHz transfer"),
    }
    dump_json(output_dir / "pretraining_audit.json", audit)
    if args.check_only:
        print(f"PASS check-only: {output_dir}"); return
    overall = {}
    for topology in ("N7-A", "N7-B"):
        print(f"Training {topology} on {device}...", flush=True)
        metrics, checkpoint, threshold_rows = train_one(
            topology, x_by_topology[topology], labels, loads, train_ids,
            validation_ids, test_ids, reference, pair_indices[topology],
            args.dataset.resolve(), dataset_hash, output_dir, args, device)
        overall[topology] = metrics
        dump_json(output_dir / f"{topology}_validation_threshold_search.json",
                  [{"macro_f1": r[0], "micro_f1": r[1], "exact_match": r[2],
                    "minimum_recall": r[3], "threshold": r[4]} for r in threshold_rows])
        print(f"{topology}: micro-F1={metrics['binary']['micro_f1']:.4f}, "
              f"pressed MAE={metrics['regression']['pressed_mae_kg']:.4f} kg", flush=True)
    dump_json(output_dir / "overall_test_metrics.json", overall)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "output_directory": output_dir,
        "training_script": SCRIPT_PATH, "training_script_sha256": sha256(SCRIPT_PATH),
        "inference_module": output_dir / INFERENCE_SOURCE.name,
        "inference_module_sha256": sha256(output_dir / INFERENCE_SOURCE.name),
        "dataset_path": args.dataset.resolve(), "dataset_sha256": dataset_hash,
        "dataset_validation_path": args.dataset.parent / "validation.json",
        "dataset_validation_sha256": sha256(args.dataset.parent / "validation.json"),
        "dataset_provenance_path": args.dataset.parent / "source_provenance_manifest.json",
        "dataset_provenance_sha256": sha256(args.dataset.parent / "source_provenance_manifest.json"),
        "generator_limitation": provenance["load_interpolation"]["limitation"],
        "split_seed": args.split_seed, "model_seed": args.seed,
        "same_splits_for_both_topologies": True,
        "training_data_policy": "simulation only; no real measurements/labels",
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": torch.__version__, "platform": platform.platform()},
        "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                  for p in output_dir.iterdir() if p.is_file()},
    }
    dump_json(output_dir / "reproducibility_manifest.json", manifest)
    print(f"Complete: {output_dir}", flush=True)


if __name__ == "__main__": main()
