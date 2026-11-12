"""Train N7-B for pressed-state classification and signed intrinsic-|Z| change.

This is a simulation-only methodological comparison.  The normalized sensor
response transferred from 5 kHz to the E2 baseline at 100 kHz has not been
experimentally validated.  The exact saved split from the frozen N7-B load
baseline is required; this script never creates a replacement split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from n7k11_dual_task_inference import (
    N7_TOPOLOGIES, ResidualBlock, normalize_features, raw_features,
    unordered_pair_indices,
)


SCRIPT_PATH = Path(__file__).resolve()
SIM_DIR = SCRIPT_PATH.parent
DEFAULT_DATASET = SIM_DIR / "E2_continuous_20k_20261111T150919" / "e2_continuous_load_20k.pkl"
DEFAULT_BASELINE_DIR = SIM_DIR / "E2_N7K11_binary_continuous_20261111T171312"
EXPECTED_BASELINE_SPLIT_SHA256 = "ced728b9a4489731601d4cd333df84b72b479b253b360b04586c5d639db98ea5"
EXPECTED_DATASET_SHA256 = "2fa0bd9d30a38fe759b41a7fa9b3d094b0a43709a9289b70bf69a4764fd4d1ae"
N7B_PAIRS = tuple(N7_TOPOLOGIES["N7-B"])
FEATURE_DEFINITION = ["abs(Z)/abs(Z0)", "sin(angle(Z/Z0))", "cos(angle(Z/Z0))"]
TRANSFER_LIMITATION = (
    "Normalized load-dependent R and C responses measured on five sensors at 5 kHz "
    "are transferred to 19 E2 unloaded baselines at 100 kHz. This is a simulation "
    "assumption and has not been independently validated at 100 kHz."
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(x):
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.ndarray): return jsonable(x.tolist())
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False,
                               allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: raise ValueError(f"No rows for {path.name}")
    keys = []
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerows([{k: jsonable(row.get(k)) for k in keys} for row in rows])


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try: torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError: pass


class N7BMagnitudeCNN(nn.Module):
    """Exact baseline layers, with an unconstrained signed regression output."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            ResidualBlock(3, 64), ResidualBlock(64, 128), nn.MaxPool1d(2),
            ResidualBlock(128, 256), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.classifier = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 19))
        self.regressor = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 19))

    def forward(self, x):
        h = self.backbone(x.transpose(1, 2))
        return self.classifier(h), self.regressor(h)


def load_inputs(dataset_path: Path, baseline_dir: Path):
    dataset_path = dataset_path.resolve(); baseline_dir = baseline_dir.resolve()
    validation_path = dataset_path.parent / "validation.json"
    provenance_path = dataset_path.parent / "source_provenance_manifest.json"
    split_path = baseline_dir / "split_ids.npz"
    metrics_path = baseline_dir / "overall_test_metrics.json"
    checkpoint_path = baseline_dir / "N7-B_best_checkpoint.pt"
    for p in (dataset_path, validation_path, provenance_path, split_path,
              metrics_path, checkpoint_path):
        if not p.is_file(): raise FileNotFoundError(p)
    dataset_hash = sha256(dataset_path)
    if dataset_hash != EXPECTED_DATASET_SHA256:
        raise ValueError(f"Dataset identity mismatch: {dataset_hash}")
    if sha256(split_path) != EXPECTED_BASELINE_SPLIT_SHA256:
        raise ValueError("Frozen baseline split file identity mismatch; refusing non-paired experiment")
    validation = json.loads(validation_path.read_text(encoding="utf-8-sig"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
    baseline_metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))["N7-B"]
    if validation.get("status") != "PASS" or not all(validation.get("checks", {}).values()):
        raise ValueError("Source dataset validation is not all-PASS")
    if validation.get("dataset_sha256") != dataset_hash:
        raise ValueError("Dataset hash differs from its validation manifest")
    with dataset_path.open("rb") as f: data = pickle.load(f)
    required = ("Z_all", "Z_all_0", "Z_unit", "Z_unit_0", "load_kg", "y_cls",
                "physical_edges", "original_pairs")
    missing = [k for k in required if k not in data]
    if missing: raise KeyError(f"Dataset fields missing: {missing}")
    z_all = np.asarray(data["Z_all"], np.complex128)
    z_all_0 = np.asarray(data["Z_all_0"], np.complex128)
    z_unit = np.asarray(data["Z_unit"], np.complex128)
    z_unit_0 = np.asarray(data["Z_unit_0"], np.complex128)
    loads = np.asarray(data["load_kg"], np.float64)
    labels = np.asarray(data["y_cls"], np.uint8)
    pairs = np.asarray(data["original_pairs"], int)
    shapes = {"Z_all": z_all.shape, "Z_all_0": z_all_0.shape,
              "Z_unit": z_unit.shape, "Z_unit_0": z_unit_0.shape,
              "load_kg": loads.shape, "y_cls": labels.shape,
              "original_pairs": pairs.shape}
    expected = {"Z_all": (20000,91), "Z_all_0": (91,), "Z_unit": (20000,19),
                "Z_unit_0": (19,), "load_kg": (20000,19), "y_cls": (20000,19),
                "original_pairs": (91,2)}
    if shapes != expected: raise ValueError(f"Unexpected shapes: {shapes}")
    for name, a in (("Z_all", z_all), ("Z_all_0", z_all_0), ("Z_unit", z_unit),
                    ("Z_unit_0", z_unit_0), ("load_kg", loads)):
        if not np.isfinite(a).all(): raise ValueError(f"Nonfinite values in {name}")
    ref_mag = np.abs(z_unit_0)
    if not np.isfinite(ref_mag).all() or np.any(ref_mag <= 0):
        raise ValueError("All 19 unloaded intrinsic magnitudes must be finite and positive")
    if not np.array_equal(labels, (loads > 0).astype(np.uint8)):
        raise ValueError("y_cls is inconsistent with load_kg > 0")
    delta = (np.abs(z_unit) - ref_mag[None, :]) / ref_mag[None, :]
    if delta.shape != (20000,19) or not np.isfinite(delta).all():
        raise ValueError("Invalid signed relative intrinsic-magnitude target")
    if not np.allclose(delta[labels == 0], 0.0, rtol=0, atol=1e-12):
        raise ValueError("Unpressed intrinsic targets are not zero within 1e-12")
    with np.load(split_path) as s:
        split = tuple(np.asarray(s[k], np.int64) for k in
                      ("train_ids", "validation_ids", "test_ids"))
    combined = np.concatenate(split)
    if len(combined) != 20000 or not np.array_equal(np.sort(combined), np.arange(20000)):
        raise ValueError("Saved split does not partition all samples exactly once")
    # Guard against exact duplicate inputs crossing the frozen split.
    membership = np.empty(20000, np.int8)
    for part, ids in enumerate(split): membership[ids] = part
    seen = {}; crossing = []
    for i, row in enumerate(np.ascontiguousarray(z_all)):
        key = row.tobytes()
        if key in seen and membership[seen[key]] != membership[i]: crossing.append((seen[key], i))
        else: seen[key] = i
    if crossing: raise ValueError(f"Exact duplicate inputs cross saved partitions: {crossing[:5]}")
    # Reading checkpoint metadata does not instantiate a model.  It proves that
    # the standalone split file is the split actually embedded in the baseline.
    baseline_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key, ids in zip(("train_ids", "validation_ids", "test_ids"), split):
        if key not in baseline_checkpoint or not np.array_equal(
                np.asarray(baseline_checkpoint[key], np.int64), ids):
            raise ValueError(f"Saved split differs from baseline checkpoint field {key}")
    if baseline_checkpoint.get("dataset_sha256") != dataset_hash:
        raise ValueError("Baseline checkpoint was trained on a different dataset")
    if tuple(map(tuple, baseline_checkpoint.get("selected_pairs", ()))) != N7B_PAIRS:
        raise ValueError("Baseline checkpoint N7-B pair selection differs")
    pair_indices = unordered_pair_indices(pairs, N7B_PAIRS)
    raw = raw_features(z_all, z_all_0, pair_indices)
    train_ids, val_ids, test_ids = split
    flat = raw[train_ids].reshape(-1, 3).astype(np.float64)
    feature_mean = flat.mean(0); feature_std = np.maximum(flat.std(0), 1e-8)
    if not np.allclose(feature_mean, baseline_checkpoint["feature_mean"], rtol=0, atol=1e-12):
        raise ValueError("Recomputed feature mean differs from baseline checkpoint")
    if not np.allclose(feature_std, baseline_checkpoint["feature_std"], rtol=0, atol=1e-12):
        raise ValueError("Recomputed feature std differs from baseline checkpoint")
    x = normalize_features(raw, feature_mean, feature_std)
    if x.shape != (20000,11,3): raise ValueError(f"Unexpected feature shape {x.shape}")
    return dict(data=data, z_all=z_all, z_all_0=z_all_0, z_unit=z_unit,
        z_unit_0=z_unit_0, ref_mag=ref_mag, loads=loads, labels=labels, delta=delta,
        pairs=pairs, pair_indices=pair_indices, x=x, feature_mean=feature_mean,
        feature_std=feature_std, split=split, validation=validation,
        provenance=provenance, baseline_metrics=baseline_metrics,
        paths=dict(dataset=dataset_path, validation=validation_path,
                   provenance=provenance_path, split=split_path,
                   baseline_metrics=metrics_path, baseline_checkpoint=checkpoint_path),
        baseline_checkpoint_metadata=baseline_checkpoint,
        hashes=dict(dataset=dataset_hash, validation=sha256(validation_path),
                    provenance=sha256(provenance_path), split=sha256(split_path),
                    baseline_checkpoint=sha256(checkpoint_path)))


def basic_stats(values: np.ndarray) -> dict:
    v = np.asarray(values, float).reshape(-1)
    return {"count": int(v.size), "minimum": float(v.min()), "maximum": float(v.max()),
            "mean": float(v.mean()), "median": float(np.median(v)),
            "standard_deviation": float(v.std()), "finite": bool(np.isfinite(v).all())}


def target_diagnostics(delta, labels):
    pressed = labels.astype(bool); rows = []
    for s in range(19):
        for subset, mask in (("all", np.ones(len(delta), bool)),
                             ("pressed", pressed[:,s]), ("unpressed", ~pressed[:,s])):
            rows.append({"sensor": f"S{s+1}", "subset": subset,
                         **basic_stats(delta[mask,s])})
    summary = {"all": basic_stats(delta), "pressed": basic_stats(delta[pressed]),
               "unpressed": basic_stats(delta[~pressed]),
               "pressed_fraction_delta_le_minus_0_99": float(np.mean(delta[pressed] <= -.99)),
               "pressed_fraction_delta_le_minus_0_999": float(np.mean(delta[pressed] <= -.999)),
               "definition": "Delta=(abs(Z_unit)-abs(Z_unit_0))/abs(Z_unit_0); signed; unclipped"}
    return rows, summary


def f1_metrics(truth, pred):
    t=np.asarray(truth,bool); p=np.asarray(pred,bool)
    tp=np.sum(t&p,0); fp=np.sum(~t&p,0); fn=np.sum(t&~p,0)
    precision=tp/np.maximum(tp+fp,1); recall=tp/np.maximum(tp+fn,1)
    f1=2*tp/np.maximum(2*tp+fp+fn,1)
    micro=float(2*tp.sum()/max(2*tp.sum()+fp.sum()+fn.sum(),1))
    return dict(micro_f1=micro, macro_f1=float(f1.mean()),
                exact_match_accuracy=float(np.all(t==p,axis=1).mean()),
                minimum_per_sensor_pressed_recall=float(recall.min()),
                precision=precision, recall=recall, f1=f1)


def select_threshold(truth, prob):
    rows=[]
    for threshold in np.linspace(.05,.95,91):
        m=f1_metrics(truth,prob>=threshold)
        rows.append({"threshold":float(threshold), "macro_f1":m["macro_f1"],
                     "micro_f1":m["micro_f1"], "exact_match_accuracy":m["exact_match_accuracy"],
                     "minimum_recall":m["minimum_per_sensor_pressed_recall"]})
    best=max(rows,key=lambda r:(r["macro_f1"],r["micro_f1"],r["exact_match_accuracy"],
                                r["minimum_recall"],r["threshold"]))
    return best["threshold"],rows


def rankdata(a):
    a=np.asarray(a); order=np.argsort(a,kind="mergesort"); ranks=np.empty(len(a),float)
    i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]: j+=1
        ranks[order[i:j]]=(i+j-1)/2+1; i=j
    return ranks


def correlation(a,b,method="pearson"):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<2: return None
    if method=="spearman": a,b=rankdata(a),rankdata(b)
    if np.std(a)==0 or np.std(b)==0: return None
    return float(np.corrcoef(a,b)[0,1])


def error_stats(truth,pred,relative_denominator_floor=None):
    t=np.asarray(truth,float); p=np.asarray(pred,float); e=p-t
    out={"count":int(t.size), "mae":float(np.mean(np.abs(e))),
         "rmse":float(np.sqrt(np.mean(e*e))), "bias":float(np.mean(e)),
         "pearson":correlation(t,p), "spearman":correlation(t,p,"spearman")}
    if relative_denominator_floor is not None:
        denom=np.maximum(np.abs(t),relative_denominator_floor)
        out["mean_absolute_relative_error"]=float(np.mean(np.abs(e)/denom))
    return out


def pairwise_ranking(truth,pred,chunk=256):
    t=np.asarray(truth,float); p=np.asarray(pred,float); good=0; valid=0
    for i in range(0,len(t),chunk):
        td=t[i:i+chunk,None]-t[None,:]; pd=p[i:i+chunk,None]-p[None,:]
        jj=np.arange(len(t))[None,:]; ii=np.arange(i,min(i+chunk,len(t)))[:,None]
        mask=(jj>ii)&(td!=0)
        valid+=int(mask.sum()); good+=int(((td*pd)>0)[mask].sum())
    return {"pairwise_ranking_accuracy": float(good/valid) if valid else None,
            "valid_comparison_pairs":valid}


def magnitude_evaluation(delta, pred, labels, loads, ref_mag):
    pressed=labels.astype(bool); unpressed=~pressed; counts=labels.sum(1)
    masks={"all_entries":np.ones(delta.shape,bool), "truly_pressed":pressed,
           "truly_unpressed":unpressed,
           "samples_1_to_4_all_entries":np.broadcast_to(((counts>=1)&(counts<=4))[:,None],delta.shape),
           "samples_1_to_4_pressed_entries":pressed&((counts>=1)&(counts<=4))[:,None]}
    regression={name:error_stats(delta[mask],pred[mask]) for name,mask in masks.items()}
    bins=[]
    for low,high in ((0,.25),(.25,.5),(.5,.75),(.75,1.0000001)):
        mask=pressed&(loads>low)&(loads<=high)
        bins.append({"load_min_kg_exclusive":low,"load_max_kg_inclusive":min(high,1.0),
                     **error_stats(delta[mask],pred[mask])})
    true_mag=ref_mag[None,:]*(1+delta); pred_mag=ref_mag[None,:]*(1+pred)
    positive_true=true_mag[true_mag>0]
    floor=max(float(np.median(positive_true)*1e-12),np.finfo(float).tiny)
    reconstructed={name:error_stats(true_mag[mask],pred_mag[mask],floor)
                   for name,mask in masks.items()}
    reconstructed["policy"]={"relative_error_denominator":"max(abs(true magnitude), floor)",
        "floor_ohm":floor,"true_entries_at_or_below_floor":int(np.sum(true_mag<=floor)),
        "invalid_predictions_1_plus_delta_below_zero":int(np.sum(1+pred<0)),
        "negative_magnitudes_not_clamped":True}
    per_sensor=[]; ranking=[]
    for s in range(19):
        pm=pressed[:,s]
        d=error_stats(delta[pm,s],pred[pm,s]); m=error_stats(true_mag[pm,s],pred_mag[pm,s],floor)
        per_sensor.append({"sensor":f"S{s+1}","pressed_support":int(pm.sum()),
            **{f"delta_{k}":v for k,v in d.items()},
            "magnitude_mae_ohm":m["mae"],"magnitude_rmse_ohm":m["rmse"],
            "magnitude_mean_absolute_relative_error":m["mean_absolute_relative_error"]})
        r=pairwise_ranking(delta[pm,s],pred[pm,s]); lr=pairwise_ranking(loads[pm,s],pred[pm,s])
        ranking.append({"sensor":f"S{s+1}","intrinsic_spearman":d["spearman"],**r,
            "load_ordering_spearman":correlation(loads[pm,s],pred[pm,s],"spearman"),
            "load_ordering_pairwise_accuracy":lr["pairwise_ranking_accuracy"],
            "load_ordering_valid_pairs":lr["valid_comparison_pairs"],
            "note":"load ordering is diagnostic only; output target is intrinsic magnitude change"})
    return regression,reconstructed,per_sensor,ranking,bins,true_mag,pred_mag


def constant_baselines(delta,labels,train_ids,test_ids):
    train=delta[train_ids]; train_cls=labels[train_ids].astype(bool); truth=delta[test_ids]
    test_cls=labels[test_ids].astype(bool); pressed=test_cls
    global_median=float(np.median(train))
    sensor_median=np.median(train,axis=0)
    pressed_median=np.array([np.median(train[train_cls[:,s],s]) for s in range(19)])
    candidates={"zero_change":{"prediction":np.zeros_like(truth),"deployable":True},
        "global_training_median":{"prediction":np.full_like(truth,global_median),"deployable":True},
        "per_sensor_training_median":{"prediction":np.broadcast_to(sensor_median,truth.shape),"deployable":True},
        "per_sensor_training_median_oracle_true_status":{"prediction":np.where(test_cls,pressed_median[None,:],0.0),"deployable":False}}
    rows=[]; zero_mae=None
    for name,item in candidates.items():
        p=item["prediction"]; mae=float(np.mean(np.abs(p[pressed]-truth[pressed])))
        if name=="zero_change": zero_mae=mae
        rows.append({"baseline":name,"deployable":item["deployable"],
                     "pressed_mae_delta":mae,"pressed_rmse_delta":float(np.sqrt(np.mean((p[pressed]-truth[pressed])**2))),
                     "all_entries_mae_delta":float(np.mean(np.abs(p-truth)))})
    for row in rows:
        row["percent_improvement_vs_zero_change_pressed_mae"]=(100*(zero_mae-row["pressed_mae_delta"])/zero_mae if zero_mae else None)
    return rows,candidates


def loss_parts(logits,pred,cls,target,pos_weight,regression_weight,zero_weight):
    classification=nn.functional.binary_cross_entropy_with_logits(logits,cls,pos_weight=pos_weight)
    pressed=cls>0.5
    pressed_loss=nn.functional.smooth_l1_loss(pred[pressed],target[pressed])
    zero_loss=nn.functional.smooth_l1_loss(pred[~pressed],target[~pressed])
    regression=pressed_loss+zero_weight*zero_loss
    return classification+regression_weight*regression,classification,regression,pressed_loss,zero_loss


@torch.no_grad()
def predict(model,x,ids,device,batch_size):
    probs=[]; values=[]
    for start in range(0,len(ids),batch_size):
        logits,p=model(torch.from_numpy(x[ids[start:start+batch_size]]).to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy()); values.append(p.cpu().numpy())
    return np.concatenate(probs),np.concatenate(values)


def figures(out,history,truth,pred,per_sensor,bins,baseline_rows):
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    e=[r["epoch"] for r in history]
    for part in ("train","validation"):
        ax[0].plot(e,[r[f"{part}_classification_loss"] for r in history],label=part)
        ax[1].plot(e,[r[f"{part}_regression_loss"] for r in history],label=part)
    ax[0].set_title("Classification loss"); ax[1].set_title("Magnitude regression loss")
    for a in ax:a.set_xlabel("Epoch");a.legend();a.grid(alpha=.25)
    save_figure(fig,out/"training_validation_losses")
    rng=np.random.default_rng(0); flat=np.arange(truth.size); ids=rng.choice(flat,min(20000,len(flat)),False)
    fig,ax=plt.subplots(figsize=(5.5,5)); ax.scatter(truth.ravel()[ids],pred.ravel()[ids],s=3,alpha=.2)
    lim=[min(truth.min(),pred.min()),max(truth.max(),pred.max())];ax.plot(lim,lim,"k--")
    ax.set(xlabel="True signed relative |Z| change",ylabel="Predicted",title="True versus predicted delta");ax.grid(alpha=.2)
    save_figure(fig,out/"true_vs_predicted_relative_magnitude")
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(np.arange(19)+1,[r["delta_mae"] for r in per_sensor])
    ax.set(xlabel="Sensor",ylabel="Pressed-entry MAE (delta)",title="Per-sensor relative-magnitude error");ax.grid(axis="y",alpha=.2)
    save_figure(fig,out/"per_sensor_relative_magnitude_errors")
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(range(len(bins)),[r["mae"] for r in bins])
    ax.set_xticks(range(len(bins)),["0-.25",".25-.50",".50-.75",".75-1.0"]);ax.set(xlabel="Load interval (kg)",ylabel="MAE (delta)",title="Pressed regression error by simulated load")
    save_figure(fig,out/"regression_errors_by_load_interval")
    fig,ax=plt.subplots(figsize=(9,4));ax.bar(range(len(baseline_rows)),[r["pressed_mae_delta"] for r in baseline_rows])
    ax.set_xticks(range(len(baseline_rows)),[r["baseline"].replace("_","\n") for r in baseline_rows],fontsize=8);ax.set(ylabel="Pressed-entry MAE (delta)",title="Constant-prediction baselines")
    save_figure(fig,out/"constant_prediction_baselines")


def save_figure(fig,path):
    fig.tight_layout();fig.savefig(path.with_suffix(".png"),dpi=180);fig.savefig(path.with_suffix(".pdf"));plt.close(fig)


def train(bundle,args,out,device):
    train_ids,val_ids,test_ids=bundle["split"];x=bundle["x"];labels=bundle["labels"];delta=bundle["delta"]
    seed_all(args.seed);model=N7BMagnitudeCNN().to(device)
    cls_train=labels[train_ids].astype(np.float32);pos=cls_train.sum(0)
    pos_weight=torch.tensor((len(train_ids)-pos)/np.maximum(pos,1),dtype=torch.float32,device=device)
    def loader(ids,shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(x[ids]),torch.from_numpy(labels[ids].astype(np.float32)),
            torch.from_numpy(delta[ids].astype(np.float32))),batch_size=args.batch_size,shuffle=shuffle,
            generator=torch.Generator().manual_seed(args.seed) if shuffle else None)
    train_loader=loader(train_ids,True);val_loader=loader(val_ids,False)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=args.lr_patience,min_lr=args.min_lr)
    best=None;best_loss=np.inf;best_epoch=0;stale=0;history=[]
    for epoch in range(1,args.epochs+1):
        row={"epoch":epoch,"lr":optimizer.param_groups[0]["lr"]}
        for name,batches in (("train",train_loader),("validation",val_loader)):
            model.train(name=="train");tot=np.zeros(5);seen=0
            with torch.set_grad_enabled(name=="train"):
                for xb,cb,tb in batches:
                    xb=xb.to(device);cb=cb.to(device);tb=tb.to(device)
                    if name=="train":optimizer.zero_grad(set_to_none=True)
                    losses=loss_parts(*model(xb),cb,tb,pos_weight,args.regression_weight,args.zero_weight)
                    if name=="train":losses[0].backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                    tot+=np.array([float(v.detach().cpu()) for v in losses])*len(xb);seen+=len(xb)
            for key,value in zip(("total","classification","regression","pressed_regression","unpressed_regression"),tot/seen):row[f"{name}_{key}_loss"]=value
        scheduler.step(row["validation_total_loss"]);history.append(row)
        if row["validation_total_loss"]<best_loss-args.min_delta:
            best_loss=row["validation_total_loss"];best_epoch=epoch;stale=0
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        print(f"epoch {epoch}: train={row['train_total_loss']:.6f}, validation={row['validation_total_loss']:.6f}",flush=True)
        if epoch>=args.min_epochs and stale>=args.patience:break
    if best is None:raise RuntimeError("No finite best model")
    model.load_state_dict(best);model.eval()
    val_prob,_=predict(model,x,val_ids,device,args.batch_size);threshold,threshold_rows=select_threshold(labels[val_ids],val_prob)
    test_prob,test_pred=predict(model,x,test_ids,device,args.batch_size);test_binary=test_prob>=threshold
    binary_full=f1_metrics(labels[test_ids],test_binary);counts=labels[test_ids].sum(1);subset=(counts>=1)&(counts<=4)
    binary_subset=f1_metrics(labels[test_ids][subset],test_binary[subset])
    regression,reconstructed,per_sensor,ranking,bins,true_mag,pred_mag=magnitude_evaluation(
        delta[test_ids],test_pred,labels[test_ids],bundle["loads"][test_ids],bundle["ref_mag"])
    baseline_rows,_=constant_baselines(delta,labels,train_ids,test_ids)
    cnn_mae=regression["truly_pressed"]["mae"]
    for row in baseline_rows:
        b=row["pressed_mae_delta"];row["cnn_pressed_mae_delta"]=cnn_mae
        row["cnn_percent_improvement_vs_baseline"]=(100*(b-cnn_mae)/b if b else None)
    checkpoint={"format_version":1,"model_state_dict":best,"model_class":"N7BMagnitudeCNN",
        "model_architecture":"baseline N7K11 residual CNN; unconstrained linear 19-channel regression output",
        "selected_pairs":N7B_PAIRS,"pair_indices":bundle["pair_indices"],"Z_all_0":bundle["z_all_0"],
        "Z_unit_0":bundle["z_unit_0"],"feature_definition":FEATURE_DEFINITION,
        "feature_mean":bundle["feature_mean"],"feature_std":bundle["feature_std"],
        "target_definition":"(abs(Z_unit)-abs(Z_unit_0))/abs(Z_unit_0), signed and unclipped",
        "target_standardization":{"used":False,"reason":"bounded dimensionless target is trained directly"},
        "classification_threshold":threshold,"best_epoch":best_epoch,"best_validation_loss":best_loss,
        "train_ids":train_ids,"validation_ids":val_ids,"test_ids":test_ids,"random_seed":args.seed,
        "dataset_sha256":bundle["hashes"]["dataset"],"split_sha256":bundle["hashes"]["split"],
        "simulation_transfer_limitation":TRANSFER_LIMITATION,"hyperparameters":vars(args)}
    torch.save(checkpoint,out/"N7-B_magnitude_best_checkpoint.pt")
    write_csv(out/"N7-B_magnitude_training_history.csv",history)
    cls_per=[]
    for s,row in enumerate(per_sensor):
        row.update(precision=binary_full["precision"][s],recall=binary_full["recall"][s],f1=binary_full["f1"][s]);cls_per.append(row)
    write_csv(out/"N7-B_magnitude_per_sensor_metrics.csv",cls_per)
    diagnostics={"primary_metric":"MAE on truly pressed entries","regression":regression,
        "reconstructed_intrinsic_magnitude_ohm":reconstructed,"load_interval_metrics":bins,
        "ranking_and_monotonicity":ranking,"constant_prediction_baselines":baseline_rows}
    dump_json(out/"N7-B_magnitude_regression_diagnostics.json",diagnostics)
    test_metrics={"binary_full_test":{k:v for k,v in binary_full.items() if not isinstance(v,np.ndarray)},
        "binary_1_to_4_pressed_samples":{k:v for k,v in binary_subset.items() if not isinstance(v,np.ndarray)},
        "classification_threshold":threshold,"threshold_selected_on":"validation macro-F1",
        "magnitude_regression":regression,"reconstructed_intrinsic_magnitude_ohm":reconstructed,
        "best_epoch":best_epoch,"best_validation_loss":best_loss}
    dump_json(out/"N7-B_magnitude_test_metrics.json",test_metrics)
    dump_json(out/"N7-B_magnitude_validation_threshold_search.json",threshold_rows)
    np.savez_compressed(out/"N7-B_magnitude_test_predictions.npz",sample_ids=test_ids,
        pressed_probability=test_prob.astype(np.float32),pressed_binary=test_binary.astype(np.uint8),
        truth_binary=labels[test_ids],truth_delta=delta[test_ids].astype(np.float32),
        predicted_delta=test_pred.astype(np.float32),truth_magnitude_ohm=true_mag,
        predicted_magnitude_ohm=pred_mag,truth_load_kg=bundle["loads"][test_ids])
    baseline=bundle["baseline_metrics"]
    comparison=[{"model":"frozen N7-B load baseline","paired_split":True,"micro_f1":baseline["binary"]["micro_f1"],
        "macro_f1":baseline["binary"]["macro_f1"],"exact_match_accuracy":baseline["binary"]["exact_match_accuracy"],
        "minimum_per_sensor_pressed_recall":baseline["binary"]["minimum_per_sensor_pressed_recall"],
        "best_epoch":baseline["best_epoch"],"training_settings":"Adam, batch 256, max 100 epochs, early stopping; load regression target"},
        {"model":"N7-B signed intrinsic magnitude","paired_split":True,"micro_f1":binary_full["micro_f1"],
        "macro_f1":binary_full["macro_f1"],"exact_match_accuracy":binary_full["exact_match_accuracy"],
        "minimum_per_sensor_pressed_recall":binary_full["minimum_per_sensor_pressed_recall"],
        "best_epoch":best_epoch,"training_settings":f"Adam, batch {args.batch_size}, max {args.epochs} epochs, early stopping; signed delta target"}]
    write_csv(out/"N7-B_baseline_comparison.csv",comparison)
    figures(out,history,delta[test_ids],test_pred,per_sensor,bins,baseline_rows)
    return checkpoint,test_metrics


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,default=DEFAULT_DATASET);p.add_argument("--baseline-dir",type=Path,default=DEFAULT_BASELINE_DIR)
    p.add_argument("--output-root",type=Path,default=SIM_DIR);p.add_argument("--seed",type=int,default=43)
    p.add_argument("--epochs",type=int,default=100);p.add_argument("--min-epochs",type=int,default=25);p.add_argument("--patience",type=int,default=15)
    p.add_argument("--batch-size",type=int,default=256);p.add_argument("--lr",type=float,default=1e-3);p.add_argument("--min-lr",type=float,default=1e-5)
    p.add_argument("--lr-patience",type=int,default=6);p.add_argument("--weight-decay",type=float,default=1e-5);p.add_argument("--min-delta",type=float,default=1e-5)
    p.add_argument("--regression-weight",type=float,default=2.0,help="Weight on pressed+weighted-unpressed SmoothL1 regression loss")
    p.add_argument("--zero-weight",type=float,default=.10,help="Weight of unpressed-target regression within the regression term")
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto");p.add_argument("--cpu-threads",type=int,default=8);p.add_argument("--check-only",action="store_true")
    return p


def main():
    args=parser().parse_args();torch.set_num_threads(args.cpu_threads)
    bundle=load_inputs(args.dataset,args.baseline_dir)
    target_rows,target_summary=target_diagnostics(bundle["delta"],bundle["labels"])
    stamp=datetime.now().strftime("%Y%m%dT%H%M%S")
    out=args.output_root.resolve()/f"E2_N7B_binary_intrinsic_magnitude_{stamp}"
    out.mkdir(parents=False,exist_ok=False)
    train_ids,val_ids,test_ids=bundle["split"]
    split_manifest={"source":"frozen baseline split_ids.npz; never regenerated",
        "source_path":bundle["paths"]["split"],"source_sha256":bundle["hashes"]["split"],
        "train_ids":train_ids,"validation_ids":val_ids,"test_ids":test_ids,
        "sizes":{"train":len(train_ids),"validation":len(val_ids),"test":len(test_ids)},
        "directly_paired_with_baseline":True}
    data_validation={"status":"PASS","dataset_fields":list(bundle["data"]),
        "dataset_sha256":bundle["hashes"]["dataset"],"shapes":{"Z_all":[20000,91],"Z_unit":[20000,19],"Z_unit_0":[19]},
        "reference_magnitude_finite_and_strictly_positive":True,"target_finite":True,
        "unpressed_target_zero_within_1e-12":True,"N7B_selected_pairs":N7B_PAIRS,
        "N7B_pair_indices":bundle["pair_indices"],"feature_shape":list(bundle["x"].shape),
        "feature_definition":FEATURE_DEFINITION,"saved_baseline_split_verified":True,
        "old_checkpoint_targeted_for_overwrite":False,"new_output_directory":out,
        "network_instantiated":False,"training_performed":False,"simulation_transfer_limitation":TRANSFER_LIMITATION}
    run_config={"created_utc":datetime.now(timezone.utc).isoformat(),"arguments":vars(args),
        "dataset_path":bundle["paths"]["dataset"],"dataset_sha256":bundle["hashes"]["dataset"],
        "baseline_directory":args.baseline_dir.resolve(),"baseline_checkpoint_sha256":bundle["hashes"]["baseline_checkpoint"],
        "model":"N7-B baseline backbone and two heads; regression output unconstrained linear",
        "target":"signed dimensionless (abs(Z_unit)-abs(Z_unit_0))/abs(Z_unit_0), not clipped or standardized",
        "loss":"BCEWithLogits + regression_weight*(pressed SmoothL1 + zero_weight*unpressed SmoothL1)",
        "feature_mean":bundle["feature_mean"],"feature_std":bundle["feature_std"],
        "random_seed":args.seed,"split_source_sha256":bundle["hashes"]["split"],
        "dependencies":{"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__,"matplotlib":plt.matplotlib.__version__,"platform":platform.platform()},
        "determinism":"fixed Python/NumPy/Torch seeds; deterministic cuDNN and deterministic-algorithm request; hardware/library differences may remain",
        "physical_scope":TRANSFER_LIMITATION}
    write_csv(out/"target_distribution_statistics.csv",target_rows)
    dump_json(out/"N7-B_magnitude_target_diagnostics.json",target_summary)
    dump_json(out/"split_manifest.json",split_manifest);dump_json(out/"run_config.json",run_config);dump_json(out/"data_validation.json",data_validation)
    if args.check_only:
        print(f"PASS --check-only: dataset, targets, N7-B mapping, features, frozen split, and output configuration verified. {out}")
        return
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device))
    if device.type=="cuda" and not torch.cuda.is_available():raise ValueError("CUDA requested but unavailable")
    checkpoint,metrics=train(bundle,args,out,device)
    data_validation["network_instantiated"]=True;data_validation["training_performed"]=True;dump_json(out/"data_validation.json",data_validation)
    run_config["completed_utc"]=datetime.now(timezone.utc).isoformat();run_config["device"]=str(device);run_config["best_epoch"]=checkpoint["best_epoch"];dump_json(out/"run_config.json",run_config)
    print(f"Complete: {out}")


if __name__=="__main__": main()
