"""Post-hoc paired evaluation for N7-B over 1..19 pressed sensors.

Uses only the frozen test predictions, the original simulation dataset and the
saved split.  It never loads a neural network, performs inference, or retrains.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, pickle
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
MAG_RUN=HERE/"E2_N7B_binary_intrinsic_magnitude_20261112T104654"
LOAD_RUN=HERE/"E2_N7K11_binary_continuous_20261111T171312"
DATASET=HERE/"E2_continuous_20k_20261111T150919"/"e2_continuous_load_20k.pkl"


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def j(x):
    if isinstance(x,dict):return {str(k):j(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [j(v) for v in x]
    if isinstance(x,Path):return str(x)
    if isinstance(x,np.ndarray):return j(x.tolist())
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def dump(path,value):path.write_text(json.dumps(j(value),indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")


def write_csv(path,rows):
    if not rows:raise ValueError(f"No rows for {path}")
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys:keys.append(key)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:j(r.get(k)) for k in keys} for r in rows])


def classification(truth,pred):
    t=np.asarray(truth,bool);p=np.asarray(pred,bool)
    tp=np.sum(t&p,axis=0);fp=np.sum(~t&p,axis=0);fn=np.sum(t&~p,axis=0)
    precision=tp/np.maximum(tp+fp,1);recall=tp/np.maximum(tp+fn,1);f1=2*tp/np.maximum(2*tp+fp+fn,1)
    supported=(tp+fn)>0
    return {"micro_f1":float(2*tp.sum()/max(2*tp.sum()+fp.sum()+fn.sum(),1)),
        "macro_f1_all_19_sensors":float(f1.mean()),
        "exact_match_accuracy":float(np.all(t==p,axis=1).mean()),
        "minimum_per_sensor_pressed_recall":float(recall[supported].min()) if supported.any() else None,
        "sensors_with_positive_recall_support":int(supported.sum()),
        "false_positive_entries":int(fp.sum()),"false_negative_entries":int(fn.sum())}


def errors(truth,pred,reference_magnitude=None):
    t=np.asarray(truth,float);p=np.asarray(pred,float);e=p-t
    row={"signed_delta_mae":float(np.mean(np.abs(e))),"signed_delta_rmse":float(np.sqrt(np.mean(e*e))),"signed_delta_bias":float(np.mean(e))}
    if reference_magnitude is not None:
        tm=reference_magnitude*(1+t);pm=reference_magnitude*(1+p);me=pm-tm
        floor=max(float(np.median(tm[tm>0]))*1e-12,np.finfo(float).tiny)
        row.update(reconstructed_magnitude_mae_ohm=float(np.mean(np.abs(me))),
            reconstructed_magnitude_rmse_ohm=float(np.sqrt(np.mean(me*me))),
            reconstructed_magnitude_bias_ohm=float(np.mean(me)),
            reconstructed_mean_absolute_relative_error=float(np.mean(np.abs(me)/np.maximum(np.abs(tm),floor))),
            invalid_reconstructed_entries=int(np.sum(1+p<0)),relative_error_floor_ohm=floor)
    return row


def load_inputs(args):
    mag_path=args.magnitude_run/"N7-B_magnitude_test_predictions.npz"
    load_path=args.baseline_run/"N7-B_test_predictions.npz"
    split_path=args.magnitude_run/"split_manifest.json"
    config_path=args.magnitude_run/"run_config.json"
    for p in (mag_path,load_path,split_path,config_path,args.dataset):
        if not p.is_file():raise FileNotFoundError(p)
    mag=np.load(mag_path);base=np.load(load_path)
    required_mag={"sample_ids","truth_binary","pressed_binary","truth_delta","predicted_delta","truth_load_kg"}
    required_base={"sample_ids","truth_binary","pressed_binary"}
    if not required_mag.issubset(mag.files):raise KeyError(f"Magnitude predictions missing {required_mag-set(mag.files)}")
    if not required_base.issubset(base.files):raise KeyError(f"Baseline predictions missing {required_base-set(base.files)}")
    if not np.array_equal(mag["sample_ids"],base["sample_ids"]) or not np.array_equal(mag["truth_binary"],base["truth_binary"]):raise ValueError("Models do not have identical paired test samples")
    with args.dataset.open("rb") as f:data=pickle.load(f)
    for key in ("Z_unit","Z_unit_0","y_cls","load_kg"):
        if key not in data:raise KeyError(f"Dataset missing {key}")
    ids=np.asarray(mag["sample_ids"],int);truth=np.asarray(mag["truth_binary"],np.uint8)
    delta_all=(np.abs(np.asarray(data["Z_unit"]))-np.abs(np.asarray(data["Z_unit_0"]))[None,:])/np.abs(np.asarray(data["Z_unit_0"]))[None,:]
    if not np.allclose(delta_all[ids],mag["truth_delta"],rtol=2e-7,atol=2e-7):raise ValueError("Saved truth_delta differs from source dataset")
    manifest=json.loads(split_path.read_text(encoding="utf-8-sig"));train_ids=np.asarray(manifest["train_ids"],int)
    if not np.array_equal(np.asarray(manifest["test_ids"],int),ids):raise ValueError("Prediction IDs differ from split manifest")
    if len(np.unique(np.concatenate((train_ids,ids))))!=len(train_ids)+len(ids):raise ValueError("Train/test overlap")
    return locals()


def constant_predictions(a):
    train=a["delta_all"][a["train_ids"]];train_cls=np.asarray(a["data"]["y_cls"])[a["train_ids"]].astype(bool)
    shape=a["mag"]["truth_delta"].shape;test_cls=a["truth"].astype(bool)
    global_median=float(np.median(train));sensor_median=np.median(train,axis=0)
    pressed_median=np.array([np.median(train[train_cls[:,s],s]) for s in range(19)])
    return {"zero_change":(np.zeros(shape),True),"global_training_median":(np.full(shape,global_median),True),
        "per_sensor_training_median":(np.broadcast_to(sensor_median,shape),True),
        "per_sensor_training_median_oracle_true_status":(np.where(test_cls,pressed_median[None,:],0.0),False)}


def savefig(fig,path):fig.tight_layout();fig.savefig(path.with_suffix(".png"),dpi=180);fig.savefig(path.with_suffix(".pdf"));plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--magnitude-run",type=Path,default=MAG_RUN);p.add_argument("--baseline-run",type=Path,default=LOAD_RUN);p.add_argument("--dataset",type=Path,default=DATASET);p.add_argument("--output-root",type=Path,default=HERE);args=p.parse_args()
    a=load_inputs(args);stamp=datetime.now().strftime("%Y%m%dT%H%M%S");out=args.output_root.resolve()/f"E2_N7B_full_1to19_analysis_{stamp}";out.mkdir(parents=False,exist_ok=False)
    truth=a["truth"];counts=truth.sum(1);delta=np.asarray(a["mag"]["truth_delta"],float);pred=np.asarray(a["mag"]["predicted_delta"],float);pressed=truth.astype(bool)
    ref=np.abs(np.asarray(a["data"]["Z_unit_0"])).reshape(1,19)
    constants=constant_predictions(a);classification_rows=[];magnitude_rows=[]
    model_predictions={"original_N7B_load_baseline":np.asarray(a["base"]["pressed_binary"],bool),"N7B_magnitude_model":np.asarray(a["mag"]["pressed_binary"],bool)}
    for count in range(1,20):
        sample_mask=counts==count;n=int(sample_mask.sum())
        if n<=0:raise ValueError(f"Test split has no samples with {count} pressed sensors")
        for model,pred_cls in model_predictions.items():classification_rows.append({"pressed_sensor_count":count,"sample_count":n,"model":model,**classification(truth[sample_mask],pred_cls[sample_mask])})
        entry_mask=pressed[sample_mask]
        truth_entries=delta[sample_mask][entry_mask];ref_entries=np.broadcast_to(ref,(n,19))[entry_mask]
        cnn=errors(truth_entries,pred[sample_mask][entry_mask],ref_entries);magnitude_rows.append({"pressed_sensor_count":count,"sample_count":n,"predictor":"N7B_magnitude_model","deployable":True,**cnn})
        for name,(constant,is_deployable) in constants.items():
            e=errors(truth_entries,constant[sample_mask][entry_mask],ref_entries)
            magnitude_rows.append({"pressed_sensor_count":count,"sample_count":n,"predictor":name,"deployable":is_deployable,**e,
                "cnn_improvement_percent_vs_predictor_mae":100*(e["signed_delta_mae"]-cnn["signed_delta_mae"])/e["signed_delta_mae"] if e["signed_delta_mae"] else None})
    write_csv(out/"classification_metrics_by_pressed_count_1to19.csv",classification_rows);write_csv(out/"magnitude_metrics_by_pressed_count_1to19.csv",magnitude_rows)
    # Full-range summaries are primary; 1-4 remains explicitly secondary.
    full_cls=[]
    for model,pred_cls in model_predictions.items():full_cls.append({"scope":"full_test_all_counts","model":model,"sample_count":len(truth),**classification(truth,pred_cls)})
    for model,pred_cls in model_predictions.items():
        mask=(counts>=1)&(counts<=4);full_cls.append({"scope":"secondary_1to4_subgroup","model":model,"sample_count":int(mask.sum()),**classification(truth[mask],pred_cls[mask])})
    write_csv(out/"classification_scope_summary.csv",full_cls)
    fig,ax=plt.subplots(2,2,figsize=(11,8),sharex=True)
    fields=(("micro_f1","Micro-F1"),("macro_f1_all_19_sensors","Macro-F1"),("exact_match_accuracy","Exact match"),("minimum_per_sensor_pressed_recall","Minimum supported-sensor recall"))
    for axis,(field,label) in zip(ax.ravel(),fields):
        for model in model_predictions:
            rows=[r for r in classification_rows if r["model"]==model];axis.plot([r["pressed_sensor_count"] for r in rows],[r[field] for r in rows],marker="o",label=model)
        axis.set_ylabel(label);axis.grid(alpha=.25);axis.set_xticks(range(1,20,2))
    ax[1,0].set_xlabel("Simultaneously pressed sensors");ax[1,1].set_xlabel("Simultaneously pressed sensors");ax[0,0].legend(fontsize=8);savefig(fig,out/"classification_performance_1to19")
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for predictor in ("N7B_magnitude_model","global_training_median","per_sensor_training_median","per_sensor_training_median_oracle_true_status"):
        rows=[r for r in magnitude_rows if r["predictor"]==predictor];ax[0].plot([r["pressed_sensor_count"] for r in rows],[r["signed_delta_mae"] for r in rows],marker="o",label=predictor);ax[1].plot([r["pressed_sensor_count"] for r in rows],[r["reconstructed_magnitude_mae_ohm"] for r in rows],marker="o",label=predictor)
    ax[0].set(ylabel="Signed-delta MAE",xlabel="Simultaneously pressed sensors");ax[1].set(ylabel="Reconstructed |Z| MAE (ohm)",xlabel="Simultaneously pressed sensors");
    for axis in ax:axis.grid(alpha=.25);axis.set_xticks(range(1,20,2));axis.legend(fontsize=7)
    savefig(fig,out/"magnitude_reconstruction_performance_1to19")
    worse=[]
    for count in range(1,20):
        cnn=next(r for r in magnitude_rows if r["pressed_sensor_count"]==count and r["predictor"]=="N7B_magnitude_model")
        med=next(r for r in magnitude_rows if r["pressed_sensor_count"]==count and r["predictor"]=="global_training_median")
        worse.append(cnn["signed_delta_mae"]>=med["signed_delta_mae"])
    decision={"scope":"full simulation test range, every count 1..19","primary_scope_is_1to19":True,"paired_test_samples":True,
        "counts_evaluated":list(range(1,20)),"magnitude_model_fails_to_beat_global_median_count_conditions":int(sum(worse)),
        "decision_rule":"GO requires stable full-range classification and useful sample-dependent magnitude reconstruction beyond deployable constants without widespread invalid magnitudes.",
        "decision":"CONDITIONAL GO for real-data domain-shift diagnostics only; quantitative real magnitude reconstruction is not validated.",
        "one_to_four_role":"secondary application-specific subgroup only"}
    dump(out/"full_range_decision.json",decision)
    provenance={"created_utc":datetime.now(timezone.utc).isoformat(),"no_training":True,"no_model_loading":True,"no_inference":True,
        "sources":{str(p):sha256(p) for p in (args.magnitude_run/"N7-B_magnitude_test_predictions.npz",args.baseline_run/"N7-B_test_predictions.npz",args.dataset,args.magnitude_run/"split_manifest.json")}}
    dump(out/"analysis_manifest.json",provenance)
    report=["# N7-B full 1–19 pressed-sensor evaluation","","Primary scope: every simulated pressed-count condition from 1 through 19.","","The 1–4 subset is secondary and is not used for model selection or the scientific decision.","",f"Decision: {decision['decision']}","",f"The CNN failed to improve on the global training median in {sum(worse)} of 19 pressed-count conditions (computed from frozen saved predictions).","","See the two per-count CSV tables and PNG/PDF figures for complete quantitative results."]
    (out/"full_range_evaluation_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(f"Complete saved-prediction analysis (no network inference): {out}")


if __name__=="__main__":main()
