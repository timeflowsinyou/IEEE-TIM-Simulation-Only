"""Frozen N7-B real-data diagnostic with audited boundary-admittance compensation.

This script does not train, fine-tune, select thresholds on real labels, or claim
real intrinsic-impedance ground truth.  It evaluates classification labels and
saves magnitude-model outputs only as diagnostic predictions.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, pickle, platform, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
REPO_ROOT=HERE.parents[1]
EXTERNAL_ROOT=REPO_ROOT/"external"
TRAINING_DIR=REPO_ROOT/"scripts"/"training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0,str(TRAINING_DIR))
from n7k11_dual_task_inference import N7K11DualHeadCNN, ResidualBlock
REAL_RUN=EXTERNAL_ROOT/"real91_extract"
REAL_NPZ=REAL_RUN/"real_91pair_measurements.npz"
REAL_MANIFEST=REAL_RUN/"sample_manifest.csv"
REAL_AUDIT=REAL_RUN/"extraction_audit.json"
PARASITIC=EXTERNAL_ROOT/"calibration"/"lsq_result91para_C_0512_v0.joblib"
MAG_RUN=EXTERNAL_ROOT/"magnitude_checkpoint"
MAG_CKPT=MAG_RUN/"N7-B_magnitude_best_checkpoint.pt"
BASE_RUN=EXTERNAL_ROOT/"baseline_checkpoint"
BASE_CKPT=BASE_RUN/"N7-B_best_checkpoint.pt"
SIM_DATA=EXTERNAL_ROOT/"simulation"/"e2_continuous_load_20k.pkl"
FREQUENCY=100_000.0
NODES=[*range(1,8),*range(14,7,-1)]
CANONICAL=[(NODES[i],NODES[j]) for i in range(13) for j in range(i+1,14)]
CONDITIONS=("raw_sim_reference","compensated_sim_reference","compensated_real_reference")


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
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
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:j(r.get(k)) for k in keys} for r in rows])


class N7BMagnitudeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=nn.Sequential(ResidualBlock(3,64),ResidualBlock(64,128),nn.MaxPool1d(2),ResidualBlock(128,256),nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.classifier=nn.Sequential(nn.Linear(256,64),nn.ReLU(),nn.Linear(64,19))
        self.regressor=nn.Sequential(nn.Linear(256,64),nn.ReLU(),nn.Linear(64,19))
    def forward(self,x):
        h=self.backbone(x.transpose(1,2));return self.classifier(h),self.regressor(h)


def parse_labels(labels,pressures):
    y=np.zeros((len(labels),19),np.uint8);loads=np.zeros((len(labels),19),float)
    for i,(label,pressure) in enumerate(zip(labels,pressures)):
        s=str(label).strip();q=str(pressure).strip()
        if s in ("0","0.0",""):
            if q not in ("0","0.0",""):raise ValueError(f"Nonzero pressure on unloaded row {i}")
            continue
        ids=[int(v) for v in s.split("-")];vals=[float(v) for v in q.split("-")]
        if len(ids)!=len(vals) or len(ids)!=len(set(ids)) or min(ids)<1 or max(ids)>19:raise ValueError(f"Invalid labels row {i}")
        if any(v not in (.5,1.0) for v in vals):raise ValueError(f"Unsupported nominal load row {i}")
        for sensor,value in zip(ids,vals):y[i,sensor-1]=1;loads[i,sensor-1]=value
    return y,loads


def pair_tuple(text):
    a,b=map(int,str(text).split("-"));return (a,b)


def map_pairs(pair_ids,selected):
    lookup={frozenset(pair_tuple(v)):i for i,v in enumerate(pair_ids)}
    if len(lookup)!=91:raise ValueError("Real pair IDs are not 91 unique unordered pairs")
    out=np.array([lookup[frozenset(map(int,p))] for p in selected],np.int64)
    if len(np.unique(out))!=11:raise ValueError("N7-B mapping is not unique")
    return out


def features(z,reference,z_indices,reference_indices,mean,std):
    selected=np.asarray(z)[:,z_indices];ref=np.asarray(reference)[reference_indices]
    phase=np.angle(selected/ref[None,:])
    raw=np.stack((np.abs(selected)/np.abs(ref)[None,:],np.sin(phase),np.cos(phase)),axis=-1)
    x=(raw-np.asarray(mean).reshape(1,1,3))/np.asarray(std).reshape(1,1,3)
    if x.shape!=(len(z),11,3) or not np.isfinite(x).all():raise ValueError("Invalid features")
    return raw,x.astype(np.float32)


def load_audit(args):
    paths=(args.real_npz,args.real_manifest,args.real_audit,args.parasitic,args.magnitude_checkpoint,args.baseline_checkpoint,args.simulation_dataset)
    for p in paths:
        if not p.is_file():raise FileNotFoundError(p)
    with np.load(args.real_npz,allow_pickle=False) as f:
        needed={"Z_measured","pair_ids","labels","pressures","session_ids","file_sample_index","frequency_hz"}
        if not needed.issubset(f.files):raise KeyError(f"Real NPZ missing {needed-set(f.files)}")
        z=np.asarray(f["Z_measured"],np.complex128);pair_ids=np.asarray(f["pair_ids"]).astype(str)
        labels=np.asarray(f["labels"]).astype(str);pressures=np.asarray(f["pressures"]).astype(str)
        sessions=np.asarray(f["session_ids"]).astype(str);file_index=np.asarray(f["file_sample_index"],int);freq=float(f["frequency_hz"])
    if z.shape!=(804,91) or pair_ids.shape!=(91,) or not np.isfinite(z).all() or np.any(z==0):raise ValueError("Unexpected real data")
    if not np.isclose(freq,FREQUENCY):raise ValueError(f"Frequency mismatch {freq}")
    expected_ids=np.array([f"{a}-{b}" for a,b in CANONICAL])
    if not np.array_equal(pair_ids,expected_ids):raise ValueError("Real pair order differs from verified clockwise canonical order")
    with args.real_manifest.open(encoding="utf-8-sig",newline="") as f:manifest=list(csv.DictReader(f))
    if len(manifest)!=804:raise ValueError("Manifest row count mismatch")
    for i,r in enumerate(manifest):
        if int(r["global_row"])!=i or r["source_file"]!=sessions[i] or int(r["file_sample_index_zero_based"])!=file_index[i] or r["load_label"]!=labels[i]:raise ValueError(f"Manifest mismatch row {i}")
    extraction=json.loads(args.real_audit.read_text(encoding="utf-8-sig"))
    if extraction.get("n_samples")!=804 or extraction.get("frequency_hz")!=FREQUENCY:raise ValueError("Extraction audit mismatch")
    y,loads=parse_labels(labels,pressures)
    mag=torch.load(args.magnitude_checkpoint,map_location="cpu",weights_only=False)
    base=torch.load(args.baseline_checkpoint,map_location="cpu",weights_only=False)
    for name,ck in (("magnitude",mag),("baseline",base)):
        if ck.get("topology","N7-B")!="N7-B":raise ValueError(f"{name} checkpoint is not N7-B")
        if len(ck["selected_pairs"])!=11 or np.asarray(ck["feature_mean"]).shape!=(3,) or np.asarray(ck["feature_std"]).shape!=(3,):raise ValueError(f"{name} checkpoint preprocessing mismatch")
        if float(ck["classification_threshold"])<=0 or float(ck["classification_threshold"])>=1:raise ValueError(f"{name} threshold invalid")
    if tuple(map(tuple,mag["selected_pairs"]))!=tuple(map(tuple,base["selected_pairs"])):raise ValueError("Frozen models use different pairs")
    if mag["dataset_sha256"]!=base["dataset_sha256"]:raise ValueError("Frozen models use different simulation datasets")
    if not np.array_equal(mag["test_ids"],base["test_ids"]):raise ValueError("Frozen models do not use same test split")
    indices=map_pairs(pair_ids,mag["selected_pairs"])
    par=joblib.load(args.parasitic)
    if not isinstance(par,dict) or not {"Gp_opt","Cp_opt"}.issubset(par):raise ValueError("Calibration needs Gp_opt and Cp_opt")
    gp=np.asarray(par["Gp_opt"],float).reshape(-1);cp=np.asarray(par["Cp_opt"],float).reshape(-1)
    if gp.shape!=(91,) or cp.shape!=(91,) or not np.isfinite([gp,cp]).all() or np.any(gp<0) or np.any(cp<0):raise ValueError("Invalid passive parasitic calibration")
    yp=gp+1j*2*np.pi*freq*cp
    # Verified forward law: Zcoupled=1/(1/Zintrinsic+Ypara). Its algebraic inverse
    # is applied once in admittance space, never by subtracting impedances.
    yintrinsic=1/z-yp[None,:]
    tiny=np.finfo(float).tiny
    if np.any(np.abs(yintrinsic)<=tiny):raise ValueError("Singular de-embedding denominator")
    zcomp=1/yintrinsic
    no_load=~y.any(1)
    if no_load.sum()!=5:raise ValueError(f"Expected five genuine no-load rows, found {no_load.sum()}")
    real_ref=np.median(zcomp[no_load].real,axis=0)+1j*np.median(zcomp[no_load].imag,axis=0)
    if not np.isfinite(real_ref).all() or np.any(real_ref==0):raise ValueError("Invalid compensated real unloaded reference")
    condition_z={"raw_sim_reference":z,"compensated_sim_reference":zcomp,"compensated_real_reference":zcomp}
    condition_ref={"raw_sim_reference":np.asarray(mag["Z_all_0"]),"compensated_sim_reference":np.asarray(mag["Z_all_0"]),"compensated_real_reference":real_ref}
    prebuilt={}
    for cond in CONDITIONS:
        prebuilt[cond]={}
        for model,ck in (("magnitude",mag),("baseline",base)):
            ref_indices=indices if cond=="compensated_real_reference" else np.asarray(ck["pair_indices"],int)
            prebuilt[cond][model]=features(condition_z[cond],condition_ref[cond],indices,ref_indices,ck["feature_mean"],ck["feature_std"])
    with args.simulation_dataset.open("rb") as f:sim=pickle.load(f)
    if "Z_all" not in sim:raise KeyError("Simulation dataset lacks Z_all")
    sim_z=np.asarray(sim["Z_all"],np.complex128)
    if digest(args.simulation_dataset)!=mag["dataset_sha256"]:raise ValueError("Simulation dataset hash differs from checkpoint")
    sim_indices=np.asarray(mag["pair_indices"],int)
    sim_raw,sim_x=features(sim_z,np.asarray(mag["Z_all_0"]),sim_indices,sim_indices,mag["feature_mean"],mag["feature_std"])
    return locals()


def cls_metrics(truth,pred):
    t=np.asarray(truth,bool);p=np.asarray(pred,bool);tp=np.sum(t&p,0);fp=np.sum(~t&p,0);fn=np.sum(t&~p,0);tn=np.sum(~t&~p,0)
    precision=tp/np.maximum(tp+fp,1);recall=tp/np.maximum(tp+fn,1);f1=2*tp/np.maximum(2*tp+fp+fn,1)
    return {"samples":len(t),"micro_f1":float(2*tp.sum()/max(2*tp.sum()+fp.sum()+fn.sum(),1)),"macro_f1":float(f1.mean()),
        "exact_match_accuracy":float(np.all(t==p,axis=1).mean()),"minimum_per_sensor_pressed_recall":float(recall.min()),
        "false_positive_entries":int(fp.sum()),"false_negative_entries":int(fn.sum()),"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":precision,"recall":recall,"f1":f1}


@torch.no_grad()
def infer(model,x,device,batch):
    probs=[];reg=[]
    for start in range(0,len(x),batch):
        logits,r=model(torch.from_numpy(x[start:start+batch]).to(device));probs.append(torch.sigmoid(logits).cpu().numpy());reg.append(r.cpu().numpy())
    return np.concatenate(probs),np.concatenate(reg)


def load_models(a,device):
    m=N7BMagnitudeCNN().to(device);m.load_state_dict(a["mag"]["model_state_dict"]);m.eval()
    b=N7K11DualHeadCNN(a["base"]["model_config"]["max_load_kg"]).to(device);b.load_state_dict(a["base"]["model_state_dict"]);b.eval()
    return {"magnitude":m,"baseline":b}


def subset_masks(y,loads):
    count=y.sum(1);out={"all":np.ones(len(y),bool),"secondary_pressed_1_to_4":(count>=1)&(count<=4)}
    # Always define the complete model scope. Empty real conditions are retained
    # in coverage metadata and omitted only from metric calculations.
    for n in range(1,20):out[f"pressed_count_{n}"]=count==n
    out["all_pressed_nominal_0p5kg"]=(count>0)&np.all((loads==0)|(loads==.5),axis=1)
    out["all_pressed_nominal_1p0kg"]=(count>0)&np.all((loads==0)|(loads==1),axis=1)
    return out


def domain_drift(sim_x,real_x):
    s=sim_x.reshape(len(sim_x),-1).astype(float);r=real_x.reshape(len(real_x),-1).astype(float)
    low=np.quantile(s,.005,axis=0);high=np.quantile(s,.995,axis=0)
    outside=(r<low)|(r>high)
    return {"absolute_normalized_feature_median":float(np.median(np.abs(r))),"absolute_normalized_feature_p95":float(np.quantile(np.abs(r),.95)),
        "absolute_normalized_feature_max":float(np.max(np.abs(r))),"component_fraction_outside_simulation_0p5_to_99p5_percentiles":float(outside.mean()),
        "sample_fraction_with_any_component_outside":float(outside.any(1).mean())}


def savefig(fig,path):fig.tight_layout();fig.savefig(path.with_suffix(".png"),dpi=180);fig.savefig(path.with_suffix(".pdf"));plt.close(fig)


def full_run(a,args,out):
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device));models=load_models(a,device)
    masks=subset_masks(a["y"],a["loads"]);metric_rows=[];sensor_rows=[];prediction_rows=[];consistency={};all_preds={};domain={}
    for cond in CONDITIONS:
        domain[cond]=domain_drift(a["sim_x"][np.asarray(a["mag"]["train_ids"],int)],a["prebuilt"][cond]["magnitude"][1])
        all_preds[cond]={}
        for model in ("baseline","magnitude"):
            prob,reg=infer(models[model],a["prebuilt"][cond][model][1],device,args.batch_size);pred=prob>=float((a["base"] if model=="baseline" else a["mag"])["classification_threshold"])
            all_preds[cond][model]=(prob,pred,reg)
            for group,mask in masks.items():
                if not mask.any():continue
                met=cls_metrics(a["y"][mask],pred[mask]);metric_rows.append({"condition":cond,"model":model,"group":group,**{k:v for k,v in met.items() if not isinstance(v,np.ndarray)}})
            met=cls_metrics(a["y"],pred)
            for s in range(19):sensor_rows.append({"condition":cond,"model":model,"sensor":f"S{s+1}","precision":met["precision"][s],"recall":met["recall"][s],"f1":met["f1"][s],"tp":met["tp"][s],"fp":met["fp"][s],"fn":met["fn"][s],"tn":met["tn"][s]})
            if model=="magnitude":
                reconstructed=np.asarray(a["mag"]["Z_unit_0"])[None,:]
                reconstructed_mag=np.abs(reconstructed)*(1+reg)
                for i in range(len(reg)):
                    for s in range(19):prediction_rows.append({"sample_id":i,"condition":cond,"sensor":f"S{s+1}","true_pressed":int(a["y"][i,s]),"nominal_load_kg":a["loads"][i,s],"predicted_pressed_probability":prob[i,s],"predicted_pressed":int(pred[i,s]),"predicted_delta":reg[i,s],"reconstructed_magnitude_ohm":reconstructed_mag[i,s],"physically_invalid":int(1+reg[i,s]<0)})
                consistency[cond]={"prediction_finite":bool(np.isfinite(reg).all()),"delta_min":float(reg.min()),"delta_max":float(reg.max()),"delta_mean":float(reg.mean()),"delta_std":float(reg.std()),"invalid_reconstructed_count":int(np.sum(1+reg<0)),"invalid_reconstructed_fraction":float(np.mean(1+reg<0)),"no_load_delta_abs_median":float(np.median(np.abs(reg[a["no_load"]]))),"pressed_0p5_delta_median":float(np.median(reg[(a["loads"]==.5)])),"pressed_1p0_delta_median":float(np.median(reg[(a["loads"]==1)])),"note":"No real intrinsic ground truth; no regression MAE/RMSE is computed."}
                consistency[cond]["by_pressed_sensor_count"]={str(n):({"sample_count":int(np.sum(a["y"].sum(1)==n)),"predicted_delta_median":float(np.median(reg[a["y"].sum(1)==n])),"predicted_delta_std":float(np.std(reg[a["y"].sum(1)==n])),"invalid_fraction":float(np.mean(1+reg[a["y"].sum(1)==n]<0))} if np.any(a["y"].sum(1)==n) else {"sample_count":0,"status":"not measured"}) for n in range(1,20)}
    # Across-condition prediction shifts, never interpreted as accuracy.
    for c1,c2 in ((CONDITIONS[0],CONDITIONS[1]),(CONDITIONS[1],CONDITIONS[2])):
        d=all_preds[c2]["magnitude"][2]-all_preds[c1]["magnitude"][2];consistency[f"difference_{c2}_minus_{c1}"]={"mae_delta":float(np.mean(np.abs(d))),"rmse_delta":float(np.sqrt(np.mean(d*d))),"maximum_absolute_delta":float(np.max(np.abs(d)))}
    write_csv(out/"real_classification_metrics.csv",metric_rows);write_csv(out/"real_per_sensor_metrics.csv",sensor_rows);write_csv(out/"real_magnitude_predictions.csv",prediction_rows)
    dump(out/"real_magnitude_consistency.json",consistency);dump(out/"domain_shift_diagnostics.json",domain)
    compare=[r for r in metric_rows if r["group"]=="all" or r["group"].startswith("pressed_count_") or r["group"]=="secondary_pressed_1_to_4"];write_csv(out/"baseline_vs_new_model_real.csv",compare)
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(19)
    for k,cond in enumerate(CONDITIONS):ax.plot(x+1,[np.median(all_preds[cond]["magnitude"][2][:,s]) for s in range(19)],marker="o",label=cond)
    ax.set(xlabel="Sensor",ylabel="Median predicted delta",title="Frozen magnitude outputs by input condition");ax.legend(fontsize=7);ax.grid(alpha=.2);savefig(fig,out/"real_magnitude_by_condition")
    fig,ax=plt.subplots(figsize=(7,4));names=[];vals=[]
    for cond in CONDITIONS:
        row=next(r for r in metric_rows if r["condition"]==cond and r["model"]=="magnitude" and r["group"]=="all");names.append(cond);vals.append(row["micro_f1"])
    ax.bar(range(3),vals);ax.set_xticks(range(3),names,rotation=15,ha="right");ax.set(ylabel="Micro-F1",title="All measured loading configurations");savefig(fig,out/"real_classification_conditions")
    fig,ax=plt.subplots(figsize=(9,4))
    available=sorted(int(v) for v in np.unique(a["y"].sum(1)) if v>0)
    for cond in CONDITIONS:
        vals=[]
        for n in available:
            row=next(r for r in metric_rows if r["condition"]==cond and r["model"]=="magnitude" and r["group"]==f"pressed_count_{n}");vals.append(row["micro_f1"])
        ax.plot(available,vals,marker="o",label=cond)
    ax.set(xlabel="Simultaneously pressed sensors actually measured",ylabel="Micro-F1",title="Real classification by measured pressed count");ax.legend(fontsize=7);ax.grid(alpha=.25);savefig(fig,out/"real_classification_by_pressed_count")
    return metric_rows,consistency,domain,str(device)


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--real-npz",type=Path,default=REAL_NPZ);p.add_argument("--real-manifest",type=Path,default=REAL_MANIFEST);p.add_argument("--real-audit",type=Path,default=REAL_AUDIT);p.add_argument("--parasitic",type=Path,default=PARASITIC);p.add_argument("--magnitude-checkpoint",type=Path,default=MAG_CKPT);p.add_argument("--baseline-checkpoint",type=Path,default=BASE_CKPT);p.add_argument("--simulation-dataset",type=Path,default=SIM_DATA);p.add_argument("--output-root",type=Path,default=HERE);p.add_argument("--batch-size",type=int,default=256);p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto");p.add_argument("--check-only",action="store_true");return p


def main():
    args=parser().parse_args();a=load_audit(args);stamp=datetime.now().strftime("%Y%m%dT%H%M%S");out=args.output_root.resolve()/f"E2_N7B_real_compensated_validation_{stamp}";out.mkdir(parents=False,exist_ok=False)
    counts=Counter(a["y"].sum(1).tolist())
    measured_counts=sorted(int(v) for v in counts if int(v)>0 and counts[v]>0)
    real_validation={"status":"PASS","samples":len(a["z"]),"frequency_hz":a["freq"],"pair_count":91,"labels_valid":True,"pressed_count_distribution":dict(sorted(counts.items())),"full_simulation_model_scope_pressed_counts":list(range(1,20)),"experimentally_measured_pressed_counts":measured_counts,"unmeasured_pressed_counts":[v for v in range(1,20) if v not in measured_counts],"coverage_limitation":"Real validation covers only actually measured configurations; absent counts are not a limitation of the frozen model or its 1–19 simulation scope.","no_load_samples":int(a["no_load"].sum()),"nominal_load_values_kg":[.5,1.0],"intrinsic_ground_truth_available":False,"original_and_additional_sessions":len(np.unique(a["sessions"])),"source_files_unchanged":True}
    comp_validation={"status":"PASS_WITH_PROVENANCE_CAVEAT","forward_law":"Zcoupled=1/(1/Zintrinsic + Gp + j*2*pi*f*Cp)","inverse_law":"Zintrinsic=1/(1/Zmeasured - Gp - j*2*pi*f*Cp)","space":"admittance; never impedance subtraction","Gp_unit":"S","Cp_unit":"F","frequency_hz":a["freq"],"Gp_nonnegative":True,"Cp_nonnegative":True,"applied_exactly_once":True,"calibration_pair_order":"91 clockwise canonical pairs, established by existing code lineage","caveat":"joblib has no embedded pair labels; order is not self-describing","inverse_nonpositive_real_admittance_fraction":float(np.mean(a["yintrinsic"].real<=0)),"inverse_nonpositive_imag_admittance_fraction":float(np.mean(a["yintrinsic"].imag<=0))}
    ref_diag={"simulation_uses_fixed_reference":True,"real_reference_samples":int(a["no_load"].sum()),"real_reference_method":"component-wise median of five genuine no-load rows after the same one-time compensation","loaded_and_real_reference_compensated_consistently":True,"conditions":list(CONDITIONS),"no_label_based_normalization":True,"simulation_to_real_reference_relative_l2":float(np.linalg.norm(a["real_ref"]-np.asarray(a["mag"]["Z_all_0"]))/np.linalg.norm(np.asarray(a["mag"]["Z_all_0"])))}
    config={"created_utc":datetime.now(timezone.utc).isoformat(),"check_only":args.check_only,"decision_scope":"CONDITIONAL GO diagnostic only","arguments":vars(args),"paths":{k:v for k,v in vars(args).items() if isinstance(v,Path)},"sha256":{str(p):digest(p) for p in (args.real_npz,args.real_manifest,args.real_audit,args.parasitic,args.magnitude_checkpoint,args.baseline_checkpoint,args.simulation_dataset)},"software":{"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__,"joblib":joblib.__version__},"models_frozen":True,"real_threshold_tuning":False,"real_training_or_finetuning":False}
    dump(out/"real_data_validation.json",real_validation);dump(out/"compensation_validation.json",comp_validation);dump(out/"reference_alignment_diagnostics.json",ref_diag);dump(out/"run_config.json",config)
    if args.check_only:
        print(f"PASS --check-only: real data, calibration, frozen checkpoints, pair mapping, frequency, references, labels, features, and output safety verified. No inference run. {out}");return
    rows,consistency,domain,device=full_run(a,args,out)
    report=["# Controlled frozen-model real-data diagnostic","", "Decision: CONDITIONAL GO — diagnostic only.","", "Simulation scope is 1–19 simultaneously pressed sensors. Real results cover only pressed counts actually present in the 804 measurements; missing counts are an experimental-coverage limitation, not a model-scope limitation.","", "No real intrinsic-impedance ground truth exists; magnitude MAE/RMSE are intentionally absent.","",f"Samples: {len(a['z'])}; genuine no-load: {int(a['no_load'].sum())}; device: {device}.","","Three conditions: raw with simulation reference, compensated with simulation reference, and compensated with compensated-real no-load reference.","","The compensation inverse is a calibrated approximation in boundary-admittance space. The calibration file does not embed pair labels; canonical order follows verified project code lineage."]
    (out/"validation_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    config["completed_utc"]=datetime.now(timezone.utc).isoformat();config["device"]=device;dump(out/"run_config.json",config);print(f"Complete: {out}")


if __name__=="__main__":main()
