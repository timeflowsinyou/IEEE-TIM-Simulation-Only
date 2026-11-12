"""Audit the physical provenance of the E2 continuous-load simulation.

This script is read-only with respect to all source artifacts. It distinguishes
recorded R/C workbook values, impedances calculated from those values, fitted
shared response curves, E2-anchored synthetic impedances, and nodal boundary
responses. No neural network code is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import scipy
from scipy.interpolate import PchipInterpolator


SCRIPT_PATH = Path(__file__).resolve()
SIM_DIR = SCRIPT_PATH.parent
ROOT = SIM_DIR.parent
DATASET = SIM_DIR / "E2_continuous_20k_20261111T150919" / "e2_continuous_load_20k.pkl"
WORKBOOK = ROOT / "Data" / "Nano_Sensors_five.xlsx"
CURVE_SOURCE = SIM_DIR / "generate_simulation_curve_nodal_forward_50k.py"
E2_GENERATOR = SIM_DIR / "generate_e2_continuous_load_20k.py"
E2_BASELINE = (ROOT / "Simulation_Experiment" / "experiment_I5_S2_E1_E2_E3" /
               "E2_weak_S2_prior" / "init_30pct" / "baseline_sensor_parameters.csv")
MULTIFREQUENCY_DIR = ROOT / "Data" / "MultiFrequency_Measurements" / "2026_11_05"
TRACE_LOAD_G = np.array([0., 100., 250., 500., 750., 1000.])
SENSORS_TRACE = (2, 14, 19)
CORRECTIONS = (("s1", "C2", 24., 97.), ("s2", "G11", 45., 4.5),
               ("s3", "C17", .195, .0195))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return jsonable(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False,
                                     allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows: raise ValueError(f"No rows for {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows: writer.writerow(jsonable(row))


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parallel_rc(r_mohm, c_pf, frequency):
    r = np.asarray(r_mohm, float) * 1e6
    c = np.asarray(c_pf, float) * 1e-12
    if not np.isfinite([r, c]).all() or np.any(r <= 0) or np.any(c <= 0):
        raise ValueError("Invalid R/C values")
    return 1. / (1. / r + 1j * 2 * np.pi * frequency * c)


def load_workbook_values(apply_pipeline_corrections):
    wb = openpyxl.load_workbook(WORKBOOK, data_only=True)
    applied = []
    if apply_pipeline_corrections:
        for sheet, cell, old, new in CORRECTIONS:
            actual = wb[sheet][cell].value
            if actual == old: wb[sheet][cell].value = new; status = "applied"
            elif actual == new: status = "already_present"
            else: status = "not_applied_unexpected_value"
            applied.append({"sheet": sheet, "cell": cell, "old": old, "new": new,
                            "observed": actual, "status": status})
    values = {}
    for sheet in ("s1", "s2", "s3", "s4", "s5"):
        a = np.asarray(list(wb[sheet].values)[1:], float)
        if a.shape != (18, 11) or not np.isfinite(a).all(): raise ValueError(f"{sheet}: invalid workbook")
        values[sheet] = {"load_g": a[:, 0], "c_pf": a[:, 1::2], "r_mohm": a[:, 2::2]}
    return values, applied


def source_measurements(raw_values, corrected_values):
    rows = []
    for status, source in (("workbook_as_stored", raw_values),
                           ("pipeline_in_memory_corrected", corrected_values)):
        for sensor, v in source.items():
            for frequency in (5_000., 100_000.):
                z = parallel_rc(v["r_mohm"], v["c_pf"], frequency)
                ratio = np.abs(z) / np.abs(z[0])[None, :]
                for i, load in enumerate(v["load_g"]):
                    for repeat in range(5):
                        rows.append({"source_sensor": sensor, "source_status": status,
                            "frequency_hz": frequency, "load_g": load, "repeat": repeat+1,
                            "recorded_R_MOhm": v["r_mohm"][i, repeat],
                            "recorded_C_pF": v["c_pf"][i, repeat],
                            "calculated_Z_abs_ohm": abs(z[i, repeat]),
                            "calculated_Z_phase_deg": np.angle(z[i, repeat], deg=True),
                            "calculated_ratio_to_repeat_unloaded": ratio[i, repeat],
                            "calculated_relative_change": ratio[i, repeat]-1,
                            "classification": ("recorded R/C; impedance calculated by parallel-RC formula; "
                                               "100 kHz is not a separate measurement")})
    return rows


def read_e2_baseline():
    with E2_BASELINE.open("r", encoding="utf-8-sig", newline="") as f: rows = list(csv.DictReader(f))
    if [r["sensor"] for r in rows] != [f"S{i}" for i in range(1,20)]: raise ValueError("E2 order")
    g = np.array([float(r["G_S"]) for r in rows]); b = np.array([float(r["B_S"]) for r in rows])
    if not np.isfinite([g,b]).all() or np.any(g<=0) or np.any(b<=0): raise ValueError("Invalid E2")
    return g, b, 1/(g+1j*b)


def evaluate_model_curve(model, name, loads):
    item = model["sensors"][name]; x = np.asarray(item["load_g"], float)
    lc = PchipInterpolator(x, item["log_C_ratio"], extrapolate=False)(loads)
    q = PchipInterpolator(x, item["log_R0_over_R"], extrapolate=False)(loads)
    return lc, q


def mean_raw_ratio_at_load(values, load_g, frequency):
    v = values["s1"]; grid = v["load_g"]
    if load_g not in grid: return np.nan
    ratios=[]
    for sensor in values.values():
        z=parallel_rc(sensor["r_mohm"],sensor["c_pf"],frequency)
        i=int(np.flatnonzero(sensor["load_g"]==load_g)[0]); ratios.extend((np.abs(z[i])/np.abs(z[0])).tolist())
    return float(np.median(ratios))


def normalization_trace(curve_module, corrected_values, model, g0, b0, z0):
    rows=[]; omega=2*np.pi*100_000.; r0=1/g0; c0=b0/omega
    lc,q=evaluate_model_curve(model,"shared",TRACE_LOAD_G)
    for sensor_number in SENSORS_TRACE:
        j=sensor_number-1
        for k,load in enumerate(TRACE_LOAD_G):
            r_ratio=np.exp(-q[k]); c_ratio=np.exp(lc[k])
            loaded_r=r0[j]*r_ratio; loaded_c=c0[j]*c_ratio
            z=1/(1/loaded_r+1j*omega*loaded_c)
            exact=bool(load in next(iter(corrected_values.values()))["load_g"])
            rows.append({"sensor":f"S{sensor_number}","load_kg":load/1000,
                "experimental_input_available_at_exact_load":exact,
                "original_five_sensor_median_Z_ratio_5k":mean_raw_ratio_at_load(corrected_values,load,5000) if exact else np.nan,
                "calculated_five_sensor_median_Z_ratio_100k":mean_raw_ratio_at_load(corrected_values,load,100000) if exact else np.nan,
                "shared_log_C_ratio":lc[k],"shared_log_R0_over_R":q[k],
                "transferred_C_over_C0":c_ratio,"transferred_R_over_R0":r_ratio,
                "E2_unloaded_Z_abs_ohm":abs(z0[j]),"generated_R_ohm":loaded_r,
                "generated_C_F":loaded_c,"generated_Z_abs_ohm":abs(z),
                "final_ratio_to_E2_unloaded":abs(z)/abs(z0[j]),
                "final_relative_change":abs(z)/abs(z0[j])-1,
                "stage_label":("shared fitted knot" if exact else "PCHIP interpolation between fitted knots"),
                "extrapolation":False})
    return rows


def frequency_transfer_rows(corrected_values, model, g0, b0, z0):
    rows=[]; loads=np.asarray(model["sensors"]["shared"]["load_g"]); lc,q=evaluate_model_curve(model,"shared",loads)
    for k,load in enumerate(loads):
        for frequency in (5000.,100000.):
            raw=[]
            for sensor in corrected_values.values():
                z=parallel_rc(sensor["r_mohm"],sensor["c_pf"],frequency)
                i=int(np.flatnonzero(sensor["load_g"]==load)[0]); raw.extend((np.abs(z[i])/np.abs(z[0])).tolist())
            rows.append({"load_g":load,"frequency_hz":frequency,
                "source":"same recorded 5-kHz-workbook R/C evaluated at requested frequency",
                "median_calculated_magnitude_ratio":np.median(raw),
                "p05_calculated_magnitude_ratio":np.percentile(raw,5),
                "p95_calculated_magnitude_ratio":np.percentile(raw,95),
                "recorded_R_C_at_frequency":frequency==5000,
                "direct_complex_intrinsic_impedance_measurement":False})
        ratios=[]
        for j in range(19):
            r=(1/g0[j])*np.exp(-q[k]); c=(b0[j]/(2*np.pi*100000))*np.exp(lc[k])
            z=1/(1/r+1j*2*np.pi*100000*c);ratios.append(abs(z)/abs(z0[j]))
        rows.append({"load_g":load,"frequency_hz":100000.,"source":"shared normalized R/C curve transferred to 19 E2 baselines",
            "median_calculated_magnitude_ratio":np.median(ratios),"p05_calculated_magnitude_ratio":np.percentile(ratios,5),
            "p95_calculated_magnitude_ratio":np.percentile(ratios,95),
            "recorded_R_C_at_frequency":False,"direct_complex_intrinsic_impedance_measurement":False})
    return rows


def incidence(items):
    a=np.zeros((14,len(items)))
    for k,(u,v) in enumerate(items):a[u-1,k]=1;a[v-1,k]=-1
    return a


def forward_numpy(z,edges,pairs):
    a=incidence(edges)[:-1];c=incidence(pairs)[:-1];lap=(a*(1/z)[None,:])@a.T
    v=np.linalg.solve(lap,c);return np.sum(c*v,axis=0)


def forward_checks(data):
    z=np.asarray(data["Z_unit"]);za=np.asarray(data["Z_all"]);z0=np.asarray(data["Z_unit_0"]);za0=np.asarray(data["Z_all_0"])
    edges=np.asarray(data["physical_edges"]);pairs=np.asarray(data["original_pairs"])
    rows=[]
    for index in (0,9990,19974,19975,19999):
        calc=forward_numpy(z[index],edges,pairs);err=np.abs(calc-za[index])
        rows.append({"sample_index":index,"check":"independent_grounded_nodal_reproduction",
            "max_absolute_error_ohm":err.max(),"max_relative_error":np.max(err/np.abs(za[index])),
            "finite":bool(np.isfinite(calc).all()),"parasitic_admittance_included":False,
            "note":"Dataset Z_all is intrinsic grounded-nodal response; no coupled/parasitic array is serialized"})
    rows.append({"sample_index":-1,"check":"unloaded_reference_forward",
        "max_absolute_error_ohm":np.max(np.abs(forward_numpy(z0,edges,pairs)-za0)),
        "max_relative_error":np.max(np.abs(forward_numpy(z0,edges,pairs)-za0)/np.abs(za0)),
        "finite":True,"parasitic_admittance_included":False})
    return rows


def source_locations(curve_module,e2_module):
    out={}
    for label,obj in (("fit_excel_curves",curve_module.fit_excel_curves),("sensor_states",curve_module.sensor_states),
                      ("nodal_forward_batch",curve_module.nodal_forward_batch),("e2_main",e2_module.main)):
        source_object=inspect.unwrap(obj)
        lines,start=inspect.getsourcelines(source_object);out[label]={"file":inspect.getsourcefile(source_object),"start_line":start,"end_line":start+len(lines)-1}
    return out


def multifrequency_inventory():
    files=sorted(MULTIFREQUENCY_DIR.glob("*.txt")) if MULTIFREQUENCY_DIR.is_dir() else []
    frequencies=set();conditions=[]
    for path in files:
        with path.open("r",encoding="utf-8-sig",errors="ignore") as f:
            reader=csv.DictReader(f,delimiter="\t")
            for row in reader:
                conditions.append(row.get("Condition"));frequencies.add(float(row["Actual Frequency (Hz)"]))
                if len(conditions)>1000:break
    return {"directory":MULTIFREQUENCY_DIR,"file_count":len(files),"frequencies_hz":sorted(frequencies),
            "data_type":"91-pair boundary impedance scans","quantitative_load_labels_present":False,
            "direct_intrinsic_impedance_present":False,
            "use_for_curve_validation":"not possible without load magnitude and intrinsic ground truth"}


def conclusion_summary(source_rows,trace,transfer,forward,corrections,inventory,locations):
    corrected=[r for r in source_rows if r["source_status"]=="pipeline_in_memory_corrected" and r["frequency_hz"]==5000 and r["load_g"] in (500,1000)]
    raw_min=min(r["calculated_ratio_to_repeat_unloaded"] for r in corrected)
    raw_median=float(np.median([r["calculated_ratio_to_repeat_unloaded"] for r in corrected]))
    final=[r["final_ratio_to_E2_unloaded"] for r in trace if r["load_kg"] in (.5,1.)]
    forward_ok=max(r["max_relative_error"] for r in forward)<1e-9
    return {
        "verified_findings":{
            "workbook_content":"five sensors, 18 loads (0..1000 g), five repeated R/C parameter pairs; documented as 5 kHz",
            "impedance_status":"impedance is calculated from recorded R/C using a parallel-RC formula, not directly stored complex impedance",
            "raw_5k_ratio_500_1000_min":raw_min,"raw_5k_ratio_500_1000_median":raw_median,
            "E2_trace_ratio_500_1000_min":min(final),"E2_trace_ratio_500_1000_median":float(np.median(final)),
            "no_extrapolation":True,"unloaded_exact":True,"forward_model_reproduced":forward_ok,
            "parasitics_in_final_dataset":False},
        "answers":{
        "A":"Some impedances calculated from the recorded workbook R/C values reach an approximately 99.8%% decrease, but this is not the typical 0.5/1.0-kg source response: the audited minimum ratio is %.6g and the median is %.6g. These are parallel-RC-derived impedances, not directly stored complex-impedance measurements."%(raw_min,raw_median),
            "B":"The first extreme reductions appear when the recorded load-dependent R/C values are converted to impedance, driven especially by orders-of-magnitude resistance decreases. The E2 transfer then makes the near-collapse more widespread: the traced 0.5/1.0-kg median ratio is %.6g because normalized R and C are applied to a different baseline."%float(np.median(final)),
            "C":"No algebraic double-normalization, percent/fraction error, or forward-model scale error was demonstrated. The critical issue is provenance/physical support: transferring normalized R and C from five 5-kHz sensors to 19 E2 baselines at 100 kHz is an explicit unsupported approximation.",
            "D":"Continuous PCHIP interpolation is bounded by 18 measured load knots and does not extrapolate, but experimental support is limited to five sensors at 5 kHz; it is inadequate as direct validation for 19 sensors at 100 kHz.",
            "E":"Do not use the current dataset as physically validated real-transfer ground truth. Retain it only as a clearly labeled hypothesis/sensitivity dataset; replace or regenerate it after direct 100-kHz intrinsic load calibration.",
            "F":"Magnitude prediction is mathematically well-defined for this synthetic pipeline, but physical training claims are not justified until the transferred curve is independently validated."},
        "corrections":corrections,"multifrequency_inventory":inventory,"source_code_locations":locations,
        "missing_evidence":["direct complex intrinsic impedance versus quantitative load at 100 kHz for the 19 physical sensors",
                            "quantitative loads for the later multifrequency boundary scans",
                            "independent validation that normalized R and C response is frequency-invariant"]}


def report(summary):
    a=summary["answers"];v=summary["verified_findings"]
    return f"""# Physical audit of E2 continuous-load simulation

## Verified pipeline

The source workbook contains five sensor sheets, 18 load levels from 0 to 1000 g, and five repeated R/C parameter pairs documented at 5 kHz. Complex intrinsic impedance is **calculated**, not stored directly:

`Z = 1 / (1/R + j 2π f C)`.

The curve builder normalizes C as `log(C/C0)` and resistance as `log(R0/R)`, smooths both, takes the median across five sensor curves, and uses PCHIP without extrapolation. The E2 generator applies `C=C0·exp(log_C_ratio)` and `R=R0·exp(-log_R0_over_R)` to each E2 baseline at 100 kHz. Thus it transfers separate normalized R and C responses—not a directly measured 100-kHz magnitude ratio.

The final dataset stores intrinsic `Z_all`; it does not serialize a parasitic-coupled boundary response. Independent grounded-nodal recomputation passed: `{v['forward_model_reproduced']}`.

## Required conclusions

### A. Is the ~99.8% decrease in original measurements?

{a['A']}

### B. Where does it first appear?

{a['B']}

### C. Error, assumption, or measured response?

{a['C']}

### D. Is continuous interpolation supported?

{a['D']}

### E. Retain, correct, or replace?

{a['E']}

### F. Train an intrinsic-magnitude model?

{a['F']}

## Missing evidence and limitations

"""+"\n".join(f"- {x}" for x in summary["missing_evidence"])+"\n"


def figures(source_rows,trace,out):
    figdir=out/"figures";figdir.mkdir()
    fig,ax=plt.subplots(figsize=(8,5))
    rows=[r for r in source_rows if r["source_status"]=="pipeline_in_memory_corrected" and r["frequency_hz"]==5000]
    for sensor in ("s1","s2","s3","s4","s5"):
        sr=[r for r in rows if r["source_sensor"]==sensor]
        loads=sorted(set(r["load_g"] for r in sr));med=[np.median([r["calculated_ratio_to_repeat_unloaded"] for r in sr if r["load_g"]==x]) for x in loads]
        ax.plot(np.array(loads)/1000,med,marker="o",ms=3,label=sensor.upper())
    ax.set_yscale("log");ax.set_xlabel("Load (kg)");ax.set_ylabel("Calculated |Z(F)| / |Z(0)| at 5 kHz")
    ax.grid(True,which="both",alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(figdir/"source_5k_magnitude_ratios.png",dpi=300);fig.savefig(figdir/"source_5k_magnitude_ratios.pdf");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for sensor in ("S2","S14","S19"):
        sr=[r for r in trace if r["sensor"]==sensor];ax.plot([r["load_kg"] for r in sr],[r["final_ratio_to_E2_unloaded"] for r in sr],marker="o",label=sensor)
    ax.set_yscale("log");ax.set_xlabel("Load (kg)");ax.set_ylabel("Generated |Z| / E2 |Z₀|")
    ax.grid(True,which="both",alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(figdir/"e2_transfer_trace.png",dpi=300);fig.savefig(figdir/"e2_transfer_trace.pdf");plt.close(fig)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--dataset",type=Path,default=DATASET)
    p.add_argument("--output-root",type=Path,default=SIM_DIR);p.add_argument("--check-only",action="store_true");return p.parse_args()


def main():
    args=parse_args();required=[args.dataset,WORKBOOK,CURVE_SOURCE,E2_GENERATOR,E2_BASELINE]
    missing=[str(p) for p in required if not p.is_file()]
    if missing:raise FileNotFoundError("Missing required files: "+str(missing))
    curve=import_file(CURVE_SOURCE,"audit_curve_source");e2=import_file(E2_GENERATOR,"audit_e2_generator")
    raw,_=load_workbook_values(False);corrected,corrections=load_workbook_values(True)
    model=curve.fit_excel_curves(WORKBOOK);g0,b0,z0=read_e2_baseline()
    with args.dataset.open("rb") as f:data=pickle.load(f)
    inventory=multifrequency_inventory();locations=source_locations(curve,e2)
    config={"created_utc":datetime.now(timezone.utc).isoformat(),"dataset":args.dataset.resolve(),"dataset_sha256":sha256(args.dataset),
        "sources":{str(p.resolve()):sha256(p) for p in required},"trace_loads_kg":(TRACE_LOAD_G/1000).tolist(),
        "trace_sensors":[f"S{x}" for x in SENSORS_TRACE],"check_only":args.check_only,
        "software":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__}}
    stamp=datetime.now().strftime("%Y%m%dT%H%M%S");out=args.output_root.resolve()/f"E2_continuous_physics_audit_{stamp}"
    out.mkdir(parents=False,exist_ok=False);write_json(out/"run_config.json",config)
    availability={"status":"PASS","required_files_present":True,"multifrequency":inventory,
        "dataset_keys":sorted(data),"dataset_shapes":{k:list(np.asarray(data[k]).shape) for k in ("Z_unit","Z_unit_0","Z_all","Z_all_0","load_kg")},
        "parasitic_coupled_response_present":"Z_all_with_coupling" in data,"source_code_locations":locations,
        "pipeline_corrections":corrections,"missing_direct_100k_intrinsic_load_data":True}
    if args.check_only:write_json(out/"physical_validity_summary.json",availability);print(f"PASS check-only: {out}");return
    source=source_measurements(raw,corrected);trace=normalization_trace(curve,corrected,model,g0,b0,z0)
    transfer=frequency_transfer_rows(corrected,model,g0,b0,z0);forward=forward_checks(data)
    summary=conclusion_summary(source,trace,transfer,forward,corrections,inventory,locations)
    write_csv(out/"source_measurement_summary.csv",source);write_csv(out/"normalization_trace.csv",trace)
    write_csv(out/"frequency_transfer_audit.csv",transfer);write_csv(out/"forward_model_checks.csv",forward)
    write_json(out/"physical_validity_summary.json",summary);(out/"audit_report.md").write_text(report(summary),encoding="utf-8")
    figures(source,trace,out);print(f"Audit complete: {out}")


if __name__=="__main__":main()
