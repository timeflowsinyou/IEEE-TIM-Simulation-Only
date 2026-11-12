"""Generate load-curve-constrained sensor states and grounded-nodal measurements.

Place beside its supporting simulation scripts; inputs default to ../Data.
Reads corrected Nano_Sensors_five(1).xlsx (g / pF / MOhm, measured at 5 kHz)
or load_response_curves.json, and optimized 19-sensor R0/C0 in SI units.

Default: shared normalized response, 1–4 loaded sensors/sample, uniform
continuous loads 10–1000 g, 50,000 nonzero samples + 25 zero samples.
Both uncoupled and coupled clean responses are stored. No random noise added.
The 5-kHz normalized response is assumed transferable to the selected frequency;
this is a simulation hypothesis, not measured 100-kHz calibration.

Requires numpy, scipy, torch, joblib; openpyxl only for Excel curve input.
Example: python generate_simulation_curve_nodal_forward_50k.py
Optional: --curve-mode templates --max-pressed 19
Optional: --load-mode discrete --load-levels 500 1000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

SCRIPT_DIR = Path(__file__).resolve().parent
PHYSICAL_EDGES = (
    (1,2),(1,8),(2,3),(2,9),(3,4),(3,10),(4,5),(4,11),
    (5,6),(5,12),(6,7),(6,13),(7,14),(8,9),(9,10),
    (10,11),(11,12),(12,13),(13,14),
)
ORIGINAL_NODES = list(range(1,8))+list(range(14,7,-1))
ORIGINAL_PAIRS = tuple((u,v) for k,u in enumerate(ORIGINAL_NODES[:-1])
                       for v in ORIGINAL_NODES[k+1:])
MEASUREMENT_PAIRS_11 = ((1,3),(1,9),(3,5),(3,9),(3,11),(5,7),
                        (5,11),(5,13),(7,13),(13,11),(11,9))
KEEP_INDICES_11 = [ORIGINAL_PAIRS.index(pair) for pair in MEASUREMENT_PAIRS_11]


def default_data_dir():
    if SCRIPT_DIR.name.lower() == 'data':
        return SCRIPT_DIR
    for candidate in (SCRIPT_DIR/'Data', SCRIPT_DIR.parent/'Data'):
        if candidate.is_dir():
            return candidate
    return SCRIPT_DIR.parent/'Data'


def incidence(pairs):
    a = np.zeros((14,len(pairs)),dtype=np.float64)
    for k,(u,v) in enumerate(pairs):
        a[u-1,k]=1.; a[v-1,k]=-1.
    return a


A_REDUCED = incidence(PHYSICAL_EDGES)[:-1]
C91_REDUCED = incidence(ORIGINAL_PAIRS)[:-1]


@torch.no_grad()
def nodal_forward_batch(z_sensors, *, y_scale, batch_size=1024, device='cpu'):
    """Same grounded complex nodal operator/order as the uploaded generator."""
    z_sensors=np.asarray(z_sensors,dtype=np.complex128)
    if z_sensors.ndim!=2 or z_sensors.shape[1]!=19 or not len(z_sensors):
        raise ValueError('Z_unit must have shape (N,19), N>0')
    if not np.isfinite(z_sensors).all() or np.any(np.abs(z_sensors)==0):
        raise ValueError('Nonfinite/zero sensor impedance')
    if not np.isfinite(y_scale) or y_scale<=0 or batch_size<1:
        raise ValueError('Invalid nodal scale or batch size')
    dev=torch.device(device)
    a=torch.tensor(A_REDUCED,dtype=torch.complex128,device=dev)
    c=torch.tensor(C91_REDUCED,dtype=torch.complex128,device=dev)
    out=[]
    for start in range(0,len(z_sensors),batch_size):
        zs=torch.tensor(z_sensors[start:start+batch_size],dtype=torch.complex128,device=dev)
        y=(1./zs)/float(y_scale)
        lap=torch.einsum('ie,be,je->bij',a,y,a)
        cb=c.expand(len(zs),-1,-1)
        v=torch.linalg.solve(lap,cb)
        out.append((torch.sum(cb*v,dim=1)/float(y_scale)).cpu().numpy())
    return np.concatenate(out,axis=0)


def smooth_knots(x,y,lam):
    t=np.log1p(x/10.)
    d=np.zeros((len(x)-2,len(x)))
    for k in range(len(x)-2):
        h1,h2=t[k+1]-t[k],t[k+2]-t[k+1]
        d[k,k:k+3]=[1/h1,-1/h1-1/h2,1/h2]
    penalty=d[:,1:]
    result=np.zeros(len(x))
    result[1:]=np.linalg.solve(np.eye(len(x)-1)+lam*penalty.T@penalty,y[1:])
    return result


def fit_repeats(x,y):
    candidates=[0.,.001,.003,.01,.03,.1,.3,1.,3.]
    scores=[]
    for lam in candidates:
        errors=[]
        for repeat in range(5):
            center=np.median(np.delete(y,repeat,axis=1),axis=1)
            predicted=smooth_knots(x,center,lam)
            errors.append(float(np.mean(np.abs(predicted[1:]-y[1:,repeat]))))
        scores.append(dict(lam=lam,mae=float(np.mean(errors)),repeat_mae=errors))
    best=min(scores,key=lambda s:s['mae'])
    return smooth_knots(x,np.median(y,axis=1),best['lam']),best,scores


def fit_excel_curves(path):
    import openpyxl  # read-only Excel access; does not edit the input
    workbook=openpyxl.load_workbook(path,data_only=True)
    models={}; diagnostics={}; corrections=[]
    # Authorized corrections from this experiment. Already corrected cells stay unchanged.
    for sheet,cell,old,new in [('s1','C2',24.,97.),('s2','G11',45.,4.5),
                                ('s3','C17',.195,.0195)]:
        actual=workbook[sheet][cell].value
        if actual==old:
            workbook[sheet][cell].value=new  # analytical in-memory correction only
            corrections.append(dict(sheet=sheet,cell=cell,old=old,new=new,status='applied_in_memory'))
        elif actual==new:
            corrections.append(dict(sheet=sheet,cell=cell,old=old,new=new,status='already_corrected'))
        else:
            corrections.append(dict(sheet=sheet,cell=cell,observed=actual,status='different_value_retained'))
    expected_x=None
    for name in ['s1','s2','s3','s4','s5']:
        a=np.asarray(list(workbook[name].values)[1:],dtype=float)
        if a.ndim!=2 or a.shape[1]!=11 or len(a)<3:
            raise ValueError(f'{name}: expected load + five C/R pairs')
        x,c,r=a[:,0],a[:,1::2],a[:,2::2]
        if not np.isfinite(a).all() or np.any(c<=0) or np.any(r<=0):
            raise ValueError(f'{name}: nonfinite or nonpositive R/C')
        if x[0]!=0 or np.any(np.diff(x)<=0):
            raise ValueError(f'{name}: loads must start at zero and strictly increase')
        if expected_x is None:expected_x=x
        elif not np.array_equal(x,expected_x):raise ValueError('Sensor load grids differ')
        logc=np.log(c/c[0]); q=np.log(r[0]/r)
        cf,cb,cs=fit_repeats(x,logc);qf,qb,qs=fit_repeats(x,q)
        models[name]=dict(load_g=x.tolist(),log_C_ratio=cf.tolist(),
            log_R0_over_R=qf.tolist(),lambda_C=cb['lam'],lambda_R=qb['lam'])
        diagnostics[name]=dict(validation_C=cs,validation_R=qs,
            repeat_C_ratio_quartiles=np.percentile(c/c[0],[25,50,75],axis=1).tolist(),
            repeat_R_ratio_quartiles=np.percentile(r/r[0],[25,50,75],axis=1).tolist(),
            raw_R_at_max_load_MOhm=r[-1].tolist())
    models['shared']=dict(load_g=expected_x.tolist(),
        log_C_ratio=np.median([s['log_C_ratio'] for s in models.values()],axis=0).tolist(),
        log_R0_over_R=np.median([s['log_R0_over_R'] for s in models.values()],axis=0).tolist())
    return dict(frequency_hz=5000,units=dict(load='g',C='pF',R='MOhm'),
        sensors=models,diagnostics=diagnostics,corrections=corrections,
        method='Per-repeat normalization, log-response medians, leave-one-repeat-out slope smoothing, PCHIP',
        shared_method='Median of the five fitted log-response knot sets')


def load_curves(path):
    if path.suffix.lower()=='.json':
        model=json.loads(path.read_text(encoding='utf-8'))
    elif path.suffix.lower()=='.xlsx':
        model=fit_excel_curves(path)
    else:raise ValueError('--curve-file must be .xlsx or .json')
    for name in ['s1','s2','s3','s4','s5','shared']:
        s=model['sensors'][name]
        x=np.asarray(s['load_g'],float)
        lc=np.asarray(s['log_C_ratio'],float);q=np.asarray(s['log_R0_over_R'],float)
        if x.ndim!=1 or len(x)<3 or x[0]!=0 or np.any(np.diff(x)<=0):
            raise ValueError(f'{name}: invalid knot grid')
        if lc.shape!=x.shape or q.shape!=x.shape or not np.isfinite([x,lc,q]).all():
            raise ValueError(f'{name}: invalid log-response arrays')
        if not np.isclose(lc[0],0,atol=1e-12) or not np.isclose(q[0],0,atol=1e-12):
            raise ValueError(f'{name}: unloaded ratios must equal one')
    return model


def vector(obj,key,n,*,positive=True):
    a=np.asarray(obj[key],dtype=float).reshape(-1)
    if a.shape!=(n,) or not np.isfinite(a).all():
        raise ValueError(f'{key}: expected {n} finite values')
    if (np.any(a<=0) if positive else np.any(a<0)):
        raise ValueError(f'{key}: values must be '+('positive' if positive else 'nonnegative'))
    return a


def sensor_states(load_g,r0,c0,model,curve_names,frequency):
    """Known fixed curve for each sensor; all variability here comes from load."""
    r=np.empty_like(load_g,dtype=np.float64);c=np.empty_like(r)
    for j,name in enumerate(curve_names):
        s=model['sensors'][name];x=np.asarray(s['load_g'])
        loads=load_g[:,j]
        if np.any(loads<x[0]) or np.any(loads>x[-1]):
            raise ValueError(f'Sensor {j+1}: load outside calibrated curve range')
        lc=PchipInterpolator(x,s['log_C_ratio'],extrapolate=False)(loads)
        q=PchipInterpolator(x,s['log_R0_over_R'],extrapolate=False)(loads)
        c[:,j]=c0[j]*np.exp(lc);r[:,j]=r0[j]*np.exp(-q)
    zero=load_g==0
    r[zero]=np.broadcast_to(r0,r.shape)[zero]
    c[zero]=np.broadcast_to(c0,c.shape)[zero]
    if not np.isfinite(r).all() or not np.isfinite(c).all() or np.any(r<=0) or np.any(c<=0):
        raise ValueError('Curve evaluation produced invalid R/C')
    z=1./(1./r+1j*2*np.pi*frequency*c)
    return r,c,z


def sample_loads(rng,n,zero_repeats,min_pressed,max_pressed,load_mode,min_load,max_load,levels):
    loads=np.zeros((n+zero_repeats,19),dtype=np.float64)
    counts=rng.integers(min_pressed,max_pressed+1,size=n)
    for i,k in enumerate(counts):
        selected=rng.choice(19,int(k),replace=False)
        values=(rng.uniform(min_load,max_load,size=int(k)) if load_mode=='continuous'
                else rng.choice(levels,size=int(k),replace=True))
        loads[i,selected]=values
    return loads


def resolve_inputs(args):
    data=args.data_dir
    if args.curve_file is None:
        candidates=[data/'Nano_Sensors_five.xlsx',data/'Nano_Sensors_five.xlsx',
                    data/'load_response_curves.json']
        args.curve_file=next((p for p in candidates if p.is_file()),candidates[0])
    if args.rc_calibration is None:args.rc_calibration=data/'lsq_result91para_0512_meas_complex_v0.joblib'
    if args.parasitic_calibration is None:args.parasitic_calibration=data/'lsq_result91para_C_0512_v0.joblib'
    if args.output is None:
        suffix=(f'{args.total_before_zero//1000}k' if args.total_before_zero%1000==0
                else str(args.total_before_zero))
        args.output=data/f'simulation_{suffix}_snr_clean_curve_{args.curve_mode}_{args.load_mode}_nodal.pkl'
    for path in [args.curve_file,args.rc_calibration]+([] if args.no_parasitics else [args.parasitic_calibration]):
        if not path.is_file():raise FileNotFoundError(f'Input not found: {path}')
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f'Output exists: {args.output}; use --overwrite to replace it')
    if args.output.resolve() in {p.resolve() for p in [args.curve_file,args.rc_calibration,args.parasitic_calibration]}:
        raise ValueError('Output cannot overwrite an input file')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir',type=Path,default=default_data_dir())
    p.add_argument('--curve-file',type=Path)
    p.add_argument('--curve-mode',choices=['shared','templates'],default='shared')
    p.add_argument('--sensor-curve-map',nargs=19,choices=['s1','s2','s3','s4','s5'],
                   help='Fixed curve assignment in PHYSICAL_EDGES order; templates mode only')
    p.add_argument('--rc-calibration',type=Path)
    p.add_argument('--parasitic-calibration',type=Path)
    p.add_argument('--no-parasitics',action='store_true',help='Save coupled response identical to intrinsic')
    p.add_argument('--frequency',type=float,default=1e5)
    p.add_argument('--total-before-zero',type=int,default=10000)
    p.add_argument('--zero-repeats',type=int,default=25)
    p.add_argument('--min-pressed',type=int,default=1)
    p.add_argument('--max-pressed',type=int,default=19)
    p.add_argument('--load-mode',choices=['continuous','discrete'],default='discrete')
    p.add_argument('--min-load-g',type=float,default=10.)
    p.add_argument('--max-load-g',type=float,default=1000.)
    p.add_argument('--load-levels',type=float,nargs='+',default=[500.,1000.])
    p.add_argument('--seed',type=int,default=24)
    p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--output',type=Path)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if not np.isfinite(args.frequency) or args.frequency<=0:raise ValueError('frequency must be positive')
    if args.total_before_zero<1 or args.zero_repeats<0 or args.batch_size<1:raise ValueError('Invalid counts')
    if not 1<=args.min_pressed<=args.max_pressed<=19:raise ValueError('Pressed counts must satisfy 1 <= min <= max <= 19')
    if args.sensor_curve_map and args.curve_mode!='templates':raise ValueError('Curve map requires templates mode')
    if args.load_mode=='continuous':
        if not np.isfinite([args.min_load_g,args.max_load_g]).all() or not 0<args.min_load_g<=args.max_load_g:
            raise ValueError('Continuous positive load range is invalid')
        selected_levels=np.array([args.min_load_g,args.max_load_g])
    else:
        selected_levels=np.asarray(args.load_levels)
        if not np.isfinite(selected_levels).all() or np.any(selected_levels<=0):raise ValueError('Discrete loads must be positive')
    resolve_inputs(args)
    model=load_curves(args.curve_file)
    curve_names=(['shared']*19 if args.curve_mode=='shared' else
                 list(args.sensor_curve_map or [f's{i%5+1}' for i in range(19)]))
    for name in set(curve_names):
        if np.max(selected_levels)>max(model['sensors'][name]['load_g']):
            raise ValueError('Requested loads exceed calibrated range; extrapolation is disabled')
    rc=joblib.load(args.rc_calibration)
    r0=vector(rc,'R0_opt',19);c0=vector(rc,'C0_opt',19)
    if args.no_parasitics:gp=np.zeros(91);cp=np.zeros(91)
    else:
        para=joblib.load(args.parasitic_calibration)
        gp=vector(para,'Gp_opt',91,positive=False);cp=vector(para,'Cp_opt',91,positive=False)
    omega=2*np.pi*args.frequency;y_para=gp+1j*omega*cp
    z0=1./(1./r0+1j*omega*c0);y_scale=float(np.median(np.abs(1./z0)))
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    rng=np.random.default_rng(args.seed)
    loads=sample_loads(rng,args.total_before_zero,args.zero_repeats,args.min_pressed,
        args.max_pressed,args.load_mode,args.min_load_g,args.max_load_g,args.load_levels)
    r,c,z_unit=sensor_states(loads,r0,c0,model,curve_names,args.frequency)
    labels=(loads>0).astype(np.float32)
    print(f'Curve input: {args.curve_file}\nR0/C0 input: {args.rc_calibration}')
    print(f'Generating {len(loads)} clean curve-based samples on {device}; {args.curve_mode} mode')
    print(f'Positive contacts/sample: {args.min_pressed}–{args.max_pressed}; load mode: {args.load_mode}')
    z_all=nodal_forward_batch(z_unit,y_scale=y_scale,batch_size=args.batch_size,device=device)
    z_coupled=1./(1./z_all+y_para[None,:]) if not args.no_parasitics else z_all.copy()
    z_boundary0=nodal_forward_batch(z0[None,:],y_scale=y_scale,batch_size=1,device=device)[0]
    if not np.isfinite(z_all).all() or not np.isfinite(z_coupled).all():raise ValueError('Nonfinite boundary response')
    if args.zero_repeats:
        if not np.allclose(z_unit[-args.zero_repeats:],z0[None,:],rtol=1e-12,atol=0):raise AssertionError('Zero-state mismatch')
        if not np.allclose(z_all[-args.zero_repeats:],z_boundary0[None,:],rtol=1e-10,atol=0):raise AssertionError('Zero boundary mismatch')
    metadata=dict(curve_mode=args.curve_mode,curve_assignment_19=curve_names,
        curve_source=str(args.curve_file),curve_source_sha256=hashlib.sha256(args.curve_file.read_bytes()).hexdigest(),
        curve_measurement_frequency_hz=float(model.get('frequency_hz',5000)),
        assumed_frequency_transfer=True,
        note='5-kHz normalized responses scaled by optimized unloaded R/C; not measured calibration of 19 sensors at 100 kHz',
        rc_calibration=str(args.rc_calibration),rc_calibration_sha256=hashlib.sha256(args.rc_calibration.read_bytes()).hexdigest(),
        no_parasitics=args.no_parasitics,no_random_noise=True,
        min_pressed=args.min_pressed,max_pressed=args.max_pressed,load_mode=args.load_mode,
        min_load_g=args.min_load_g,max_load_g=args.max_load_g,load_levels_g=args.load_levels,
        template_assignment_note='Fixed assumed mapping; five templates reused across 19 edges, not 19 measured curves',
        units=dict(load_g='g',R_unit='ohm',C_unit='F',Z_unit='ohm',Z_all='ohm'))
    dataset=dict(Z_all=z_all,Z_all_with_coupling=z_coupled,Z_unit=z_unit,y_cls=labels,
        load_g=loads,y_load_g=loads,R_unit=r,C_unit=c,R0_opt=r0,C0_opt=c0,Gp_opt=gp,Cp_opt=cp,
        Z_unit_0=z0,Z_all_0=z_boundary0,Z_all_with_coupling_0=1./(1./z_boundary0+y_para),
        Z_unit_ratio=z_unit/z0[None,:],Z_unit_relative_change=(z_unit-z0[None,:])/np.abs(z0[None,:]),
        frequency=float(args.frequency),physical_edges=PHYSICAL_EDGES,original_nodes=ORIGINAL_NODES,
        original_pairs=ORIGINAL_PAIRS,measurement_pairs_11=MEASUREMENT_PAIRS_11,keep_indices_11=KEEP_INDICES_11,
        generator='curve_constrained_grounded_nodal_forward',seed=args.seed,n_measured_seed_samples=0,
        n_total=len(z_unit),zero_repeats=args.zero_repeats,curve_model=model,generation_metadata=metadata)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('wb') as f:pickle.dump(dataset,f,protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved: {args.output}')
    for key in ['Z_all','Z_all_with_coupling','Z_unit','y_cls','load_g','R_unit','C_unit']:
        print(f'{key:22s}: {dataset[key].shape} {dataset[key].dtype}')
    print(f'Loaded sensor R range: {r[loads>0].min():.6g}–{r[loads>0].max():.6g} ohm')
    print(f'Loaded sensor C range: {c[loads>0].min()*1e12:.6g}–{c[loads>0].max()*1e12:.6g} pF')
    print('Existing reconstruction scripts can select Z_all (--coupling without) or Z_all_with_coupling (--coupling with).')
    print('Continuous load reconstruction requires a load-output model using load_g and these same fixed curves.')


if __name__=='__main__':main()
