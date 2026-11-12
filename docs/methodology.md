# Methodology and Reproducibility Notes

## Scope

This archive covers the E2 continuous-load simulation, N7-A/N7-B dual-task training, physical-validity diagnostics, N7-B intrinsic-magnitude experiment, full 1–19 pressed-count evaluation, and frozen real-data diagnostic. It intentionally excludes unrelated historical studies.

## Electrical network and measurement ordering

The physical network contains 14 nodes and 19 sensor edges in fixed S1–S19 order. All unordered boundary-node pairs are retained during data generation, giving `14 choose 2 = 91` complex boundary impedances. The simulation dataset stores their canonical ordering in `original_pairs` rather than relying on an implicit CSV order.

N7-B uses the following 11 physical pairs:

```text
(2,4), (2,10), (2,8), (4,6), (4,12), (4,10),
(6,14), (6,12), (14,12), (12,10), (10,8)
```

Pair lookup is unordered: orientation in a stored pair does not change column selection. Real data use the acquisition's clockwise canonical node sequence `[1..7, 14..8]`; real and simulation pair indices are mapped separately by physical node IDs.

## E2 simulation generation

The generator creates exactly 20,000 samples at 100 kHz:

- 19,975 loaded samples with a uniformly selected pressed count from 1 through 19.
- 25 unloaded samples.
- One fixed E2 unloaded intrinsic reference containing 19 complex impedances.
- All 91 grounded-nodal boundary responses.

Unpressed sensors retain the E2 baseline and zero load. Pressed sensors receive positive continuous loads and intrinsic impedances from a shared load-response library. The library fits normalized resistance and capacitance responses from five sensors measured at 5 kHz and transfers those normalized curves to the 100 kHz E2 baselines. PCHIP interpolation remains within the measured 0–1 kg range.

This frequency transfer is an unvalidated simulation assumption. The generated dataset excludes parasitic boundary admittance.

Expected dataset fields include `Z_all (N,91)`, `Z_all_0 (91,)`, `Z_unit (N,19)`, `Z_unit_0 (19,)`, `load_kg (N,19)`, `y_cls (N,19)`, `physical_edges (19,2)`, and `original_pairs (91,2)`.

## Input preprocessing

For each selected measurement pair, the neural-network input is:

1. `abs(Z) / abs(Z0)`
2. `sin(angle(Z / Z0))`
3. `cos(angle(Z / Z0))`

This produces an `N × 11 × 3` tensor. Feature mean and standard deviation are fitted on the training partition only, across all selected pairs, independently for the three feature channels. Validation, test, and real inputs use the frozen checkpoint statistics.

## Targets

Classification uses 19 binary pressed-state labels.

The original dual-task experiment regresses 19 continuous loads in kilograms. The magnitude experiment instead predicts

```text
Delta = (abs(Z_unit) - abs(Z_unit_0)) / abs(Z_unit_0)
```

The magnitude target is signed and unclipped. Its regression head is linear and unconstrained. Reconstructed magnitude is `abs(Z_unit_0) * (1 + predicted_Delta)`; predictions below `Delta = -1` are recorded as invalid and are not silently clamped.

## Splitting and reproducibility

Byte-identical 91-dimensional complex inputs are grouped before splitting so duplicates cannot cross partitions. Conflicting binary or continuous targets within an identical-input group cause a hard failure.

The grouped split is approximately 70/15/15, stratified by pressed-sensor count:

- Split seed: 24.
- Model seed: 43.
- Batch size: 256.
- Maximum epochs: 100.
- Minimum epochs: 25.
- Early-stopping patience: 15.

The magnitude experiment requires the exact saved split from the frozen original N7-B experiment and checks its file hash and checkpoint-embedded indices. Both models therefore have paired test samples.

## CNN and training

The shared backbone contains residual Conv1d blocks with GroupNorm and ReLU:

```text
3 → 64 → 128 → max-pool → 256 → adaptive average pool
```

Each output head is `Linear(256,64) → ReLU → Linear(64,19)`. Classification uses BCE with logits and training-set class weights. Regression uses Smooth-L1 on pressed entries plus a 0.10-weighted unpressed term. Its combined-loss weight is 2.0. Optimization uses Adam (`lr=1e-3`, weight decay `1e-5`) and ReduceLROnPlateau (`factor=0.5`, patience 6, minimum LR `1e-5`). The best checkpoint is selected by total validation loss. Classification threshold selection uses validation data only.

## Evaluation

The primary simulation evaluation reports every pressed count from 1 through 19. For each count it reports sample count, Micro-F1, Macro-F1 across all 19 sensors, Exact Match Accuracy, and minimum recall among sensors with positive support. Magnitude metrics include signed-change MAE/RMSE/bias, reconstructed magnitude errors in ohms, relative error, invalid reconstructions, and comparisons with:

- zero change;
- global training-set median;
- per-sensor training-set median;
- per-sensor median conditioned on true pressed state (nondeployable oracle diagnostic).

The 1–4 subgroup is secondary and is not a model-selection criterion.

## Real-data diagnostic

Real validation uses frozen checkpoints and thresholds. No network is retrained or fine-tuned, and real labels do not select preprocessing statistics or thresholds. The inspected archive contains 804 samples at 100 kHz, all 91 complex boundary pairs, nominal 0.5/1.0 kg labels, and five genuine no-load samples. Pressed counts 1–11 occur, but counts 7 and 11 each have one sample; counts 12–19 are absent.

Three conditions are evaluated:

1. raw measurement + simulation unloaded reference;
2. parasitic-compensated measurement + simulation unloaded reference;
3. parasitic-compensated measurement + compensated real unloaded reference.

The real unloaded reference is the component-wise median of the five compensated no-load rows. Raw + real reference was not evaluated, so compensation and reference replacement are not independently identified.

## Parasitic compensation

The historical calibration contains 91 conductances `Gp_opt` in siemens and capacitances `Cp_opt` in farads. At frequency `f`, the project model uses

```text
Ypara = Gp_opt + j*2*pi*f*Cp_opt
Zcoupled = 1 / (1/Zintrinsic + Ypara)
```

The diagnostic inverse is therefore

```text
Zintrinsic = 1 / (1/Zmeasured - Ypara)
```

It is applied exactly once in admittance space, never as impedance subtraction. This remains a calibrated approximation. The joblib file has no embedded pair labels; its clockwise 91-pair order is established through project code lineage. A substantial fraction of inferred real admittances had nonpositive real parts, limiting physical interpretation.

## Supported commands

The archived scripts retain their original command-line interfaces. From the original project layout, representative commands are:

```powershell
python generate_e2_continuous_load_20k.py --check-only
python train_e2_n7k11_binary_continuous_reviewed.py --check-only
python train_e2_n7b_binary_intrinsic_magnitude.py --check-only
python analyze_e2_n7b_full_pressed_count.py
python validate_e2_n7b_magnitude_real_compensated.py --check-only
python validate_e2_n7b_magnitude_real_compensated.py
```

Generation, training, and inference are not necessary to inspect the lightweight archived results. Script defaults contain historical local paths; use their supported path arguments when running elsewhere.

## External inputs and privacy

Full reproduction requires external inputs that are not published here:

- the 19-sensor E2 calibration and provenance files;
- experimental resistance/capacitance workbook used for the normalized response library;
- 20,000-sample complex simulation dataset;
- frozen `.pt` checkpoints;
- 804-sample real 91-pair measurement archive and manifest;
- historical 91-path parasitic `.joblib` calibration.

Without these private or large artifacts, the repository supports source inspection and verification of published summary tables, not complete end-to-end reproduction.
