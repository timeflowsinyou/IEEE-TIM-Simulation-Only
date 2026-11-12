# Repository Preparation Report

## Purpose

The private repository was prepared as a focused archive of the completed E2 simulation and sim-to-real investigation. Existing research scripts, datasets, checkpoints, calibration files, and timestamped output directories were not moved, renamed, deleted, or overwritten.

## Added archive material

- Scientific English README with verified simulation and real-data results.
- Methodology and reproducibility documentation.
- Dependency-name list without invented version constraints.
- Archival copies of the selected simulation, training, diagnostic, and validation scripts.
- Compact result CSV/JSON files, four copied figures, and one documentation-only sensor-recall figure generated from the saved CSV.
- Expanded ignore rules for binary data, checkpoints, private inputs, logs, and timestamped experimental outputs.

The prior untracked README described a different historical clean-simulation study. It is preserved locally as `README.previous.local.md` and ignored; it is not proposed for this focused archive.

## Scientific status

The project is archived as a completed exploration. Simulation classification is count-dependent. The magnitude model does not demonstrate quantitative reconstruction, and frozen real-data evaluation reveals major reference and sensor-specific domain shift. No real intrinsic-impedance ground truth is available.

## Excluded material

- raw experimental TXT/NPZ measurements;
- synthetic PKL/NPZ datasets;
- model checkpoints;
- joblib calibration files;
- complete timestamped result directories;
- unrelated historical experiment scripts and outputs;
- local logs, caches, and personal workstation paths contained in run manifests.

## Review policy

Only the explicit files listed in the final staging plan should be added. Do not use `git add .` or `git add -A`.
