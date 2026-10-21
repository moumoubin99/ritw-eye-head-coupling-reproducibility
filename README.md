# Gaze stabilization during walking: reproducibility materials

This anonymous code and aggregate-results release contains slow-phase gaze analysis, hand-referenced analysis, simulation checks, and selected published-result illustrations. It does not identify the study authors.

## Start with the offline check

From the repository root, run:

```sh
python tools/verify_exports.py
```

This needs Python 3.10 or later and no third-party libraries or network access. It checks the file inventory, full-file SHA256 hashes, numeric fingerprints of exported aggregate/synthetic JSON, synthetic CSV shapes, and Python syntax. It verifies the release contents; it does not rerun the scientific analysis.

For analysis or plotting, install the dependencies listed in `requirements.txt` in a separate environment. The dependency list is not an exact reconstruction of the original execution environment. Frozen result integrity and full numerical rerun equivalence are different checks.

## Included and excluded

- `src/ritw_kappa/`: analysis modules.
- `scripts/`: study measurement, statistics, calibration and simulation programs.
- `results/`: aggregate scientific statistics and synthetic experiments. Retained numerical values match the source exports. A record-linked inclusion-detail branch is intentionally omitted.
- `paper/figures/`: selected PNG/SVG illustrations and figure/table source code. Raster image pixels are unchanged; identifying metadata is removed.
- `EXPORTS.json` and `SHA256SUMS`: public content-integrity records.

Raw eye/head/hand signals, real recording identifiers and dates, participant-linking tables, metadata/annotations, private workflow documents, credentials, signed URLs, logs, caches, manuscript source, identified PDFs and repository history are not included. The real-recording Figure 3 trace and its trace metadata are not bundled.

## Reproducing the analyses

See `DATA_ACCESS.md` for official dataset access and the expected local directory layout. Run the analysis from the repository root. With authorized Seattle data and upstream annotations in place, the study sequence begins with:

```sh
python scripts/m3_cohort.py --root data/raw/seattle
python scripts/m3_measure.py --root data/raw/seattle
python scripts/m3_stats.py
python scripts/m3t_hand_coverage.py
python scripts/m3t_measure.py --root data/raw/seattle
python scripts/m3t_mask_calibration.py
python scripts/m3t_stats.py
python scripts/posthoc/m3t_decomposition_checks.py --root data/raw/seattle
```

Inspect each script's arguments and configuration before running. These commands create local recording-level tables, cohort files and outputs that should remain outside a public commit. They can overwrite included result files; use a separate working copy and retain the frozen export for comparison.

The included aggregate results support result inspection. Several figure/table generators require the excluded recording-level measurements and authorized raw data; this bundle does **not** regenerate every figure offline. Figure 3 additionally requires `--example-recording-id` supplied from the user's authorized local cohort. Its public source has no embedded example identifier and omits source identifiers from trace metadata and plot labels. Table 2 computes its rigid-comparison count from regenerated scientific CSVs instead of a private workflow log.

Synthetic source programs include `scripts/m3_synthetic.py`, `scripts/m3t_synthetic.py`, `scripts/run_v1.py` and `scripts/run_v1_rotation.py`. They are CPU analyses and can take time; the offline integrity check does not run them.

## Licensing and attribution

Public availability here is for reproducibility. No new general copyright license is asserted for original analysis code whose license was not specified by its authors. See `LICENSE.md`. Existing third-party terms remain in force: upstream Reading in the Wild Seattle data/code and the Columbus dataset card specify CC BY-NC 4.0. This release does not relicense them. See `THIRD_PARTY_NOTICES.md` and `LICENSES/CC-BY-NC-4.0.txt` for attribution and upstream terms.
