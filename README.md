# xdomain-measurability

Pre-registration, evaluation and analysis scripts, manifests, digests and analysis summaries accompanying

> *Measurability of mechanism effects in cross-domain few-shot building extraction: a pre-registered multi-seed protocol and six empirical regularities* (manuscript submitted to ISPRS Journal of Photogrammetry and Remote Sensing, 2026).

## Layout

| Path | Content |
|---|---|
| `docs/preregistration_zh.md` | The pre-registration document with all appendices and its dated change log (original language: Chinese; the criteria are restated in English in Section 3 of the paper) |
| `docs/PROVENANCE.md` | Date of every revision of the pre-registration document, from the authors' private version control |
| `scripts/` | Training drivers, evaluation scripts and analysis scripts (`*_analyze.py`); each analysis starts with the gates of Section 3.6 of the paper and stops on failure |
| `src/` | The GAPL-SegNet training code as used; `src_forked_readonly/` holds the unmodified fork source, checked by `scripts/verify_fork.py`; the fork archive is kept byte-for-byte (its digests are verified), so its comments remain in the original language, whereas all comments in `scripts/` and `src/` are in English |
| `results/summaries/` | The analysis summaries (`*_summary.json` and related files) from which every table and number in the paper is taken |
| `results/manifests/` | Support-set manifests, query-view manifests with per-file SHA-256 digests, and checkpoint digests |

Datasets (WHU Building, Inria Aerial Image Labeling, Massachusetts Buildings) are public and are not redistributed here; `scripts/make_data_view.py` and the view manifests describe how the evaluation views are built. The per-evaluation result files (one JSON per checkpoint, support draw and view), the judgement documents and the 96 checkpoints (≈5 GB) are available from the corresponding author on request and will be deposited with the archived release.

## Result archive

The per-evaluation result files behind every table (3,609 files: per-evaluation JSONs with checkpoint digests, widened sweeps, per-sample records, calibration records, manifests and the analysis summaries) are archived on Zenodo: <https://doi.org/10.5281/zenodo.22793181> (CC BY 4.0). Its README maps every directory to a paper section; the analysis scripts here recompute every table from it without a GPU.

## Reproducing

The summaries in `results/summaries/` are the outputs of the analysis scripts (for example `scripts/stage1_analyze.py`, `scripts/mass_analyze.py`, `scripts/w_analyze.py`, `scripts/y_analyze.py`, `scripts/z_analyze.py`); each script reads the per-evaluation result files, checks provenance (checkpoint digest, frozen threshold, support list), cross-machine calibration and label consistency, and writes the summary. The evaluation scripts (`scripts/eval_crossdomain.py`, `scripts/eval_stratified.py`, `scripts/threshold_range_diag.py`) regenerate the per-evaluation files from a checkpoint, a data view and a manifest.

## License

- Source code (`scripts/`, `src/`, `src_forked_readonly/`): [MIT License](LICENSE).
- Documents (`docs/`) and results (`results/`): [Creative Commons Attribution 4.0 International](LICENSE-CC-BY-4.0). Please cite the paper when reusing the pre-registration or the summaries.
