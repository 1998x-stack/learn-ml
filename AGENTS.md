# learn-ml — 从零实现 mini-scikit-learn（NumPy 递进式教程）

Progressive chapter-based tutorial (s00_overview + s01–s20) building a mini scikit-learn from scratch in NumPy, following learn-claude-code's invariants: a fixed minimum core (`Estimator Protocol`) that every chapter extends by exactly one mechanism. Each chapter dir has a Chinese `README.md` (14-section template) + self-contained runnable `code.py`.

This is a **teaching repository**, not an application. Chapters read in order (s00 → s20). The source-of-truth version tree lives in `versions.md`; each chapter table links back via `[返回根目录](../README.md)`. Do not modify unless explicitly asked.

## Architecture

### Dual tracks

- **Teaching track** `sXX/code.py` — each is self-contained & runnable (`python sXX_*/code.py`), re-implements pieces it needs rather than importing earlier chapters. Conceptual dependency only.
- **Framework track** `src/mini_sklearn/` — formal shared package, type-annotated, reused by all chapters; behavior verified by `tests/` (`api/`, `numerical/`, later `parity/`+`integration/`).

Conceptual dependency (v0.1 chapters):

```
s01_minimal_estimator   fit→self · training state · `_`-suffix
        ↓ concept
s02_estimator_protocol  BaseEstimator · get_params/set_params · clone · Mixins
        ↓ concept
s03_data_validation     check_array / check_X_y / check_is_fitted
        ↓ concept
s04_evaluation_loop     train_test_split · MSE/MAE/R²/accuracy · make_* datasets
```

Data flows conceptually: `X,y → check → fit → fitted_attrs → predict/score`.

## Environment

No external API keys; all chapters run on CPU with synthetic datasets. Deps: numpy, scipy, sklearn (dev parity only).

## Commands

```bash
pip install -r requirements.txt
python s01_minimal_estimator/code.py   # run a chapter
python -m pytest tests/ -q             # framework-track tests
python -c "import ast; ast.parse(open('s01_minimal_estimator/code.py').read())"  # syntax only
```

## Key conventions

- README template: versions.md §7 (14 sections); start from `[返回根目录](../README.md)`.
- `code.py` self-contained & runnable; version-tag comments `# --- v0.1.x: ... ---`.
- Framework: `fit` returns `self`; learned attrs end `_`; `get_params` returns only hyperparameters; `clone` deep-copies hyperparameters only.
- Testing: `tests/` via pytest on the framework track; each `code.py` self-verifies in `__main__` (prints `[PASS]`). No pytest coupling on the teaching track.

## Git

- **Independent git repo** initialized at `learn-ml/.git/` — not the parent `Agent-Tutorials` repo.
- `versions.md` is the source-of-truth and committed first.
- Commit only module/work files under `learn-ml/`; never stage the parent repo as a nested gitlink.
