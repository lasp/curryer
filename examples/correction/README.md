# Correction Package Examples

Reference examples and configuration templates for `curryer.correction`.
Copy and adapt them when integrating geolocation correction or verification
into a new mission repository.

**Full documentation:** [`docs/source/correction_user_guide.md`](../../docs/source/correction_user_guide.md)

## Contents

| File                        | Status   | Description                                                                    |
| --------------------------- | -------- | ------------------------------------------------------------------------------ |
| `example_config.json`       | —        | Generic config template (setup/sweep/output) — copy and adapt for your mission |
| `clarreo_config.json`       | —        | CLARREO config — loadable by `load_config_files()`                             |
| `clarreo_config.py`         | —        | CLARREO config factory using the Python API; use as a new-mission template     |
| `example_verification.py`   | Runnable | End-to-end verification demo (real CLARREO data or synthetic fallback)         |
| `example_run_correction.py` | Template | Correction loop template — exits cleanly if data/tools are missing             |
| `regrid_gcp_chips.py`       | Runnable | Batch-regrid HDF GCP chips to NetCDF (see `docs/source/gcp_regridding.md`)     |

## Config surface

The correction API uses three config objects, all importable from `curryer.correction`:

- **`GeolocationSetup`** — durable mission setup (SPICE kernels + instrument via `geo`,
  pass/fail `requirements`, optional `calibration`/`data_config`, and the image-matching
  variable-name fields). Built once and reused across many runs.
- **`Sweep`** — the lightweight parameter experiment you vary between runs. Cheap-to-copy
  helpers `sweep.update_param(selector, **spec_changes)` and
  `sweep.with_strategy(strategy, **changes)` return re-validated copies.
- **`OutputConfig`** — NetCDF metadata and output filename.

Load all three from one JSON file (three top-level sections `"setup"`, `"sweep"`,
optional `"output"`):

```python
from curryer.correction import load_config_files, run_correction

setup, sweep, output = load_config_files("examples/correction/clarreo_config.json")
# NOTE arg order: inputs BEFORE work_dir.
result = run_correction(setup, sweep, inputs, work_dir, output)
```

Or build them programmatically with the CLARREO factory
(`create_clarreo_config()` returns `(setup, sweep, output)`). Standalone compliance
checks use `verify(setup, ...)`.

**Runnable** — executes immediately against committed test data; no extra setup required.
**Template** — documents the API; requires SPICE tools and mission data not in the repo.

## Prerequisites

```bash
pip install -e ".[test,dev]"
```

## Quick Start

```bash
# Verification (fully self-contained):
python examples/correction/example_verification.py

# Correction loop template (dry-run when prerequisites are absent):
python examples/correction/example_run_correction.py
```

See the [Correction & Verification User Guide](../../docs/source/correction_user_guide.md)
for the architecture overview, new-mission checklist, configuration reference,
and troubleshooting guide.
