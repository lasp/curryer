# Correction Package Examples

Reference examples and configuration templates for `curryer.correction`.
Copy and adapt them when integrating geolocation correction or verification
into a new mission repository.

**Full documentation:** [`docs/source/correction_user_guide.md`](../../docs/source/correction_user_guide.md)

## Contents

| File                        | Status   | Description                                                                |
| --------------------------- | -------- | -------------------------------------------------------------------------- |
| `example_config.json`       | —        | Generic config template — copy and adapt for your mission                  |
| `clarreo_config.json`       | —        | CLARREO config — loadable by `load_config_from_json()`                     |
| `clarreo_config.py`         | —        | CLARREO config factory using the Python API; use as a new-mission template |
| `example_verification.py`   | Runnable | End-to-end verification demo (real CLARREO data or synthetic fallback)     |
| `example_run_correction.py` | Template | Correction loop template — exits cleanly if data/tools are missing         |
| `regrid_gcp_chips.py`       | Runnable | Batch-regrid HDF GCP chips to NetCDF (see `docs/source/gcp_regridding.md`) |

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
