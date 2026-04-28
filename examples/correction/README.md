# Correction Package Examples

Ready-to-run examples and copy-paste-ready configuration templates for the
`curryer.correction` package.

## Contents

| File                        | Runnable?   | Description                                                                 |
| --------------------------- | ----------- | --------------------------------------------------------------------------- |
| `README.md`                 | —           | This file                                                                   |
| `example_config.json`       | —           | Generic config template — copy and fill in your mission's values            |
| `clarreo_config.json`       | —           | CLARREO-specific config — loadable by `load_config_from_json()`             |
| `clarreo_config.py`         | —           | CLARREO config factory using the Python API                                 |
| `example_verification.py`   | ✅ Yes      | End-to-end verification demo (real CLARREO test data or synthetic fallback) |
| `example_run_correction.py` | ⚠️ Template | Correction loop template — dry-run if SPICE tools / data are missing        |

**✅ Runnable** — works immediately with committed test data, no extra setup.
**⚠️ Template** — shows the API pattern; requires SPICE tools and data not included in the repo.

## Quick Start

### Verification (runs immediately)

```bash
# From the repo root:
python examples/correction/example_verification.py
```

Loads real CLARREO image-matching test data from `tests/data/clarreo/image_match/`
when available, or falls back to synthetic data automatically. Always produces
output so you can see the full API in action.

### Correction Loop (template — requires external data)

```bash
python examples/correction/example_run_correction.py
# Or with your config file:
python examples/correction/example_run_correction.py \
    --config examples/correction/clarreo_config.json
```

The correction loop requires:

- Preprocessed telemetry + science CSV files
- GCP `.mat` image-chip files
- Generic SPICE kernels in `data/generic/` (see `docs/source/users.md` for download links)
- SPICE kernel creation tools (`mkspk`, `msopck`) on `PATH`

Without these the script runs in **dry-run** mode and prints the API pattern.

For CLARREO, raw telemetry must first be preprocessed:

```bash
python scripts/clarreo_preprocess.py --help
```

## Prerequisites

Install `lasp-curryer` with test and dev extras:

```bash
pip install -e ".[test,dev]"
```

All runtime dependencies (including `scipy`) are declared in `pyproject.toml`.
No extra `pip install` is needed.

For the CLARREO verification example you also need:

- `tests/data/clarreo/image_match/` — the `.mat` image-chip data files
  (committed to this repo under `tests/data/`)

## Using a Config File

Copy `example_config.json` and fill in your mission's values:

```python
from curryer.correction import load_config_from_json

config = load_config_from_json("examples/correction/my_mission_config.json")
print(config.n_iterations)
print(config.performance_threshold_m)
```

Required top-level JSON sections: `mission_config`, `correction`, `geolocation`.

## Further Reading

- **User Guide:** `docs/source/correction_user_guide.md`
- **API Reference:** `curryer/correction/__init__.py` (`__all__` list)
- **CLARREO test fixtures:** `tests/test_correction/clarreo/`
- **CLARREO operational script:** `scripts/clarreo_weekly_verification.py`
