# Developer Documentation

## Environment Setup

Development environment requires Poetry 2 or greater.

```bash
poetry install
```

## Testing

To run tests using Docker:

```bash
docker build --tag "curryer-tests" --target test . && docker run "curryer-tests"
```

To run locally, you will need to download the third-party files defined in the README, as well as install SPICE binaries.

### Running additional tests

Curryer has some extra tests which rely on DEM files. These do not run by default. To run them, first download the DEM files locally - you can use the [download script](https://github.com/lasp/curryer/blob/main/bin/download_dem.py),
or if you are part of the CLARREO development team, download from the S3 bucket under `clarreo/geolocation/dems`.

Note: these files are temporarily excluded from the repository and not run in our Github actions tests.

To run all the tests that require those files:

```bash
docker build --tag "curryer-tests" --target test . && docker run "curryer-tests" --run-extra
```

## Building Documentation with Sphinx

## Making a Pull Request

## Release Process
