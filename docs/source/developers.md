# Developer Documentation

## Environment Setup

Development environment requires Poetry 2 or greater. To install all dev dependencies using Poetry:

```bash
poetry install --all-extras
```

We strongly recommend that Poetry is used for all development. However, you can install using Pip:

```bash
pip install .[dev,test]
# If you're using zsh, instead:
pip install .'[dev,test]'
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

Curryer is set up with automatic release tooling for your convenience. We use the Github Release tooling and git tags to start the process, and a GitHub actions step (found under release.yml) to build and send the release to PyPi.

### Release Steps

1. [Determine the version for a release](https://semver.org/). We follow standard semantic versioning, meaning a major version update indicates a breaking API change, a minor version update is for backwards compatible functionality updates, and a patch version is for backwards compatible bug fixes or very minor changes.
2. Update the version number within pyproject.toml. At this point, also update the changelog.md file if it is not already updated.
3. On the Curryer GitHub page, click into the Releases page and then hit "Draft new release." This will open your draft release. Start by creating a new tag with your version, then hit "Generate Release Notes" to automatically fill in all the PRs since the last update.
4. Update the release notes with any other information you want to include. Then, submit the release.
5. Verify that the release github action succeeds and that the new version is on Pypi. You're done!
