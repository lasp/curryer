# Curryer Repository Management and DevOps

## Release Process

Curryer is set up with automatic release tooling for your convenience. We use the Github Release tooling and git tags to start the process, and a GitHub actions step (found under release.yml) to build and send the release to PyPi.

### Release Steps

1. [Determine the version for a release](https://semver.org/). We follow standard semantic versioning, meaning a major version update indicates a breaking API change, a minor version update is for backwards compatible functionality updates, and a patch version is for backwards compatible bug fixes or very minor changes.
2. Update the version number within pyproject.toml. At this point, also update the changelog.md file if it is not already updated.
3. On the Curryer GitHub page, click into the Releases page and then hit "Draft new release." This will open your draft release. Start by creating a new tag with your version, then hit "Generate Release Notes" to automatically fill in all the PRs since the last update.
4. Update the release notes with any other information you want to include. Then, submit the release.
5. Verify that the release github action succeeds and that the new version is on Pypi. You're done!

## Test Workflows

In addition to the workflow managing releases, Curryer also has a workflow for running tests.

These tests run in a "matrix" to test multiple versions of Python or multiple operating systems.

Right now, Curryer uses different versions of Python:

`strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]`

Each of these versions gets their own job to run.

It is also possible to only run a certain set of tests (for example very time intensive tests) or to test against other supported operating systems. As an example, [IMAP](https://github.com/IMAP-Science-Operations-Center/imap_processing/blob/cc5f47924db09996d854b1cb3c737b25703f1092/.github/workflows/test.yml) has a more complex version of the same setup if you are looking for future places to expand.

## Dependency Management and Poetry

Curryer uses Poetry for dependency management. This is also used as the build system. All dependency and build configuration is in the pyproject.toml file.

We have 3 sets of optional dependencies for dev, test, and docs.

`dev` is for dependencies that are ONLY used in development, but not in testing. Right now, this is primarily the pre-commit/linted tools.
`docs` is for documentation dependencies.
`test` is for dependencies that are only used in testing.

Adding new optional dependency groups can be useful for large dependencies that are only used for a subset of features, but keep in mind that users may be using pip to install the project. Pip doesn't support optional dependency groups the same way Poetry does, so if you want to provide optional dependency groups to end users you will need to use `extras` instead.

## Linting, Pre-commit, and Codecov

We use several tools for pre-merge gating to help enforce some code quality rules.

For linting, we use `ruff`. This has configuration within the pyproject.toml file for optional exclusions if some of the rules are too harsh, or you can do per-line exclusions. This should be the main style formatter, to ensure we don't conflict.

Also in pre-commit, we have a few other tools for cleaning up code style. These settings are in .pre-commit-config.yaml and run automatically as a check on every PR.

We use codecov for code coverage. This is used as a tool, not a hard enforcer of test, and it is up to individual developers to determine if the coverage is sufficient on a given PR. As such, these checks are not required and can be merged past if they fail.

Finally, we have a documentation build, which checks that the PR documentation works correctly by running a test build in ReadTheDocs. This is managed in .readthedocs.yaml. The Curryer project on readthedocs.org also has some management settings, but broadly speaking this should run with minimal interference.
