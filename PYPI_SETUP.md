# PyPI Publishing Setup

This document explains how to set up automatic publishing to PyPI for SpiPy releases.

## Overview

The repository includes a GitHub Actions workflow (`.github/workflows/publish-pypi.yml`) that automatically builds and publishes the package to PyPI when you create a GitHub release.

## Setup Steps

### 1. Register on PyPI

1. Create an account at [https://pypi.org/account/register/](https://pypi.org/account/register/)
2. Verify your email address
3. Enable 2FA (required for new projects)

### 2. Set Up Trusted Publishing (Recommended)

**Trusted Publishing** is the modern, secure way to publish to PyPI without API tokens. It uses OpenID Connect (OIDC) to verify that the package is being uploaded from an authorized GitHub Actions workflow.

#### On PyPI:

1. Log into [https://pypi.org](https://pypi.org)
2. Go to "Your projects" → "Publishing" (or go directly to [https://pypi.org/manage/account/publishing/](https://pypi.org/manage/account/publishing/))
3. Click "Add a new pending publisher"
4. Fill in the form:
   - **PyPI Project Name**: `spires`
   - **Owner**: `edwardbair`
   - **Repository name**: `SpiPy`
   - **Workflow name**: `publish-pypi.yml`
   - **Environment name**: (leave empty)
5. Click "Add"

#### On GitHub:

No additional setup needed! The workflow already has the required `permissions: id-token: write`.

### 3. Alternative: API Token (Legacy Method)

If you prefer using an API token instead of trusted publishing:

1. Go to [https://pypi.org/manage/account/token/](https://pypi.org/manage/account/token/)
2. Click "Add API token"
3. Name it (e.g., "GitHub Actions - SpiPy")
4. Scope: "Project: spires" (or "Entire account" if the project doesn't exist yet)
5. Copy the token (starts with `pypi-`)

Then add it to GitHub:

1. Go to the GitHub repository settings
2. Navigate to "Secrets and variables" → "Actions"
3. Click "New repository secret"
4. Name: `PYPI_API_TOKEN`
5. Value: paste your PyPI token
6. Click "Add secret"

Finally, modify `.github/workflows/publish-pypi.yml`:

```yaml
- name: Publish to PyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    password: ${{ secrets.PYPI_API_TOKEN }}  # Add this line
    verbose: true
```

## Testing with TestPyPI (Recommended)

Before publishing to the real PyPI, test with TestPyPI:

1. Create an account at [https://test.pypi.org](https://test.pypi.org)
2. Set up trusted publishing there (same steps as above)
3. In `.github/workflows/publish-pypi.yml`, uncomment the line:
   ```yaml
   repository-url: https://test.pypi.org/legacy/
   ```
4. Create a test release
5. Verify the package appears at `https://test.pypi.org/project/spires/`
6. Try installing: `pip install --index-url https://test.pypi.org/simple/ spires`
7. Once confirmed working, remove the `repository-url` line to publish to real PyPI

## Publishing a Release

Once setup is complete, publishing is automatic:

1. **Create a git tag:**
   ```bash
   git tag v0.2.2
   git push upstream v0.2.2
   ```

2. **Create a GitHub Release:**
   - Go to [https://github.com/edwardbair/SpiPy/releases/new](https://github.com/edwardbair/SpiPy/releases/new)
   - Choose your tag (e.g., `v0.2.2`)
   - Fill in release notes
   - Click "Publish release"

3. **Automatic workflow:**
   - GitHub Actions will automatically trigger
   - Builds wheels for Linux and macOS
   - Builds source distribution
   - Publishes to PyPI
   - Check progress at: [https://github.com/edwardbair/SpiPy/actions](https://github.com/edwardbair/SpiPy/actions)

4. **Verify publication:**
   - Package appears at: [https://pypi.org/project/spires/](https://pypi.org/project/spires/)
   - Install with: `pip install spires`

## Workflow Files

- **`.github/workflows/publish-pypi.yml`**: Builds wheels and publishes to PyPI on releases
- **`.github/workflows/build.yml`**: Tests builds on every push/PR (does not publish)
- **`.github/workflows/docs.yml`**: Builds documentation

## Build Details

The workflow uses:

- **cibuildwheel**: Builds binary wheels for multiple Python versions and platforms
- **setuptools-scm**: Automatically determines version from git tags
- **SWIG**: Generates Python bindings for C++ code
- **nlopt**: Required C++ dependency for optimization

Wheels are built for:
- Python 3.9, 3.10, 3.11, 3.12
- Linux (manylinux)
- macOS (ARM64 and x86_64)

## Troubleshooting

### Build fails with "nlopt not found"

The workflow installs nlopt automatically. If builds fail:
- Check system package installation in workflow logs
- Verify nlopt is available: `yum list nlopt-devel` (Linux) or `brew info nlopt` (macOS)

### "Filename already exists" error

You're trying to upload a version that already exists on PyPI. Versions are immutable:
- Delete the git tag: `git tag -d v0.2.2 && git push --delete upstream v0.2.2`
- Create a new version: `git tag v0.2.3`
- Or use post-release versioning (automatic with setuptools-scm)

### Testing locally

Build and test the package locally:

```bash
# Build wheel
pip install build
python -m build --wheel

# Install and test
pip install dist/spires-*.whl
python -c "import spires; print(spires.__version__)"
```

## Resources

- [PyPI Trusted Publishing Guide](https://docs.pypi.org/trusted-publishers/)
- [cibuildwheel documentation](https://cibuildwheel.readthedocs.io/)
- [setuptools-scm](https://github.com/pypa/setuptools_scm)
