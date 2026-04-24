# Publishing Guide

This guide covers automated and manual publishing for `RHEED-tools`.

## Automated Release

The workflow `.github/workflows/main.yml` releases from `main` when the head
commit message contains one of these tokens:

- `#major`
- `#minor`
- `#patch`

The action runs a build check, bumps the version in `pyproject.toml` and
`src/rheed_tools/__init__.py`, creates a release commit, tags `vX.Y.Z`, builds,
and publishes to PyPI.

## PyPI Trusted Publishing

Configure PyPI Trusted Publishing with:

- Repository: `yig319/RHEED-tools`
- Workflow: `main.yml`
- Environment name: any / unset

## Local Release Check

```bash
pip install -r requirements-dev.txt
pip install -e ".[dev]"
pytest -q
python -m build
twine check dist/*
```

## Manual Upload

```bash
twine upload dist/*
```

## Post-Release Check

```bash
pip install --upgrade RHEED-tools
python -c "import rheed_tools; print(rheed_tools.__version__)"
```
