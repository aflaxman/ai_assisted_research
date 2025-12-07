# mpld3 Development Environment Setup

This guide covers setting up a development environment for [mpld3](https://github.com/mpld3/mpld3) and [mplexporter](https://github.com/mpld3/mplexporter).

## Overview

- **mpld3**: Brings together Matplotlib and D3.js to create interactive web-based visualizations
- **mplexporter**: A general Matplotlib exporter framework used by mpld3

## Prerequisites

- Python 3.8+ (3.11 recommended)
- Git
- [uv](https://github.com/astral-sh/uv) - fast Python package installer
- Node.js and npm (for JavaScript development/building)

## Environment Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create Project Directory and Virtual Environment

```bash
mkdir mpld3-dev && cd mpld3-dev
uv venv --python 3.11
source .venv/bin/activate
```

### 3. Clone the Repositories

```bash
git clone https://github.com/mpld3/mpld3.git
git clone https://github.com/mpld3/mplexporter.git
```

### 4. Install Dependencies

The setup.py files for these packages have build-time dependencies, so we need to install prerequisites first and disable build isolation:

```bash
# Install build dependencies first
uv pip install matplotlib numpy setuptools

# Install mplexporter in editable mode (requires --no-build-isolation)
uv pip install --no-build-isolation -e ./mplexporter

# Install mpld3 in editable mode
uv pip install --no-build-isolation -e ./mpld3

# Install testing and development tools
uv pip install pytest jinja2
```

### 5. Install JavaScript Dependencies (optional, for JS development)

```bash
cd mpld3
npm install
cd ..
```

## Verification

**Important**: You must run Python from **outside** the `mpld3-dev` directory (e.g., from `mpld3_notes/`). The `mpld3-dev/` directory contains the cloned `mpld3/` repo, which shadows the installed package due to Python adding the current directory to `sys.path`.

```bash
# IMPORTANT: run from OUTSIDE mpld3-dev (the parent directory)
cd /path/to/mpld3_notes
source mpld3-dev/.venv/bin/activate
python -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpld3

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
ax.set_title('Test Plot')

html = mpld3.fig_to_html(fig)
print('SUCCESS: mpld3 is working!')
print(f'Generated HTML length: {len(html)} characters')
"
```

## Running Tests

```bash
# From mpld3_notes directory (parent of mpld3-dev)
source mpld3-dev/.venv/bin/activate

# Run mpld3 tests
cd mpld3-dev/mpld3
pytest mpld3/tests/ -v
cd ../..

# Run mplexporter tests
cd mpld3-dev/mplexporter
pytest -v
cd ../..
```

## Testing GitHub Pull Requests

### Testing a Single PR (mpld3 or mplexporter)

```bash
# Start from mpld3_notes directory
cd /path/to/mpld3_notes
source mpld3-dev/.venv/bin/activate

# For an mpld3 PR:
cd mpld3-dev/mpld3
git fetch origin pull/<PR_NUMBER>/head:pr-<PR_NUMBER>
git checkout pr-<PR_NUMBER>
cd ..

# Reinstall to pick up changes
uv pip install --no-build-isolation -e ./mpld3

# Run tests
cd mpld3 && pytest mpld3/tests/ -v && cd ../..
```

```bash
# For an mplexporter PR:
cd mpld3-dev/mplexporter
git fetch origin pull/<PR_NUMBER>/head:pr-<PR_NUMBER>
git checkout pr-<PR_NUMBER>
cd ..

# Reinstall to pick up changes
uv pip install --no-build-isolation -e ./mplexporter

# Run tests
cd mplexporter && pytest -v && cd ../..
```

### Testing Related PRs Together

Sometimes PRs in mpld3 and mplexporter need to be tested together (e.g., when mpld3 changes depend on mplexporter changes):

```bash
# Start from mpld3_notes directory
cd /path/to/mpld3_notes
source mpld3-dev/.venv/bin/activate

# Checkout mplexporter PR first (mpld3 depends on it)
cd mpld3-dev/mplexporter
git fetch origin pull/<MPLEXPORTER_PR>/head:pr-<MPLEXPORTER_PR>
git checkout pr-<MPLEXPORTER_PR>
cd ..

# Checkout mpld3 PR
cd mpld3
git fetch origin pull/<MPLD3_PR>/head:pr-<MPLD3_PR>
git checkout pr-<MPLD3_PR>
cd ..

# Reinstall both in correct order
uv pip install --no-build-isolation -e ./mplexporter
uv pip install --no-build-isolation -e ./mpld3

# Run both test suites
cd mplexporter && pytest -v && cd ..
cd mpld3 && pytest mpld3/tests/ -v && cd ../..
```

### Returning to Main Branch

```bash
cd mpld3-dev/mpld3 && git checkout main && cd ..
cd mplexporter && git checkout master && cd ..

# Reinstall clean versions
uv pip install --no-build-isolation -e ./mplexporter
uv pip install --no-build-isolation -e ./mpld3
cd ..
```

### Using gh CLI (Alternative)

If you have the GitHub CLI installed:

```bash
# Checkout PR directly
cd mpld3-dev/mpld3
gh pr checkout <PR_NUMBER>
cd ..
uv pip install --no-build-isolation -e ./mpld3
cd ..
```

## Building mpld3 JavaScript

```bash
cd mpld3

# Build the JavaScript bundle
npm run build

# Watch for changes during development
npm run watch
```

## Troubleshooting

### Import Errors / `AttributeError: module 'mpld3' has no attribute 'fig_to_html'`
- **Most common cause**: Running Python from inside `mpld3-dev/` directory
- Python adds the current directory (`''`) to `sys.path` first, so the `mpld3/` repo directory shadows the installed package
- **Solution**: Run from `mpld3_notes/` (parent of `mpld3-dev/`), not from inside it
- Ensure the virtual environment is activated: `source mpld3-dev/.venv/bin/activate`

### Build Errors During Install
Make sure to install build dependencies first:
```bash
uv pip install matplotlib numpy setuptools
```

Then use `--no-build-isolation` flag:
```bash
uv pip install --no-build-isolation -e ./mplexporter
```

### JavaScript Not Loading
Rebuild the JS bundle:
```bash
cd mpld3 && npm run build
```

### Matplotlib Backend Issues
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
```

## Resources

- [mpld3 Documentation](http://mpld3.github.io/)
- [mpld3 GitHub](https://github.com/mpld3/mpld3)
- [mplexporter GitHub](https://github.com/mpld3/mplexporter)
- [D3.js Documentation](https://d3js.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
