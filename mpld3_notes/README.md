# mpld3 Development Environment Setup

This guide covers setting up a development environment for [mpld3](https://github.com/mpld3/mpld3).

## Overview

- **mpld3**: Brings together Matplotlib and D3.js to create interactive web-based visualizations
- **mplexporter**: A general Matplotlib exporter framework. **Note:** mpld3 includes a bundled copy at `mpld3/mplexporter/` - the standalone [mplexporter repo](https://github.com/mpld3/mplexporter) is only needed when testing mplexporter PRs.

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

### 3. Clone Repositories

```bash
git clone https://github.com/mpld3/mpld3.git
git clone https://github.com/mpld3/mplexporter.git
```

**Note:** mpld3 bundles its own copy of mplexporter at `mpld3/mpld3/mplexporter/`. The standalone mplexporter clone is needed when testing mplexporter PRs - you'll copy files from it to the bundled location.

### 4. Install Dependencies

```bash
# Install build dependencies first
uv pip install matplotlib numpy setuptools

# Install mpld3 in editable mode
uv pip install --no-build-isolation -e ./mpld3

# Install testing and development tools
uv pip install pytest jinja2 pandas
```

### 5. Install JavaScript Dependencies (optional, for JS development)

```bash
cd mpld3
npm install
cd ..
```

## Verification

**Important**: You can avoid path shadowing issues by using `python -P` which disables adding the current directory to `sys.path`.

```bash
cd /home/abie/ai_assisted_research/mpld3_notes/mpld3-dev
source .venv/bin/activate
python -P -c "
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
cd /home/abie/ai_assisted_research/mpld3_notes/mpld3-dev
source .venv/bin/activate

# Run mpld3 tests
cd mpld3
python -P -m pytest mpld3/tests/ -v
cd ..
```

## Testing GitHub Pull Requests

### Testing an mpld3 PR

```bash
cd /home/abie/ai_assisted_research/mpld3_notes/mpld3-dev
source .venv/bin/activate

cd mpld3
git fetch origin pull/<PR_NUMBER>/head:pr-<PR_NUMBER>
git checkout pr-<PR_NUMBER>
cd ..

# Reinstall to pick up changes
uv pip install --no-build-isolation -e ./mpld3

# Run tests
cd mpld3 && python -P -m pytest mpld3/tests/ -v && cd ..
```

### Testing an mplexporter PR

**Important:** mpld3 uses a **bundled copy** of mplexporter at `mpld3/mpld3/mplexporter/`. The standalone mplexporter package is NOT used by mpld3. To test mplexporter PRs, you must copy files from the standalone clone to mpld3's bundled copy.

```bash
cd /home/abie/ai_assisted_research/mpld3_notes/mpld3-dev
source .venv/bin/activate

# Fetch and checkout the PR
cd mplexporter
git fetch origin pull/<PR_NUMBER>/head:pr-<PR_NUMBER>
git checkout pr-<PR_NUMBER>
cd ..

# Copy the changed files to mpld3's bundled mplexporter
cp mplexporter/mplexporter/exporter.py mpld3/mpld3/mplexporter/exporter.py
# (copy other changed files as needed, e.g., renderers/base.py, utils.py)

# Test
cd mpld3 && python -P -m pytest mpld3/tests/ -v && cd ..
```

### Restoring mpld3's Bundled mplexporter

```bash
cd mpld3
git checkout -- mpld3/mplexporter/
cd ..
```

### Using gh CLI (Alternative)

If you have the GitHub CLI installed:

```bash
cd mpld3
gh pr checkout <PR_NUMBER>
cd ..
uv pip install --no-build-isolation -e ./mpld3
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
- **Cause**: The `mpld3/` repo directory shadows the installed package
- **Solution**: Use `python -P` to disable adding current directory to `sys.path`
- Ensure the virtual environment is activated: `source .venv/bin/activate`

### Build Errors During Install
Make sure to install build dependencies first:
```bash
uv pip install matplotlib numpy setuptools
```

Then use `--no-build-isolation` flag:
```bash
uv pip install --no-build-isolation -e ./mpld3
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
- [mplexporter GitHub](https://github.com/mpld3/mplexporter) (standalone repo, mpld3 bundles its own copy)
- [D3.js Documentation](https://d3js.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
