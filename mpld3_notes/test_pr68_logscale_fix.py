#!/usr/bin/env python
"""
Test script for mplexporter PR #68: Fix broken scatter/fill_between in log-scale axis
https://github.com/mpld3/mplexporter/pull/68

This PR fixes a bug where scatter plots and fill_between visualizations render
incorrectly on logarithmic-scale axes. Before the fix, these elements would not
respond properly to zooming and panning in the exported visualization.

SETUP INSTRUCTIONS
==================

# From the mpld3_notes directory:
cd /home/abie/ai_assisted_research/mpld3_notes
source mpld3-dev/.venv/bin/activate

# First, test WITHOUT the PR (should show broken behavior):
# Using specific commit SHAs so this works even after PR is merged
cd mpld3-dev/mplexporter
git fetch origin
git checkout 967e0a4   # master before PR #68
cd ../mpld3
git fetch origin
git checkout 3aad00b   # main as of 2024-12-07
cd ..
uv pip install --no-build-isolation -e ./mplexporter
uv pip install --no-build-isolation -e ./mpld3
cd ..
python -P test_pr68_logscale_fix.py

# Then, apply the PR and test again (should show fixed behavior):
cd mpld3-dev/mplexporter
git fetch origin pull/68/head:pr-68
git checkout pr-68
cd ..
uv pip install --no-build-isolation -e ./mplexporter
cd ..
python -P test_pr68_logscale_fix.py

# Alternative using gh CLI (if authenticated):
# cd mpld3-dev/mplexporter
# gh pr checkout 68
# cd ..
# uv pip install --no-build-isolation -e ./mplexporter

EXPECTED BEHAVIOR
=================

WITHOUT PR (master branch):
- The scatter points and fill_between region may not zoom/pan correctly
- Coordinates may be in wrong space (display vs data)

WITH PR (pr-68 branch):
- Scatter points and fill_between should zoom and pan correctly
- Interactive controls should work as expected on log-scale axes
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import mpld3
import webbrowser
import tempfile
import os

def create_test_figure():
    """Create a figure with scatter and fill_between on log-scale axes."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Test 1: Scatter on log-log axes
    ax1 = axes[0]
    x = np.logspace(0, 3, 50)  # 1 to 1000
    y = x ** 1.5 * (1 + 0.3 * np.random.randn(50))
    ax1.scatter(x, y, c=np.log10(x), cmap='viridis', s=50, alpha=0.7)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('X (log scale)')
    ax1.set_ylabel('Y (log scale)')
    ax1.set_title('Scatter on Log-Log Axes')
    ax1.grid(True, alpha=0.3)

    # Test 2: fill_between on semilog-y axes
    ax2 = axes[1]
    x = np.linspace(0, 10, 100)
    y = np.exp(x / 3)
    y_lower = y * 0.7
    y_upper = y * 1.3
    ax2.fill_between(x, y_lower, y_upper, alpha=0.3, label='Confidence band')
    ax2.plot(x, y, 'b-', linewidth=2, label='Mean')
    ax2.set_yscale('log')
    ax2.set_xlabel('X (linear)')
    ax2.set_ylabel('Y (log scale)')
    ax2.set_title('fill_between on Semilog-Y')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Test 3: Scatter on semilog-x axes
    ax3 = axes[2]
    x = np.logspace(-1, 2, 40)  # 0.1 to 100
    y = np.sin(np.log10(x) * np.pi) + np.random.randn(40) * 0.1
    colors = np.random.rand(40)
    sizes = np.random.rand(40) * 100 + 20
    ax3.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='plasma')
    ax3.set_xscale('log')
    ax3.set_xlabel('X (log scale)')
    ax3.set_ylabel('Y (linear)')
    ax3.set_title('Scatter on Semilog-X')
    ax3.grid(True, alpha=0.3)

    fig.suptitle('PR #68 Test: Log-scale Scatter and fill_between\n'
                 'Try zooming and panning - elements should follow correctly',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    return fig

def main():
    # Print version info
    print("Testing mplexporter PR #68: Log-scale scatter/fill_between fix")
    print("=" * 60)

    try:
        import mplexporter
        exporter_path = os.path.dirname(mplexporter.__file__)
        print(f"mplexporter location: {exporter_path}")
    except ImportError:
        print("ERROR: mplexporter not installed")
        return

    print(f"mpld3 version: {mpld3.__version__}")
    print(f"matplotlib version: {matplotlib.__version__}")
    print()

    # Check which branch mplexporter is on
    mplexporter_git_dir = os.path.join(os.path.dirname(exporter_path), '.git')
    if os.path.exists(mplexporter_git_dir):
        import subprocess
        result = subprocess.run(
            ['git', 'branch', '--show-current'],
            cwd=os.path.dirname(exporter_path),
            capture_output=True, text=True
        )
        branch = result.stdout.strip() or "(detached HEAD)"
        print(f"mplexporter git branch: {branch}")
        if branch == "master":
            print("WARNING: On master branch - this is BEFORE the fix")
        elif "68" in branch or "pr-68" in branch:
            print("INFO: On PR branch - this should have the fix")
    print()

    # Create and export figure
    print("Creating test figure...")
    fig = create_test_figure()

    # Generate HTML using local dev JS files (not CDN)
    # This ensures we test with the actual dev code, not released versions
    print("Exporting to HTML with mpld3...")

    # Find the local JS files
    mpld3_dir = os.path.dirname(mpld3.__file__)
    d3_js_path = os.path.join(mpld3_dir, 'js', 'd3.v5.min.js')
    mpld3_js_path = os.path.join(mpld3_dir, 'js', 'mpld3.v0.5.13-dev.js')

    # Read the JS files and embed inline
    with open(d3_js_path, 'r') as f:
        d3_js = f.read()
    with open(mpld3_js_path, 'r') as f:
        mpld3_js = f.read()

    # Generate HTML without library includes, then inject our own
    html = mpld3.fig_to_html(fig, include_libraries=False)

    # Wrap with inline JS
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PR #68 Test: Log-scale scatter/fill_between</title>
    <script>
    {d3_js}
    </script>
    <script>
    {mpld3_js}
    </script>
</head>
<body>
{html}
</body>
</html>
"""

    # Save to the mpld3_notes directory
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_pr68_output.html')
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Saved to: {output_path}")
    print()
    print("TEST: Try zooming and panning on each subplot.")
    print("  - WITHOUT the fix: scatter points/fill regions may not follow correctly")
    print("  - WITH the fix: all elements should zoom and pan together")
    print()

    # Try to open in browser (with WSL support)
    opened = False

    # Check if running on WSL
    is_wsl = 'microsoft' in os.uname().release.lower() if hasattr(os, 'uname') else False

    if is_wsl:
        # On WSL, use explorer.exe to open in Windows browser
        import subprocess
        # Convert Linux path to Windows path
        try:
            result = subprocess.run(
                ['wslpath', '-w', output_path],
                capture_output=True, text=True
            )
            win_path = result.stdout.strip()
            subprocess.run(['explorer.exe', win_path], check=False)
            print(f"Opened in Windows browser via explorer.exe")
            opened = True
        except Exception as e:
            print(f"Could not open via WSL: {e}")

    if not opened:
        try:
            webbrowser.open(f'file://{output_path}')
            print("Opened in browser")
            opened = True
        except Exception as e:
            print(f"Could not open browser: {e}")

    if not opened:
        print(f"\nManually open this file in your browser:\n  {output_path}")

    plt.close(fig)

if __name__ == '__main__':
    main()
