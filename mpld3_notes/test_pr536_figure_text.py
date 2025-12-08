#!/usr/bin/env python
"""
Test script for mpld3 PR #536: Render figure-level text objects
https://github.com/mpld3/mpld3/pull/536

This PR fixes a bug where figure-level text elements like fig.suptitle() and
fig.text() were not appearing in mpld3 visualizations.

Fixes issue #296: https://github.com/mpld3/mpld3/issues/296

SETUP INSTRUCTIONS
==================

IMPORTANT: This fix requires BOTH:
  - mpld3 PR #536: https://github.com/mpld3/mpld3/pull/536
  - mplexporter PR #70: https://github.com/mpld3/mplexporter/pull/70

Since mpld3 bundles mplexporter, you must apply PR #70 to the bundled copy.

# From the mpld3_notes directory:
cd /home/abie/ai_assisted_research/mpld3_notes
source mpld3-dev/.venv/bin/activate

# First, test WITHOUT the PRs (should show missing figure text):
cd mpld3-dev/mpld3
git fetch origin
git checkout 3aad00b   # main as of 2024-12-07, before PR #536
git checkout -- mpld3/mplexporter/  # restore bundled mplexporter
cd ../..
python -P test_pr536_figure_text.py

# Then, apply BOTH PRs:
# 1. Apply mpld3 PR #536
cd mpld3-dev/mpld3
git fetch origin pull/536/head:pr-536
git checkout pr-536
cd ..

# 2. Apply mplexporter PR #70 to the BUNDLED mplexporter
cd mplexporter
git fetch origin pull/70/head:pr-70
git checkout pr-70
cd ..
cp mplexporter/mplexporter/exporter.py mpld3/mpld3/mplexporter/exporter.py
cp mplexporter/mplexporter/renderers/base.py mpld3/mpld3/mplexporter/renderers/base.py
cp mplexporter/mplexporter/renderers/fake_renderer.py mpld3/mpld3/mplexporter/renderers/fake_renderer.py
cp mplexporter/mplexporter/utils.py mpld3/mpld3/mplexporter/utils.py
cd ..

# Test with both PRs applied:
python -P test_pr536_figure_text.py

# To return to master:
cd /home/abie/ai_assisted_research/mpld3_notes/mpld3-dev/mpld3
git checkout master
git checkout -- mpld3/mplexporter/ 2>/dev/null || true  # restore bundled copy if modified
cd ../mplexporter
git checkout master

EXPECTED BEHAVIOR
=================

WITHOUT PR (main branch):
- fig.suptitle() text will NOT appear in the rendered HTML
- fig.text() annotations will NOT appear
- Only axes-level text (ax.set_title, ax.set_xlabel, etc.) will render

WITH PR (pr-536 branch):
- fig.suptitle() should appear at the top of the figure
- fig.text() annotations should appear at specified figure coordinates
- All text elements should be visible and properly positioned
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import mpld3
import os
import webbrowser
import subprocess


def create_test_figure():
    """Create a figure with various figure-level and axes-level text elements."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Add figure-level suptitle (THIS IS WHAT PR #536 FIXES)
    fig.suptitle('DARK BLUE: Figure Suptitle - Should Appear With PR #536',
                 fontsize=16, fontweight='bold', color='darkblue')

    # Add figure-level text annotations (ALSO FIXED BY PR #536)
    fig.text(0.5, 0.02, 'DARK GREEN: Figure-level text at bottom (fig.text)',
             ha='center', fontsize=10, style='italic', color='darkgreen')
    fig.text(0.02, 0.5, 'DARK RED: Left side fig.text',
             va='center', rotation=90, fontsize=10, color='darkred')
    fig.text(0.98, 0.5, 'PURPLE: Right side fig.text',
             va='center', rotation=-90, fontsize=10, color='purple')

    # Left subplot - line plot
    ax1 = axes[0]
    x = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(x, np.sin(x), 'b-', label='sin(x)')
    ax1.plot(x, np.cos(x), 'r--', label='cos(x)')
    ax1.set_xlabel('X axis label (axes-level)')
    ax1.set_ylabel('Y axis label (axes-level)')
    ax1.set_title('Axes Title (axes-level) - Always Works')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right subplot - scatter plot
    ax2 = axes[1]
    np.random.seed(42)
    x = np.random.randn(50)
    y = np.random.randn(50)
    colors = np.random.rand(50)
    ax2.scatter(x, y, c=colors, cmap='viridis', s=100, alpha=0.6)
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_title('Scatter Plot (axes-level title)')

    # Add axes-level text annotation (this should always work)
    ax2.text(0, 0, 'ax.text (axes-level)\nAlways works',
             ha='center', va='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0.03, 0.05, 0.97, 0.95])  # Leave room for figure text

    return fig


def check_figure_text_in_dict(fig_dict):
    """Check if figure-level text appears in the exported dict."""
    # Look for texts at the figure level (not inside axes)
    figure_texts = fig_dict.get('texts', [])
    axes_texts = []
    for ax in fig_dict.get('axes', []):
        axes_texts.extend(ax.get('texts', []))

    return {
        'figure_level_texts': len(figure_texts),
        'axes_level_texts': len(axes_texts),
        'has_figure_texts': len(figure_texts) > 0
    }


def main():
    # Print version info
    print("Testing mpld3 PR #536: Render figure-level text objects")
    print("=" * 60)

    print(f"mpld3 version: {mpld3.__version__}")
    print(f"matplotlib version: {matplotlib.__version__}")

    # Check which branch mpld3 is on
    mpld3_dir = os.path.dirname(os.path.dirname(mpld3.__file__))
    git_dir = os.path.join(mpld3_dir, '.git')
    if os.path.exists(git_dir):
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=mpld3_dir,
            capture_output=True, text=True
        )
        commit = result.stdout.strip()

        result2 = subprocess.run(
            ['git', 'branch', '--show-current'],
            cwd=mpld3_dir,
            capture_output=True, text=True
        )
        branch = result2.stdout.strip() or f"(detached at {commit})"
        print(f"mpld3 git: {branch} ({commit})")

        if commit == "3aad00b" or branch == "main":
            print("STATUS: On main/original - figure text will NOT appear")
        elif "536" in branch:
            print("STATUS: On PR #536 branch - figure text SHOULD appear")
    print()

    # Create and export figure
    print("Creating test figure with fig.suptitle() and fig.text()...")
    fig = create_test_figure()

    # Check what gets exported
    print("Exporting to dict to check figure-level text...")
    fig_dict = mpld3.fig_to_dict(fig)
    text_info = check_figure_text_in_dict(fig_dict)

    print(f"  Figure-level texts found: {text_info['figure_level_texts']}")
    print(f"  Axes-level texts found: {text_info['axes_level_texts']}")
    if text_info['has_figure_texts']:
        print("  ✓ Figure-level text IS being exported (PR #536 fix working)")
    else:
        print("  ✗ Figure-level text NOT exported (PR #536 fix NOT applied)")
    print()

    # Generate HTML using local dev JS files
    print("Exporting to HTML with mpld3...")
    mpld3_dir = os.path.dirname(mpld3.__file__)
    d3_js_path = os.path.join(mpld3_dir, 'js', 'd3.v5.min.js')
    mpld3_js_path = os.path.join(mpld3_dir, 'js', 'mpld3.v0.5.13-dev.js')

    with open(d3_js_path, 'r') as f:
        d3_js = f.read()
    with open(mpld3_js_path, 'r') as f:
        mpld3_js = f.read()

    html_content = mpld3.fig_to_html(fig, include_libraries=False)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PR #536 Test: Figure-level Text</title>
    <script>
    {d3_js}
    </script>
    <script>
    {mpld3_js}
    </script>
</head>
<body>
<h2>mpld3 PR #536 Test: Figure-level Text Objects</h2>
<p><strong>What to look for:</strong></p>
<ul>
    <li>Blue "Figure Suptitle" at the top (fig.suptitle)</li>
    <li>Green italic text at the bottom (fig.text)</li>
    <li>Red rotated text on the left side (fig.text)</li>
    <li>Purple rotated text on the right side (fig.text)</li>
</ul>
<p><em>If these are missing, the PR fix is not applied.</em></p>
<hr>
{html_content}
</body>
</html>
"""

    # Save to file
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_pr536_output.html')
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Saved to: {output_path}")
    print()
    print("TEST: Look for figure-level text elements:")
    print("  - Blue suptitle at top")
    print("  - Green text at bottom")
    print("  - Red/purple rotated text on sides")
    print()

    # Try to open in browser (with WSL support)
    opened = False
    is_wsl = 'microsoft' in os.uname().release.lower() if hasattr(os, 'uname') else False

    if is_wsl:
        try:
            result = subprocess.run(
                ['wslpath', '-w', output_path],
                capture_output=True, text=True
            )
            win_path = result.stdout.strip()
            subprocess.run(['explorer.exe', win_path], check=False)
            print("Opened in Windows browser via explorer.exe")
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
