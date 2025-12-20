#!/usr/bin/env python
"""
Test script for mpld3 PR #545: Set `text` attribute to preserve space
https://github.com/mpld3/mpld3/pull/545

This PR fixes a bug where SVG text rendering collapses multiple spaces and
trims leading/trailing whitespace. Matplotlib preserves these spaces, but
SVG's default behavior removes them. PR #545 adds an SVG attribute to text
elements that prevents this space-collapsing behavior.

SETUP INSTRUCTIONS
==================

# From the mpld3_notes directory:
cd /home/abie/ai_assisted_research/mpld3_notes
source mpld3-dev/.venv/bin/activate

# First, test WITHOUT the PR (should show collapsed whitespace):
cd mpld3-dev/mpld3
git fetch origin
git checkout master
make clean && make javascript   # Rebuild JS from source
cd ../..
python -P test_pr545_whitespace_preserve.py

# Then, apply PR #545:
cd mpld3-dev/mpld3
git fetch origin pull/545/head:pr-545
git checkout pr-545
make clean && make javascript   # IMPORTANT: Rebuild JS with PR changes
cd ../..
python -P test_pr545_whitespace_preserve.py

# To return to master:
cd /home/abie/ai_assisted_research/mpld3_notes/mpld3-dev/mpld3
git checkout master
make clean && make javascript   # Rebuild JS for master

EXPECTED BEHAVIOR
=================

WITHOUT PR (master branch):
- Multiple spaces between words will be collapsed to single spaces
- Leading and trailing spaces will be trimmed
- Text alignment using spaces will look wrong

WITH PR (pr-545 branch):
- Multiple spaces between words are preserved
- Leading spaces are preserved
- Text with intentional spacing should render as intended
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
    """Create a figure with various whitespace patterns in text elements."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Test 1: Multiple spaces between words
    ax1 = axes[0, 0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.text(5, 8, 'One    Four    Spaces', ha='center', fontsize=14,
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax1.text(5, 6, 'Two  Spaces  Between', ha='center', fontsize=14,
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax1.text(5, 4, 'Normal Single Spaces', ha='center', fontsize=14,
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax1.text(5, 2, 'Compare: spaces should differ in length above',
             ha='center', fontsize=10, style='italic')
    ax1.set_title('Multiple Spaces Between Words')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Test 2: Leading and trailing spaces
    ax2 = axes[0, 1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.text(5, 8, '    Leading spaces', ha='center', fontsize=14,
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax2.text(5, 6, 'Trailing spaces    ', ha='center', fontsize=14,
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax2.text(5, 4, '    Both ends    ', ha='center', fontsize=14,
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax2.text(5, 2, 'Text position should reflect the spaces',
             ha='center', fontsize=10, style='italic')
    ax2.set_title('Leading/Trailing Spaces')
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Test 3: Alignment with spaces (practical use case)
    ax3 = axes[1, 0]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    # Simulating aligned text using spaces
    ax3.text(1, 8, 'Name:     Alice', ha='left', fontsize=12,
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lavender'))
    ax3.text(1, 6, 'Age:      25', ha='left', fontsize=12,
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lavender'))
    ax3.text(1, 4, 'Location: New York', ha='left', fontsize=12,
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lavender'))
    ax3.text(1, 2, 'Values should be vertically aligned',
             ha='left', fontsize=10, style='italic')
    ax3.set_title('Alignment Using Spaces')
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Test 4: Scatter plot with spaced labels
    ax4 = axes[1, 1]
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 3, 5, 4])
    ax4.scatter(x, y, s=100, c='coral')
    # Labels with intentional spacing
    ax4.text(1, 2.3, 'Point  A', ha='center', fontsize=10)
    ax4.text(2, 4.3, 'Point  B', ha='center', fontsize=10)
    ax4.text(3, 3.3, 'Point  C', ha='center', fontsize=10)
    ax4.text(4, 5.3, 'Point  D', ha='center', fontsize=10)
    ax4.text(5, 4.3, 'Point  E', ha='center', fontsize=10)
    ax4.set_xlabel('X Axis')
    ax4.set_ylabel('Y Axis')
    ax4.set_title('Scatter with Spaced Labels')
    ax4.grid(True, alpha=0.3)

    fig.suptitle('PR #545 Test: Whitespace Preservation in SVG Text\n'
                 'Check if multiple spaces are preserved (not collapsed to single space)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    return fig


def check_whitespace_in_html(html_content):
    """Check if the HTML contains text-rendering attributes for whitespace preservation."""
    # PR #545 should add xml:space="preserve" or similar attribute
    has_preserve = 'xml:space' in html_content or 'white-space' in html_content
    return has_preserve


def main():
    # Print version info
    print("Testing mpld3 PR #545: Whitespace preservation in SVG text")
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

        if "545" in branch:
            print("STATUS: On PR #545 branch - whitespace SHOULD be preserved")
        else:
            print("STATUS: On master/other - whitespace may be collapsed")
    print()

    # Create and export figure
    print("Creating test figure with various whitespace patterns...")
    fig = create_test_figure()

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

    # Check if whitespace preservation is in the HTML
    if check_whitespace_in_html(html_content):
        print("  Found whitespace preservation attributes in output")
    else:
        print("  No whitespace preservation attributes found (PR may not be applied)")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PR #545 Test: Whitespace Preservation</title>
    <script>
    {d3_js}
    </script>
    <script>
    {mpld3_js}
    </script>
    <style>
        .test-info {{
            font-family: monospace;
            margin: 20px;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 5px;
        }}
        .test-info code {{
            background: #e0e0e0;
            padding: 2px 5px;
            border-radius: 3px;
        }}
        /* Force monospace font for all mpld3 text elements */
        .mpld3-text {{
            font-family: "Courier New", Courier, monospace !important;
        }}
    </style>
</head>
<body>
<h2>mpld3 PR #545 Test: Whitespace Preservation in SVG Text</h2>

<div class="test-info">
<p><strong>What to look for:</strong></p>
<ul>
    <li><strong>Top-left:</strong> "One    Four    Spaces" should have 4 spaces visible between words</li>
    <li><strong>Top-left:</strong> "Two  Spaces  Between" should have 2 spaces visible</li>
    <li><strong>Top-right:</strong> Leading/trailing spaces should affect text position in boxes</li>
    <li><strong>Bottom-left:</strong> Values (Alice, 25, New York) should be vertically aligned</li>
</ul>
<p><strong>Without PR #545:</strong> Multiple spaces collapse to single space, alignment breaks</p>
<p><strong>With PR #545:</strong> All spaces are preserved, text looks as intended</p>
</div>

<hr>
{html_content}
</body>
</html>
"""

    # Save to file
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_pr545_output.html')
    with open(output_path, 'w') as f:
        f.write(html)

    print()
    print(f"Saved to: {output_path}")
    print()
    print("TEST: Look for whitespace preservation:")
    print("  - Multiple spaces should NOT collapse to single space")
    print("  - 'One    Four' should have 4 visible spaces")
    print("  - Aligned text should stay aligned")
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
