#!/usr/bin/env python3
"""
Visualization for SEIR Simulation Results

Creates plots showing the epidemic dynamics from the Monte Carlo simulations.
"""

import json
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not installed. Install with: pip install matplotlib")
    print("Falling back to text-based visualization.\n")


def load_results(filename: str = "simulation_results.json") -> dict:
    """Load simulation results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def text_visualization(data: dict):
    """Simple text-based visualization of results."""
    print("\n" + "="*70)
    print("SEIR EPIDEMIC DYNAMICS (Text Visualization)")
    print("="*70)

    n_steps = len(data['times'])
    # Sample every 10th time step for display
    step_size = max(1, n_steps // 20)

    print(f"\n{'Step':>6} {'S':>10} {'E':>10} {'I':>10} {'R':>10}")
    print("-" * 50)

    for i in range(0, n_steps, step_size):
        s = data['S_mean'][i]
        e = data['E_mean'][i]
        inf = data['I_mean'][i]
        r = data['R_mean'][i]
        print(f"{i:>6} {s:>10.1f} {e:>10.1f} {inf:>10.1f} {r:>10.1f}")

    # Show final state
    print("-" * 50)
    print(f"{'Final':>6} {data['S_mean'][-1]:>10.1f} {data['E_mean'][-1]:>10.1f} "
          f"{data['I_mean'][-1]:>10.1f} {data['R_mean'][-1]:>10.1f}")

    # ASCII bar chart for final state
    print("\n\nFinal State Distribution:")
    total = data['S_mean'][-1] + data['E_mean'][-1] + data['I_mean'][-1] + data['R_mean'][-1]

    for state, label in [('S', 'Susceptible'), ('E', 'Exposed'),
                         ('I', 'Infectious'), ('R', 'Recovered')]:
        value = data[f'{state}_mean'][-1]
        pct = value / total * 100
        bar_len = int(pct / 2)
        bar = '#' * bar_len
        print(f"  {label:12} [{bar:<50}] {pct:5.1f}%")

    # Epidemic curve ASCII art
    print("\n\nInfectious Curve (I):")
    max_I = max(data['I_mean'])
    height = 10

    for row in range(height, 0, -1):
        threshold = max_I * row / height
        line = "  "
        for i in range(0, n_steps, max(1, n_steps // 60)):
            if data['I_mean'][i] >= threshold:
                line += "*"
            else:
                line += " "
        print(line)
    print("  " + "-" * min(60, n_steps))
    print("  Time -->")


def plot_results(data: dict, output_file: str = "seir_plot.png"):
    """Create matplotlib visualization of results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SEIR Disease Simulation on Temporal Contact Network\n'
                 '(Replay Framework Demonstration)', fontsize=14, fontweight='bold')

    # Parse times
    times = list(range(len(data['times'])))

    # Colors for SEIR states
    colors = {'S': '#3498db', 'E': '#f39c12', 'I': '#e74c3c', 'R': '#2ecc71'}

    # Plot 1: All states over time
    ax1 = axes[0, 0]
    for state in ['S', 'E', 'I', 'R']:
        mean = data[f'{state}_mean']
        std = data[f'{state}_std']
        ax1.plot(times, mean, label=state, color=colors[state], linewidth=2)
        ax1.fill_between(times,
                         [m - s for m, s in zip(mean, std)],
                         [m + s for m, s in zip(mean, std)],
                         color=colors[state], alpha=0.2)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Number of Individuals')
    ax1.set_title('SEIR Dynamics Over Time')
    ax1.legend(loc='right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Stacked area chart
    ax2 = axes[0, 1]
    ax2.stackplot(times,
                  data['S_mean'], data['E_mean'], data['I_mean'], data['R_mean'],
                  labels=['Susceptible', 'Exposed', 'Infectious', 'Recovered'],
                  colors=[colors['S'], colors['E'], colors['I'], colors['R']],
                  alpha=0.8)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Number of Individuals')
    ax2.set_title('Population Composition Over Time')
    ax2.legend(loc='upper right')

    # Plot 3: Epidemic curve (Infectious only) with uncertainty
    ax3 = axes[1, 0]
    mean_I = data['I_mean']
    std_I = data['I_std']
    ax3.plot(times, mean_I, color=colors['I'], linewidth=2, label='Mean')
    ax3.fill_between(times,
                     [m - 2*s for m, s in zip(mean_I, std_I)],
                     [m + 2*s for m, s in zip(mean_I, std_I)],
                     color=colors['I'], alpha=0.3, label='95% CI')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Infectious Individuals')
    ax3.set_title('Epidemic Curve (Infectious)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Find peak
    peak_idx = mean_I.index(max(mean_I))
    ax3.axvline(x=peak_idx, color='gray', linestyle='--', alpha=0.5)
    ax3.annotate(f'Peak: {max(mean_I):.1f}',
                 xy=(peak_idx, max(mean_I)),
                 xytext=(peak_idx + 5, max(mean_I) * 0.9),
                 fontsize=10)

    # Plot 4: Final state pie chart
    ax4 = axes[1, 1]
    final_values = [data['S_mean'][-1], data['E_mean'][-1],
                    data['I_mean'][-1], data['R_mean'][-1]]
    labels = ['Susceptible', 'Exposed', 'Infectious', 'Recovered']
    pie_colors = [colors['S'], colors['E'], colors['I'], colors['R']]

    # Only show states with non-zero values
    nonzero = [(v, l, c) for v, l, c in zip(final_values, labels, pie_colors) if v > 0.5]
    if nonzero:
        values, labels, pie_colors = zip(*nonzero)
        ax4.pie(values, labels=labels, colors=pie_colors, autopct='%1.1f%%',
                startangle=90, explode=[0.02] * len(values))
    ax4.set_title('Final State Distribution')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    # Also save as HTML using a simple approach
    plt.savefig(output_file.replace('.png', '.svg'), format='svg')
    print(f"SVG saved to: {output_file.replace('.png', '.svg')}")

    return fig


def main():
    print("="*60)
    print("SEIR Simulation Results Visualization")
    print("="*60)

    # Load results
    try:
        data = load_results("simulation_results.json")
        print(f"\nLoaded results with {len(data['times'])} time steps")
    except FileNotFoundError:
        print("\nError: simulation_results.json not found!")
        print("Run seir_contact_simulation.py first to generate results.")
        sys.exit(1)

    # Create visualization
    if HAS_MATPLOTLIB:
        print("\nCreating matplotlib visualization...")
        plot_results(data)
        print("\nTo view the plot, open seir_plot.png or seir_plot.svg")
    else:
        text_visualization(data)

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == "__main__":
    main()
