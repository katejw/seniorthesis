"""
figures.py

All figure-generating functions. Each saves a .png to the working directory.
Generative AI tools were consulted in the development of this code.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from trial_runners import (
    run_trials, run_trials_local,
    run_trials_decomposed, run_trials_local_decomposed,
)

# ── Aesthetics ────────────────────────────────────────────────────────────────

SEQ_COLORS_LOCAL = ['#AED6F1', '#5DADE2', '#2E86C1', '#1B4F72']

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         12,
    "axes.labelsize":    13,
    "legend.fontsize":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        200,
})


# ── Shared helpers ────────────────────────────────────────────────────────────

def shade(ax, x, means, cis, color, label=None, linestyle='-'):
    ax.plot(x, means, color=color, label=label, linestyle=linestyle,
            marker='o', markersize=4, markevery=2,
            markerfacecolor='white', markeredgewidth=1.2, zorder=3)
    ax.fill_between(x, means - cis, means + cis,
                    color=color, alpha=0.15, linewidth=0, zorder=2)


def finalize_plot(ax, xlabel, ylabel, loca='lower right', show_legend=True):
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=1)
    ax.set_xlim(1, 30)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    if show_legend:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.15)
        ax.legend(loc=loca, frameon=True, framealpha=1.0,
                  facecolor='white', edgecolor='#cccccc', borderaxespad=1.0)
    plt.tight_layout()


# ── Global-norm figures (§5.1, §5.3) ─────────────────────────────────────────

def plot_baseline():
    M_vals = np.arange(1, 31)
    means, cis = zip(*[run_trials(100, M, 0.5)
                       for M in tqdm(M_vals, desc="Baseline")])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    shade(ax, M_vals, np.array(means), np.array(cis), '#5D6D7E')
    ax.axhline(0, color='black', lw=0.8, alpha=0.3)
    finalize_plot(ax, 'Number of Dimensions (M)', 'Average Individual Utility',
                  show_legend=False)
    plt.savefig('fig_baseline.png')
    plt.show()


def plot_decomposition():
    M_vals = np.arange(1, 31)
    totals, r_vals, s_vals, tc, rc, sc = [], [], [], [], [], []
    for M in tqdm(M_vals, desc="Decomposition"):
        m, c = run_trials_decomposed(100, M, 0.5)
        totals.append(m[0]);  r_vals.append(m[1]);  s_vals.append(-m[2])
        tc.append(c[0]);      rc.append(c[1]);       sc.append(c[2])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    shade(ax, M_vals, np.array(r_vals), np.array(rc), '#16A34A', 'Rank Utility')
    shade(ax, M_vals, np.array(s_vals), np.array(sc), '#E67E22', '−Stress Cost')
    shade(ax, M_vals, np.array(totals), np.array(tc), '#5D6D7E', 'Total Utility')
    ax.axhline(0, color='black', lw=0.8, alpha=0.3)
    finalize_plot(ax, 'Number of Dimensions (M)', 'Average Individual Utility',
                  loca='upper right')
    plt.savefig('fig_decomposition.png')
    plt.show()


def plot_lambda():
    M_vals = np.arange(1, 31)
    l_vals = [0.0, 0.3, 0.5, 0.7, 1.0]
    colors = ['#F7DC6F', '#D4AC0D', '#B7950B', '#927709', '#6E5504']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lv, color in zip(l_vals, colors):
        means, cis = zip(*[run_trials(100, M, lv)
                           for M in tqdm(M_vals, desc=f"λ={lv}")])
        shade(ax, M_vals, np.array(means), np.array(cis), color, label=rf'λ={lv}')
    finalize_plot(ax, 'Number of Dimensions (M)', 'Average Individual Utility',
                  loca='upper right')
    plt.savefig('fig_lambda.png')
    plt.show()


def plot_population_size():
    M_vals = np.arange(1, 31)
    N_vals = [10, 100, 500]
    colors = ['#6C3483', '#A569BD', '#D7BDE2']
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for N, color in zip(N_vals, colors):
        means, cis = zip(*[run_trials(N, M, 0.5, n_trials=10)
                           for M in tqdm(M_vals, desc=f"N={N}")])
        shade(ax, M_vals, np.array(means), np.array(cis), color, label=f'N={N}')
    finalize_plot(ax, 'Number of Dimensions (M)', 'Average Individual Utility',
                  loca='lower right')
    plt.savefig('fig_population.png')
    plt.show()


# ── Local-norm figures (§5.2, §5.3) ──────────────────────────────────────────

def plot_neighbor_effects():
    N, lamb   = 100, 0.5
    M_vals    = np.arange(1, 31)
    neighbors = [3, 10, 20, 50]
    fig, ax   = plt.subplots(figsize=(7, 4.5))
    for d, color in zip(neighbors, SEQ_COLORS_LOCAL):
        means, cis = zip(*[run_trials_local(N, M, lamb, d=d)
                           for M in tqdm(M_vals, desc=f'd={d}')])
        shade(ax, M_vals, np.array(means), np.array(cis),
              color=color, linestyle='-', label=f'd = {d}')
    means_g, cis_g = zip(*[run_trials_local(N, M, lamb, d=N)
                            for M in tqdm(M_vals, desc='d=N (Global)')])
    shade(ax, M_vals, np.array(means_g), np.array(cis_g),
          color='#5D6D7E', linestyle='--', label='d = N (Global)')
    ax.axhline(0, color='black', lw=0.8, alpha=0.3)
    finalize_plot(ax, 'Number of Dimensions (M)', 'Average Individual Utility',
                  loca='lower right')
    plt.savefig('fig_neighbor_effects.png', dpi=200)
    plt.show()


def plot_local_decomp_combined():
    """Welfare decomposition for local norm at d=3 and d=50 (two separate files)."""
    N, lamb = 100, 0.5
    M_vals  = np.arange(1, 31)
    configs = [
        (3,  'fig_local_decomp_d3.png'),
        (50, 'fig_local_decomp_d50.png'),
    ]
    for d_val, filename in configs:
        totals, r_vals, s_vals = [], [], []
        tc,     rc,     sc     = [], [], []
        for M in tqdm(M_vals, desc=f'Local decomp d={d_val}'):
            m, c = run_trials_local_decomposed(N, M, lamb, d=d_val)
            totals.append(m[0]);  r_vals.append(m[1]);  s_vals.append(-m[2])
            tc.append(c[0]);      rc.append(c[1]);       sc.append(c[2])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        shade(ax, M_vals, np.array(r_vals), np.array(rc), '#16A34A', 'Rank Utility')
        shade(ax, M_vals, np.array(s_vals), np.array(sc), '#E67E22', '−Stress Cost')
        shade(ax, M_vals, np.array(totals), np.array(tc), '#5D6D7E', 'Total Utility')
        ax.axhline(0, color='black', lw=0.8, alpha=0.3)
        finalize_plot(ax, 'Number of Dimensions (M)', 'Average Individual Utility',
                      loca='center right')
        plt.savefig(filename, dpi=200)
        plt.show()


def plot_lambda_d_interaction():
    M_vals     = np.arange(1, 31)
    N          = 100
    conditions = [
        (0.3,  5, 'Low λ, Local  (λ=0.3, d=5)',  '#D4AF2C', '--'),
        (0.3,  N, 'Low λ, Global (λ=0.3, d=N)',  '#5DADE2', '--'),
        (0.7,  5, 'High λ, Local  (λ=0.7, d=5)', '#927709', '-'),
        (0.7,  N, 'High λ, Global (λ=0.7, d=N)', '#1B4F72', '-'),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for lv, d, label, color, ls in conditions:
        means, cis = zip(*[run_trials_local(N, M, lv, d=d)
                           for M in tqdm(M_vals, desc=label)])
        means, cis = np.array(means), np.array(cis)
        ax.plot(M_vals, means, color=color, label=label, linestyle=ls,
                marker='o', markersize=4, markevery=3,
                markerfacecolor='white', markeredgewidth=1.2, zorder=3)
        ax.fill_between(M_vals, means - cis, means + cis,
                        color=color, alpha=0.12, linewidth=0, zorder=2)
    ax.axhline(0, color='black', lw=0.8, alpha=0.3)
    ax.set_xlim(1, 30)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel('Number of Dimensions (M)', labelpad=10)
    ax.set_ylabel('Average Individual Utility', labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=1)
    ax.legend(loc='lower right', frameon=True, framealpha=1.0,
              facecolor='white', edgecolor='#cccccc')
    plt.tight_layout()
    plt.savefig('fig_lambda_d_interaction.png', dpi=200)
    plt.show()


# ── Appendix B figures ────────────────────────────────────────────────────────


def plot_local_N_sensitivity():
    """Local norm welfare across population sizes N ∈ {10, 100, 500} (Appendix B.2)."""
    M_vals = np.arange(1, 31)
    lamb   = 0.5
    N_vals = [10, 100, 500]
    colors = ['#6C3483', '#A569BD', '#D7BDE2']

    # Fixed d=3
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for N, color in zip(N_vals, colors):
        means, cis = zip(*[run_trials_local(N, M, lamb, d=3, n_trials=10)
                           for M in tqdm(M_vals, desc=f'N={N}, d=3')])
        shade(ax, M_vals, np.array(means), np.array(cis), color=color, label=f'N={N}')
    ax.axhline(0, color='black', lw=0.8, alpha=0.3)
    finalize_plot(ax, 'Number of Dimensions (M)', 'Average Individual Utility',
                  loca='lower right')
    plt.savefig('fig_local_N_sensitivity_d3.png', dpi=200)
    plt.show()

    # d = 10% of N
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for N, color in zip(N_vals, colors):
        d_pct = max(1, round(0.10 * N))
        means, cis = zip(*[run_trials_local(N, M, lamb, d=d_pct, n_trials=10)
                           for M in tqdm(M_vals, desc=f'N={N}, d={d_pct} (10%)')])
        shade(ax, M_vals, np.array(means), np.array(cis),
              color=color, label=f'N={N}, d={d_pct}')
    ax.axhline(0, color='black', lw=0.8, alpha=0.3)
    finalize_plot(ax, 'Number of Dimensions (M)', 'Average Individual Utility',
                  loca='lower right')
    plt.savefig('fig_local_N_sensitivity_d10pct.png', dpi=200)
    plt.show()


# ── Summary table ─────────────────────────────────────────────────────────────

def print_peak_table():
    M_vals = np.arange(1, 31)
    N, lamb = 100, 0.5
    conditions = [
        (N,  'Global norm (d=N)'),
        (3,  'Local norm,  d=3'),
        (10, 'Local norm,  d=10'),
        (50, 'Local norm,  d=50'),
    ]
    peak_results = []
    for d_val, label in conditions:
        means = np.array([
            run_trials_local(N, M, lamb, d=d_val, n_trials=15)[0]
            for M in tqdm(M_vals, desc=label)
        ])
        peak_idx = int(np.argmax(means))
        peak_results.append((label, means[peak_idx], M_vals[peak_idx], d_val))

    global_peak = peak_results[0][1]
    print("\n" + "=" * 70)
    print(f"{'Condition':<25} {'Peak Utility':>14} {'M at Peak':>12} {'% Gain':>12}")
    print("=" * 70)
    for label, peak_val, peak_M, d_val in peak_results:
        gain_str = "—" if d_val == N else f"+{((peak_val - global_peak) / abs(global_peak)) * 100:.0f}%"
        print(f"{label:<25} {peak_val:>14.3f} {peak_M:>12} {gain_str:>12}")
    print("=" * 70 + "\n")
