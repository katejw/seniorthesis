"""
trial_runners.py

Runs replicated trials and returns (mean, 95% CI half-width). See  §4.6.
Generative AI tools were consulted in the development of this code.
"""

import numpy as np
from simulators import Simulator, LocalSimulator

N_TRIALS = 50
Z95      = 1.96


def run_trials(N, M, lamb, n_trials=N_TRIALS):
    """Runs n_trials replications of the global-norm simulator.
    Returns (mean, 95% CI half-width)."""
    results = np.array([Simulator(N, M, lamb, s).run_simulation() / N
                        for s in range(n_trials)])
    return results.mean(), Z95 * results.std(ddof=1) / np.sqrt(n_trials)
    

def run_trials_local(N, M, lamb, d, n_trials=N_TRIALS):
    """Same as run_trials but for the local-norm simulator; adds neighbor count d."""
    results = np.array([LocalSimulator(N, M, lamb, d, s).run_simulation() / N
                        for s in range(n_trials)])
    return results.mean(), Z95 * results.std(ddof=1) / np.sqrt(n_trials)


def run_trials_decomposed(N, M, lamb, n_trials=N_TRIALS):
    """Returns decomposed (total, rank utility, stress cost) means and CIs."""
    results = []
    for s in range(n_trials):
        t, r, st = Simulator(N, M, lamb, s).run_simulation_decomposed()
        results.append([t / N, r / N, st / N])
    arr = np.array(results)
    return arr.mean(axis=0), Z95 * arr.std(axis=0, ddof=1) / np.sqrt(n_trials)


def run_trials_local_decomposed(N, M, lamb, d, n_trials=N_TRIALS):
    """Same as run_trials_decomposed but for the local-norm simulator."""
    results = []
    for s in range(n_trials):
        sim = LocalSimulator(N, M, lamb, d, s)
        sim.run_simulation()
        u_rank, u_stress = sim.get_utilities()
        results.append([
            float(np.sum(u_rank - u_stress)) / N,
            float(np.sum(u_rank))            / N,
            float(np.sum(u_stress))          / N,
        ])
    arr = np.array(results)
    return arr.mean(axis=0), Z95 * arr.std(axis=0, ddof=1) / np.sqrt(n_trials)
