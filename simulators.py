"""
simulators.py

Simulator classes:
  Simulator — global norm
  LocalSimulator — local norm

Generative AI tools were consulted in the development of this code.
"""

import numpy as np

class Simulator:
    """
    Global-norm, L2 attention constraint (main model).

    N: Number of individuals
    M: Number of dimensions
    Lambda: Social pressure parameter
    """

    def __init__(self, N, M, lamb, seed):
        np.random.seed(seed)
        self.N    = N
        self.M    = M
        self.lamb = lamb
        self.R    = self._rank()
        rand_w    = np.random.rand(N, M)
        self.W    = rand_w / np.linalg.norm(rand_w, axis=1, keepdims=True)
        self._update_norm()

    def _rank(self):
        """Generate ranks via equally-spaced grid on [-1, 1],
        independently shuffled per dimension. See §4.1."""
        ranks = np.zeros((self.N, self.M))
        dist  = np.linspace(-1.0, 1.0, self.N)
        for k in range(self.M):
            col = dist.copy()
            np.random.shuffle(col)
            ranks[:, k] = col
        return ranks

    def _update_norm(self):
        """Calculate social norm (W_bar). See §4.4."""
        avg      = np.mean(self.W, axis=0)
        norm_val = np.linalg.norm(avg)
        self.W_bar = (avg / norm_val if norm_val > 1e-9
                      else np.full(self.M, 1.0 / np.sqrt(self.M)))

    def get_utilities(self):
        u_rank   = (1.0 / np.sqrt(self.M)) * np.sum(self.W * self.R, axis=1)
        u_stress = self.lamb * np.sum((self.W - self.W_bar) ** 2, axis=1)
        return u_rank, u_stress

    def _optimize(self):
        """Optimize using analytic best-response on the
        non-negative unit sphere. See §4.3."""
        g     = (self.R / np.sqrt(self.M)) + 2.0 * self.lamb * self.W_bar
        w_pos = np.maximum(g, 0.0)
        all_neg = np.all(g <= 0, axis=1)
        if all_neg.any():
            rows = np.where(all_neg)[0]
            w_pos[rows, np.argmax(g[rows], axis=1)] = 1.0
        self.W = w_pos / np.linalg.norm(w_pos, axis=1, keepdims=True)

    def run_simulation(self, max_iter=100, tolerance=5e-3):
        """Iterates best-response and norm update until convergence.
        See §4.4."""
        for _ in range(max_iter):
            old_W_bar = self.W_bar.copy()
            self._optimize()
            self._update_norm()
            if np.linalg.norm(self.W_bar - old_W_bar) < tolerance:
                break
        u_rank, u_stress = self.get_utilities()
        return float(np.sum(u_rank - u_stress))

    def run_simulation_decomposed(self, max_iter=100, tolerance=5e-3):
        self.run_simulation(max_iter, tolerance)
        u_rank, u_stress = self.get_utilities()
        total = float(np.sum(u_rank - u_stress))
        return total, float(np.sum(u_rank)), float(np.sum(u_stress))


class LocalSimulator:
    """
    Local-norm, L2 attention constraint (§4.2.2, §5.2).
    Each agent's norm is the normalized mean of its d nearest neighbors.
    """

    def __init__(self, N, M, lamb, d, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.N, self.M, self.lamb = N, M, lamb
        self.d = min(d, N - 1)
        self.R = self._rank()
        rand_w = np.random.rand(N, M)
        self.W = rand_w / np.linalg.norm(rand_w, axis=1, keepdims=True)
        self._local_norms = None

    def _rank(self):
        """Same as Simulator._rank. See §4.1."""
        ranks = np.zeros((self.N, self.M))
        dist  = np.linspace(-1.0, 1.0, self.N)
        for k in range(self.M):
            col = dist.copy()
            np.random.shuffle(col)
            ranks[:, k] = col
        return ranks

    def get_utilities(self):
        u_rank   = (1.0 / np.sqrt(self.M)) * np.sum(self.W * self.R, axis=1)
        u_stress = self.lamb * np.sum((self.W - self._local_norms) ** 2, axis=1)
        return u_rank, u_stress

    def _optimize_local(self):
        """Computes each agent's local norm over d+1 nearest neighbors,
        then applies analytic best-response. See §4.3, §4.5.2."""
        dot      = np.dot(self.W, self.W.T)
        dists_sq = np.maximum(2.0 - 2.0 * dot, 0.0)
        nb_idx   = np.argpartition(dists_sq, self.d, axis=1)[:, :self.d + 1]
        avg      = np.mean(self.W[nb_idx], axis=1)
        mags     = np.linalg.norm(avg, axis=1, keepdims=True)
        self._local_norms = np.where(mags > 1e-9,
                                     avg / mags,
                                     1.0 / np.sqrt(self.M))
        g     = (self.R / np.sqrt(self.M)) + 2.0 * self.lamb * self._local_norms
        w_pos = np.maximum(g, 0.0)
        all_neg = np.all(g <= 0, axis=1)
        if all_neg.any():
            rows = np.where(all_neg)[0]
            w_pos[rows, np.argmax(g[rows], axis=1)] = 1.0
        self.W = w_pos / np.linalg.norm(w_pos, axis=1, keepdims=True)

    def run_simulation(self, max_iter=100, tolerance=5e-3):
        """Iterates best-response and local norm update until convergence.
        See §4.5.2."""
        for _ in range(max_iter):
            old_local_norms = (self._local_norms.copy()
                               if self._local_norms is not None else None)
            self._optimize_local()
            if (old_local_norms is not None and
                np.linalg.norm(self._local_norms - old_local_norms) /
                np.sqrt(self.N) < tolerance):
                break
        u_rank, u_stress = self.get_utilities()
        return float(np.sum(u_rank - u_stress))
