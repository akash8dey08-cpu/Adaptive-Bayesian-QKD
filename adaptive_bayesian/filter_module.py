import numpy as np
from scipy.stats import binom


class BayesianDriftFilter:
    

    def __init__(
        self,
        sigma_drift: float = 0.05,
        depol_prob: float = 0.015,
        drift_bias: float = 0.018,
        init_theta: float = 0.0,
        init_std: float = 0.05,
        grid_min: float = -np.pi,
        grid_max: float = np.pi,
        grid_size: int = 601,
        x_basis_only: bool = True,
    ):
        
        self.sigma_drift = float(sigma_drift)
        self.depol_prob = float(depol_prob)
        self.e_0 = (2.0 / 3.0) * self.depol_prob
        self.drift_bias = float(drift_bias)
        self.x_basis_only = bool(x_basis_only)

        # Use endpoint=False when spanning a full 2*pi period to avoid duplicate boundary point
        is_full_circle = np.isclose(grid_max - grid_min, 2.0 * np.pi)
        self.grid = np.linspace(grid_min, grid_max, grid_size, endpoint=not is_full_circle)
        self.dtheta = float(self.grid[1] - self.grid[0])

        # Initialize wrapped Gaussian prior centered at init_theta
        init_diff = np.arctan2(
            np.sin(self.grid - init_theta),
            np.cos(self.grid - init_theta),
        )
        self.posterior = np.exp(-0.5 * (init_diff / max(init_std, 1e-3)) ** 2)
        self.posterior /= np.sum(self.posterior)

        self.estimated_theta = float(init_theta)
        self.variance = float(np.sum((init_diff ** 2) * self.posterior))

        # Precompute wrapped Markov transition matrix P(theta_t | theta_{t-1})
        self._transition_matrix = self._build_transition_matrix()

    def _build_transition_matrix(self) -> np.ndarray:
        
        sigma = max(self.sigma_drift, 1e-4)
        raw_diff = self.grid[:, None] - (self.grid[None, :] + self.drift_bias)
        wrapped_diff = np.arctan2(np.sin(raw_diff), np.cos(raw_diff))
        t_mat = np.exp(-0.5 * (wrapped_diff / sigma) ** 2)
        col_sums = np.sum(t_mat, axis=0, keepdims=True)
        col_sums = np.where(col_sums > 0, col_sums, 1.0)
        return t_mat / col_sums

    def expected_error_rate(self, candidate_theta: np.ndarray, applied_theta: float) -> np.ndarray:
        
        residual = candidate_theta - float(applied_theta)
        sin_sq = np.sin(residual / 2.0) ** 2

        if self.x_basis_only:
            p_err = self.e_0 + (1.0 - 2.0 * self.e_0) * sin_sq
        else:
            p_err = self.e_0 + (0.5 - self.e_0) * sin_sq

        return np.clip(p_err, 1e-9, 1.0 - 1e-9)

    def expected_error_rate_y(self, candidate_theta: np.ndarray, applied_theta: float) -> np.ndarray:
        
        residual = candidate_theta - float(applied_theta)
        p_err_y = 0.5 * (1.0 - (1.0 - 2.0 * self.e_0) * np.sin(residual))
        return np.clip(p_err_y, 1e-9, 1.0 - 1e-9)

    def predict(self) -> np.ndarray:
        
        prior = self._transition_matrix @ self.posterior
        total = np.sum(prior)
        if total > 0:
            prior /= total
        return prior

    def _compute_circular_readout(self, reference_theta: float) -> tuple:
        
        sin_mean = float(np.sum(np.sin(self.grid) * self.posterior))
        cos_mean = float(np.sum(np.cos(self.grid) * self.posterior))
        wrapped_theta = float(np.arctan2(sin_mean, cos_mean))

        # Phase-unwrap relative to reference_theta + drift_bias
        expected_ref = float(reference_theta + self.drift_bias)
        delta_from_ref = float(
            np.arctan2(np.sin(wrapped_theta - expected_ref), np.cos(wrapped_theta - expected_ref))
        )
        unwrapped_theta = float(expected_ref + delta_from_ref)

        ang_res = np.arctan2(
            np.sin(self.grid - wrapped_theta),
            np.cos(self.grid - wrapped_theta),
        )
        variance = float(np.sum((ang_res ** 2) * self.posterior))
        return unwrapped_theta, variance

    def update(
        self,
        test_qber: float,
        num_test_bits: int = 50,
        applied_theta: float = None,
        y_test_qber: float = None,
        num_y_test_bits: int = 0,
    ) -> tuple:

        if applied_theta is None:
            applied_theta = self.estimated_theta

        # 1. Time update (prior)
        prior = self.predict()

        # If 0 sifted test bits were available in this block, rely on prediction prior
        if (num_test_bits is None or num_test_bits <= 0) and (num_y_test_bits <= 0):
            self.posterior = prior
            self.estimated_theta, self.variance = self._compute_circular_readout(applied_theta)
            return self.estimated_theta, self.variance

        # 2. Basis-conditioned grid binomial likelihood L(k_{X,t} | theta_t, hat{theta}_{t-1})
        likelihood = np.ones_like(self.grid, dtype=float)

        if num_test_bits is not None and num_test_bits > 0:
            n_bits = int(max(1, round(num_test_bits)))
            k_errors = int(np.clip(round(float(test_qber) * n_bits), 0, n_bits))
            p_grid = self.expected_error_rate(self.grid, applied_theta)
            likelihood *= binom.pmf(k_errors, n_bits, p_grid)

        if y_test_qber is not None and num_y_test_bits is not None and num_y_test_bits > 0:
            n_y = int(max(1, round(num_y_test_bits)))
            k_y = int(np.clip(round(float(y_test_qber) * n_y), 0, n_y))
            p_y_grid = self.expected_error_rate_y(self.grid, applied_theta)
            likelihood *= binom.pmf(k_y, n_y, p_y_grid)

        # 3. Measurement update (Bayes' rule)
        unnormalized_post = prior * likelihood
        norm = np.sum(unnormalized_post)
        if norm > 1e-300:
            self.posterior = unnormalized_post / norm
        else:
            self.posterior = prior

        # 4. Circular MMSE posterior mean (unwrapped) and circular variance
        self.estimated_theta, self.variance = self._compute_circular_readout(applied_theta)
        return self.estimated_theta, self.variance


def split_test_and_data_bits(
    alice_bits: np.ndarray,
    alice_bases: np.ndarray,
    bob_bases: np.ndarray,
    bob_results: list,
    test_fraction: float = 0.10,
    x_basis_only_for_test: bool = True,
) -> dict:
    alice_bits_arr = np.asarray(alice_bits, dtype=int)
    alice_bases_arr = np.asarray(alice_bases, dtype=int)
    bob_bases_arr = np.asarray(bob_bases, dtype=int)
    bob_results_arr = np.asarray(bob_results, dtype=int)

    n_total = len(alice_bits_arr)
    n_test_raw = max(1, int(round(n_total * test_fraction)))

    # First n_test_raw qubits used as public test sample; remaining 90% as data bits
    test_a_bits = alice_bits_arr[:n_test_raw]
    test_a_bases = alice_bases_arr[:n_test_raw]
    test_b_bases = bob_bases_arr[:n_test_raw]
    test_b_results = bob_results_arr[:n_test_raw]

    data_a_bits = alice_bits_arr[n_test_raw:]
    data_a_bases = alice_bases_arr[n_test_raw:]
    data_b_bases = bob_bases_arr[n_test_raw:]
    data_b_results = bob_results_arr[n_test_raw:]

    # Separate Z-basis test monitor (stationary hardware floor e_0)
    z_mask = (test_a_bases == test_b_bases) & (test_a_bases == 0)
    num_sifted_test_z = int(np.sum(z_mask))
    test_qber_z = (
        float(np.mean(test_a_bits[z_mask] != test_b_results[z_mask]))
        if num_sifted_test_z > 0
        else 0.0
    )

    # Informative test mask for phase drift tracking
    if x_basis_only_for_test:
        sift_mask = (test_a_bases == test_b_bases) & (test_a_bases == 1)
    else:
        sift_mask = test_a_bases == test_b_bases

    sifted_test_alice = test_a_bits[sift_mask]
    sifted_test_bob = test_b_results[sift_mask]
    num_sifted_test = int(len(sifted_test_alice))

    if num_sifted_test > 0:
        k_test_errors = int(np.sum(sifted_test_alice != sifted_test_bob))
        test_qber = float(k_test_errors / num_sifted_test)
    else:
        k_test_errors = 0
        test_qber = 0.0

    return {
        "test_qber": test_qber,
        "k_test_errors": k_test_errors,
        "num_sifted_test": num_sifted_test,
        "test_qber_z": test_qber_z,
        "num_sifted_test_z": num_sifted_test_z,
        "data_alice_bits": data_a_bits,
        "data_alice_bases": data_a_bases,
        "data_bob_bases": data_b_bases,
        "data_bob_results": data_b_results,
    }