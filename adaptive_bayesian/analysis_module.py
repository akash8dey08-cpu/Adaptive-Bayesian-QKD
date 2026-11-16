import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


def sift_key(alice_bits, alice_bases, bob_bases, bob_results):
    sifted_key_alice = []
    sifted_key_bob = []

    for i in range(len(alice_bits)):
        if alice_bases[i] == bob_bases[i]:
            sifted_key_alice.append(int(alice_bits[i]))
            sifted_key_bob.append(int(bob_results[i]))

    return sifted_key_alice, sifted_key_bob


def calculate_qber(alice_key, bob_key):
    errors = 0

    for i in range(len(alice_key)):
        if alice_key[i] != bob_key[i]:
            errors += 1

    if len(alice_key) == 0:
        return 0.0  # avoid division by zero

    qber = errors / len(alice_key)
    return float(qber)


def binary_entropy(qber):
    
    e = float(np.clip(qber, 0.0, 0.5))
    if e <= 0.0:
        return 0.0
    if e >= 0.5:
        return 1.0
    return float(-e * np.log2(e) - (1.0 - e) * np.log2(1.0 - e))


def calculate_skr(qber, sifting_factor=1.0, f_ec=1.0):
    h2 = binary_entropy(qber)
    raw_rate = sifting_factor * (1.0 - (1.0 + f_ec) * h2)
    return float(max(0.0, raw_rate))


def compute_rolling_average(series, window=5):
    arr = np.asarray(series, dtype=float)
    if window <= 1 or len(arr) == 0:
        return arr.copy()
    rolled = np.empty_like(arr)
    for i in range(len(arr)):
        start_idx = max(0, i - window + 1)
        rolled[i] = np.mean(arr[start_idx : i + 1])
    return rolled


def _apply_publication_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
    })


def plot_tracking_performance(
    true_drift_history,
    estimated_drift_history,
    variance_history=None,
    save_path="tracking_performance.png",
):
    _apply_publication_style()
    t_steps = np.arange(len(true_drift_history))
    true_arr = np.asarray(true_drift_history, dtype=float)
    est_arr = np.asarray(estimated_drift_history, dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(
        t_steps,
        true_arr,
        color="#1B365D",
        linewidth=2.3,
        linestyle="-",
        label=r"True Channel Drift ($\theta_t$)",
    )
    ax.plot(
        t_steps,
        est_arr,
        color="#D95F02",
        linewidth=2.2,
        linestyle="--",
        label=r"Bayesian Estimate ($\hat{\theta}_t$)",
    )

    if variance_history is not None:
        std_arr = np.sqrt(np.maximum(0.0, np.asarray(variance_history, dtype=float)))
        ax.fill_between(
            t_steps,
            est_arr - std_arr,
            est_arr + std_arr,
            color="#D95F02",
            alpha=0.16,
            label=r"Posterior $\pm 1\sigma$ Credible Band",
        )

    rmse = float(np.sqrt(np.mean((true_arr - est_arr) ** 2)))
    ax.set_title(
        f"Bayesian Polarization Drift Tracking (RMSE = {rmse:.3f} rad)",
        pad=12,
    )
    ax.set_xlabel("Time Step ($t$)")
    ax.set_ylabel("Polarization Drift Angle (rad)")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_qber_stabilization(
    uncompensated_qber_history,
    compensated_qber_history,
    abort_threshold=0.11,
    save_path="qber_stabilization.png",
):
    """
    Phase 3 Plot 2: Plots time (t) vs. QBER (%), comparing uncompensated QBER
    against Bayesian-compensated QBER and the 11% BB84 abort threshold.
    """
    _apply_publication_style()
    t_steps = np.arange(len(uncompensated_qber_history))
    uncomp_pct = np.asarray(uncompensated_qber_history, dtype=float) * 100.0
    comp_pct = np.asarray(compensated_qber_history, dtype=float) * 100.0
    thresh_pct = abort_threshold * 100.0

    fig, ax = plt.subplots(figsize=(8.5, 4.6))

    y_max = max(float(np.max(uncomp_pct)) * 1.12, thresh_pct + 5.0)
    ax.axhspan(
        thresh_pct,
        y_max,
        color="#FDE0DD",
        alpha=0.45,
        label="Insecure Region (Protocol Abort)",
    )

    ax.plot(
        t_steps,
        uncomp_pct,
        color="#C0392B",
        linewidth=1.8,
        alpha=0.85,
        label=f"Uncompensated QBER (Mean: {np.mean(uncomp_pct):.1f}%)",
    )
    ax.plot(
        t_steps,
        comp_pct,
        color="#0072B2",
        linewidth=2.2,
        label=f"Bayesian-Compensated QBER (Mean: {np.mean(comp_pct):.1f}%)",
    )
    ax.axhline(
        thresh_pct,
        color="#8B0000",
        linestyle=":",
        linewidth=2.0,
        label=f"BB84 Security Abort Threshold ({thresh_pct:.0f}%)",
    )

    ax.set_ylim(0.0, y_max)
    ax.set_title("QBER Stabilization via Active Bayesian Rz Compensation", pad=12)
    ax.set_xlabel("Time Step ($t$)")
    ax.set_ylabel("Quantum Bit Error Rate (%)")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_rolling_skr(
    uncompensated_skr_history,
    compensated_skr_history,
    window=5,
    save_path="rolling_skr.png",
):
    _apply_publication_style()
    t_steps = np.arange(len(compensated_skr_history))
    uncomp_roll = compute_rolling_average(uncompensated_skr_history, window=window)
    comp_roll = compute_rolling_average(compensated_skr_history, window=window)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(
        t_steps,
        uncomp_roll,
        color="#C0392B",
        linewidth=2.0,
        linestyle="--",
        label=f"Uncompensated Rolling SKR ({window}-step avg)",
    )
    ax.plot(
        t_steps,
        comp_roll,
        color="#009E73",
        linewidth=2.4,
        linestyle="-",
        label=f"Bayesian-Compensated Rolling SKR ({window}-step avg)",
    )
    ax.fill_between(
        t_steps,
        uncomp_roll,
        comp_roll,
        where=(comp_roll >= uncomp_roll),
        color="#009E73",
        alpha=0.15,
        label="Net Key Rate Gain from Active Compensation",
    )

    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Sustained Secret Key Rate (SKR) Under Channel Drift", pad=12)
    ax.set_xlabel("Time Step ($t$)")
    ax.set_ylabel("Secret Key Rate (bits / sifted bit)")
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_combined_dashboard(
    true_drift_history,
    estimated_drift_history,
    uncompensated_qber_history,
    compensated_qber_history,
    uncompensated_skr_history,
    compensated_skr_history,
    variance_history=None,
    abort_threshold=0.11,
    rolling_window=5,
    save_path="adaptive_qkd_summary.png",
):
    _apply_publication_style()
    t_steps = np.arange(len(true_drift_history))
    true_arr = np.asarray(true_drift_history, dtype=float)
    est_arr = np.asarray(estimated_drift_history, dtype=float)
    uncomp_pct = np.asarray(uncompensated_qber_history, dtype=float) * 100.0
    comp_pct = np.asarray(compensated_qber_history, dtype=float) * 100.0
    thresh_pct = abort_threshold * 100.0
    uncomp_skr_roll = compute_rolling_average(uncompensated_skr_history, window=rolling_window)
    comp_skr_roll = compute_rolling_average(compensated_skr_history, window=rolling_window)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 11.2), sharex=True)

    ax0 = axes[0]
    ax0.plot(t_steps, true_arr, color="#1B365D", lw=2.3, label=r"True Drift ($\theta_t$)")
    ax0.plot(
        t_steps,
        est_arr,
        color="#D95F02",
        lw=2.2,
        ls="--",
        label=r"Bayesian Estimate ($\hat{\theta}_t$)",
    )
    if variance_history is not None:
        std_arr = np.sqrt(np.maximum(0.0, np.asarray(variance_history, dtype=float)))
        ax0.fill_between(
            t_steps,
            est_arr - std_arr,
            est_arr + std_arr,
            color="#D95F02",
            alpha=0.16,
            label=r"Posterior $\pm 1\sigma$ Band",
        )
    rmse = float(np.sqrt(np.mean((true_arr - est_arr) ** 2)))
    ax0.set_title(f"(a) Recursive Bayesian Drift Tracking (RMSE = {rmse:.3f} rad)", pad=8)
    ax0.set_ylabel("Angle (rad)")
    ax0.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    ax1 = axes[1]
    y_max = max(float(np.max(uncomp_pct)) * 1.12, thresh_pct + 5.0)
    ax1.axhspan(thresh_pct, y_max, color="#FDE0DD", alpha=0.45, label="Abort Zone (>11%)")
    ax1.plot(
        t_steps,
        uncomp_pct,
        color="#C0392B",
        lw=1.8,
        alpha=0.85,
        label=f"Uncompensated QBER (Mean: {np.mean(uncomp_pct):.1f}%)",
    )
    ax1.plot(
        t_steps,
        comp_pct,
        color="#0072B2",
        lw=2.2,
        label=f"Bayesian-Compensated QBER (Mean: {np.mean(comp_pct):.1f}%)",
    )
    ax1.axhline(
        thresh_pct,
        color="#8B0000",
        ls=":",
        lw=2.0,
        label="11% Security Abort Threshold",
    )
    ax1.set_ylim(0.0, y_max)
    ax1.set_title("(b) Quantum Bit Error Rate (QBER) Stabilization", pad=8)
    ax1.set_ylabel("QBER (%)")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    ax2 = axes[2]
    ax2.plot(
        t_steps,
        uncomp_skr_roll,
        color="#C0392B",
        lw=2.0,
        ls="--",
        label=f"Uncompensated Rolling SKR ({rolling_window}-step avg)",
    )
    ax2.plot(
        t_steps,
        comp_skr_roll,
        color="#009E73",
        lw=2.4,
        label=f"Bayesian-Compensated Rolling SKR ({rolling_window}-step avg)",
    )
    ax2.fill_between(
        t_steps,
        uncomp_skr_roll,
        comp_skr_roll,
        where=(comp_skr_roll >= uncomp_skr_roll),
        color="#009E73",
        alpha=0.15,
        label="Sustained Key Generation Gain",
    )
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_title("(c) Rolling Asymptotic Secret Key Rate (SKR)", pad=8)
    ax2.set_xlabel("Time Step ($t$)")
    ax2.set_ylabel("SKR (bits / sifted bit)")
    ax2.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path