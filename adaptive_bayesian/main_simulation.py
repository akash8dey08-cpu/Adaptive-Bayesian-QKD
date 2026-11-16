import os
import numpy as np

import alice_module
import channel_module
import bob_module
import filter_module
import analysis_module


def run_adaptive_qkd_simulation(
    num_steps: int = 100,
    qubits_per_step: int = 800,
    sigma_drift: float = 0.04,
    drift_bias: float = 0.018,
    depol_prob: float = 0.015,
    test_fraction: float = 0.10,
    seed: int = 42,
    output_dir: str = ".",
) -> dict:
    """
    Executes the full time-series Adaptive Bayesian BB84 QKD simulation across
    T = `num_steps` steps (Phases 1, 2, and 3 of the project blueprint).
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1: Initialization & Setup
    theta_t = 0.0
    bayes_filter = filter_module.BayesianDriftFilter(
        sigma_drift=sigma_drift,
        depol_prob=depol_prob,
        drift_bias=drift_bias,
        init_theta=0.0,
        init_std=0.05,
    )

    true_drift_history = []
    estimated_drift_history = []
    posterior_variance_history = []
    uncompensated_qber_history = []
    compensated_qber_history = []
    uncompensated_skr_history = []
    compensated_skr_history = []

    # Phase 2: The Time-Step Loop (t = 0 to T - 1)
    for t in range(num_steps):
        # 1. Alice Prepares N Qubits
        alice_bits, alice_bases, alice_circuits = alice_module.prepare_qubits(qubits_per_step)

        # 2. Channel Applies Time-Varying Drift R_z(theta_t) + Depolarizing Noise
        theta_t, channel_circuits = channel_module.transmit_through_channel(
            circuits=alice_circuits,
            prev_theta=theta_t,
            sigma_drift=sigma_drift,
            depol_prob=depol_prob,
            drift_bias=drift_bias,
        )

        # 3 & 4. Bob Applies Active Correction R_z(-hat{theta}_{t-1}) & Measures
        bob_bases = bob_module.generate_bases(qubits_per_step)
        applied_theta = bayes_filter.estimated_theta

        bob_results_comp = bob_module.measure_qubits(
            circuits=channel_circuits,
            bases=bob_bases,
            estimated_theta=applied_theta,
        )
        bob_results_uncomp = bob_module.measure_qubits(
            circuits=channel_circuits,
            bases=bob_bases,
            estimated_theta=0.0,
        )

        # 5. Test Bits (10%) & Recursive Bayesian Update
        comp_partition = filter_module.split_test_and_data_bits(
            alice_bits=alice_bits,
            alice_bases=alice_bases,
            bob_bases=bob_bases,
            bob_results=bob_results_comp,
            test_fraction=test_fraction,
        )
        uncomp_partition = filter_module.split_test_and_data_bits(
            alice_bits=alice_bits,
            alice_bases=alice_bases,
            bob_bases=bob_bases,
            bob_results=bob_results_uncomp,
            test_fraction=test_fraction,
        )

        refined_theta, post_var = bayes_filter.update(
            test_qber=comp_partition["test_qber"],
            num_test_bits=comp_partition["num_sifted_test"],
            applied_theta=applied_theta,
        )

        # 6. Key Distillation on Remaining 90% Data Bits
        sift_a_comp, sift_b_comp = analysis_module.sift_key(
            comp_partition["data_alice_bits"],
            comp_partition["data_alice_bases"],
            comp_partition["data_bob_bases"],
            comp_partition["data_bob_results"],
        )
        sift_a_uncomp, sift_b_uncomp = analysis_module.sift_key(
            uncomp_partition["data_alice_bits"],
            uncomp_partition["data_alice_bases"],
            uncomp_partition["data_bob_bases"],
            uncomp_partition["data_bob_results"],
        )

        qber_comp = analysis_module.calculate_qber(sift_a_comp, sift_b_comp)
        qber_uncomp = analysis_module.calculate_qber(sift_a_uncomp, sift_b_uncomp)

        skr_comp = analysis_module.calculate_skr(qber_comp)
        skr_uncomp = analysis_module.calculate_skr(qber_uncomp)

        true_drift_history.append(theta_t)
        estimated_drift_history.append(refined_theta)
        posterior_variance_history.append(post_var)
        uncompensated_qber_history.append(qber_uncomp)
        compensated_qber_history.append(qber_comp)
        uncompensated_skr_history.append(skr_uncomp)
        compensated_skr_history.append(skr_comp)

    # Phase 3: Publication-Ready Visualizations
    tracking_plot = analysis_module.plot_tracking_performance(
        true_drift_history=true_drift_history,
        estimated_drift_history=estimated_drift_history,
        variance_history=posterior_variance_history,
        save_path=os.path.join(output_dir, "tracking_performance.png"),
    )
    qber_plot = analysis_module.plot_qber_stabilization(
        uncompensated_qber_history=uncompensated_qber_history,
        compensated_qber_history=compensated_qber_history,
        abort_threshold=0.11,
        save_path=os.path.join(output_dir, "qber_stabilization.png"),
    )
    skr_plot = analysis_module.plot_rolling_skr(
        uncompensated_skr_history=uncompensated_skr_history,
        compensated_skr_history=compensated_skr_history,
        window=5,
        save_path=os.path.join(output_dir, "rolling_skr.png"),
    )
    summary_plot = analysis_module.plot_combined_dashboard(
        true_drift_history=true_drift_history,
        estimated_drift_history=estimated_drift_history,
        uncompensated_qber_history=uncompensated_qber_history,
        compensated_qber_history=compensated_qber_history,
        uncompensated_skr_history=uncompensated_skr_history,
        compensated_skr_history=compensated_skr_history,
        variance_history=posterior_variance_history,
        abort_threshold=0.11,
        rolling_window=5,
        save_path=os.path.join(output_dir, "adaptive_qkd_summary.png"),
    )

    return {
        "true_drift_history": true_drift_history,
        "estimated_drift_history": estimated_drift_history,
        "posterior_variance_history": posterior_variance_history,
        "uncompensated_qber_history": uncompensated_qber_history,
        "compensated_qber_history": compensated_qber_history,
        "uncompensated_skr_history": uncompensated_skr_history,
        "compensated_skr_history": compensated_skr_history,
        "plots": {
            "tracking": tracking_plot,
            "qber": qber_plot,
            "skr": skr_plot,
            "summary": summary_plot,
        },
    }


if __name__ == "__main__":
    results = run_adaptive_qkd_simulation()

    # Extract metrics
    u_qber = np.array(results["uncompensated_qber_history"]) * 100.0
    c_qber = np.array(results["compensated_qber_history"]) * 100.0
    u_skr = np.array(results["uncompensated_skr_history"])
    c_skr = np.array(results["compensated_skr_history"])
    true_th = np.array(results["true_drift_history"])
    est_th = np.array(results["estimated_drift_history"])

    # Calculate summary statistics
    tracking_rmse_rad = float(np.sqrt(np.mean((true_th - est_th) ** 2)))
    tracking_rmse_deg = tracking_rmse_rad * 180.0 / np.pi
    u_aborts = int(np.sum(u_qber >= 11.0))
    c_aborts = int(np.sum(c_qber >= 11.0))
    u_uptime = float(np.mean(u_qber < 11.0) * 100.0)
    c_uptime = float(np.mean(c_qber < 11.0) * 100.0)

    # Print formatted comparison table
    print("\n" + "=" * 82)
    print(" ADAPTIVE BAYESIAN BB84 SIMULATION RESULTS (Single Session: T = 100 steps)")
    print("=" * 82)
    print(f"{'Performance Metric':<36} | {'Uncompensated Baseline':<21} | {'Bayesian Compensation':<21}")
    print("-" * 82)
    print(f"{'Tracking RMSE (rad)':<36} | {'—':<21} | {tracking_rmse_rad:.3f} rad")
    print(f"{'Tracking RMSE (deg)':<36} | {'—':<21} | {tracking_rmse_deg:.1f}°")
    print(f"{'Mean QBER (%)':<36} | {np.mean(u_qber):.2f}%{'':<15} | {np.mean(c_qber):.2f}%")
    print(f"{'Peak QBER (%)':<36} | {np.max(u_qber):.2f}%{'':<15} | {np.max(c_qber):.2f}%")
    print(f"{'Steps Above 11% Abort Threshold':<36} | {u_aborts:>2} / 100 steps{'':<8} | {c_aborts:>2} / 100 steps")
    print(f"{'Link Availability / Uptime (%)':<36} | {u_uptime:.1f}%{'':<16} | {c_uptime:.1f}%")
    print(f"{'Mean Secret Key Rate (SKR)':<36} | {np.mean(u_skr):.3f} bits/sifted bit | {np.mean(c_skr):.3f} bits/sifted bit")
    print("=" * 82)
    print("Generated Plots:")
    for name, path in results["plots"].items():
        print(f"  • {name:<10}: {path}")
    print("=" * 82 + "\n")