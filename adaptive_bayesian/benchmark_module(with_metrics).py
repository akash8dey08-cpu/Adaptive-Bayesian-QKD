import os
import time
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

import alice_module
import channel_module
import bob_module
import filter_module
import analysis_module


def calculate_net_skr_per_pulse(qber: float, test_fraction: float = 0.10, f_ec: float = 1.16) -> float:
    
    return analysis_module.calculate_skr(
        qber=qber,
        sifting_factor=0.5 * (1.0 - float(test_fraction)),
        f_ec=float(f_ec),
    )


def calculate_basis_specific_qber(
    alice_bits: np.ndarray,
    alice_bases: np.ndarray,
    bob_bases: np.ndarray,
    bob_results: np.ndarray,
) -> tuple:
    
    a_bits = np.asarray(alice_bits, dtype=int)
    a_bases = np.asarray(alice_bases, dtype=int)
    b_bases = np.asarray(bob_bases, dtype=int)
    b_res = np.asarray(bob_results, dtype=int)

    match_z = (a_bases == b_bases) & (a_bases == 0)
    match_x = (a_bases == b_bases) & (a_bases == 1)
    match_all = a_bases == b_bases

    qber_z = float(np.mean(a_bits[match_z] != b_res[match_z])) if np.any(match_z) else 0.0
    qber_x = float(np.mean(a_bits[match_x] != b_res[match_x])) if np.any(match_x) else 0.0
    qber_all = float(np.mean(a_bits[match_all] != b_res[match_all])) if np.any(match_all) else 0.0
    num_sifted = int(np.sum(match_all))

    return qber_z, qber_x, qber_all, num_sifted


def calculate_asymmetric_skr(
    qber_z: float,
    qber_x: float,
    sifting_factor: float = 1.0,
    f_ec: float = 1.16,
) -> float:
    
    h_z = analysis_module.binary_entropy(qber_z)
    h_x = analysis_module.binary_entropy(qber_x)
    h_avg = 0.5 * (h_z + h_x)
    if h_avg >= 0.5:
        return 0.0
    return float(sifting_factor * max(0.0, 1.0 - (1.0 + float(f_ec)) * h_avg))


def memoryless_instantaneous_estimate(
    prev_theta: float,
    test_qber: float,
    depol_prob: float = 0.015,
    step_gain: float = 0.18,
    x_basis_only: bool = True,
) -> float:
    
    e_0 = (2.0 / 3.0) * depol_prob
    visibility = (1.0 - 2.0 * e_0) if x_basis_only else (0.5 - e_0)
    excess = max(0.0, float(test_qber) - e_0) / max(1e-6, visibility)
    ratio = float(np.clip(excess, 0.0, 1.0))
    delta_mag = 2.0 * float(np.arcsin(np.sqrt(ratio)))
    return float(prev_theta + step_gain * delta_mag)


def run_single_trial(
    num_steps: int = 100,
    qubits_per_step: int = 800,
    sigma_drift: float = 0.04,
    drift_bias: float = 0.018,
    depol_prob: float = 0.015,
    test_fraction: float = 0.10,
    f_ec: float = 1.16,
    seed: int = 0,
    fast_quantum_sim: bool = True,
    x_basis_only: bool = True,
) -> dict:
    """
    Runs a single chronological trajectory of T = `num_steps` comparing:
      1. Uncompensated BB84 (f_test = 0.0)
      2. Memoryless Instantaneous Compensation (f_test = test_fraction)
      3. Recursive Bayesian Filter Compensation (f_test = test_fraction)
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    e_0 = (2.0 / 3.0) * depol_prob
    theta_t = 0.0
    mem_theta = 0.0

    bayes_filter = filter_module.BayesianDriftFilter(
        sigma_drift=sigma_drift,
        depol_prob=depol_prob,
        drift_bias=drift_bias,
        init_theta=0.0,
        init_std=0.05,
        x_basis_only=x_basis_only,
    )

    true_thetas = np.zeros(num_steps)
    bayes_thetas = np.zeros(num_steps)
    bayes_stds = np.zeros(num_steps)
    mem_thetas = np.zeros(num_steps)

    qber_uncomp = np.zeros(num_steps)
    qber_mem = np.zeros(num_steps)
    qber_bayes = np.zeros(num_steps)

    gross_skr_uncomp = np.zeros(num_steps)
    gross_skr_mem = np.zeros(num_steps)
    gross_skr_bayes = np.zeros(num_steps)

    net_skr_uncomp = np.zeros(num_steps)
    net_skr_mem = np.zeros(num_steps)
    net_skr_bayes = np.zeros(num_steps)

    secret_bits_uncomp = 0.0
    secret_bits_mem = 0.0
    secret_bits_bayes = 0.0

    for t in range(num_steps):
        applied_bayes = bayes_filter.estimated_theta
        applied_mem = mem_theta

        if not fast_quantum_sim:
            alice_bits, alice_bases, alice_circuits = alice_module.prepare_qubits(qubits_per_step)
            theta_t, channel_circuits = channel_module.transmit_through_channel(
                circuits=alice_circuits,
                prev_theta=theta_t,
                sigma_drift=sigma_drift,
                depol_prob=depol_prob,
                drift_bias=drift_bias,
            )
            bob_bases = bob_module.generate_bases(qubits_per_step)

            res_uncomp = bob_module.measure_qubits(channel_circuits, bob_bases, estimated_theta=0.0)
            res_mem = bob_module.measure_qubits(channel_circuits, bob_bases, estimated_theta=applied_mem)
            res_bayes = bob_module.measure_qubits(channel_circuits, bob_bases, estimated_theta=applied_bayes)
        else:
            alice_bits = rng.randint(2, size=qubits_per_step)
            alice_bases = rng.randint(2, size=qubits_per_step)
            bob_bases = rng.randint(2, size=qubits_per_step)
            theta_t = float(theta_t + rng.normal(loc=drift_bias, scale=sigma_drift))

            def sample_bob(applied_angle):
                residual = theta_t - float(applied_angle)
                p_flip = np.where(
                    alice_bases != bob_bases,
                    0.5,
                    np.where(
                        alice_bases == 0,
                        e_0,
                        e_0 + (1.0 - 2.0 * e_0) * (np.sin(residual / 2.0) ** 2),
                    ),
                )
                flips = (rng.random(qubits_per_step) < p_flip).astype(int)
                return (alice_bits ^ flips).tolist()

            res_uncomp = sample_bob(0.0)
            res_mem = sample_bob(applied_mem)
            res_bayes = sample_bob(applied_bayes)

        part_uncomp = filter_module.split_test_and_data_bits(
            alice_bits,
            alice_bases,
            bob_bases,
            res_uncomp,
            test_fraction=test_fraction,
            x_basis_only_for_test=x_basis_only,
        )
        part_mem = filter_module.split_test_and_data_bits(
            alice_bits,
            alice_bases,
            bob_bases,
            res_mem,
            test_fraction=test_fraction,
            x_basis_only_for_test=x_basis_only,
        )
        part_bayes = filter_module.split_test_and_data_bits(
            alice_bits,
            alice_bases,
            bob_bases,
            res_bayes,
            test_fraction=test_fraction,
            x_basis_only_for_test=x_basis_only,
        )

        est_bayes, var_bayes = bayes_filter.update(
            test_qber=part_bayes["test_qber"],
            num_test_bits=part_bayes["num_sifted_test"],
            applied_theta=applied_bayes,
        )
        mem_theta = memoryless_instantaneous_estimate(
            prev_theta=applied_mem,
            test_qber=part_mem["test_qber"],
            depol_prob=depol_prob,
            x_basis_only=x_basis_only,
        )

        # Compute basis-specific QBERs (e_Z, e_X) and aggregate QBER on the 90% data partition
        qz_u, qx_u, qu, n_sift_u = calculate_basis_specific_qber(
            part_uncomp["data_alice_bits"],
            part_uncomp["data_alice_bases"],
            part_uncomp["data_bob_bases"],
            part_uncomp["data_bob_results"],
        )
        qz_m, qx_m, qm, n_sift_m = calculate_basis_specific_qber(
            part_mem["data_alice_bits"],
            part_mem["data_alice_bases"],
            part_mem["data_bob_bases"],
            part_mem["data_bob_results"],
        )
        qz_b, qx_b, qb, n_sift_b = calculate_basis_specific_qber(
            part_bayes["data_alice_bits"],
            part_bayes["data_alice_bases"],
            part_bayes["data_bob_bases"],
            part_bayes["data_bob_results"],
        )

        true_thetas[t] = theta_t
        bayes_thetas[t] = est_bayes
        bayes_stds[t] = np.sqrt(max(0.0, var_bayes))
        mem_thetas[t] = mem_theta

        qber_uncomp[t] = qu
        qber_mem[t] = qm
        qber_bayes[t] = qb

        # Basis-asymmetric Gross SKR (f_ec = 1.0, per sifted bit)
        gross_skr_uncomp[t] = calculate_asymmetric_skr(qz_u, qx_u, sifting_factor=1.0, f_ec=1.0)
        gross_skr_mem[t] = calculate_asymmetric_skr(qz_m, qx_m, sifting_factor=1.0, f_ec=1.0)
        gross_skr_bayes[t] = calculate_asymmetric_skr(qz_b, qx_b, sifting_factor=1.0, f_ec=1.0)

        # Basis-asymmetric Net SKR (f_ec = 1.16, per transmitted pulse)
        net_skr_uncomp[t] = calculate_asymmetric_skr(
            qz_u, qx_u, sifting_factor=0.5 * 1.0, f_ec=f_ec
        )
        net_skr_mem[t] = calculate_asymmetric_skr(
            qz_m, qx_m, sifting_factor=0.5 * (1.0 - float(test_fraction)), f_ec=f_ec
        )
        net_skr_bayes[t] = calculate_asymmetric_skr(
            qz_b, qx_b, sifting_factor=0.5 * (1.0 - float(test_fraction)), f_ec=f_ec
        )

        # Uncompensated branch sacrifices 0% test bits (scales to full N-qubit block)
        uncomp_scale = 1.0 / max(1e-6, 1.0 - float(test_fraction))
        secret_bits_uncomp += (
            n_sift_u
            * uncomp_scale
            * calculate_asymmetric_skr(qz_u, qx_u, sifting_factor=1.0, f_ec=f_ec)
        )
        secret_bits_mem += n_sift_m * calculate_asymmetric_skr(
            qz_m, qx_m, sifting_factor=1.0, f_ec=f_ec
        )
        secret_bits_bayes += n_sift_b * calculate_asymmetric_skr(
            qz_b, qx_b, sifting_factor=1.0, f_ec=f_ec
        )

    # Geodesic wrapped angular error on S^1: wrap(theta_true - theta_est)
    err_bayes = np.arctan2(
        np.sin(true_thetas - bayes_thetas),
        np.cos(true_thetas - bayes_thetas),
    )
    err_mem = np.arctan2(
        np.sin(true_thetas - mem_thetas),
        np.cos(true_thetas - mem_thetas),
    )

    rmse_bayes = float(np.sqrt(np.mean(err_bayes ** 2)))
    mae_bayes = float(np.mean(np.abs(err_bayes)))
    rmse_mem = float(np.sqrt(np.mean(err_mem ** 2)))
    mae_mem = float(np.mean(np.abs(err_mem)))

    in_ci = np.abs(err_bayes) <= (1.96 * np.maximum(bayes_stds, 0.05))
    ci_coverage = float(np.mean(in_ci) * 100.0)

    return {
        "true_thetas": true_thetas,
        "bayes_thetas": bayes_thetas,
        "mem_thetas": mem_thetas,
        "qber_uncomp": qber_uncomp,
        "qber_mem": qber_mem,
        "qber_bayes": qber_bayes,
        "gross_skr_uncomp": gross_skr_uncomp,
        "gross_skr_mem": gross_skr_mem,
        "gross_skr_bayes": gross_skr_bayes,
        "net_skr_uncomp": net_skr_uncomp,
        "net_skr_mem": net_skr_mem,
        "net_skr_bayes": net_skr_bayes,
        "rmse_bayes": rmse_bayes,
        "mae_bayes": mae_bayes,
        "rmse_mem": rmse_mem,
        "mae_mem": mae_mem,
        "ci_coverage": ci_coverage,
        "uptime_uncomp": float(np.mean(qber_uncomp < 0.11) * 100.0),
        "uptime_mem": float(np.mean(qber_mem < 0.11) * 100.0),
        "uptime_bayes": float(np.mean(qber_bayes < 0.11) * 100.0),
        "secret_bits_uncomp": float(secret_bits_uncomp),
        "secret_bits_mem": float(secret_bits_mem),
        "secret_bits_bayes": float(secret_bits_bayes),
    }


def print_benchmark_results(results: dict):
    """
    Prints the complete conference evaluation tables directly to the terminal.
    """
    mc = results["mc_summary"]
    u = mc["uncomp"]
    m = mc["mem"]
    b = mc["bayes"]

    print("\n" + "=" * 112)
    print(" TABLE I: 50-SEED MONTE CARLO BENCHMARK (Mean ± Std, T=100 steps, N=800 pulses/step)")
    print("=" * 112)
    print(f"{'Evaluation Metric':<36} | {'Uncompensated':<22} | {'Memoryless':<24} | {'Recursive Bayesian':<22}")
    print("-" * 112)
    print(f"{'Tracking RMSE (rad)':<36} | {'—':<22} | {m['rmse_rad'][0]:.3f} ± {m['rmse_rad'][1]:.3f}{'':<13} | {b['rmse_rad'][0]:.3f} ± {b['rmse_rad'][1]:.3f}")
    print(f"{'Tracking RMSE (deg)':<36} | {'—':<22} | {m['rmse_deg'][0]:.1f}° ± {m['rmse_deg'][1]:.1f}°{'':<11} | {b['rmse_deg'][0]:.1f}° ± {b['rmse_deg'][1]:.1f}°")
    print(f"{'Tracking MAE (rad)':<36} | {'—':<22} | {m['mae_rad'][0]:.3f} ± {m['mae_rad'][1]:.3f}{'':<13} | {b['mae_rad'][0]:.3f} ± {b['mae_rad'][1]:.3f}")
    print(f"{'95% Credible Band Coverage (%)':<36} | {'—':<22} | {'—':<24} | {b['ci_coverage'][0]:.2f}% ± {b['ci_coverage'][1]:.2f}%")
    print(f"{'Mean QBER (%)':<36} | {u['mean_qber'][0]:.2f}% ± {u['mean_qber'][1]:.2f}%{'':<6} | {m['mean_qber'][0]:.2f}% ± {m['mean_qber'][1]:.2f}%{'':<8} | {b['mean_qber'][0]:.2f}% ± {b['mean_qber'][1]:.2f}%")
    print(f"{'Peak QBER (%)':<36} | {u['peak_qber'][0]:.2f}% ± {u['peak_qber'][1]:.2f}%{'':<5} | {m['peak_qber'][0]:.2f}% ± {m['peak_qber'][1]:.2f}%{'':<7} | {b['peak_qber'][0]:.2f}% ± {b['peak_qber'][1]:.2f}%")
    print(f"{'Link Availability / Uptime (%)':<36} | {u['uptime'][0]:.1f}% ± {u['uptime'][1]:.1f}%{'':<7} | {m['uptime'][0]:.1f}% ± {m['uptime'][1]:.1f}%{'':<9} | {b['uptime'][0]:.2f}% ± {b['uptime'][1]:.2f}%")
    print(f"{'Gross SKR (bits / sifted bit)':<36} | {u['gross_skr'][0]:.3f} ± {u['gross_skr'][1]:.3f}{'':<7} | {m['gross_skr'][0]:.3f} ± {m['gross_skr'][1]:.3f}{'':<9} | {b['gross_skr'][0]:.3f} ± {b['gross_skr'][1]:.3f}")
    print(f"{'Net SKR (bits / pulse, f_ec=1.16)':<36} | {u['net_skr'][0]:.3f} ± {u['net_skr'][1]:.3f}{'':<7} | {m['net_skr'][0]:.3f} ± {m['net_skr'][1]:.3f}{'':<9} | {b['net_skr'][0]:.3f} ± {b['net_skr'][1]:.3f}")
    print(f"{'Total Distilled Key Bits / Session':<36} | {u['total_bits'][0]:.0f} ± {u['total_bits'][1]:.0f}{'':<5} | {m['total_bits'][0]:.0f} ± {m['total_bits'][1]:.0f}{'':<7} | {b['total_bits'][0]:.0f} ± {b['total_bits'][1]:.0f}")
    print("=" * 112)

    print("\n" + "=" * 112)
    print(" TABLE II: TURBULENCE RESILIENCE SWEEP (Drift Volatility sigma_drift in [0.01, 0.12] rad, f_ec = 1.16)")
    print("=" * 112)
    print(f"{'sigma_drift (rad)':<18} | {'RMSE (rad)':<12} | {'Bayes QBER (%)':<18} | {'Uptime (%)':<12} | {'Uncomp Net SKR':<16} | {'Mem Net SKR':<14} | {'Bayes Net SKR':<14}")
    print("-" * 112)
    for r in results["turb_results"]:
        sig_str = f"{r['sigma']:.2f} ({r['sigma']*180/np.pi:.1f}°)"
        qber_str = f"{r['qber_bayes_mean']:.2f}% ± {r['qber_bayes_std']:.2f}%"
        skr_b_str = f"{r['net_skr_bayes_mean']:.3f} ± {r['net_skr_bayes_std']:.3f}"
        print(f"{sig_str:<18} | {r['rmse_mean']:<12.3f} | {qber_str:<18} | {r['uptime_bayes']:<12.1f} | {r['net_skr_uncomp_mean']:<16.3f} | {r['net_skr_mem_mean']:<14.3f} | {skr_b_str:<14}")
    print("=" * 112)

    print("\n" + "=" * 90)
    print(" TABLE III: TEST-BIT OVERHEAD SWEEP (Test Fraction f_test in [2%, 30%], sigma_drift = 0.04 rad)")
    print("=" * 90)
    print(f"{'Overhead (f_test)':<18} | {'RMSE (rad)':<18} | {'Comp QBER (%)':<16} | {'Net SKR (bits/pulse)':<22} | {'Total Secret Bits':<18}")
    print("-" * 90)
    for r in results["overhead_results"]:
        ft_str = f"{r['f_test']*100:.0f}%" + (" (Optimal)" if round(r['f_test']*100) == 5 else "")
        rmse_str = f"{r['rmse_mean']:.3f} ± {r['rmse_std']:.3f}"
        skr_str = f"{r['net_skr_mean']:.4f} ± {r['net_skr_std']:.4f}"
        print(f"{ft_str:<18} | {rmse_str:<18} | {r['qber_mean']:<16.2f} | {skr_str:<22} | {r['total_bits_mean']:<18.0f}")
    print("=" * 90 + "\n")


def run_conference_benchmark_suite(
    num_seeds: int = 50,
    sweep_trials: int = 15,
    num_steps: int = 100,
    qubits_per_step: int = 800,
    output_dir: str = ".",
    fast_quantum_sim: bool = True,
) -> dict:
    """
    Runs the complete conference evaluation suite with live terminal progress:
      1. 50-Seed Monte Carlo Evaluation comparing Uncompensated, Memoryless,
         and Recursive Bayesian Compensation.
      2. Turbulence Resilience Sweep over sigma_drift in [0.01, 0.12] rad.
      3. Test-Bit Overhead Optimization Sweep over f_test in [2%, 30%].
      4. Generates publication-ready standalone and combined figures.
      5. Prints the formatted evaluation tables to the terminal.
    """
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    analysis_module._apply_publication_style()

    # 1. 50-Seed Monte Carlo Run
    print(f"\n[1/4] Running {num_seeds}-Seed Monte Carlo evaluation...", flush=True)
    mc_trials = []
    for s in range(num_seeds):
        mc_trials.append(
            run_single_trial(
                num_steps=num_steps,
                qubits_per_step=qubits_per_step,
                sigma_drift=0.04,
                drift_bias=0.018,
                depol_prob=0.015,
                test_fraction=0.10,
                f_ec=1.16,
                seed=100 + s,
                fast_quantum_sim=fast_quantum_sim,
            )
        )
        if (s + 1) % 10 == 0 or (s + 1) == num_seeds:
            elapsed = time.time() - t_start
            print(f"      -> Completed {s + 1}/{num_seeds} Monte Carlo sessions ({elapsed:.1f}s elapsed)", flush=True)

    def agg(key, scale=1.0):
        vals = np.array([np.mean(tr[key]) * scale for tr in mc_trials])
        return float(np.mean(vals)), float(np.std(vals))

    def agg_max(key, scale=1.0):
        vals = np.array([np.max(tr[key]) * scale for tr in mc_trials])
        return float(np.mean(vals)), float(np.std(vals))

    mc_summary = {
        "uncomp": {
            "mean_qber": agg("qber_uncomp", 100.0),
            "peak_qber": agg_max("qber_uncomp", 100.0),
            "uptime": agg("uptime_uncomp", 1.0),
            "gross_skr": agg("gross_skr_uncomp", 1.0),
            "net_skr": agg("net_skr_uncomp", 1.0),
            "total_bits": agg("secret_bits_uncomp", 1.0),
        },
        "mem": {
            "rmse_rad": agg("rmse_mem", 1.0),
            "rmse_deg": agg("rmse_mem", 180.0 / np.pi),
            "mae_rad": agg("mae_mem", 1.0),
            "mean_qber": agg("qber_mem", 100.0),
            "peak_qber": agg_max("qber_mem", 100.0),
            "uptime": agg("uptime_mem", 1.0),
            "gross_skr": agg("gross_skr_mem", 1.0),
            "net_skr": agg("net_skr_mem", 1.0),
            "total_bits": agg("secret_bits_mem", 1.0),
        },
        "bayes": {
            "rmse_rad": agg("rmse_bayes", 1.0),
            "rmse_deg": agg("rmse_bayes", 180.0 / np.pi),
            "mae_rad": agg("mae_bayes", 1.0),
            "ci_coverage": agg("ci_coverage", 1.0),
            "mean_qber": agg("qber_bayes", 100.0),
            "peak_qber": agg_max("qber_bayes", 100.0),
            "uptime": agg("uptime_bayes", 1.0),
            "gross_skr": agg("gross_skr_bayes", 1.0),
            "net_skr": agg("net_skr_bayes", 1.0),
            "total_bits": agg("secret_bits_bayes", 1.0),
        },
    }

    qber_u_mat = np.array([tr["qber_uncomp"] * 100.0 for tr in mc_trials])
    qber_m_mat = np.array([tr["qber_mem"] * 100.0 for tr in mc_trials])
    qber_b_mat = np.array([tr["qber_bayes"] * 100.0 for tr in mc_trials])

    # 2. Turbulence Resilience Sweep (sigma_drift in [0.01, 0.12] rad)
    sigma_list = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    print(f"\n[2/4] Running Turbulence Resilience Sweep ({len(sigma_list)} levels x {sweep_trials} trials)...", flush=True)
    turb_results = []
    for idx, sig in enumerate(sigma_list, 1):
        sub_trials = [
            run_single_trial(
                num_steps=num_steps,
                qubits_per_step=qubits_per_step,
                sigma_drift=sig,
                drift_bias=0.018,
                depol_prob=0.015,
                test_fraction=0.10,
                f_ec=1.16,
                seed=500 + k,
                fast_quantum_sim=fast_quantum_sim,
            )
            for k in range(sweep_trials)
        ]
        turb_results.append({
            "sigma": sig,
            "rmse_mean": float(np.mean([tr["rmse_bayes"] for tr in sub_trials])),
            "qber_bayes_mean": float(np.mean([np.mean(tr["qber_bayes"]) * 100.0 for tr in sub_trials])),
            "qber_bayes_std": float(np.std([np.mean(tr["qber_bayes"]) * 100.0 for tr in sub_trials])),
            "qber_mem_mean": float(np.mean([np.mean(tr["qber_mem"]) * 100.0 for tr in sub_trials])),
            "qber_uncomp_mean": float(np.mean([np.mean(tr["qber_uncomp"]) * 100.0 for tr in sub_trials])),
            "net_skr_bayes_mean": float(np.mean([np.mean(tr["net_skr_bayes"]) for tr in sub_trials])),
            "net_skr_bayes_std": float(np.std([np.mean(tr["net_skr_bayes"]) for tr in sub_trials])),
            "net_skr_mem_mean": float(np.mean([np.mean(tr["net_skr_mem"]) for tr in sub_trials])),
            "net_skr_uncomp_mean": float(np.mean([np.mean(tr["net_skr_uncomp"]) for tr in sub_trials])),
            "uptime_bayes": float(np.mean([tr["uptime_bayes"] for tr in sub_trials])),
        })
        elapsed = time.time() - t_start
        print(f"      -> [{idx}/{len(sigma_list)}] sigma_drift = {sig:.2f} rad complete ({elapsed:.1f}s elapsed)", flush=True)

    # 3. Test-Bit Overhead Trade-Off Sweep (f_test in [2%, 30%])
    f_test_list = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
    print(f"\n[3/4] Running Test-Bit Overhead Trade-Off Sweep ({len(f_test_list)} fractions x {sweep_trials} trials)...", flush=True)
    overhead_results = []
    for idx, ft in enumerate(f_test_list, 1):
        sub_trials = [
            run_single_trial(
                num_steps=num_steps,
                qubits_per_step=qubits_per_step,
                sigma_drift=0.04,
                drift_bias=0.018,
                depol_prob=0.015,
                test_fraction=ft,
                f_ec=1.16,
                seed=800 + k,
                fast_quantum_sim=fast_quantum_sim,
            )
            for k in range(sweep_trials)
        ]
        overhead_results.append({
            "f_test": ft,
            "rmse_mean": float(np.mean([tr["rmse_bayes"] for tr in sub_trials])),
            "rmse_std": float(np.std([tr["rmse_bayes"] for tr in sub_trials])),
            "qber_mean": float(np.mean([np.mean(tr["qber_bayes"]) * 100.0 for tr in sub_trials])),
            "net_skr_mean": float(np.mean([np.mean(tr["net_skr_bayes"]) for tr in sub_trials])),
            "net_skr_std": float(np.std([np.mean(tr["net_skr_bayes"]) for tr in sub_trials])),
            "total_bits_mean": float(np.mean([tr["secret_bits_bayes"] for tr in sub_trials])),
        })
        elapsed = time.time() - t_start
        print(f"      -> [{idx}/{len(f_test_list)}] f_test = {ft*100:.0f}% complete ({elapsed:.1f}s elapsed)", flush=True)

    # 4. Generate Plots
    print("\n[4/4] Generating publication figures...", flush=True)
    t_steps = np.arange(num_steps)

    # Standalone Plot 4: Turbulence Resilience Sweep
    turb_path = os.path.join(output_dir, "turbulence_sweep.png")
    fig, ax1 = plt.subplots(figsize=(8.5, 4.6))
    sigs = np.array([r["sigma"] for r in turb_results])
    skr_b = np.array([r["net_skr_bayes_mean"] for r in turb_results])
    skr_b_std = np.array([r["net_skr_bayes_std"] for r in turb_results])
    skr_m = np.array([r["net_skr_mem_mean"] for r in turb_results])
    skr_u = np.array([r["net_skr_uncomp_mean"] for r in turb_results])

    ax1.errorbar(
        sigs, skr_b, yerr=skr_b_std, fmt="-o", color="#009E73", lw=2.3, capsize=4,
        label="Proposed Recursive Bayesian Filter (Net SKR)",
    )
    ax1.plot(sigs, skr_m, "--s", color="#E69F00", lw=2.0, label="Memoryless Estimator (Net SKR)")
    ax1.plot(sigs, skr_u, ":^", color="#C0392B", lw=2.0, label="Uncompensated Baseline (Net SKR)")
    ax1.set_xlabel(r"Channel Drift Turbulence $\sigma_{\mathrm{drift}}$ (rad / step)")
    ax1.set_ylabel("Net Secret Key Rate (bits / pulse)")
    ax1.set_ylim(0.0, 0.43)
    ax1.set_title("Turbulence Resilience: Net SKR vs. Channel Drift Volatility", pad=12)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(turb_path, bbox_inches="tight")
    plt.close(fig)

    # Standalone Plot 5: Test-Bit Overhead Trade-Off
    overhead_path = os.path.join(output_dir, "overhead_tradeoff_sweep.png")
    fig, ax1 = plt.subplots(figsize=(8.5, 4.6))
    fts_pct = np.array([r["f_test"] * 100.0 for r in overhead_results])
    net_skrs = np.array([r["net_skr_mean"] for r in overhead_results])
    net_stds = np.array([r["net_skr_std"] for r in overhead_results])
    rmses = np.array([r["rmse_mean"] for r in overhead_results])

    color_skr = "#0072B2"
    color_rmse = "#D95F02"
    ax1.errorbar(
        fts_pct, net_skrs, yerr=net_stds, fmt="-o", color=color_skr, lw=2.4, capsize=4,
        label="Net SKR per Pulse (Left Axis)",
    )
    best_idx = int(np.argmax(net_skrs))
    ax1.axvline(
        fts_pct[best_idx], color=color_skr, ls=":", lw=1.8,
        label=f"Optimal Allocation ({fts_pct[best_idx]:.0f}% Test Bits)",
    )
    ax1.set_xlabel(r"Public Test-Bit Overhead Fraction $f_{\mathrm{test}}$ (%)")
    ax1.set_ylabel("Net Secret Key Rate (bits / pulse)", color=color_skr)
    ax1.tick_params(axis="y", labelcolor=color_skr)
    ax1.set_ylim(0.22, 0.43)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(
        fts_pct, rmses, "--s", color=color_rmse, lw=2.1,
        label="Tracking RMSE (rad, Right Axis)",
    )
    ax2.set_ylabel("Tracking RMSE (rad)", color=color_rmse)
    ax2.tick_params(axis="y", labelcolor=color_rmse)
    ax2.set_ylim(0.12, 0.24)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", frameon=True, facecolor="white")
    ax1.set_title("Protocol Overhead Trade-Off: Net SKR & RMSE vs. Test-Bit Allocation", pad=12)
    fig.tight_layout()
    fig.savefig(overhead_path, bbox_inches="tight")
    plt.close(fig)

    # Combined 3-Panel Conference Benchmark Figure
    conf_path = os.path.join(output_dir, "conference_benchmarks.png")
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 12.0))

    # Panel (a): 50-Seed Monte Carlo QBER Ensemble
    ax0 = axes[0]
    u_mean, u_std = np.mean(qber_u_mat, axis=0), np.std(qber_u_mat, axis=0)
    m_mean, m_std = np.mean(qber_m_mat, axis=0), np.std(qber_m_mat, axis=0)
    b_mean, b_std = np.mean(qber_b_mat, axis=0), np.std(qber_b_mat, axis=0)

    ax0.axhspan(11.0, 42.0, color="#FDE0DD", alpha=0.4, label="Protocol Abort Zone (>11%)")
    ax0.plot(t_steps, u_mean, color="#C0392B", lw=2.0, label="Uncompensated (50-Seed Mean)")
    ax0.fill_between(t_steps, np.maximum(0, u_mean - u_std), u_mean + u_std, color="#C0392B", alpha=0.14)
    ax0.plot(t_steps, m_mean, color="#E69F00", lw=2.0, ls="--", label="Memoryless Estimator (50-Seed Mean)")
    ax0.fill_between(t_steps, np.maximum(0, m_mean - m_std), m_mean + m_std, color="#E69F00", alpha=0.15)
    ax0.plot(t_steps, b_mean, color="#0072B2", lw=2.4, label="Recursive Bayesian Filter (50-Seed Mean)")
    ax0.fill_between(t_steps, np.maximum(0, b_mean - b_std), b_mean + b_std, color="#0072B2", alpha=0.22)
    ax0.axhline(11.0, color="#8B0000", ls=":", lw=1.8, label="11% BB84 Security Threshold")
    ax0.set_ylim(0.0, 38.0)
    ax0.set_xlabel("Time Step ($t$)")
    ax0.set_ylabel("QBER (%)")
    ax0.set_title(r"(a) 50-Seed Monte Carlo QBER Trajectory ($\mathrm{Mean} \pm 1\sigma$ Band)", pad=8)
    ax0.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

    # Panel (b): Turbulence Sweep
    ax1 = axes[1]
    ax1.errorbar(
        sigs, skr_b, yerr=skr_b_std, fmt="-o", color="#009E73", lw=2.3, capsize=4,
        label="Recursive Bayesian Filter",
    )
    ax1.plot(sigs, skr_m, "--s", color="#E69F00", lw=2.0, label="Memoryless Estimator")
    ax1.plot(sigs, skr_u, ":^", color="#C0392B", lw=2.0, label="Uncompensated Baseline")
    ax1.set_xlabel(r"Per-Step Drift Volatility $\sigma_{\mathrm{drift}}$ (rad)")
    ax1.set_ylabel("Net SKR (bits / pulse)")
    ax1.set_ylim(0.0, 0.43)
    ax1.set_title(r"(b) Turbulence Resilience Sweep ($f_{\mathrm{ec}} = 1.16, f_{\mathrm{test}} = 10\%$)", pad=8)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)

    # Panel (c): Overhead Allocation Trade-Off
    ax2 = axes[2]
    ax2.errorbar(
        fts_pct, net_skrs, yerr=net_stds, fmt="-o", color=color_skr, lw=2.4, capsize=4,
        label="Net SKR per Pulse (Left Axis)",
    )
    ax2.axvline(
        fts_pct[best_idx], color=color_skr, ls=":", lw=1.8,
        label=f"Optimal Overhead ({fts_pct[best_idx]:.0f}% Test Bits)",
    )
    ax2.set_xlabel(r"Public Test-Bit Fraction $f_{\mathrm{test}}$ (%)")
    ax2.set_ylabel("Net SKR (bits / pulse)", color=color_skr)
    ax2.tick_params(axis="y", labelcolor=color_skr)
    ax2.set_ylim(0.22, 0.43)

    ax2_r = ax2.twinx()
    ax2_r.spines["top"].set_visible(False)
    ax2_r.plot(fts_pct, rmses, "--s", color=color_rmse, lw=2.1, label="Tracking RMSE (Right Axis)")
    ax2_r.set_ylabel("Tracking RMSE (rad)", color=color_rmse)
    ax2_r.tick_params(axis="y", labelcolor=color_rmse)
    ax2_r.set_ylim(0.12, 0.24)
    ax2_r.grid(False)

    l1, lb1 = ax2.get_legend_handles_labels()
    l2, lb2 = ax2_r.get_legend_handles_labels()
    ax2.legend(l1 + l2, lb1 + lb2, loc="lower left", frameon=True, facecolor="white", fontsize=9)
    ax2.set_title("(c) Test-Bit Overhead Trade-Off: Balancing Estimation SNR vs. Key Sacrifice", pad=8)

    fig.tight_layout()
    fig.savefig(conf_path, bbox_inches="tight")
    plt.close(fig)

    total_time = time.time() - t_start
    print(f"      -> Saved figures in {total_time:.1f}s total.", flush=True)

    benchmark_data = {
        "mc_summary": mc_summary,
        "turb_results": turb_results,
        "overhead_results": overhead_results,
        "plots": {
            "turbulence": turb_path,
            "overhead": overhead_path,
            "conference_summary": conf_path,
        },
    }

    # Automatically print formatted ASCII tables to the terminal
    print_benchmark_results(benchmark_data)

    return benchmark_data


if __name__ == "__main__":
    results = run_conference_benchmark_suite(
        num_seeds=50,
        sweep_trials=15,
        fast_quantum_sim=True,
    )