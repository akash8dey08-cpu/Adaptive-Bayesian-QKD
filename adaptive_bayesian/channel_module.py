import numpy as np
from qiskit import QuantumCircuit

try:
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    HAS_AER_NOISE = True
except ImportError:
    HAS_AER_NOISE = False


def update_drift_angle(
    prev_theta: float,
    sigma_drift: float,
    drift_bias: float = 0.018,
) -> float:
    
    delta_theta = np.random.normal(loc=drift_bias, scale=sigma_drift)
    return float(prev_theta + delta_theta)


def apply_depolarizing_channel_circuit(qc: QuantumCircuit, depol_prob: float) -> None:
    
    if depol_prob <= 0.0:
        return

    r = np.random.random()
    if r < depol_prob / 3.0:
        qc.x(0)
    elif r < 2.0 * depol_prob / 3.0:
        qc.y(0)
    elif r < depol_prob:
        qc.z(0)


def build_aer_depolarizing_noise_model(depol_prob: float):
    
    if not HAS_AER_NOISE:
        raise ImportError("qiskit_aer is required to build an Aer NoiseModel.")

    noise_model = NoiseModel()
    if depol_prob > 0.0:
        error_1q = depolarizing_error(depol_prob, 1)
        noise_model.add_all_qubit_quantum_error(error_1q, ["id", "x", "h", "rz"])
    return noise_model


def apply_channel_effects(
    circuits: list,
    theta_t: float,
    depol_prob: float = 0.0,
    use_circuit_depol: bool = True,
    copy_circuits: bool = True,
) -> list:
    
    noisy_circuits = []

    for qc in circuits:
        channel_qc = qc.copy() if copy_circuits else qc

        # Apply time-varying polarization/phase drift R_z(theta_t)
        if theta_t != 0.0:
            channel_qc.rz(float(theta_t), 0)

        # Apply baseline depolarizing noise
        if use_circuit_depol and depol_prob > 0.0:
            apply_depolarizing_channel_circuit(channel_qc, depol_prob)

        noisy_circuits.append(channel_qc)

    return noisy_circuits


def transmit_through_channel(
    circuits: list,
    prev_theta: float,
    sigma_drift: float,
    depol_prob: float = 0.0,
    drift_bias: float = 0.018,
    use_circuit_depol: bool = True,
    copy_circuits: bool = True,
) -> tuple:
    
    theta_t = update_drift_angle(
        prev_theta=prev_theta,
        sigma_drift=sigma_drift,
        drift_bias=drift_bias,
    )
    noisy_circuits = apply_channel_effects(
        circuits=circuits,
        theta_t=theta_t,
        depol_prob=depol_prob,
        use_circuit_depol=use_circuit_depol,
        copy_circuits=copy_circuits,
    )
    return theta_t, noisy_circuits