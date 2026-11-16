import numpy as np
from qiskit_aer import Aer


def generate_bases(n):
    return np.random.randint(2, size=n)


def apply_active_correction(circuits, estimated_theta):
    
    corrected_circuits = []

    for i in range(len(circuits)):
        qc = circuits[i].copy()

        if estimated_theta is not None and float(estimated_theta) != 0.0:
            qc.rz(-float(estimated_theta), 0)

        corrected_circuits.append(qc)

    return corrected_circuits


def measure_qubits(circuits, bases, estimated_theta=None):
    
    backend = Aer.get_backend('aer_simulator')

    measured_circuits = []

    # STEP 1 — Prepare all circuits (optional active R_z compensation + basis + measurement)
    for i in range(len(circuits)):

        qc = circuits[i].copy()

        # Apply active physical-layer R_z(-hat{theta}_{t-1}) compensation if requested
        if estimated_theta is not None and float(estimated_theta) != 0.0:
            qc.rz(-float(estimated_theta), 0)

        # Bob applies measurement basis
        if bases[i] == 1:
            qc.h(0)

        # Measure qubit
        qc.measure(0, 0)

        measured_circuits.append(qc)

    # STEP 2 — Execute ALL circuits together
    job = backend.run(measured_circuits, shots=1)

    result = job.result()

    # STEP 3 — Extract results
    results = []

    for i in range(len(measured_circuits)):

        counts = result.get_counts(i)

        measured_bit = int(list(counts.keys())[0])

        results.append(measured_bit)

    return results