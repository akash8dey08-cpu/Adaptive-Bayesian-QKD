import numpy as np
from qiskit import QuantumCircuit


def generate_bits_bases(n):
    bits = np.random.randint(2, size=n)
    bases = np.random.randint(2, size=n)
    return bits, bases

def encode_qubits(bits, bases):
    circuits = []

    for i in range(len(bits)):
        qc = QuantumCircuit(1, 1)

        # Encode bit
        if bits[i] == 1:
            qc.x(0)

        # Apply basis
        if bases[i] == 1:  # diagonal basis
            qc.h(0)

        circuits.append(qc)

    return circuits


# Wrapper function 
def prepare_qubits(n):
    bits, bases = generate_bits_bases(n)
    circuits = encode_qubits(bits, bases)

    return bits, bases, circuits
