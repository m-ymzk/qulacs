"""
build a QuantumVolume Circuit with parameter
"""

from qulacs import QuantumCircuit
import numpy as np

def build_circuit(nqubits=20, global_nqubits=2, depth=10, verbose=False, random_gen=""):
    local_nqubits = nqubits - global_nqubits
    if random_gen == "":
        rng = np.random.default_rng()
    else:
        rng = random_gen

    circuit = QuantumCircuit(nqubits)

    for d in range(depth):
        for q in range(nqubits):
            circuit.add_H_gate(q)

    if verbose: print("circuit=", circuit)

    return circuit

