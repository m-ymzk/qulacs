# Generated from Cirq v0.8.0

import numpy as np
from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import Identity, U1

def func(st, circ):

    # Qubits: [0,  1,  2,  3,  4,  5,  6,  7
    #nqubits=8
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
    #st = QuantumState(nqubits)                         # qreg q[8];
    #circ = QuantumCircuit(nqubits)                     #
    #
    circ.add_X_gate(0)                                 # x q[0];
    circ.add_H_gate(1)                                 # h q[1];
    circ.add_X_gate(2)                                 # x q[2];
    circ.add_X_gate(3)                                 # x q[3];
    circ.add_X_gate(4)                                 # x q[4];
    circ.add_X_gate(5)                                 # x q[5];
    circ.add_H_gate(7)                                 # h q[7];
    circ.add_H_gate(5)                                 # h q[5];
    circ.add_H_gate(1)                                 # h q[1];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_H_gate(4)                                 # h q[4];
    circ.add_H_gate(7)                                 # h q[7];
    circ.add_X_gate(0)                                 # x q[0];
    circ.add_H_gate(1)                                 # h q[1];
    circ.add_X_gate(2)                                 # x q[2];
    circ.add_X_gate(3)                                 # x q[3];
    circ.add_X_gate(4)                                 # x q[4];
    circ.add_H_gate(7)                                 # h q[7];
    circ.add_H_gate(5)                                 # h q[5];
    circ.add_H_gate(6)                                 # h q[6];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_H_gate(4)                                 # h q[4];
    circ.add_H_gate(1)                                 # h q[1];
    circ.add_H_gate(3)                                 # h q[3];
    circ.add_H_gate(7)                                 # h q[7];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_H_gate(4)                                 # h q[4];
    circ.update_quantum_state(st)
#    with open('Qulacs.log', 'w') as f:
#        print(np.around(st.get_vector(),14)+0, file=f)

