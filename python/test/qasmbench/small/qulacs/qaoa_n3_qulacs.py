# Finding the max of the cost function: C = -1 + z(0)z(2) - 2 z(0)z(1)z(2) - 3 z(1)
# Starting with p = 1
# Generated from Cirq v0.8.0

import numpy as np
from qulacs import QuantumCircuit, QuantumState

def func(st, circ):
    # Qubits: [(0,  0),  (1,  0),  (2,  0)
#    nqubits=3
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)                         # qreg q[3];
#    circ = QuantumCircuit(nqubits)

    circ.add_H_gate(0)                                 # h q[0];
    circ.add_H_gate(1)                                 # h q[1];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_CNOT_gate(0, 2)                           # cx q[0],q[2];
    circ.add_RZ_gate(2, np.pi*1.79986*-1)              # rz(pi*1.79986) q[2];
    circ.add_CNOT_gate(0, 2)                           # cx q[0],q[2];
    circ.add_CNOT_gate(0, 1)                           # cx q[0],q[1];
    circ.add_CNOT_gate(1, 2)                           # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*-3.59973*-1)             # rz(pi*-3.59973) q[2];
    circ.add_CNOT_gate(1, 2)                           # cx q[1],q[2];
    circ.add_CNOT_gate(0, 1)                           # cx q[0],q[1];
    circ.add_RX_gate(2, np.pi*0.545344*-1)             # rx(pi*0.545344) q[2];
    circ.add_RZ_gate(1, np.pi*-5.39959*-1)             # rz(pi*-5.39959) q[1];
    circ.add_RX_gate(0, np.pi*0.545344*-1)             # rx(pi*0.545344) q[0];

    circ.add_RX_gate(1, np.pi*0.545344*-1)             # rx(pi*0.545344) q[1];

    circ.update_quantum_state(st)
#    with open('Qulacs.log', 'w') as f:
#        print(np.around(st.get_vector(),14)+0, file=f)
