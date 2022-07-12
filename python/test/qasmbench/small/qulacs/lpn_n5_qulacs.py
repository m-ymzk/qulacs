# Name of Experiment: LPN circuit 2 v1

import numpy as np
from qulacs import QuantumCircuit, QuantumState

def func(st, circ):
#    nqubits=5
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)     # qreg q[5];
#    circ = QuantumCircuit(nqubits)

    circ.add_H_gate(0)             # h q[0];
    circ.add_H_gate(1)             # h q[1];
    circ.add_H_gate(3)             # h q[3];
    circ.add_H_gate(4)             # h q[4];
    circ.add_CNOT_gate(3, 2)       # cx q[3], q[2];
    circ.add_CNOT_gate(0, 2)       # cx q[0], q[2];
    circ.add_H_gate(0)             # h q[0];
    circ.add_H_gate(1)             # h q[1];
    circ.add_H_gate(2)             # h q[2];
    circ.add_H_gate(3)             # h q[3];
    circ.add_H_gate(4)             # h q[4];
    circ.update_quantum_state(st)
#    with open('Qulacs.log', 'w') as f:
#        print(np.around(st.get_vector(),14)+0, file=f)
