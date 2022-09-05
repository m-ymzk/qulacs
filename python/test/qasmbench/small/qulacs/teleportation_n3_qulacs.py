# Teleportation using 3 qubits.
# Description: Based on the example given by S. Fedortchenko (https://arxiv.org/pdf/1607.02398.pdf) 

import numpy as np
from qulacs import QuantumCircuit, QuantumState

def func(st, circ):
#    nqubits=3
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)     # qreg q[3];
#    circ = QuantumCircuit(nqubits)

    circ.add_H_gate(0)             # h q[0];
    circ.add_T_gate(0)             # t q[0];
    circ.add_H_gate(0)             # h q[0];
    circ.add_H_gate(2)             # h q[2];
    circ.add_S_gate(0)             # s q[0];
    circ.add_CNOT_gate(2, 1)       # cx q[2],q[1];
    circ.add_CNOT_gate(0, 1)       # cx q[0],q[1];
    circ.add_H_gate(0)             # h q[0];
    circ.update_quantum_state(st)
#    with open('Qulacs.log', 'w') as f:
#        print(np.around(st.get_vector(),14)+0, file=f)
