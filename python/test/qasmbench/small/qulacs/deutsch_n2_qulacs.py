# Implementation of Deutsch algorithm with two qubits for f(x)=x
import numpy as np
from qulacs import QuantumCircuit, QuantumState
def func(st, circ):
#    nqubits=2
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)     # qreg q[2];
#    circ = QuantumCircuit(nqubits)

    circ.add_X_gate(1)             # x q[1];
    circ.add_H_gate(0)             # h q[0];
    circ.add_H_gate(1)             # h q[1];
    circ.add_CNOT_gate(0, 1)       # cx q[0],q[1];
    circ.add_H_gate(0)             # h q[0];
    circ.update_quantum_state(st)
#    with open('Qulacs.log', 'w') as f:
#        print(np.around(st.get_vector(),14)+0, file=f)
