import numpy as np
from qulacs import QuantumCircuit, QuantumState

def func(st, circ):
#    nqubits=4
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)     # qreg q[4];
#    circ = QuantumCircuit(nqubits)

    circ.add_H_gate(0)             # h q[0];
    circ.add_H_gate(1)             # h q[1];
    circ.add_H_gate(2)             # h q[2];
    circ.add_H_gate(3)             # h q[3];
    circ.update_quantum_state(st)
#    with open('Qulacs.log', 'w') as f:
#        print(np.around(st.get_vector(),14)+0, file=f)
