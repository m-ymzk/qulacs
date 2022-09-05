import numpy as np
from qulacs import QuantumCircuit, QuantumState

def func(st, circ):
#    nqubits=4
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)     # qreg bits[4];
#    circ = QuantumCircuit(nqubits)
    
    circ.add_H_gate(0)             # h bits[0];
    circ.add_CNOT_gate(0, 1)       # cx bits[0],bits[1];
    circ.add_CNOT_gate(1, 2)       # cx bits[1],bits[2];
    circ.add_CNOT_gate(2, 3)       # cx bits[2],bits[3];
    circ.update_quantum_state(st)
#    with open('Qulacs.log', 'w') as f:
#        print(np.around(st.get_vector(),14)+0, file=f)
