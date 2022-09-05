import numpy as np
from qulacs import QuantumCircuit, QuantumState

def func(st, circ):
#    nqubits=3
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)     # qreg a[3];
#    circ = QuantumCircuit(nqubits)

    circ.add_X_gate(0)             # x a[0];
    circ.add_X_gate(1)             # x a[1];
    circ.add_H_gate(2)             # h a[2];
    circ.add_CNOT_gate(1, 2)       # cx a[1],a[2];
    circ.add_Tdag_gate(2)          # tdg a[2];
    circ.add_CNOT_gate(0, 2)       # cx a[0],a[2];
    circ.add_T_gate(2)             # t a[2];
    circ.add_CNOT_gate(1, 2)       # cx a[1],a[2];
    circ.add_Tdag_gate(2)          # tdg a[2];
    circ.add_CNOT_gate(0, 2)       # cx a[0],a[2];
    circ.add_Tdag_gate(1)          # tdg a[1];
    circ.add_T_gate(2)             # t a[2];
    circ.add_CNOT_gate(0, 1)       # cx a[0],a[1];
    circ.add_H_gate(2)             # h a[2];
    circ.add_Tdag_gate(1)          # tdg a[1];
    circ.add_CNOT_gate(0, 1)       # cx a[0],a[1];
    circ.add_T_gate(0)             # t a[0];
    circ.add_S_gate(1)             # s a[1];
    circ.update_quantum_state(st)
#    with open('Qulacs.log', 'w') as f:
#        print(np.around(st.get_vector(),14)+0, file=f)
