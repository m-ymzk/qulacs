import numpy as np
from qulacs import QuantumCircuit, QuantumState

def func(st, circ):
#    nqubits=3
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)     # qreg q[3];
#    circ = QuantumCircuit(nqubits)

    circ.add_X_gate(0)             # x q[0];
    circ.add_X_gate(1)             # x q[1];
    circ.add_CNOT_gate(2, 1)       # cx q[2],q[1];
    circ.add_CNOT_gate(0, 1)       # cx q[0],q[1];
    circ.add_H_gate(2)             # h q[2];
    circ.add_T_gate(0)             # t q[0];
    circ.add_Tdag_gate(1)          # tdg q[1];
    circ.add_T_gate(2)             # t q[2];
    circ.add_CNOT_gate(2, 1)       # cx q[2],q[1];
    circ.add_CNOT_gate(0, 2)       # cx q[0],q[2];
    circ.add_T_gate(1)             # t q[1];
    circ.add_CNOT_gate(0, 1)       # cx q[0],q[1];
    circ.add_Tdag_gate(2)          # tdg q[2];
    circ.add_Tdag_gate(1)          # tdg q[1];
    circ.add_CNOT_gate(0, 2)       # cx q[0],q[2];
    circ.add_CNOT_gate(2, 1)       # cx q[2],q[1];
    circ.add_T_gate(1)             # t q[1];
    circ.add_H_gate(2)             # h q[2];
    circ.add_CNOT_gate(2, 1)       # cx q[2],q[1];
    circ.update_quantum_state(st)
#    with open('Qulacs.log', 'w') as f:
#        print(np.around(st.get_vector(),14)+0, file=f)
