import numpy as np
from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import Identity, U1, TOFFOLI, FREDKIN

def func(st, circ):
#    nqubits=6
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)                         # qreg q[6];
#    circ = QuantumCircuit(nqubits)                     #
                                                       # // This initializes 6 quantum registers and 6 classical registers.
                                                       #
    circ.add_H_gate(0)                                 # h q[0];
    circ.add_H_gate(1)                                 # h q[1];
    circ.add_H_gate(2)                                 # h q[2];
                                                       # // The first 3 qubits are put into superposition states.
                                                       #
                                                       # barrier q;
    circ.add_CNOT_gate(2,  4)                          # cx q[2], q[4];
    circ.add_X_gate(3)                                 # x q[3];
    circ.add_CNOT_gate(2,  3)                          # cx q[2], q[3];
    circ.add_gate(TOFFOLI(0,  1,  3))                  # ccx q[0], q[1], q[3];
    circ.add_X_gate(0)                                 # x q[0];
    circ.add_X_gate(1)                                 # x q[1];
    circ.add_gate(TOFFOLI(0,  1,  3))                  # ccx q[0], q[1], q[3];
    circ.add_X_gate(0)                                 # x q[0];
    circ.add_X_gate(1)                                 # x q[1];
    circ.add_X_gate(3)                                 # x q[3];
                                                       # // This applies the secret structure: s=110.
                                                       #
                                                       # barrier q;
    circ.add_H_gate(0)                                 # h q[0];
    circ.add_H_gate(1)                                 # h q[1];
    circ.add_H_gate(2)                                 # h q[2];


    circ.update_quantum_state(st)
#    with open('Qulacs.log', 'w') as f:
#        print(np.around(st.get_vector(),14)+0, file=f)
