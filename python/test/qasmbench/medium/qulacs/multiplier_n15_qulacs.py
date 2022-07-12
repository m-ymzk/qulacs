# Generated from Cirq v0.8.0

import numpy as np
from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import Identity, U1, TOFFOLI, FREDKIN

def func(st, circ):
    # Qubits: [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14
#    nqubits=15
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)                              # qreg q[15];
#    circ = QuantumCircuit(nqubits)                          #
                                                            #
    circ.add_X_gate(13)                                     # x q[13];
    circ.add_X_gate(12)                                     # x q[12];
    circ.add_X_gate(10)                                     # x q[10];
    circ.add_X_gate(9)                                      # x q[9];
                                                            #
    # Gate: <__main__.Multiplier object at 0x7f0de092ec40>  # // Gate: <__main__.Multiplier object at 0x7f0de092ec40>
    circ.add_gate(TOFFOLI(12, 9, 1))                        # ccx q[12],q[9],q[1];
    circ.add_gate(TOFFOLI(12, 10, 4))                       # ccx q[12],q[10],q[4];
    circ.add_gate(TOFFOLI(12, 11, 7))                       # ccx q[12],q[11],q[7];
    circ.add_gate(TOFFOLI(1, 2, 3))                         # ccx q[1],q[2],q[3];
    circ.add_CNOT_gate(1, 2)                                # cx q[1],q[2];
    circ.add_gate(TOFFOLI(0, 2, 3))                         # ccx q[0],q[2],q[3];
    circ.add_gate(TOFFOLI(4, 5, 6))                         # ccx q[4],q[5],q[6];
    circ.add_CNOT_gate(4, 5)                                # cx q[4],q[5];
    circ.add_gate(TOFFOLI(3, 5, 6))                         # ccx q[3],q[5],q[6];
    circ.add_CNOT_gate(7, 8)                                # cx q[7],q[8];
    circ.add_CNOT_gate(6, 8)                                # cx q[6],q[8];
    circ.add_gate(TOFFOLI(3, 5, 6))                         # ccx q[3],q[5],q[6];
    circ.add_CNOT_gate(4, 5)                                # cx q[4],q[5];
    circ.add_gate(TOFFOLI(4, 5, 6))                         # ccx q[4],q[5],q[6];
    circ.add_CNOT_gate(4, 5)                                # cx q[4],q[5];
    circ.add_CNOT_gate(3, 5)                                # cx q[3],q[5];
    circ.add_gate(TOFFOLI(0, 2, 3))                         # ccx q[0],q[2],q[3];
    circ.add_CNOT_gate(1, 2)                                # cx q[1],q[2];
    circ.add_gate(TOFFOLI(1, 2, 3))                         # ccx q[1],q[2],q[3];
    circ.add_CNOT_gate(1, 2)                                # cx q[1],q[2];
    circ.add_CNOT_gate(0, 2)                                # cx q[0],q[2];
    circ.add_gate(TOFFOLI(12, 9, 1))                        # ccx q[12],q[9],q[1];
    circ.add_gate(TOFFOLI(12, 10, 4))                       # ccx q[12],q[10],q[4];
    circ.add_gate(TOFFOLI(12, 11, 7))                       # ccx q[12],q[11],q[7];
    circ.add_gate(TOFFOLI(13, 9, 4))                        # ccx q[13],q[9],q[4];
    circ.add_gate(TOFFOLI(13, 10, 7))                       # ccx q[13],q[10],q[7];
    circ.add_gate(TOFFOLI(1, 2, 3))                         # ccx q[1],q[2],q[3];
    circ.add_CNOT_gate(1, 2)                                # cx q[1],q[2];
    circ.add_gate(TOFFOLI(0, 2, 3))                         # ccx q[0],q[2],q[3];
    circ.add_gate(TOFFOLI(4, 5, 6))                         # ccx q[4],q[5],q[6];
    circ.add_CNOT_gate(4, 5)                                # cx q[4],q[5];
    circ.add_gate(TOFFOLI(3, 5, 6))                         # ccx q[3],q[5],q[6];
    circ.add_CNOT_gate(7, 8)                                # cx q[7],q[8];
    circ.add_CNOT_gate(6, 8)                                # cx q[6],q[8];
    circ.add_gate(TOFFOLI(3, 5, 6))                         # ccx q[3],q[5],q[6];
    circ.add_CNOT_gate(4, 5)                                # cx q[4],q[5];
    circ.add_gate(TOFFOLI(4, 5, 6))                         # ccx q[4],q[5],q[6];
    circ.add_CNOT_gate(4, 5)                                # cx q[4],q[5];
    circ.add_CNOT_gate(3, 5)                                # cx q[3],q[5];
    circ.add_gate(TOFFOLI(0, 2, 3))                         # ccx q[0],q[2],q[3];
    circ.add_CNOT_gate(1, 2)                                # cx q[1],q[2];
    circ.add_gate(TOFFOLI(1, 2, 3))                         # ccx q[1],q[2],q[3];
    circ.add_CNOT_gate(1, 2)                                # cx q[1],q[2];
    circ.add_CNOT_gate(0, 2)                                # cx q[0],q[2];
    circ.add_gate(TOFFOLI(13, 9, 4))                        # ccx q[13],q[9],q[4];
    circ.add_gate(TOFFOLI(13, 10, 7))                       # ccx q[13],q[10],q[7];
    circ.add_gate(TOFFOLI(14, 9, 7))                        # ccx q[14],q[9],q[7];
    circ.add_gate(TOFFOLI(1, 2, 3))                         # ccx q[1],q[2],q[3];
    circ.add_CNOT_gate(1, 2)                                # cx q[1],q[2];
    circ.add_gate(TOFFOLI(0, 2, 3))                         # ccx q[0],q[2],q[3];
    circ.add_gate(TOFFOLI(4, 5, 6))                         # ccx q[4],q[5],q[6];
    circ.add_CNOT_gate(4, 5)                                # cx q[4],q[5];
    circ.add_gate(TOFFOLI(3, 5, 6))                         # ccx q[3],q[5],q[6];
    circ.add_CNOT_gate(7, 8)                                # cx q[7],q[8];
    circ.add_CNOT_gate(6, 8)                                # cx q[6],q[8];
    circ.add_gate(TOFFOLI(3, 5, 6))                         # ccx q[3],q[5],q[6];
    circ.add_CNOT_gate(4, 5)                                # cx q[4],q[5];
    circ.add_gate(TOFFOLI(4, 5, 6))                         # ccx q[4],q[5],q[6];
    circ.add_CNOT_gate(4, 5)                                # cx q[4],q[5];
    circ.add_CNOT_gate(3, 5)                                # cx q[3],q[5];
    circ.add_gate(TOFFOLI(0, 2, 3))                         # ccx q[0],q[2],q[3];
    circ.add_CNOT_gate(1, 2)                                # cx q[1],q[2];
    circ.add_gate(TOFFOLI(1, 2, 3))                         # ccx q[1],q[2],q[3];
    circ.add_CNOT_gate(1, 2)                                # cx q[1],q[2];
    circ.add_CNOT_gate(0, 2)                                # cx q[0],q[2];
    circ.add_gate(TOFFOLI(14, 9, 7))                        # ccx q[14],q[9],q[7];

    circ.update_quantum_state(st)
#    with open('Qulacs.log', 'w') as f:
#        print(np.around(st.get_vector(),14)+0, file=f)
