import numpy as np
from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import Identity, U1, TOFFOLI, FREDKIN

def func(st, circ):
#    nqubits=11
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)                         # qreg q[11];
#    circ = QuantumCircuit(nqubits)                     #
                                                       #
    circ.add_Z_gate(0)                                 # z q[0];
    circ.add_H_gate(0) # secret unitary: hz            # h q[0]; // secret unitary: hz
                                                       #
                                                       # barrier q; // Shor's error correction algorithm
    circ.add_CNOT_gate(0,  3)                          # cx q[0], q[3];
    circ.add_CNOT_gate(0,  6)                          # cx q[0], q[6];
    circ.add_CZ_gate(0,  3)                            # cz q[0], q[3];
    circ.add_CZ_gate(0,  6)                            # cz q[0], q[6];
    circ.add_H_gate(0)                                 # h q[0];
    circ.add_H_gate(3)                                 # h q[3];
    circ.add_H_gate(6)                                 # h q[6];
    circ.add_Z_gate(0)                                 # z q[0];
    circ.add_Z_gate(3)                                 # z q[3];
    circ.add_Z_gate(6)                                 # z q[6];
    circ.add_CNOT_gate(0,  1)                          # cx q[0], q[1];
    circ.add_CNOT_gate(0,  2)                          # cx q[0], q[2];
    circ.add_CNOT_gate(3,  4)                          # cx q[3], q[4];
    circ.add_CNOT_gate(3,  5)                          # cx q[3], q[5];
    circ.add_CNOT_gate(6,  7)                          # cx q[6], q[7];
    circ.add_CNOT_gate(6,  8)                          # cx q[6], q[8];
    circ.add_CZ_gate(0,  1)                            # cz q[0], q[1];
    circ.add_CZ_gate(0,  2)                            # cz q[0], q[2];
    circ.add_CZ_gate(3,  4)                            # cz q[3], q[4];
    circ.add_CZ_gate(3,  5)                            # cz q[3], q[5];
    circ.add_CZ_gate(6,  7)                            # cz q[6], q[7];
    circ.add_CZ_gate(6,  8)                            # cz q[6], q[8];
                                                       #
                                                       # // Alice starts with qubit 9.
                                                       # // Bob starts with qubit 10.
                                                       # // Alice is given qubit 0.
                                                       # // Bob is given error-correcting qubits 1-8.
                                                       # // Alice and Bob do not know what has been done to qubit 0.
                                                       #
                                                       # barrier q; // Alice and Bob entangle their starting qubits.
    circ.add_H_gate(9)                                 # h q[9];
    circ.add_CNOT_gate(9,  10)                         # cx q[9], q[10];
                                                       #
    # Alice keeps qubits 0 and 9.                      # // Alice keeps qubits 0 and 9.
    # Bob leaves with qubits 1-8 and 10.               # // Bob leaves with qubits 1-8 and 10.
                                                       #
                                                       # barrier q; // Alice teleports the quantum state of qubit 0 to Bob's qubit.
    circ.add_CNOT_gate(0,  9)                          # cx q[0], q[9];

    circ.add_H_gate(0)                                 # h q[0];
    circ.add_CNOT_gate(9,  10)                         # cx q[9], q[10];

    circ.add_CZ_gate(0,  10)                           # cz q[0], q[10];
                                                       #
                                                       # barrier q; // Bob corrects for bit flips and sign flips
    circ.add_CNOT_gate(10,  1)                         # cx q[10], q[1];
    circ.add_CNOT_gate(10,  2)                         # cx q[10], q[2];
    circ.add_CNOT_gate(3,  4)                          # cx q[3], q[4];
    circ.add_CNOT_gate(3,  5)                          # cx q[3], q[5];
    circ.add_CNOT_gate(6,  7)                          # cx q[6], q[7];
    circ.add_CNOT_gate(6,  8)                          # cx q[6], q[8];
    circ.add_CZ_gate(10,  1)                           # cz q[10], q[1];
    circ.add_CZ_gate(10,  2)                           # cz q[10], q[2];
    circ.add_CZ_gate(3,  4)                            # cz q[3], q[4];
    circ.add_CZ_gate(3,  5)                            # cz q[3], q[5];
    circ.add_CZ_gate(6,  7)                            # cz q[6], q[7];
    circ.add_CZ_gate(6,  8)                            # cz q[6], q[8];
    circ.add_gate(TOFFOLI(1,  2,  10))                 # ccx q[1], q[2], q[10];
    circ.add_gate(TOFFOLI(5,  4,  3))                  # ccx q[5], q[4], q[3];
    circ.add_gate(TOFFOLI(8,  7,  6))                  # ccx q[8], q[7], q[6];
                                                       # barrier q; // start CCZ gates
    circ.add_H_gate(10)                                # h q[10];
    circ.add_gate(TOFFOLI(1,  2,  10))                 # ccx q[1], q[2], q[10];
    circ.add_H_gate(10)                                # h q[10];
    circ.add_H_gate(3)                                 # h q[3];
    circ.add_gate(TOFFOLI(5,  4,  3))                  # ccx q[5], q[4], q[3];
    circ.add_H_gate(3)                                 # h q[3];
    circ.add_H_gate(6)                                 # h q[6];
    circ.add_gate(TOFFOLI(8,  7,  6))                  # ccx q[8], q[7], q[6];
    circ.add_H_gate(6)                                 # h q[6];
                                                       # barrier q; // end CCZ gates
    circ.add_H_gate(10)                                # h q[10];
    circ.add_H_gate(3)                                 # h q[3];
    circ.add_H_gate(6)                                 # h q[6];
    circ.add_Z_gate(10)                                # z q[10];
    circ.add_Z_gate(3)                                 # z q[3];
    circ.add_Z_gate(6)                                 # z q[6];
    circ.add_CNOT_gate(10,  3)                         # cx q[10], q[3];
    circ.add_CNOT_gate(10,  6)                         # cx q[10], q[6];
    circ.add_CZ_gate(10,  3)                           # cz q[10], q[3];
    circ.add_CZ_gate(10,  6)                           # cz q[10], q[6];
    circ.add_gate(TOFFOLI(3,  6,  10))                 # ccx q[3], q[6], q[10];
    circ.add_H_gate(10)                                # h q[10];
    circ.add_gate(TOFFOLI(3,  6,  10))                 # ccx q[3], q[6], q[10];
    circ.add_H_gate(10)                                # h q[10];
                                                       #
                                                       # barrier q; // Based on Alice's measurements, Bob reverses the secret unitary.
    # 00 do nothing                                    # // 00 do nothing
    # 01 apply X                                       # // 01 apply X
    # 10 apply Z                                       # // 10 apply Z
    # 11 apply ZX                                      # // 11 apply ZX
    circ.add_H_gate(10)                                # h q[10];
    circ.add_Z_gate(10)                                # z q[10];


    circ.update_quantum_state(st)
#    with open('Qulacs.log', 'w') as f:
#        print(np.around(st.get_vector(),14)+0, file=f)
