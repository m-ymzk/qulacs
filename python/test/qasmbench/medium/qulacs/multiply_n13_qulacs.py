import numpy as np
from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import Identity, U1, TOFFOLI, FREDKIN

def func(st, circ):
    # This initializes 13 quantum registers and 4 classical registers.
#    nqubits=13
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)                                           # qreg q[13];
#    circ = QuantumCircuit(nqubits)                                       #
                                                                         #
    circ.add_X_gate(0)                                                   # x q[0];
    circ.add_X_gate(1) # 1st 2 qubits 11 (3)                             # x q[1]; // 1st 2 qubits 11 (3)
    circ.add_X_gate(2)                                                   # x q[2];
    circ.add_X_gate(4) # next 3 qubits 101 (5)                           # x q[4]; // next 3 qubits 101 (5)
    # All qubits start at a ground state of 0 this changes 4 of the first 5 qubits to 1 so that I could use a binary 11 (digital 3) and a binary 101 (digital 5).#
                                                                         #
                                                                         # barrier q; // multiply
    circ.add_gate(TOFFOLI(2,  0,  5)) # LSQ                              # ccx q[2], q[0], q[5]; // LSQ
    circ.add_gate(TOFFOLI(2,  1,  6))                                    # ccx q[2], q[1], q[6];
    circ.add_gate(TOFFOLI(3,  0,  7))                                    # ccx q[3], q[0], q[7];
    circ.add_gate(TOFFOLI(3,  1,  8))                                    # ccx q[3], q[1], q[8];
    circ.add_gate(TOFFOLI(4,  0,  9))                                    # ccx q[4], q[0], q[9];
    circ.add_gate(TOFFOLI(4,  1,  10)) # MSQ                             # ccx q[4], q[1], q[10]; // MSQ
    # Multiplication is all AND gates: 1 x 1 = 1 all else is 0.          # // Multiplication is all AND gates: 1 x 1 = 1; all else is 0.
                                                                         #
                                                                         # barrier q; // add
    circ.add_CNOT_gate(6,  11)                                           # cx q[6], q[11];
    circ.add_CNOT_gate(7,  11) # 2nd digit                               # cx q[7], q[11]; // 2nd digit
    circ.add_CNOT_gate(8,  12)                                           # cx q[8], q[12];
    circ.add_CNOT_gate(9,  12) # 3rd digit                               # cx q[9], q[12]; // 3rd digit
    # With 3 x 5,  all addition can be done with simple XOR gates.       # // With 3 x 5, all addition can be done with simple XOR gates.
                                                                         #
                                                                         # barrier q; // measure

    circ.update_quantum_state(st)
    with open('Qulacs.log', 'w') as f:
        print(np.around(st.get_vector(),14)+0, file=f)

    # This measures the appropriate qubits and sends the output to the classical registers for display as a histogram.
