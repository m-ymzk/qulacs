# EIGHT QUBIT DEEP (16 dimensions )
# Generated from Cirq v0.8.0

import numpy as np
from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import Identity, U1

def func(st, circ):

    # Qubits: [(0,  0),  (0,  1),  (0,  2),  (0,  3),  (0,  4),  (0,  5),  (0,  6),  (0,  7)
#    nqubits=8
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)                            # qreg q[8];
#    circ = QuantumCircuit(nqubits)                        #
                                                          #
    circ.add_RX_gate(0, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[0];
    circ.add_RY_gate(0, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[0];
    circ.add_RZ_gate(0, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[0];
    circ.add_RX_gate(1, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[1];
    circ.add_RY_gate(1, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[1];
    circ.add_RZ_gate(1, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[1];
                                                          #
    # Gate: ZZ**1.1                                       # // Gate: ZZ**1.1
    circ.add_RZ_gate(0, np.pi*1.1*-1)                     # rz(pi*1.1) q[0];
    circ.add_RZ_gate(1, np.pi*1.1*-1)                     # rz(pi*1.1) q[1];
    circ.add_U3_gate(0, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                     # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.4*-1)                     # rx(pi*0.4) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                     # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                     # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.65, np.pi*1.0) # u3(pi*0.5,pi*0.65,pi*1.0) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.15, 0)         # u3(pi*0.5,pi*0.15,0) q[1];
                                                          #
    # Gate: YY**1.1                                       # // Gate: YY**1.1
    circ.add_U3_gate(0, 0, np.pi*1.0, np.pi*0.5)          # u3(0,pi*1.0,pi*0.5) q[0];
    circ.add_U3_gate(1, 0, 0, np.pi*0.5)                  # u3(0,0,pi*0.5) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                     # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.4*-1)                     # rx(pi*0.4) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                     # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                     # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*1.0, 0, np.pi*0.5)          # u3(pi*1.0,0,pi*0.5) q[0];
    circ.add_U3_gate(1, np.pi*1.0, 0, np.pi*1.5)          # u3(pi*1.0,0,pi*1.5) q[1];
                                                          #
    # Gate: XX**1.1                                       # // Gate: XX**1.1
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.5, np.pi*1.5)  # u3(pi*0.5,pi*1.5,pi*1.5) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                     # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.4*-1)                     # rx(pi*0.4) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                     # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                     # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.5, np.pi*0.5)  # u3(pi*0.5,pi*0.5,pi*0.5) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[1];
                                                          #
    circ.add_RX_gate(0, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[0];
    circ.add_RY_gate(0, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[0];
    circ.add_RZ_gate(0, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[0];
    circ.add_RX_gate(1, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[1];
    circ.add_RY_gate(1, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[1];
    circ.add_RZ_gate(1, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[1];
    circ.add_RX_gate(2, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[2];
    circ.add_RY_gate(2, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[2];
    circ.add_RZ_gate(2, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[2];
    circ.add_RX_gate(3, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[3];
    circ.add_RY_gate(3, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[3];
    circ.add_RZ_gate(3, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[3];
                                                          #
    # Gate: ZZ**1.1                                       # // Gate: ZZ**1.1
    circ.add_RZ_gate(2, np.pi*1.1*-1)                     # rz(pi*1.1) q[2];
    circ.add_RZ_gate(3, np.pi*1.1*-1)                     # rz(pi*1.1) q[3];
    circ.add_U3_gate(2, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[3];
    circ.add_RX_gate(2, np.pi*0.5*-1)                     # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                              # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*0.4*-1)                     # rx(pi*0.4) q[2];
    circ.add_RY_gate(3, np.pi*0.5*-1)                     # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                              # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                     # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(2, 3)                              # cx q[2],q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*0.65, np.pi*1.0) # u3(pi*0.5,pi*0.65,pi*1.0) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.15, 0)         # u3(pi*0.5,pi*0.15,0) q[3];
                                                          #
    # Gate: YY**1.1                                       # // Gate: YY**1.1
    circ.add_U3_gate(2, 0, np.pi*1.0, np.pi*0.5)          # u3(0,pi*1.0,pi*0.5) q[2];
    circ.add_U3_gate(3, 0, 0, np.pi*0.5)                  # u3(0,0,pi*0.5) q[3];
    circ.add_RX_gate(2, np.pi*0.5*-1)                     # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                              # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*0.4*-1)                     # rx(pi*0.4) q[2];
    circ.add_RY_gate(3, np.pi*0.5*-1)                     # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                              # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                     # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(2, 3)                              # cx q[2],q[3];
    circ.add_U3_gate(2, np.pi*1.0, 0, np.pi*0.5)          # u3(pi*1.0,0,pi*0.5) q[2];
    circ.add_U3_gate(3, np.pi*1.0, 0, np.pi*1.5)          # u3(pi*1.0,0,pi*1.5) q[3];
                                                          #
    # Gate: XX**1.1                                       # // Gate: XX**1.1
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.5, np.pi*1.5)  # u3(pi*0.5,pi*1.5,pi*1.5) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[3];
    circ.add_RX_gate(2, np.pi*0.5*-1)                     # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                              # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*0.4*-1)                     # rx(pi*0.4) q[2];
    circ.add_RY_gate(3, np.pi*0.5*-1)                     # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                              # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                     # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(2, 3)                              # cx q[2],q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*0.5, np.pi*0.5)  # u3(pi*0.5,pi*0.5,pi*0.5) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[3];
                                                          #
    circ.add_RX_gate(2, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[2];
    circ.add_RY_gate(2, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[2];
    circ.add_RZ_gate(2, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[2];
    circ.add_RX_gate(3, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[3];
    circ.add_RY_gate(3, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[3];
    circ.add_RZ_gate(3, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[3];
    circ.add_RX_gate(4, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[4];
    circ.add_RY_gate(4, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[4];
    circ.add_RZ_gate(4, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[4];
    circ.add_RX_gate(5, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[5];
    circ.add_RY_gate(5, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[5];
    circ.add_RZ_gate(5, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[5];
                                                          #
    # Gate: ZZ**1.1                                       # // Gate: ZZ**1.1
    circ.add_RZ_gate(4, np.pi*1.1*-1)                     # rz(pi*1.1) q[4];
    circ.add_RZ_gate(5, np.pi*1.1*-1)                     # rz(pi*1.1) q[5];
    circ.add_U3_gate(4, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[4];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[5];
    circ.add_RX_gate(4, np.pi*0.5*-1)                     # rx(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 5)                              # cx q[4],q[5];
    circ.add_RX_gate(4, np.pi*0.4*-1)                     # rx(pi*0.4) q[4];
    circ.add_RY_gate(5, np.pi*0.5*-1)                     # ry(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 4)                              # cx q[5],q[4];
    circ.add_RX_gate(5, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[5];
    circ.add_RZ_gate(5, np.pi*0.5*-1)                     # rz(pi*0.5) q[5];
    circ.add_CNOT_gate(4, 5)                              # cx q[4],q[5];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*0.65, np.pi*1.0) # u3(pi*0.5,pi*0.65,pi*1.0) q[4];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*0.15, 0)         # u3(pi*0.5,pi*0.15,0) q[5];
                                                          #
    # Gate: YY**1.1                                       # // Gate: YY**1.1
    circ.add_U3_gate(4, 0, np.pi*1.0, np.pi*0.5)          # u3(0,pi*1.0,pi*0.5) q[4];
    circ.add_U3_gate(5, 0, 0, np.pi*0.5)                  # u3(0,0,pi*0.5) q[5];
    circ.add_RX_gate(4, np.pi*0.5*-1)                     # rx(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 5)                              # cx q[4],q[5];
    circ.add_RX_gate(4, np.pi*0.4*-1)                     # rx(pi*0.4) q[4];
    circ.add_RY_gate(5, np.pi*0.5*-1)                     # ry(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 4)                              # cx q[5],q[4];
    circ.add_RX_gate(5, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[5];
    circ.add_RZ_gate(5, np.pi*0.5*-1)                     # rz(pi*0.5) q[5];
    circ.add_CNOT_gate(4, 5)                              # cx q[4],q[5];
    circ.add_U3_gate(4, np.pi*1.0, 0, np.pi*0.5)          # u3(pi*1.0,0,pi*0.5) q[4];
    circ.add_U3_gate(5, np.pi*1.0, 0, np.pi*1.5)          # u3(pi*1.0,0,pi*1.5) q[5];
                                                          #
    # Gate: XX**1.1                                       # // Gate: XX**1.1
    circ.add_U3_gate(4, np.pi*0.5, np.pi*1.5, np.pi*1.5)  # u3(pi*0.5,pi*1.5,pi*1.5) q[4];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[5];
    circ.add_RX_gate(4, np.pi*0.5*-1)                     # rx(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 5)                              # cx q[4],q[5];
    circ.add_RX_gate(4, np.pi*0.4*-1)                     # rx(pi*0.4) q[4];
    circ.add_RY_gate(5, np.pi*0.5*-1)                     # ry(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 4)                              # cx q[5],q[4];
    circ.add_RX_gate(5, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[5];
    circ.add_RZ_gate(5, np.pi*0.5*-1)                     # rz(pi*0.5) q[5];
    circ.add_CNOT_gate(4, 5)                              # cx q[4],q[5];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*0.5, np.pi*0.5)  # u3(pi*0.5,pi*0.5,pi*0.5) q[4];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[5];
                                                          #
    circ.add_RX_gate(4, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[4];
    circ.add_RY_gate(4, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[4];
    circ.add_RZ_gate(4, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[4];
    circ.add_RX_gate(5, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[5];
    circ.add_RY_gate(5, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[5];
    circ.add_RZ_gate(5, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[5];
    circ.add_RX_gate(6, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[6];
    circ.add_RY_gate(6, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[6];
    circ.add_RZ_gate(6, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[6];
    circ.add_RX_gate(7, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[7];
    circ.add_RY_gate(7, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[7];
    circ.add_RZ_gate(7, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[7];
                                                          #
    # Gate: ZZ**1.1                                       # // Gate: ZZ**1.1
    circ.add_RZ_gate(6, np.pi*1.1*-1)                     # rz(pi*1.1) q[6];
    circ.add_RZ_gate(7, np.pi*1.1*-1)                     # rz(pi*1.1) q[7];
    circ.add_U3_gate(6, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[6];
    circ.add_U3_gate(7, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[7];
    circ.add_RX_gate(6, np.pi*0.5*-1)                     # rx(pi*0.5) q[6];
    circ.add_CNOT_gate(6, 7)                              # cx q[6],q[7];
    circ.add_RX_gate(6, np.pi*0.4*-1)                     # rx(pi*0.4) q[6];
    circ.add_RY_gate(7, np.pi*0.5*-1)                     # ry(pi*0.5) q[7];
    circ.add_CNOT_gate(7, 6)                              # cx q[7],q[6];
    circ.add_RX_gate(7, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[7];
    circ.add_RZ_gate(7, np.pi*0.5*-1)                     # rz(pi*0.5) q[7];
    circ.add_CNOT_gate(6, 7)                              # cx q[6],q[7];
    circ.add_U3_gate(6, np.pi*0.5, np.pi*0.65, np.pi*1.0) # u3(pi*0.5,pi*0.65,pi*1.0) q[6];
    circ.add_U3_gate(7, np.pi*0.5, np.pi*0.15, 0)         # u3(pi*0.5,pi*0.15,0) q[7];
                                                          #
    # Gate: YY**1.1                                       # // Gate: YY**1.1
    circ.add_U3_gate(6, 0, np.pi*1.0, np.pi*0.5)          # u3(0,pi*1.0,pi*0.5) q[6];
    circ.add_U3_gate(7, 0, 0, np.pi*0.5)                  # u3(0,0,pi*0.5) q[7];
    circ.add_RX_gate(6, np.pi*0.5*-1)                     # rx(pi*0.5) q[6];
    circ.add_CNOT_gate(6, 7)                              # cx q[6],q[7];
    circ.add_RX_gate(6, np.pi*0.4*-1)                     # rx(pi*0.4) q[6];
    circ.add_RY_gate(7, np.pi*0.5*-1)                     # ry(pi*0.5) q[7];
    circ.add_CNOT_gate(7, 6)                              # cx q[7],q[6];
    circ.add_RX_gate(7, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[7];
    circ.add_RZ_gate(7, np.pi*0.5*-1)                     # rz(pi*0.5) q[7];
    circ.add_CNOT_gate(6, 7)                              # cx q[6],q[7];
    circ.add_U3_gate(6, np.pi*1.0, 0, np.pi*0.5)          # u3(pi*1.0,0,pi*0.5) q[6];
    circ.add_U3_gate(7, np.pi*1.0, 0, np.pi*1.5)          # u3(pi*1.0,0,pi*1.5) q[7];
                                                          #
    # Gate: XX**1.1                                       # // Gate: XX**1.1
    circ.add_U3_gate(6, np.pi*0.5, np.pi*1.5, np.pi*1.5)  # u3(pi*0.5,pi*1.5,pi*1.5) q[6];
    circ.add_U3_gate(7, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[7];
    circ.add_RX_gate(6, np.pi*0.5*-1)                     # rx(pi*0.5) q[6];
    circ.add_CNOT_gate(6, 7)                              # cx q[6],q[7];
    circ.add_RX_gate(6, np.pi*0.4*-1)                     # rx(pi*0.4) q[6];
    circ.add_RY_gate(7, np.pi*0.5*-1)                     # ry(pi*0.5) q[7];
    circ.add_CNOT_gate(7, 6)                              # cx q[7],q[6];
    circ.add_RX_gate(7, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[7];
    circ.add_RZ_gate(7, np.pi*0.5*-1)                     # rz(pi*0.5) q[7];
    circ.add_CNOT_gate(6, 7)                              # cx q[6],q[7];
    circ.add_U3_gate(6, np.pi*0.5, np.pi*0.5, np.pi*0.5)  # u3(pi*0.5,pi*0.5,pi*0.5) q[6];
    circ.add_U3_gate(7, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[7];
                                                          #
    circ.add_RX_gate(6, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[6];
    circ.add_RY_gate(6, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[6];
    circ.add_RZ_gate(6, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[6];
    circ.add_RX_gate(7, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[7];
    circ.add_RY_gate(7, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[7];
    circ.add_RZ_gate(7, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[7];
    circ.add_RX_gate(1, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[1];
    circ.add_RY_gate(1, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[1];
    circ.add_RZ_gate(1, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[1];
    circ.add_RX_gate(2, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[2];
    circ.add_RY_gate(2, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[2];
    circ.add_RZ_gate(2, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[2];
                                                          #
    # Gate: ZZ**1.1                                       # // Gate: ZZ**1.1
    circ.add_RZ_gate(1, np.pi*1.1*-1)                     # rz(pi*1.1) q[1];
    circ.add_RZ_gate(2, np.pi*1.1*-1)                     # rz(pi*1.1) q[2];
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                     # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                              # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.4*-1)                     # rx(pi*0.4) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                     # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                              # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                     # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                              # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.65, np.pi*1.0) # u3(pi*0.5,pi*0.65,pi*1.0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*0.15, 0)         # u3(pi*0.5,pi*0.15,0) q[2];
                                                          #
    # Gate: YY**1.1                                       # // Gate: YY**1.1
    circ.add_U3_gate(1, 0, np.pi*1.0, np.pi*0.5)          # u3(0,pi*1.0,pi*0.5) q[1];
    circ.add_U3_gate(2, 0, 0, np.pi*0.5)                  # u3(0,0,pi*0.5) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                     # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                              # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.4*-1)                     # rx(pi*0.4) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                     # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                              # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                     # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                              # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*1.0, 0, np.pi*0.5)          # u3(pi*1.0,0,pi*0.5) q[1];
    circ.add_U3_gate(2, np.pi*1.0, 0, np.pi*1.5)          # u3(pi*1.0,0,pi*1.5) q[2];
                                                          #
    # Gate: XX**1.1                                       # // Gate: XX**1.1
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.5, np.pi*1.5)  # u3(pi*0.5,pi*1.5,pi*1.5) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                     # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                              # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.4*-1)                     # rx(pi*0.4) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                     # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                              # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                     # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                              # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.5, np.pi*0.5)  # u3(pi*0.5,pi*0.5,pi*0.5) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[2];
                                                          #
    circ.add_RX_gate(1, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[1];
    circ.add_RY_gate(1, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[1];
    circ.add_RZ_gate(1, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[1];
    circ.add_RX_gate(2, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[2];
    circ.add_RY_gate(2, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[2];
    circ.add_RZ_gate(2, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[2];
    circ.add_RX_gate(3, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[3];
    circ.add_RY_gate(3, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[3];
    circ.add_RZ_gate(3, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[3];
    circ.add_RX_gate(4, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[4];
    circ.add_RY_gate(4, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[4];
    circ.add_RZ_gate(4, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[4];
                                                          #
    # Gate: ZZ**1.1                                       # // Gate: ZZ**1.1
    circ.add_RZ_gate(3, np.pi*1.1*-1)                     # rz(pi*1.1) q[3];
    circ.add_RZ_gate(4, np.pi*1.1*-1)                     # rz(pi*1.1) q[4];
    circ.add_U3_gate(3, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[3];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[4];
    circ.add_RX_gate(3, np.pi*0.5*-1)                     # rx(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 4)                              # cx q[3],q[4];
    circ.add_RX_gate(3, np.pi*0.4*-1)                     # rx(pi*0.4) q[3];
    circ.add_RY_gate(4, np.pi*0.5*-1)                     # ry(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 3)                              # cx q[4],q[3];
    circ.add_RX_gate(4, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[4];
    circ.add_RZ_gate(4, np.pi*0.5*-1)                     # rz(pi*0.5) q[4];
    circ.add_CNOT_gate(3, 4)                              # cx q[3],q[4];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.65, np.pi*1.0) # u3(pi*0.5,pi*0.65,pi*1.0) q[3];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*0.15, 0)         # u3(pi*0.5,pi*0.15,0) q[4];
                                                          #
    # Gate: YY**1.1                                       # // Gate: YY**1.1
    circ.add_U3_gate(3, 0, np.pi*1.0, np.pi*0.5)          # u3(0,pi*1.0,pi*0.5) q[3];
    circ.add_U3_gate(4, 0, 0, np.pi*0.5)                  # u3(0,0,pi*0.5) q[4];
    circ.add_RX_gate(3, np.pi*0.5*-1)                     # rx(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 4)                              # cx q[3],q[4];
    circ.add_RX_gate(3, np.pi*0.4*-1)                     # rx(pi*0.4) q[3];
    circ.add_RY_gate(4, np.pi*0.5*-1)                     # ry(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 3)                              # cx q[4],q[3];
    circ.add_RX_gate(4, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[4];
    circ.add_RZ_gate(4, np.pi*0.5*-1)                     # rz(pi*0.5) q[4];
    circ.add_CNOT_gate(3, 4)                              # cx q[3],q[4];
    circ.add_U3_gate(3, np.pi*1.0, 0, np.pi*0.5)          # u3(pi*1.0,0,pi*0.5) q[3];
    circ.add_U3_gate(4, np.pi*1.0, 0, np.pi*1.5)          # u3(pi*1.0,0,pi*1.5) q[4];
                                                          #
    # Gate: XX**1.1                                       # // Gate: XX**1.1
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.5, np.pi*1.5)  # u3(pi*0.5,pi*1.5,pi*1.5) q[3];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[4];
    circ.add_RX_gate(3, np.pi*0.5*-1)                     # rx(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 4)                              # cx q[3],q[4];
    circ.add_RX_gate(3, np.pi*0.4*-1)                     # rx(pi*0.4) q[3];
    circ.add_RY_gate(4, np.pi*0.5*-1)                     # ry(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 3)                              # cx q[4],q[3];
    circ.add_RX_gate(4, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[4];
    circ.add_RZ_gate(4, np.pi*0.5*-1)                     # rz(pi*0.5) q[4];
    circ.add_CNOT_gate(3, 4)                              # cx q[3],q[4];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.5, np.pi*0.5)  # u3(pi*0.5,pi*0.5,pi*0.5) q[3];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[4];
                                                          #
    circ.add_RX_gate(3, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[3];
    circ.add_RY_gate(3, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[3];
    circ.add_RZ_gate(3, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[3];
    circ.add_RX_gate(4, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[4];
    circ.add_RY_gate(4, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[4];
    circ.add_RZ_gate(4, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[4];
    circ.add_RX_gate(5, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[5];
    circ.add_RY_gate(5, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[5];
    circ.add_RZ_gate(5, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[5];
    circ.add_RX_gate(6, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[6];
    circ.add_RY_gate(6, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[6];
    circ.add_RZ_gate(6, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[6];
                                                          #
    # Gate: ZZ**1.1                                       # // Gate: ZZ**1.1
    circ.add_RZ_gate(5, np.pi*1.1*-1)                     # rz(pi*1.1) q[5];
    circ.add_RZ_gate(6, np.pi*1.1*-1)                     # rz(pi*1.1) q[6];
    circ.add_U3_gate(5, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[5];
    circ.add_U3_gate(6, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[6];
    circ.add_RX_gate(5, np.pi*0.5*-1)                     # rx(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 6)                              # cx q[5],q[6];
    circ.add_RX_gate(5, np.pi*0.4*-1)                     # rx(pi*0.4) q[5];
    circ.add_RY_gate(6, np.pi*0.5*-1)                     # ry(pi*0.5) q[6];
    circ.add_CNOT_gate(6, 5)                              # cx q[6],q[5];
    circ.add_RX_gate(6, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[6];
    circ.add_RZ_gate(6, np.pi*0.5*-1)                     # rz(pi*0.5) q[6];
    circ.add_CNOT_gate(5, 6)                              # cx q[5],q[6];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*0.65, np.pi*1.0) # u3(pi*0.5,pi*0.65,pi*1.0) q[5];
    circ.add_U3_gate(6, np.pi*0.5, np.pi*0.15, 0)         # u3(pi*0.5,pi*0.15,0) q[6];
                                                          #
    # Gate: YY**1.1                                       # // Gate: YY**1.1
    circ.add_U3_gate(5, 0, np.pi*1.0, np.pi*0.5)          # u3(0,pi*1.0,pi*0.5) q[5];
    circ.add_U3_gate(6, 0, 0, np.pi*0.5)                  # u3(0,0,pi*0.5) q[6];
    circ.add_RX_gate(5, np.pi*0.5*-1)                     # rx(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 6)                              # cx q[5],q[6];
    circ.add_RX_gate(5, np.pi*0.4*-1)                     # rx(pi*0.4) q[5];
    circ.add_RY_gate(6, np.pi*0.5*-1)                     # ry(pi*0.5) q[6];
    circ.add_CNOT_gate(6, 5)                              # cx q[6],q[5];
    circ.add_RX_gate(6, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[6];
    circ.add_RZ_gate(6, np.pi*0.5*-1)                     # rz(pi*0.5) q[6];
    circ.add_CNOT_gate(5, 6)                              # cx q[5],q[6];
    circ.add_U3_gate(5, np.pi*1.0, 0, np.pi*0.5)          # u3(pi*1.0,0,pi*0.5) q[5];
    circ.add_U3_gate(6, np.pi*1.0, 0, np.pi*1.5)          # u3(pi*1.0,0,pi*1.5) q[6];
                                                          #
    # Gate: XX**1.1                                       # // Gate: XX**1.1
    circ.add_U3_gate(5, np.pi*0.5, np.pi*1.5, np.pi*1.5)  # u3(pi*0.5,pi*1.5,pi*1.5) q[5];
    circ.add_U3_gate(6, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[6];
    circ.add_RX_gate(5, np.pi*0.5*-1)                     # rx(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 6)                              # cx q[5],q[6];
    circ.add_RX_gate(5, np.pi*0.4*-1)                     # rx(pi*0.4) q[5];
    circ.add_RY_gate(6, np.pi*0.5*-1)                     # ry(pi*0.5) q[6];
    circ.add_CNOT_gate(6, 5)                              # cx q[6],q[5];
    circ.add_RX_gate(6, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[6];
    circ.add_RZ_gate(6, np.pi*0.5*-1)                     # rz(pi*0.5) q[6];
    circ.add_CNOT_gate(5, 6)                              # cx q[5],q[6];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*0.5, np.pi*0.5)  # u3(pi*0.5,pi*0.5,pi*0.5) q[5];
    circ.add_U3_gate(6, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[6];
                                                          #
    circ.add_RX_gate(5, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[5];
    circ.add_RY_gate(5, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[5];
    circ.add_RZ_gate(5, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[5];
    circ.add_RX_gate(6, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[6];
    circ.add_RY_gate(6, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[6];
    circ.add_RZ_gate(6, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[6];
    circ.add_RX_gate(7, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[7];
    circ.add_RY_gate(7, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[7];
    circ.add_RZ_gate(7, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[7];
    circ.add_RX_gate(0, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[0];
    circ.add_RY_gate(0, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[0];
    circ.add_RZ_gate(0, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[0];
                                                          #
    # Gate: ZZ**1.1                                       # // Gate: ZZ**1.1
    circ.add_RZ_gate(7, np.pi*1.1*-1)                     # rz(pi*1.1) q[7];
    circ.add_RZ_gate(0, np.pi*1.1*-1)                     # rz(pi*1.1) q[0];
    circ.add_U3_gate(7, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[7];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[0];
    circ.add_RX_gate(7, np.pi*0.5*-1)                     # rx(pi*0.5) q[7];
    circ.add_CNOT_gate(7, 0)                              # cx q[7],q[0];
    circ.add_RX_gate(7, np.pi*0.4*-1)                     # rx(pi*0.4) q[7];
    circ.add_RY_gate(0, np.pi*0.5*-1)                     # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 7)                              # cx q[0],q[7];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                     # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(7, 0)                              # cx q[7],q[0];
    circ.add_U3_gate(7, np.pi*0.5, np.pi*0.65, np.pi*1.0) # u3(pi*0.5,pi*0.65,pi*1.0) q[7];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.15, 0)         # u3(pi*0.5,pi*0.15,0) q[0];
                                                          #
    # Gate: YY**1.1                                       # // Gate: YY**1.1
    circ.add_U3_gate(7, 0, np.pi*1.0, np.pi*0.5)          # u3(0,pi*1.0,pi*0.5) q[7];
    circ.add_U3_gate(0, 0, 0, np.pi*0.5)                  # u3(0,0,pi*0.5) q[0];
    circ.add_RX_gate(7, np.pi*0.5*-1)                     # rx(pi*0.5) q[7];
    circ.add_CNOT_gate(7, 0)                              # cx q[7],q[0];
    circ.add_RX_gate(7, np.pi*0.4*-1)                     # rx(pi*0.4) q[7];
    circ.add_RY_gate(0, np.pi*0.5*-1)                     # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 7)                              # cx q[0],q[7];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                     # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(7, 0)                              # cx q[7],q[0];
    circ.add_U3_gate(7, np.pi*1.0, 0, np.pi*0.5)          # u3(pi*1.0,0,pi*0.5) q[7];
    circ.add_U3_gate(0, np.pi*1.0, 0, np.pi*1.5)          # u3(pi*1.0,0,pi*1.5) q[0];
                                                          #
    # Gate: XX**1.1                                       # // Gate: XX**1.1
    circ.add_U3_gate(7, np.pi*0.5, np.pi*1.5, np.pi*1.5)  # u3(pi*0.5,pi*1.5,pi*1.5) q[7];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[0];
    circ.add_RX_gate(7, np.pi*0.5*-1)                     # rx(pi*0.5) q[7];
    circ.add_CNOT_gate(7, 0)                              # cx q[7],q[0];
    circ.add_RX_gate(7, np.pi*0.4*-1)                     # rx(pi*0.4) q[7];
    circ.add_RY_gate(0, np.pi*0.5*-1)                     # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 7)                              # cx q[0],q[7];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                     # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(7, 0)                              # cx q[7],q[0];
    circ.add_U3_gate(7, np.pi*0.5, np.pi*0.5, np.pi*0.5)  # u3(pi*0.5,pi*0.5,pi*0.5) q[7];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[0];
                                                          #
    circ.add_RX_gate(7, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[7];
    circ.add_RY_gate(7, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[7];
    circ.add_RZ_gate(7, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[7];
    circ.add_RX_gate(0, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[0];
    circ.add_RY_gate(0, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[0];
    circ.add_RZ_gate(0, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[0];
                                                          #
    # Gate: CNOT**1.1                                     # // Gate: CNOT**1.1
    circ.add_RY_gate(1, np.pi*-0.5*-1)                    # ry(pi*-0.5) q[1];
    circ.add_U3_gate(0, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                     # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.05*-1)                    # rx(pi*0.05) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                     # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                     # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.3, np.pi*1.0)  # u3(pi*0.5,pi*0.3,pi*1.0) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.8, 0)          # u3(pi*0.5,pi*1.8,0) q[1];
    circ.add_RY_gate(1, np.pi*0.5*-1)                     # ry(pi*0.5) q[1];
                                                          #
    # Gate: CNOT**1.1                                     # // Gate: CNOT**1.1
    circ.add_RY_gate(3, np.pi*-0.5*-1)                    # ry(pi*-0.5) q[3];
    circ.add_U3_gate(2, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[3];
    circ.add_RX_gate(2, np.pi*0.5*-1)                     # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                              # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*0.05*-1)                    # rx(pi*0.05) q[2];
    circ.add_RY_gate(3, np.pi*0.5*-1)                     # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                              # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                     # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(2, 3)                              # cx q[2],q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*0.3, np.pi*1.0)  # u3(pi*0.5,pi*0.3,pi*1.0) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.8, 0)          # u3(pi*0.5,pi*1.8,0) q[3];
    circ.add_RY_gate(3, np.pi*0.5*-1)                     # ry(pi*0.5) q[3];
                                                          #
    # Gate: CNOT**1.1                                     # // Gate: CNOT**1.1
    circ.add_RY_gate(5, np.pi*-0.5*-1)                    # ry(pi*-0.5) q[5];
    circ.add_U3_gate(4, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[4];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[5];
    circ.add_RX_gate(4, np.pi*0.5*-1)                     # rx(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 5)                              # cx q[4],q[5];
    circ.add_RX_gate(4, np.pi*0.05*-1)                    # rx(pi*0.05) q[4];
    circ.add_RY_gate(5, np.pi*0.5*-1)                     # ry(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 4)                              # cx q[5],q[4];
    circ.add_RX_gate(5, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[5];
    circ.add_RZ_gate(5, np.pi*0.5*-1)                     # rz(pi*0.5) q[5];
    circ.add_CNOT_gate(4, 5)                              # cx q[4],q[5];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*0.3, np.pi*1.0)  # u3(pi*0.5,pi*0.3,pi*1.0) q[4];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*1.8, 0)          # u3(pi*0.5,pi*1.8,0) q[5];
    circ.add_RY_gate(5, np.pi*0.5*-1)                     # ry(pi*0.5) q[5];
                                                          #
    # Gate: CNOT**1.1                                     # // Gate: CNOT**1.1
    circ.add_RY_gate(7, np.pi*-0.5*-1)                    # ry(pi*-0.5) q[7];
    circ.add_U3_gate(6, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[6];
    circ.add_U3_gate(7, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[7];
    circ.add_RX_gate(6, np.pi*0.5*-1)                     # rx(pi*0.5) q[6];
    circ.add_CNOT_gate(6, 7)                              # cx q[6],q[7];
    circ.add_RX_gate(6, np.pi*0.05*-1)                    # rx(pi*0.05) q[6];
    circ.add_RY_gate(7, np.pi*0.5*-1)                     # ry(pi*0.5) q[7];
    circ.add_CNOT_gate(7, 6)                              # cx q[7],q[6];
    circ.add_RX_gate(7, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[7];
    circ.add_RZ_gate(7, np.pi*0.5*-1)                     # rz(pi*0.5) q[7];
    circ.add_CNOT_gate(6, 7)                              # cx q[6],q[7];
    circ.add_U3_gate(6, np.pi*0.5, np.pi*0.3, np.pi*1.0)  # u3(pi*0.5,pi*0.3,pi*1.0) q[6];
    circ.add_U3_gate(7, np.pi*0.5, np.pi*1.8, 0)          # u3(pi*0.5,pi*1.8,0) q[7];
    circ.add_RY_gate(7, np.pi*0.5*-1)                     # ry(pi*0.5) q[7];
                                                          #
    # Gate: CZ**1.1                                       # // Gate: CZ**1.1
    circ.add_U3_gate(0, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                     # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.05*-1)                    # rx(pi*0.05) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                     # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                     # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.3, np.pi*1.0)  # u3(pi*0.5,pi*0.3,pi*1.0) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.8, 0)          # u3(pi*0.5,pi*1.8,0) q[1];
                                                          #
    # Gate: CZ**1.1                                       # // Gate: CZ**1.1
    circ.add_U3_gate(2, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[3];
    circ.add_RX_gate(2, np.pi*0.5*-1)                     # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                              # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*0.05*-1)                    # rx(pi*0.05) q[2];
    circ.add_RY_gate(3, np.pi*0.5*-1)                     # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                              # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                     # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(2, 3)                              # cx q[2],q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*0.3, np.pi*1.0)  # u3(pi*0.5,pi*0.3,pi*1.0) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.8, 0)          # u3(pi*0.5,pi*1.8,0) q[3];
                                                          #
    # Gate: CZ**1.1                                       # // Gate: CZ**1.1
    circ.add_U3_gate(4, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[4];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[5];
    circ.add_RX_gate(4, np.pi*0.5*-1)                     # rx(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 5)                              # cx q[4],q[5];
    circ.add_RX_gate(4, np.pi*0.05*-1)                    # rx(pi*0.05) q[4];
    circ.add_RY_gate(5, np.pi*0.5*-1)                     # ry(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 4)                              # cx q[5],q[4];
    circ.add_RX_gate(5, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[5];
    circ.add_RZ_gate(5, np.pi*0.5*-1)                     # rz(pi*0.5) q[5];
    circ.add_CNOT_gate(4, 5)                              # cx q[4],q[5];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*0.3, np.pi*1.0)  # u3(pi*0.5,pi*0.3,pi*1.0) q[4];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*1.8, 0)          # u3(pi*0.5,pi*1.8,0) q[5];
                                                          #
    # Gate: CZ**1.1                                       # // Gate: CZ**1.1
    circ.add_U3_gate(6, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[6];
    circ.add_U3_gate(7, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[7];
    circ.add_RX_gate(6, np.pi*0.5*-1)                     # rx(pi*0.5) q[6];
    circ.add_CNOT_gate(6, 7)                              # cx q[6],q[7];
    circ.add_RX_gate(6, np.pi*0.05*-1)                    # rx(pi*0.05) q[6];
    circ.add_RY_gate(7, np.pi*0.5*-1)                     # ry(pi*0.5) q[7];
    circ.add_CNOT_gate(7, 6)                              # cx q[7],q[6];
    circ.add_RX_gate(7, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[7];
    circ.add_RZ_gate(7, np.pi*0.5*-1)                     # rz(pi*0.5) q[7];
    circ.add_CNOT_gate(6, 7)                              # cx q[6],q[7];
    circ.add_U3_gate(6, np.pi*0.5, np.pi*0.3, np.pi*1.0)  # u3(pi*0.5,pi*0.3,pi*1.0) q[6];
    circ.add_U3_gate(7, np.pi*0.5, np.pi*1.8, 0)          # u3(pi*0.5,pi*1.8,0) q[7];
                                                          #
    # Gate: CNOT**1.1                                     # // Gate: CNOT**1.1
    circ.add_RY_gate(2, np.pi*-0.5*-1)                    # ry(pi*-0.5) q[2];
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                     # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                              # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.05*-1)                    # rx(pi*0.05) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                     # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                              # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                     # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                              # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.3, np.pi*1.0)  # u3(pi*0.5,pi*0.3,pi*1.0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.8, 0)          # u3(pi*0.5,pi*1.8,0) q[2];
    circ.add_RY_gate(2, np.pi*0.5*-1)                     # ry(pi*0.5) q[2];
                                                          #
    # Gate: CNOT**1.1                                     # // Gate: CNOT**1.1
    circ.add_RY_gate(4, np.pi*-0.5*-1)                    # ry(pi*-0.5) q[4];
    circ.add_U3_gate(3, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[3];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[4];
    circ.add_RX_gate(3, np.pi*0.5*-1)                     # rx(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 4)                              # cx q[3],q[4];
    circ.add_RX_gate(3, np.pi*0.05*-1)                    # rx(pi*0.05) q[3];
    circ.add_RY_gate(4, np.pi*0.5*-1)                     # ry(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 3)                              # cx q[4],q[3];
    circ.add_RX_gate(4, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[4];
    circ.add_RZ_gate(4, np.pi*0.5*-1)                     # rz(pi*0.5) q[4];
    circ.add_CNOT_gate(3, 4)                              # cx q[3],q[4];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.3, np.pi*1.0)  # u3(pi*0.5,pi*0.3,pi*1.0) q[3];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*1.8, 0)          # u3(pi*0.5,pi*1.8,0) q[4];
    circ.add_RY_gate(4, np.pi*0.5*-1)                     # ry(pi*0.5) q[4];
                                                          #
    # Gate: CNOT**1.1                                     # // Gate: CNOT**1.1
    circ.add_RY_gate(6, np.pi*-0.5*-1)                    # ry(pi*-0.5) q[6];
    circ.add_U3_gate(5, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[5];
    circ.add_U3_gate(6, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[6];
    circ.add_RX_gate(5, np.pi*0.5*-1)                     # rx(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 6)                              # cx q[5],q[6];
    circ.add_RX_gate(5, np.pi*0.05*-1)                    # rx(pi*0.05) q[5];
    circ.add_RY_gate(6, np.pi*0.5*-1)                     # ry(pi*0.5) q[6];
    circ.add_CNOT_gate(6, 5)                              # cx q[6],q[5];
    circ.add_RX_gate(6, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[6];
    circ.add_RZ_gate(6, np.pi*0.5*-1)                     # rz(pi*0.5) q[6];
    circ.add_CNOT_gate(5, 6)                              # cx q[5],q[6];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*0.3, np.pi*1.0)  # u3(pi*0.5,pi*0.3,pi*1.0) q[5];
    circ.add_U3_gate(6, np.pi*0.5, np.pi*1.8, 0)          # u3(pi*0.5,pi*1.8,0) q[6];
    circ.add_RY_gate(6, np.pi*0.5*-1)                     # ry(pi*0.5) q[6];
                                                          #
    # Gate: CNOT**1.1                                     # // Gate: CNOT**1.1
    circ.add_RY_gate(0, np.pi*-0.5*-1)                    # ry(pi*-0.5) q[0];
    circ.add_U3_gate(7, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[7];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[0];
    circ.add_RX_gate(7, np.pi*0.5*-1)                     # rx(pi*0.5) q[7];
    circ.add_CNOT_gate(7, 0)                              # cx q[7],q[0];
    circ.add_RX_gate(7, np.pi*0.05*-1)                    # rx(pi*0.05) q[7];
    circ.add_RY_gate(0, np.pi*0.5*-1)                     # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 7)                              # cx q[0],q[7];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                     # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(7, 0)                              # cx q[7],q[0];
    circ.add_U3_gate(7, np.pi*0.5, np.pi*0.3, np.pi*1.0)  # u3(pi*0.5,pi*0.3,pi*1.0) q[7];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.8, 0)          # u3(pi*0.5,pi*1.8,0) q[0];
    circ.add_RY_gate(0, np.pi*0.5*-1)                     # ry(pi*0.5) q[0];
                                                          #
    # Gate: CZ**1.1                                       # // Gate: CZ**1.1
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                     # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                              # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.05*-1)                    # rx(pi*0.05) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                     # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                              # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                     # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                              # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.3, np.pi*1.0)  # u3(pi*0.5,pi*0.3,pi*1.0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.8, 0)          # u3(pi*0.5,pi*1.8,0) q[2];
                                                          #
    # Gate: CZ**1.1                                       # // Gate: CZ**1.1
    circ.add_U3_gate(3, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[3];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[4];
    circ.add_RX_gate(3, np.pi*0.5*-1)                     # rx(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 4)                              # cx q[3],q[4];
    circ.add_RX_gate(3, np.pi*0.05*-1)                    # rx(pi*0.05) q[3];
    circ.add_RY_gate(4, np.pi*0.5*-1)                     # ry(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 3)                              # cx q[4],q[3];
    circ.add_RX_gate(4, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[4];
    circ.add_RZ_gate(4, np.pi*0.5*-1)                     # rz(pi*0.5) q[4];
    circ.add_CNOT_gate(3, 4)                              # cx q[3],q[4];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.3, np.pi*1.0)  # u3(pi*0.5,pi*0.3,pi*1.0) q[3];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*1.8, 0)          # u3(pi*0.5,pi*1.8,0) q[4];
                                                          #
    # Gate: CZ**1.1                                       # // Gate: CZ**1.1
    circ.add_U3_gate(5, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[5];
    circ.add_U3_gate(6, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[6];
    circ.add_RX_gate(5, np.pi*0.5*-1)                     # rx(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 6)                              # cx q[5],q[6];
    circ.add_RX_gate(5, np.pi*0.05*-1)                    # rx(pi*0.05) q[5];
    circ.add_RY_gate(6, np.pi*0.5*-1)                     # ry(pi*0.5) q[6];
    circ.add_CNOT_gate(6, 5)                              # cx q[6],q[5];
    circ.add_RX_gate(6, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[6];
    circ.add_RZ_gate(6, np.pi*0.5*-1)                     # rz(pi*0.5) q[6];
    circ.add_CNOT_gate(5, 6)                              # cx q[5],q[6];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*0.3, np.pi*1.0)  # u3(pi*0.5,pi*0.3,pi*1.0) q[5];
    circ.add_U3_gate(6, np.pi*0.5, np.pi*1.8, 0)          # u3(pi*0.5,pi*1.8,0) q[6];
                                                          #
    # Gate: CZ**1.1                                       # // Gate: CZ**1.1
    circ.add_U3_gate(7, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[7];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[0];
    circ.add_RX_gate(7, np.pi*0.5*-1)                     # rx(pi*0.5) q[7];
    circ.add_CNOT_gate(7, 0)                              # cx q[7],q[0];
    circ.add_RX_gate(7, np.pi*0.05*-1)                    # rx(pi*0.05) q[7];
    circ.add_RY_gate(0, np.pi*0.5*-1)                     # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 7)                              # cx q[0],q[7];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                     # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(7, 0)                              # cx q[7],q[0];
    circ.add_U3_gate(7, np.pi*0.5, np.pi*0.3, np.pi*1.0)  # u3(pi*0.5,pi*0.3,pi*1.0) q[7];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.8, 0)          # u3(pi*0.5,pi*1.8,0) q[0];
                                                          #
    circ.add_RX_gate(0, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[0];
    circ.add_RY_gate(0, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[0];
    circ.add_RZ_gate(0, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[0];
    circ.add_RX_gate(1, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[1];
    circ.add_RY_gate(1, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[1];
    circ.add_RZ_gate(1, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[1];
                                                          #
    # Gate: ZZ**1.1                                       # // Gate: ZZ**1.1
    circ.add_RZ_gate(0, np.pi*1.1*-1)                     # rz(pi*1.1) q[0];
    circ.add_RZ_gate(1, np.pi*1.1*-1)                     # rz(pi*1.1) q[1];
    circ.add_U3_gate(0, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                     # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.4*-1)                     # rx(pi*0.4) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                     # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                     # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.65, np.pi*1.0) # u3(pi*0.5,pi*0.65,pi*1.0) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.15, 0)         # u3(pi*0.5,pi*0.15,0) q[1];
                                                          #
    # Gate: YY**1.1                                       # // Gate: YY**1.1
    circ.add_U3_gate(0, 0, np.pi*1.0, np.pi*0.5)          # u3(0,pi*1.0,pi*0.5) q[0];
    circ.add_U3_gate(1, 0, 0, np.pi*0.5)                  # u3(0,0,pi*0.5) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                     # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.4*-1)                     # rx(pi*0.4) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                     # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                     # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*1.0, 0, np.pi*0.5)          # u3(pi*1.0,0,pi*0.5) q[0];
    circ.add_U3_gate(1, np.pi*1.0, 0, np.pi*1.5)          # u3(pi*1.0,0,pi*1.5) q[1];
                                                          #
    # Gate: XX**1.1                                       # // Gate: XX**1.1
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.5, np.pi*1.5)  # u3(pi*0.5,pi*1.5,pi*1.5) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                     # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.4*-1)                     # rx(pi*0.4) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                     # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                     # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.5, np.pi*0.5)  # u3(pi*0.5,pi*0.5,pi*0.5) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[1];
                                                          #
    circ.add_RX_gate(0, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[0];
    circ.add_RY_gate(0, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[0];
    circ.add_RZ_gate(0, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[0];
    circ.add_RX_gate(1, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[1];
    circ.add_RY_gate(1, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[1];
    circ.add_RZ_gate(1, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[1];
    circ.add_RX_gate(2, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[2];
    circ.add_RY_gate(2, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[2];
    circ.add_RZ_gate(2, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[2];
    circ.add_RX_gate(3, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[3];
    circ.add_RY_gate(3, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[3];
    circ.add_RZ_gate(3, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[3];
                                                          #
    # Gate: ZZ**1.1                                       # // Gate: ZZ**1.1
    circ.add_RZ_gate(2, np.pi*1.1*-1)                     # rz(pi*1.1) q[2];
    circ.add_RZ_gate(3, np.pi*1.1*-1)                     # rz(pi*1.1) q[3];
    circ.add_U3_gate(2, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[3];
    circ.add_RX_gate(2, np.pi*0.5*-1)                     # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                              # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*0.4*-1)                     # rx(pi*0.4) q[2];
    circ.add_RY_gate(3, np.pi*0.5*-1)                     # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                              # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                     # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(2, 3)                              # cx q[2],q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*0.65, np.pi*1.0) # u3(pi*0.5,pi*0.65,pi*1.0) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.15, 0)         # u3(pi*0.5,pi*0.15,0) q[3];
                                                          #
    # Gate: YY**1.1                                       # // Gate: YY**1.1
    circ.add_U3_gate(2, 0, np.pi*1.0, np.pi*0.5)          # u3(0,pi*1.0,pi*0.5) q[2];
    circ.add_U3_gate(3, 0, 0, np.pi*0.5)                  # u3(0,0,pi*0.5) q[3];
    circ.add_RX_gate(2, np.pi*0.5*-1)                     # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                              # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*0.4*-1)                     # rx(pi*0.4) q[2];
    circ.add_RY_gate(3, np.pi*0.5*-1)                     # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                              # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                     # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(2, 3)                              # cx q[2],q[3];
    circ.add_U3_gate(2, np.pi*1.0, 0, np.pi*0.5)          # u3(pi*1.0,0,pi*0.5) q[2];
    circ.add_U3_gate(3, np.pi*1.0, 0, np.pi*1.5)          # u3(pi*1.0,0,pi*1.5) q[3];
                                                          #
    # Gate: XX**1.1                                       # // Gate: XX**1.1
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.5, np.pi*1.5)  # u3(pi*0.5,pi*1.5,pi*1.5) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[3];
    circ.add_RX_gate(2, np.pi*0.5*-1)                     # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                              # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*0.4*-1)                     # rx(pi*0.4) q[2];
    circ.add_RY_gate(3, np.pi*0.5*-1)                     # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                              # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                     # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(2, 3)                              # cx q[2],q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*0.5, np.pi*0.5)  # u3(pi*0.5,pi*0.5,pi*0.5) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[3];
                                                          #
    circ.add_RX_gate(2, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[2];
    circ.add_RY_gate(2, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[2];
    circ.add_RZ_gate(2, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[2];
    circ.add_RX_gate(3, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[3];
    circ.add_RY_gate(3, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[3];
    circ.add_RZ_gate(3, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[3];
    circ.add_RX_gate(4, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[4];
    circ.add_RY_gate(4, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[4];
    circ.add_RZ_gate(4, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[4];
    circ.add_RX_gate(5, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[5];
    circ.add_RY_gate(5, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[5];
    circ.add_RZ_gate(5, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[5];
                                                          #
    # Gate: ZZ**1.1                                       # // Gate: ZZ**1.1
    circ.add_RZ_gate(4, np.pi*1.1*-1)                     # rz(pi*1.1) q[4];
    circ.add_RZ_gate(5, np.pi*1.1*-1)                     # rz(pi*1.1) q[5];
    circ.add_U3_gate(4, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[4];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[5];
    circ.add_RX_gate(4, np.pi*0.5*-1)                     # rx(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 5)                              # cx q[4],q[5];
    circ.add_RX_gate(4, np.pi*0.4*-1)                     # rx(pi*0.4) q[4];
    circ.add_RY_gate(5, np.pi*0.5*-1)                     # ry(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 4)                              # cx q[5],q[4];
    circ.add_RX_gate(5, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[5];
    circ.add_RZ_gate(5, np.pi*0.5*-1)                     # rz(pi*0.5) q[5];
    circ.add_CNOT_gate(4, 5)                              # cx q[4],q[5];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*0.65, np.pi*1.0) # u3(pi*0.5,pi*0.65,pi*1.0) q[4];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*0.15, 0)         # u3(pi*0.5,pi*0.15,0) q[5];
                                                          #
    # Gate: YY**1.1                                       # // Gate: YY**1.1
    circ.add_U3_gate(4, 0, np.pi*1.0, np.pi*0.5)          # u3(0,pi*1.0,pi*0.5) q[4];
    circ.add_U3_gate(5, 0, 0, np.pi*0.5)                  # u3(0,0,pi*0.5) q[5];
    circ.add_RX_gate(4, np.pi*0.5*-1)                     # rx(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 5)                              # cx q[4],q[5];
    circ.add_RX_gate(4, np.pi*0.4*-1)                     # rx(pi*0.4) q[4];
    circ.add_RY_gate(5, np.pi*0.5*-1)                     # ry(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 4)                              # cx q[5],q[4];
    circ.add_RX_gate(5, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[5];
    circ.add_RZ_gate(5, np.pi*0.5*-1)                     # rz(pi*0.5) q[5];
    circ.add_CNOT_gate(4, 5)                              # cx q[4],q[5];
    circ.add_U3_gate(4, np.pi*1.0, 0, np.pi*0.5)          # u3(pi*1.0,0,pi*0.5) q[4];
    circ.add_U3_gate(5, np.pi*1.0, 0, np.pi*1.5)          # u3(pi*1.0,0,pi*1.5) q[5];
                                                          #
    # Gate: XX**1.1                                       # // Gate: XX**1.1
    circ.add_U3_gate(4, np.pi*0.5, np.pi*1.5, np.pi*1.5)  # u3(pi*0.5,pi*1.5,pi*1.5) q[4];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[5];
    circ.add_RX_gate(4, np.pi*0.5*-1)                     # rx(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 5)                              # cx q[4],q[5];
    circ.add_RX_gate(4, np.pi*0.4*-1)                     # rx(pi*0.4) q[4];
    circ.add_RY_gate(5, np.pi*0.5*-1)                     # ry(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 4)                              # cx q[5],q[4];
    circ.add_RX_gate(5, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[5];
    circ.add_RZ_gate(5, np.pi*0.5*-1)                     # rz(pi*0.5) q[5];
    circ.add_CNOT_gate(4, 5)                              # cx q[4],q[5];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*0.5, np.pi*0.5)  # u3(pi*0.5,pi*0.5,pi*0.5) q[4];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[5];
                                                          #
    circ.add_RX_gate(4, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[4];
    circ.add_RY_gate(4, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[4];
    circ.add_RZ_gate(4, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[4];
    circ.add_RX_gate(5, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[5];
    circ.add_RY_gate(5, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[5];
    circ.add_RZ_gate(5, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[5];
    circ.add_RX_gate(6, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[6];
    circ.add_RY_gate(6, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[6];
    circ.add_RZ_gate(6, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[6];
    circ.add_RX_gate(7, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[7];
    circ.add_RY_gate(7, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[7];
    circ.add_RZ_gate(7, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[7];
                                                          #
    # Gate: ZZ**1.1                                       # // Gate: ZZ**1.1
    circ.add_RZ_gate(6, np.pi*1.1*-1)                     # rz(pi*1.1) q[6];
    circ.add_RZ_gate(7, np.pi*1.1*-1)                     # rz(pi*1.1) q[7];
    circ.add_U3_gate(6, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[6];
    circ.add_U3_gate(7, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[7];
    circ.add_RX_gate(6, np.pi*0.5*-1)                     # rx(pi*0.5) q[6];
    circ.add_CNOT_gate(6, 7)                              # cx q[6],q[7];
    circ.add_RX_gate(6, np.pi*0.4*-1)                     # rx(pi*0.4) q[6];
    circ.add_RY_gate(7, np.pi*0.5*-1)                     # ry(pi*0.5) q[7];
    circ.add_CNOT_gate(7, 6)                              # cx q[7],q[6];
    circ.add_RX_gate(7, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[7];
    circ.add_RZ_gate(7, np.pi*0.5*-1)                     # rz(pi*0.5) q[7];
    circ.add_CNOT_gate(6, 7)                              # cx q[6],q[7];
    circ.add_U3_gate(6, np.pi*0.5, np.pi*0.65, np.pi*1.0) # u3(pi*0.5,pi*0.65,pi*1.0) q[6];
    circ.add_U3_gate(7, np.pi*0.5, np.pi*0.15, 0)         # u3(pi*0.5,pi*0.15,0) q[7];
                                                          #
    # Gate: YY**1.1                                       # // Gate: YY**1.1
    circ.add_U3_gate(6, 0, np.pi*1.0, np.pi*0.5)          # u3(0,pi*1.0,pi*0.5) q[6];
    circ.add_U3_gate(7, 0, 0, np.pi*0.5)                  # u3(0,0,pi*0.5) q[7];
    circ.add_RX_gate(6, np.pi*0.5*-1)                     # rx(pi*0.5) q[6];
    circ.add_CNOT_gate(6, 7)                              # cx q[6],q[7];
    circ.add_RX_gate(6, np.pi*0.4*-1)                     # rx(pi*0.4) q[6];
    circ.add_RY_gate(7, np.pi*0.5*-1)                     # ry(pi*0.5) q[7];
    circ.add_CNOT_gate(7, 6)                              # cx q[7],q[6];
    circ.add_RX_gate(7, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[7];
    circ.add_RZ_gate(7, np.pi*0.5*-1)                     # rz(pi*0.5) q[7];
    circ.add_CNOT_gate(6, 7)                              # cx q[6],q[7];
    circ.add_U3_gate(6, np.pi*1.0, 0, np.pi*0.5)          # u3(pi*1.0,0,pi*0.5) q[6];
    circ.add_U3_gate(7, np.pi*1.0, 0, np.pi*1.5)          # u3(pi*1.0,0,pi*1.5) q[7];
                                                          #
    # Gate: XX**1.1                                       # // Gate: XX**1.1
    circ.add_U3_gate(6, np.pi*0.5, np.pi*1.5, np.pi*1.5)  # u3(pi*0.5,pi*1.5,pi*1.5) q[6];
    circ.add_U3_gate(7, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[7];
    circ.add_RX_gate(6, np.pi*0.5*-1)                     # rx(pi*0.5) q[6];
    circ.add_CNOT_gate(6, 7)                              # cx q[6],q[7];
    circ.add_RX_gate(6, np.pi*0.4*-1)                     # rx(pi*0.4) q[6];
    circ.add_RY_gate(7, np.pi*0.5*-1)                     # ry(pi*0.5) q[7];
    circ.add_CNOT_gate(7, 6)                              # cx q[7],q[6];
    circ.add_RX_gate(7, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[7];
    circ.add_RZ_gate(7, np.pi*0.5*-1)                     # rz(pi*0.5) q[7];
    circ.add_CNOT_gate(6, 7)                              # cx q[6],q[7];
    circ.add_U3_gate(6, np.pi*0.5, np.pi*0.5, np.pi*0.5)  # u3(pi*0.5,pi*0.5,pi*0.5) q[6];
    circ.add_U3_gate(7, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[7];
                                                          #
    circ.add_RX_gate(6, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[6];
    circ.add_RY_gate(6, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[6];
    circ.add_RZ_gate(6, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[6];
    circ.add_RX_gate(7, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[7];
    circ.add_RY_gate(7, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[7];
    circ.add_RZ_gate(7, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[7];
    circ.add_RX_gate(1, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[1];
    circ.add_RY_gate(1, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[1];
    circ.add_RZ_gate(1, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[1];
    circ.add_RX_gate(2, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[2];
    circ.add_RY_gate(2, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[2];
    circ.add_RZ_gate(2, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[2];
                                                          #
    # Gate: ZZ**1.1                                       # // Gate: ZZ**1.1
    circ.add_RZ_gate(1, np.pi*1.1*-1)                     # rz(pi*1.1) q[1];
    circ.add_RZ_gate(2, np.pi*1.1*-1)                     # rz(pi*1.1) q[2];
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                     # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                              # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.4*-1)                     # rx(pi*0.4) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                     # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                              # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                     # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                              # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.65, np.pi*1.0) # u3(pi*0.5,pi*0.65,pi*1.0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*0.15, 0)         # u3(pi*0.5,pi*0.15,0) q[2];
                                                          #
    # Gate: YY**1.1                                       # // Gate: YY**1.1
    circ.add_U3_gate(1, 0, np.pi*1.0, np.pi*0.5)          # u3(0,pi*1.0,pi*0.5) q[1];
    circ.add_U3_gate(2, 0, 0, np.pi*0.5)                  # u3(0,0,pi*0.5) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                     # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                              # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.4*-1)                     # rx(pi*0.4) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                     # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                              # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                     # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                              # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*1.0, 0, np.pi*0.5)          # u3(pi*1.0,0,pi*0.5) q[1];
    circ.add_U3_gate(2, np.pi*1.0, 0, np.pi*1.5)          # u3(pi*1.0,0,pi*1.5) q[2];
                                                          #
    # Gate: XX**1.1                                       # // Gate: XX**1.1
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.5, np.pi*1.5)  # u3(pi*0.5,pi*1.5,pi*1.5) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                     # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                              # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.4*-1)                     # rx(pi*0.4) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                     # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                              # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                     # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                              # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.5, np.pi*0.5)  # u3(pi*0.5,pi*0.5,pi*0.5) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[2];
                                                          #
    circ.add_RX_gate(1, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[1];
    circ.add_RY_gate(1, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[1];
    circ.add_RZ_gate(1, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[1];
    circ.add_RX_gate(2, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[2];
    circ.add_RY_gate(2, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[2];
    circ.add_RZ_gate(2, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[2];
    circ.add_RX_gate(3, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[3];
    circ.add_RY_gate(3, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[3];
    circ.add_RZ_gate(3, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[3];
    circ.add_RX_gate(4, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[4];
    circ.add_RY_gate(4, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[4];
    circ.add_RZ_gate(4, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[4];
                                                          #
    # Gate: ZZ**1.1                                       # // Gate: ZZ**1.1
    circ.add_RZ_gate(3, np.pi*1.1*-1)                     # rz(pi*1.1) q[3];
    circ.add_RZ_gate(4, np.pi*1.1*-1)                     # rz(pi*1.1) q[4];
    circ.add_U3_gate(3, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[3];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[4];
    circ.add_RX_gate(3, np.pi*0.5*-1)                     # rx(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 4)                              # cx q[3],q[4];
    circ.add_RX_gate(3, np.pi*0.4*-1)                     # rx(pi*0.4) q[3];
    circ.add_RY_gate(4, np.pi*0.5*-1)                     # ry(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 3)                              # cx q[4],q[3];
    circ.add_RX_gate(4, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[4];
    circ.add_RZ_gate(4, np.pi*0.5*-1)                     # rz(pi*0.5) q[4];
    circ.add_CNOT_gate(3, 4)                              # cx q[3],q[4];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.65, np.pi*1.0) # u3(pi*0.5,pi*0.65,pi*1.0) q[3];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*0.15, 0)         # u3(pi*0.5,pi*0.15,0) q[4];
                                                          #
    # Gate: YY**1.1                                       # // Gate: YY**1.1
    circ.add_U3_gate(3, 0, np.pi*1.0, np.pi*0.5)          # u3(0,pi*1.0,pi*0.5) q[3];
    circ.add_U3_gate(4, 0, 0, np.pi*0.5)                  # u3(0,0,pi*0.5) q[4];
    circ.add_RX_gate(3, np.pi*0.5*-1)                     # rx(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 4)                              # cx q[3],q[4];
    circ.add_RX_gate(3, np.pi*0.4*-1)                     # rx(pi*0.4) q[3];
    circ.add_RY_gate(4, np.pi*0.5*-1)                     # ry(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 3)                              # cx q[4],q[3];
    circ.add_RX_gate(4, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[4];
    circ.add_RZ_gate(4, np.pi*0.5*-1)                     # rz(pi*0.5) q[4];
    circ.add_CNOT_gate(3, 4)                              # cx q[3],q[4];
    circ.add_U3_gate(3, np.pi*1.0, 0, np.pi*0.5)          # u3(pi*1.0,0,pi*0.5) q[3];
    circ.add_U3_gate(4, np.pi*1.0, 0, np.pi*1.5)          # u3(pi*1.0,0,pi*1.5) q[4];
                                                          #
    # Gate: XX**1.1                                       # // Gate: XX**1.1
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.5, np.pi*1.5)  # u3(pi*0.5,pi*1.5,pi*1.5) q[3];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[4];
    circ.add_RX_gate(3, np.pi*0.5*-1)                     # rx(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 4)                              # cx q[3],q[4];
    circ.add_RX_gate(3, np.pi*0.4*-1)                     # rx(pi*0.4) q[3];
    circ.add_RY_gate(4, np.pi*0.5*-1)                     # ry(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 3)                              # cx q[4],q[3];
    circ.add_RX_gate(4, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[4];
    circ.add_RZ_gate(4, np.pi*0.5*-1)                     # rz(pi*0.5) q[4];
    circ.add_CNOT_gate(3, 4)                              # cx q[3],q[4];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.5, np.pi*0.5)  # u3(pi*0.5,pi*0.5,pi*0.5) q[3];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[4];
                                                          #
    circ.add_RX_gate(3, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[3];
    circ.add_RY_gate(3, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[3];
    circ.add_RZ_gate(3, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[3];
    circ.add_RX_gate(4, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[4];
    circ.add_RY_gate(4, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[4];
    circ.add_RZ_gate(4, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[4];
    circ.add_RX_gate(5, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[5];
    circ.add_RY_gate(5, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[5];
    circ.add_RZ_gate(5, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[5];
    circ.add_RX_gate(6, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[6];
    circ.add_RY_gate(6, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[6];
    circ.add_RZ_gate(6, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[6];
                                                          #
    # Gate: ZZ**1.1                                       # // Gate: ZZ**1.1
    circ.add_RZ_gate(5, np.pi*1.1*-1)                     # rz(pi*1.1) q[5];
    circ.add_RZ_gate(6, np.pi*1.1*-1)                     # rz(pi*1.1) q[6];
    circ.add_U3_gate(5, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[5];
    circ.add_U3_gate(6, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[6];
    circ.add_RX_gate(5, np.pi*0.5*-1)                     # rx(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 6)                              # cx q[5],q[6];
    circ.add_RX_gate(5, np.pi*0.4*-1)                     # rx(pi*0.4) q[5];
    circ.add_RY_gate(6, np.pi*0.5*-1)                     # ry(pi*0.5) q[6];
    circ.add_CNOT_gate(6, 5)                              # cx q[6],q[5];
    circ.add_RX_gate(6, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[6];
    circ.add_RZ_gate(6, np.pi*0.5*-1)                     # rz(pi*0.5) q[6];
    circ.add_CNOT_gate(5, 6)                              # cx q[5],q[6];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*0.65, np.pi*1.0) # u3(pi*0.5,pi*0.65,pi*1.0) q[5];
    circ.add_U3_gate(6, np.pi*0.5, np.pi*0.15, 0)         # u3(pi*0.5,pi*0.15,0) q[6];
                                                          #
    # Gate: YY**1.1                                       # // Gate: YY**1.1
    circ.add_U3_gate(5, 0, np.pi*1.0, np.pi*0.5)          # u3(0,pi*1.0,pi*0.5) q[5];
    circ.add_U3_gate(6, 0, 0, np.pi*0.5)                  # u3(0,0,pi*0.5) q[6];
    circ.add_RX_gate(5, np.pi*0.5*-1)                     # rx(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 6)                              # cx q[5],q[6];
    circ.add_RX_gate(5, np.pi*0.4*-1)                     # rx(pi*0.4) q[5];
    circ.add_RY_gate(6, np.pi*0.5*-1)                     # ry(pi*0.5) q[6];
    circ.add_CNOT_gate(6, 5)                              # cx q[6],q[5];
    circ.add_RX_gate(6, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[6];
    circ.add_RZ_gate(6, np.pi*0.5*-1)                     # rz(pi*0.5) q[6];
    circ.add_CNOT_gate(5, 6)                              # cx q[5],q[6];
    circ.add_U3_gate(5, np.pi*1.0, 0, np.pi*0.5)          # u3(pi*1.0,0,pi*0.5) q[5];
    circ.add_U3_gate(6, np.pi*1.0, 0, np.pi*1.5)          # u3(pi*1.0,0,pi*1.5) q[6];
                                                          #
    # Gate: XX**1.1                                       # // Gate: XX**1.1
    circ.add_U3_gate(5, np.pi*0.5, np.pi*1.5, np.pi*1.5)  # u3(pi*0.5,pi*1.5,pi*1.5) q[5];
    circ.add_U3_gate(6, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[6];
    circ.add_RX_gate(5, np.pi*0.5*-1)                     # rx(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 6)                              # cx q[5],q[6];
    circ.add_RX_gate(5, np.pi*0.4*-1)                     # rx(pi*0.4) q[5];
    circ.add_RY_gate(6, np.pi*0.5*-1)                     # ry(pi*0.5) q[6];
    circ.add_CNOT_gate(6, 5)                              # cx q[6],q[5];
    circ.add_RX_gate(6, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[6];
    circ.add_RZ_gate(6, np.pi*0.5*-1)                     # rz(pi*0.5) q[6];
    circ.add_CNOT_gate(5, 6)                              # cx q[5],q[6];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*0.5, np.pi*0.5)  # u3(pi*0.5,pi*0.5,pi*0.5) q[5];
    circ.add_U3_gate(6, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[6];
                                                          #
    circ.add_RX_gate(5, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[5];
    circ.add_RY_gate(5, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[5];
    circ.add_RZ_gate(5, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[5];
    circ.add_RX_gate(6, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[6];
    circ.add_RY_gate(6, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[6];
    circ.add_RZ_gate(6, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[6];
    circ.add_RX_gate(7, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[7];
    circ.add_RY_gate(7, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[7];
    circ.add_RZ_gate(7, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[7];
    circ.add_RX_gate(0, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[0];
    circ.add_RY_gate(0, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[0];
    circ.add_RZ_gate(0, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[0];
                                                          #
    # Gate: ZZ**1.1                                       # // Gate: ZZ**1.1
    circ.add_RZ_gate(7, np.pi*1.1*-1)                     # rz(pi*1.1) q[7];
    circ.add_RZ_gate(0, np.pi*1.1*-1)                     # rz(pi*1.1) q[0];
    circ.add_U3_gate(7, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[7];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[0];
    circ.add_RX_gate(7, np.pi*0.5*-1)                     # rx(pi*0.5) q[7];
    circ.add_CNOT_gate(7, 0)                              # cx q[7],q[0];
    circ.add_RX_gate(7, np.pi*0.4*-1)                     # rx(pi*0.4) q[7];
    circ.add_RY_gate(0, np.pi*0.5*-1)                     # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 7)                              # cx q[0],q[7];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                     # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(7, 0)                              # cx q[7],q[0];
    circ.add_U3_gate(7, np.pi*0.5, np.pi*0.65, np.pi*1.0) # u3(pi*0.5,pi*0.65,pi*1.0) q[7];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.15, 0)         # u3(pi*0.5,pi*0.15,0) q[0];
                                                          #
    # Gate: YY**1.1                                       # // Gate: YY**1.1
    circ.add_U3_gate(7, 0, np.pi*1.0, np.pi*0.5)          # u3(0,pi*1.0,pi*0.5) q[7];
    circ.add_U3_gate(0, 0, 0, np.pi*0.5)                  # u3(0,0,pi*0.5) q[0];
    circ.add_RX_gate(7, np.pi*0.5*-1)                     # rx(pi*0.5) q[7];
    circ.add_CNOT_gate(7, 0)                              # cx q[7],q[0];
    circ.add_RX_gate(7, np.pi*0.4*-1)                     # rx(pi*0.4) q[7];
    circ.add_RY_gate(0, np.pi*0.5*-1)                     # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 7)                              # cx q[0],q[7];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                     # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(7, 0)                              # cx q[7],q[0];
    circ.add_U3_gate(7, np.pi*1.0, 0, np.pi*0.5)          # u3(pi*1.0,0,pi*0.5) q[7];
    circ.add_U3_gate(0, np.pi*1.0, 0, np.pi*1.5)          # u3(pi*1.0,0,pi*1.5) q[0];
                                                          #
    # Gate: XX**1.1                                       # // Gate: XX**1.1
    circ.add_U3_gate(7, np.pi*0.5, np.pi*1.5, np.pi*1.5)  # u3(pi*0.5,pi*1.5,pi*1.5) q[7];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[0];
    circ.add_RX_gate(7, np.pi*0.5*-1)                     # rx(pi*0.5) q[7];
    circ.add_CNOT_gate(7, 0)                              # cx q[7],q[0];
    circ.add_RX_gate(7, np.pi*0.4*-1)                     # rx(pi*0.4) q[7];
    circ.add_RY_gate(0, np.pi*0.5*-1)                     # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 7)                              # cx q[0],q[7];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                     # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(7, 0)                              # cx q[7],q[0];
    circ.add_U3_gate(7, np.pi*0.5, np.pi*0.5, np.pi*0.5)  # u3(pi*0.5,pi*0.5,pi*0.5) q[7];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[0];
                                                          #
    circ.add_RX_gate(7, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[7];
    circ.add_RY_gate(7, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[7];
    circ.add_RZ_gate(7, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[7];
    circ.add_RX_gate(0, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[0];
    circ.add_RY_gate(0, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[0];
    circ.add_RZ_gate(0, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[0];



    circ.update_quantum_state(st)
#    with open('Qulacs.log', 'w') as f:
#        print(np.around(st.get_vector(),14)+0, file=f)
