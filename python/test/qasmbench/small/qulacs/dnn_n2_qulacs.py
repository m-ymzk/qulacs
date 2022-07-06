# A TWO QUBIT CIRCIUT 3 LAYERS DEEP
# Generated from Cirq v0.8.0

import numpy as np
from qulacs import QuantumCircuit, QuantumState

def func(st, circ):
    # Qubits: [(0,  0),  (0,  1)
#    nqubits=2
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)
#    circ = QuantumCircuit(nqubits)                        # qreg q[2];
                                                          #
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
    circ.add_RX_gate(1, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[1];
    circ.add_RY_gate(1, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[1];
    circ.add_RZ_gate(1, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[1];
    circ.add_RX_gate(0, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[0];
    circ.add_RY_gate(0, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[0];
    circ.add_RZ_gate(0, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[0];
                                                          #
    # Gate: ZZ**1.1                                       # // Gate: ZZ**1.1
    circ.add_RZ_gate(1, np.pi*1.1*-1)                     # rz(pi*1.1) q[1];
    circ.add_RZ_gate(0, np.pi*1.1*-1)                     # rz(pi*1.1) q[0];
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[0];
    circ.add_RX_gate(1, np.pi*0.5*-1)                     # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*0.4*-1)                     # rx(pi*0.4) q[1];
    circ.add_RY_gate(0, np.pi*0.5*-1)                     # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                     # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.65, np.pi*1.0) # u3(pi*0.5,pi*0.65,pi*1.0) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.15, 0)         # u3(pi*0.5,pi*0.15,0) q[0];
                                                          #
    # Gate: YY**1.1                                       # // Gate: YY**1.1
    circ.add_U3_gate(1, 0, np.pi*1.0, np.pi*0.5)          # u3(0,pi*1.0,pi*0.5) q[1];
    circ.add_U3_gate(0, 0, 0, np.pi*0.5)                  # u3(0,0,pi*0.5) q[0];
    circ.add_RX_gate(1, np.pi*0.5*-1)                     # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*0.4*-1)                     # rx(pi*0.4) q[1];
    circ.add_RY_gate(0, np.pi*0.5*-1)                     # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                     # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_U3_gate(1, np.pi*1.0, 0, np.pi*0.5)          # u3(pi*1.0,0,pi*0.5) q[1];
    circ.add_U3_gate(0, np.pi*1.0, 0, np.pi*1.5)          # u3(pi*1.0,0,pi*1.5) q[0];
                                                          #
    # Gate: XX**1.1                                       # // Gate: XX**1.1
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.5, np.pi*1.5)  # u3(pi*0.5,pi*1.5,pi*1.5) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[0];
    circ.add_RX_gate(1, np.pi*0.5*-1)                     # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*0.4*-1)                     # rx(pi*0.4) q[1];
    circ.add_RY_gate(0, np.pi*0.5*-1)                     # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                     # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.5, np.pi*0.5)  # u3(pi*0.5,pi*0.5,pi*0.5) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[0];
                                                          #
    circ.add_RX_gate(1, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[1];
    circ.add_RY_gate(1, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[1];
    circ.add_RZ_gate(1, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[1];
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
    circ.add_RX_gate(1, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[1];
    circ.add_RY_gate(1, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[1];
    circ.add_RZ_gate(1, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[1];
    circ.add_RX_gate(0, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[0];
    circ.add_RY_gate(0, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[0];
    circ.add_RZ_gate(0, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[0];
                                                          #
    # Gate: ZZ**1.1                                       # // Gate: ZZ**1.1
    circ.add_RZ_gate(1, np.pi*1.1*-1)                     # rz(pi*1.1) q[1];
    circ.add_RZ_gate(0, np.pi*1.1*-1)                     # rz(pi*1.1) q[0];
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*0.25)         # u3(pi*0.5,0,pi*0.25) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0, np.pi*0.75) # u3(pi*0.5,pi*1.0,pi*0.75) q[0];
    circ.add_RX_gate(1, np.pi*0.5*-1)                     # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*0.4*-1)                     # rx(pi*0.4) q[1];
    circ.add_RY_gate(0, np.pi*0.5*-1)                     # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                     # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.65, np.pi*1.0) # u3(pi*0.5,pi*0.65,pi*1.0) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.15, 0)         # u3(pi*0.5,pi*0.15,0) q[0];
                                                          #
    # Gate: YY**1.1                                       # // Gate: YY**1.1
    circ.add_U3_gate(1, 0, np.pi*1.0, np.pi*0.5)          # u3(0,pi*1.0,pi*0.5) q[1];
    circ.add_U3_gate(0, 0, 0, np.pi*0.5)                  # u3(0,0,pi*0.5) q[0];
    circ.add_RX_gate(1, np.pi*0.5*-1)                     # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*0.4*-1)                     # rx(pi*0.4) q[1];
    circ.add_RY_gate(0, np.pi*0.5*-1)                     # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                     # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_U3_gate(1, np.pi*1.0, 0, np.pi*0.5)          # u3(pi*1.0,0,pi*0.5) q[1];
    circ.add_U3_gate(0, np.pi*1.0, 0, np.pi*1.5)          # u3(pi*1.0,0,pi*1.5) q[0];
                                                          #
    # Gate: XX**1.1                                       # // Gate: XX**1.1
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.5, np.pi*1.5)  # u3(pi*0.5,pi*1.5,pi*1.5) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[0];
    circ.add_RX_gate(1, np.pi*0.5*-1)                     # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*0.4*-1)                     # rx(pi*0.4) q[1];
    circ.add_RY_gate(0, np.pi*0.5*-1)                     # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                              # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                    # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                     # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(1, 0)                              # cx q[1],q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.5, np.pi*0.5)  # u3(pi*0.5,pi*0.5,pi*0.5) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.5, np.pi*1.5)  # u3(pi*0.5,pi*0.5,pi*1.5) q[0];
                                                          #
    circ.add_RX_gate(1, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[1];
    circ.add_RY_gate(1, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[1];
    circ.add_RZ_gate(1, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[1];
    circ.add_RX_gate(0, np.pi*0.3501408748*-1)            # rx(pi*0.3501408748) q[0];
    circ.add_RY_gate(0, np.pi*0.3501408748*-1)            # ry(pi*0.3501408748) q[0];
    circ.add_RZ_gate(0, np.pi*0.3501408748*-1)            # rz(pi*0.3501408748) q[0];

    circ.update_quantum_state(st)
#    with open('Qulacs.log', 'w') as f:
#        print(np.around(st.get_vector(),14)+0, file=f)
