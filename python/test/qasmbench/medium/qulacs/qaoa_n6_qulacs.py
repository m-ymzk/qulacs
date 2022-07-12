# Generated from Cirq v0.8.0

import numpy as np
from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import Identity, U1

def func(st, circ):
    # Qubits: [0,  1,  2,  3,  4,  5
#    nqubits=6
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)                                    # qreg q[6];
#    circ = QuantumCircuit(nqubits)                                #
                                                                  #
    circ.add_H_gate(0)                                            # h q[0];
    circ.add_H_gate(1)                                            # h q[1];
    circ.add_H_gate(2)                                            # h q[2];
    circ.add_H_gate(3)                                            # h q[3];
    circ.add_H_gate(4)                                            # h q[4];
    circ.add_H_gate(5)                                            # h q[5];
                                                                  #
    # Gate: ZZ**-0.9153964902652879                               # // Gate: ZZ**-0.9153964902652879
    circ.add_RZ_gate(0, np.pi*-0.9153964903*-1)                   # rz(pi*-0.9153964903) q[0];
    circ.add_RZ_gate(1, np.pi*-0.9153964903*-1)                   # rz(pi*-0.9153964903) q[1];
    circ.add_U3_gate(0, np.pi*0.5, 0, 0)                          # u3(pi*0.5,0,0) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, 0)                  # u3(pi*0.5,pi*1.0,0) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                             # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                      # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.4153964903*-1)                    # rx(pi*0.4153964903) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                             # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                      # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                            # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                             # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                                      # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.9153964903, np.pi*1.0) # u3(pi*0.5,pi*0.9153964903,pi*1.0) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.9153964903, 0)         # u3(pi*0.5,pi*0.9153964903,0) q[1];
                                                                  #
    # Gate: ZZ**-0.9153964902652879                               # // Gate: ZZ**-0.9153964902652879
    circ.add_RZ_gate(0, np.pi*-0.9153964903*-1)                   # rz(pi*-0.9153964903) q[0];
    circ.add_RZ_gate(2, np.pi*-0.9153964903*-1)                   # rz(pi*-0.9153964903) q[2];
    circ.add_U3_gate(0, np.pi*0.5, 0, 0)                          # u3(pi*0.5,0,0) q[0];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, 0)                  # u3(pi*0.5,pi*1.0,0) q[2];
    circ.add_RX_gate(0, np.pi*0.5*-1)                             # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 2)                                      # cx q[0],q[2];
    circ.add_RX_gate(0, np.pi*0.4153964903*-1)                    # rx(pi*0.4153964903) q[0];
    circ.add_RY_gate(2, np.pi*0.5*-1)                             # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 0)                                      # cx q[2],q[0];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                            # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                             # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(0, 2)                                      # cx q[0],q[2];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.9153964903, np.pi*1.0) # u3(pi*0.5,pi*0.9153964903,pi*1.0) q[0];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*0.9153964903, 0)         # u3(pi*0.5,pi*0.9153964903,0) q[2];
                                                                  #
    # Gate: ZZ**-0.9153964902652879                               # // Gate: ZZ**-0.9153964902652879
    circ.add_RZ_gate(0, np.pi*-0.9153964903*-1)                   # rz(pi*-0.9153964903) q[0];
    circ.add_RZ_gate(5, np.pi*-0.9153964903*-1)                   # rz(pi*-0.9153964903) q[5];
    circ.add_U3_gate(0, np.pi*0.5, 0, 0)                          # u3(pi*0.5,0,0) q[0];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*1.0, 0)                  # u3(pi*0.5,pi*1.0,0) q[5];
    circ.add_RX_gate(0, np.pi*0.5*-1)                             # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 5)                                      # cx q[0],q[5];
    circ.add_RX_gate(0, np.pi*0.4153964903*-1)                    # rx(pi*0.4153964903) q[0];
    circ.add_RY_gate(5, np.pi*0.5*-1)                             # ry(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 0)                                      # cx q[5],q[0];
    circ.add_RX_gate(5, np.pi*-0.5*-1)                            # rx(pi*-0.5) q[5];
    circ.add_RZ_gate(5, np.pi*0.5*-1)                             # rz(pi*0.5) q[5];
    circ.add_CNOT_gate(0, 5)                                      # cx q[0],q[5];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.9153964903, np.pi*1.0) # u3(pi*0.5,pi*0.9153964903,pi*1.0) q[0];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*0.9153964903, 0)         # u3(pi*0.5,pi*0.9153964903,0) q[5];
                                                                  #
    # Gate: ZZ**-0.9153964902652879                               # // Gate: ZZ**-0.9153964902652879
    circ.add_RZ_gate(1, np.pi*-0.9153964903*-1)                   # rz(pi*-0.9153964903) q[1];
    circ.add_RZ_gate(2, np.pi*-0.9153964903*-1)                   # rz(pi*-0.9153964903) q[2];
    circ.add_U3_gate(1, np.pi*0.5, 0, 0)                          # u3(pi*0.5,0,0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, 0)                  # u3(pi*0.5,pi*1.0,0) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                             # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                      # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.4153964903*-1)                    # rx(pi*0.4153964903) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                             # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                      # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                            # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                             # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                      # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.9153964903, np.pi*1.0) # u3(pi*0.5,pi*0.9153964903,pi*1.0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*0.9153964903, 0)         # u3(pi*0.5,pi*0.9153964903,0) q[2];
                                                                  #
    # Gate: ZZ**-0.9153964902652879                               # // Gate: ZZ**-0.9153964902652879
    circ.add_RZ_gate(1, np.pi*-0.9153964903*-1)                   # rz(pi*-0.9153964903) q[1];
    circ.add_RZ_gate(3, np.pi*-0.9153964903*-1)                   # rz(pi*-0.9153964903) q[3];
    circ.add_U3_gate(1, np.pi*0.5, 0, 0)                          # u3(pi*0.5,0,0) q[1];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0, 0)                  # u3(pi*0.5,pi*1.0,0) q[3];
    circ.add_RX_gate(1, np.pi*0.5*-1)                             # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 3)                                      # cx q[1],q[3];
    circ.add_RX_gate(1, np.pi*0.4153964903*-1)                    # rx(pi*0.4153964903) q[1];
    circ.add_RY_gate(3, np.pi*0.5*-1)                             # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 1)                                      # cx q[3],q[1];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                            # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                             # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(1, 3)                                      # cx q[1],q[3];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.9153964903, np.pi*1.0) # u3(pi*0.5,pi*0.9153964903,pi*1.0) q[1];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.9153964903, 0)         # u3(pi*0.5,pi*0.9153964903,0) q[3];
                                                                  #
    # Gate: ZZ**-0.9153964902652879                               # // Gate: ZZ**-0.9153964902652879
    circ.add_RZ_gate(2, np.pi*-0.9153964903*-1)                   # rz(pi*-0.9153964903) q[2];
    circ.add_RZ_gate(4, np.pi*-0.9153964903*-1)                   # rz(pi*-0.9153964903) q[4];
    circ.add_U3_gate(2, np.pi*0.5, 0, 0)                          # u3(pi*0.5,0,0) q[2];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*1.0, 0)                  # u3(pi*0.5,pi*1.0,0) q[4];
    circ.add_RX_gate(2, np.pi*0.5*-1)                             # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 4)                                      # cx q[2],q[4];
    circ.add_RX_gate(2, np.pi*0.4153964903*-1)                    # rx(pi*0.4153964903) q[2];
    circ.add_RY_gate(4, np.pi*0.5*-1)                             # ry(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 2)                                      # cx q[4],q[2];
    circ.add_RX_gate(4, np.pi*-0.5*-1)                            # rx(pi*-0.5) q[4];
    circ.add_RZ_gate(4, np.pi*0.5*-1)                             # rz(pi*0.5) q[4];
    circ.add_CNOT_gate(2, 4)                                      # cx q[2],q[4];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*0.9153964903, np.pi*1.0) # u3(pi*0.5,pi*0.9153964903,pi*1.0) q[2];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*0.9153964903, 0)         # u3(pi*0.5,pi*0.9153964903,0) q[4];
                                                                  #
    circ.add_RX_gate(0, np.pi*-0.6320733477*-1)                   # rx(pi*-0.6320733477) q[0];
                                                                  #
    # Gate: ZZ**-0.9153964902652879                               # // Gate: ZZ**-0.9153964902652879
    circ.add_RZ_gate(4, np.pi*-0.9153964903*-1)                   # rz(pi*-0.9153964903) q[4];
    circ.add_RZ_gate(3, np.pi*-0.9153964903*-1)                   # rz(pi*-0.9153964903) q[3];
    circ.add_U3_gate(4, np.pi*0.5, 0, 0)                          # u3(pi*0.5,0,0) q[4];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0, 0)                  # u3(pi*0.5,pi*1.0,0) q[3];
    circ.add_RX_gate(4, np.pi*0.5*-1)                             # rx(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 3)                                      # cx q[4],q[3];
    circ.add_RX_gate(4, np.pi*0.4153964903*-1)                    # rx(pi*0.4153964903) q[4];
    circ.add_RY_gate(3, np.pi*0.5*-1)                             # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 4)                                      # cx q[3],q[4];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                            # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                             # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(4, 3)                                      # cx q[4],q[3];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*0.9153964903, np.pi*1.0) # u3(pi*0.5,pi*0.9153964903,pi*1.0) q[4];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.9153964903, 0)         # u3(pi*0.5,pi*0.9153964903,0) q[3];
                                                                  #
    circ.add_RX_gate(1, np.pi*-0.6320733477*-1)                   # rx(pi*-0.6320733477) q[1];
    circ.add_RX_gate(2, np.pi*-0.6320733477*-1)                   # rx(pi*-0.6320733477) q[2];
                                                                  #
    # Gate: ZZ**-0.9153964902652879                               # // Gate: ZZ**-0.9153964902652879
    circ.add_RZ_gate(4, np.pi*-0.9153964903*-1)                   # rz(pi*-0.9153964903) q[4];
    circ.add_RZ_gate(5, np.pi*-0.9153964903*-1)                   # rz(pi*-0.9153964903) q[5];
    circ.add_U3_gate(4, np.pi*0.5, 0, 0)                          # u3(pi*0.5,0,0) q[4];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*1.0, 0)                  # u3(pi*0.5,pi*1.0,0) q[5];
    circ.add_RX_gate(4, np.pi*0.5*-1)                             # rx(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 5)                                      # cx q[4],q[5];
    circ.add_RX_gate(4, np.pi*0.4153964903*-1)                    # rx(pi*0.4153964903) q[4];
    circ.add_RY_gate(5, np.pi*0.5*-1)                             # ry(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 4)                                      # cx q[5],q[4];
    circ.add_RX_gate(5, np.pi*-0.5*-1)                            # rx(pi*-0.5) q[5];
    circ.add_RZ_gate(5, np.pi*0.5*-1)                             # rz(pi*0.5) q[5];
    circ.add_CNOT_gate(4, 5)                                      # cx q[4],q[5];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*0.9153964903, np.pi*1.0) # u3(pi*0.5,pi*0.9153964903,pi*1.0) q[4];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*0.9153964903, 0)         # u3(pi*0.5,pi*0.9153964903,0) q[5];
                                                                  #
    # Gate: ZZ**0.14873770971193984                               # // Gate: ZZ**0.14873770971193984
    circ.add_RZ_gate(0, np.pi*0.1487377097*-1)                    # rz(pi*0.1487377097) q[0];
    circ.add_RZ_gate(1, np.pi*0.1487377097*-1)                    # rz(pi*0.1487377097) q[1];
    circ.add_U3_gate(0, np.pi*0.5, 0, np.pi*1.8013661765)         # u3(pi*0.5,0,pi*1.8013661765) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.8013661765) # u3(pi*0.5,pi*1.0,pi*1.8013661765) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                             # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                      # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.3512622903*-1)                    # rx(pi*0.3512622903) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                             # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                      # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                            # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                             # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                                      # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0498961138, np.pi*1.0) # u3(pi*0.5,pi*1.0498961138,pi*1.0) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0498961138, 0)         # u3(pi*0.5,pi*1.0498961138,0) q[1];
                                                                  #
    # Gate: ZZ**-0.9153964902652879                               # // Gate: ZZ**-0.9153964902652879
    circ.add_RZ_gate(3, np.pi*-0.9153964903*-1)                   # rz(pi*-0.9153964903) q[3];
    circ.add_RZ_gate(5, np.pi*-0.9153964903*-1)                   # rz(pi*-0.9153964903) q[5];
    circ.add_U3_gate(3, np.pi*0.5, 0, 0)                          # u3(pi*0.5,0,0) q[3];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*1.0, 0)                  # u3(pi*0.5,pi*1.0,0) q[5];
    circ.add_RX_gate(3, np.pi*0.5*-1)                             # rx(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 5)                                      # cx q[3],q[5];
    circ.add_RX_gate(3, np.pi*0.4153964903*-1)                    # rx(pi*0.4153964903) q[3];
    circ.add_RY_gate(5, np.pi*0.5*-1)                             # ry(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 3)                                      # cx q[5],q[3];
    circ.add_RX_gate(5, np.pi*-0.5*-1)                            # rx(pi*-0.5) q[5];
    circ.add_RZ_gate(5, np.pi*0.5*-1)                             # rz(pi*0.5) q[5];
    circ.add_CNOT_gate(3, 5)                                      # cx q[3],q[5];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.9153964903, np.pi*1.0) # u3(pi*0.5,pi*0.9153964903,pi*1.0) q[3];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*0.9153964903, 0)         # u3(pi*0.5,pi*0.9153964903,0) q[5];
                                                                  #
    circ.add_RX_gate(4, np.pi*-0.6320733477*-1)                   # rx(pi*-0.6320733477) q[4];
                                                                  #
    # Gate: ZZ**0.14873770971193984                               # // Gate: ZZ**0.14873770971193984
    circ.add_RZ_gate(0, np.pi*0.1487377097*-1)                    # rz(pi*0.1487377097) q[0];
    circ.add_RZ_gate(2, np.pi*0.1487377097*-1)                    # rz(pi*0.1487377097) q[2];
    circ.add_U3_gate(0, np.pi*0.5, 0, np.pi*1.8013661765)         # u3(pi*0.5,0,pi*1.8013661765) q[0];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.8013661765) # u3(pi*0.5,pi*1.0,pi*1.8013661765) q[2];
    circ.add_RX_gate(0, np.pi*0.5*-1)                             # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 2)                                      # cx q[0],q[2];
    circ.add_RX_gate(0, np.pi*0.3512622903*-1)                    # rx(pi*0.3512622903) q[0];
    circ.add_RY_gate(2, np.pi*0.5*-1)                             # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 0)                                      # cx q[2],q[0];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                            # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                             # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(0, 2)                                      # cx q[0],q[2];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0498961138, np.pi*1.0) # u3(pi*0.5,pi*1.0498961138,pi*1.0) q[0];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0498961138, 0)         # u3(pi*0.5,pi*1.0498961138,0) q[2];
                                                                  #
    circ.add_RX_gate(3, np.pi*-0.6320733477*-1)                   # rx(pi*-0.6320733477) q[3];
    circ.add_RX_gate(5, np.pi*-0.6320733477*-1)                   # rx(pi*-0.6320733477) q[5];
                                                                  #
    # Gate: ZZ**0.14873770971193984                               # // Gate: ZZ**0.14873770971193984
    circ.add_RZ_gate(1, np.pi*0.1487377097*-1)                    # rz(pi*0.1487377097) q[1];
    circ.add_RZ_gate(2, np.pi*0.1487377097*-1)                    # rz(pi*0.1487377097) q[2];
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*1.8013661765)         # u3(pi*0.5,0,pi*1.8013661765) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.8013661765) # u3(pi*0.5,pi*1.0,pi*1.8013661765) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                             # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                      # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.3512622903*-1)                    # rx(pi*0.3512622903) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                             # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                      # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                            # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                             # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                      # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0498961138, np.pi*1.0) # u3(pi*0.5,pi*1.0498961138,pi*1.0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0498961138, 0)         # u3(pi*0.5,pi*1.0498961138,0) q[2];
                                                                  #
    # Gate: ZZ**0.14873770971193984                               # // Gate: ZZ**0.14873770971193984
    circ.add_RZ_gate(0, np.pi*0.1487377097*-1)                    # rz(pi*0.1487377097) q[0];
    circ.add_RZ_gate(5, np.pi*0.1487377097*-1)                    # rz(pi*0.1487377097) q[5];
    circ.add_U3_gate(0, np.pi*0.5, 0, np.pi*1.8013661765)         # u3(pi*0.5,0,pi*1.8013661765) q[0];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*1.0, np.pi*1.8013661765) # u3(pi*0.5,pi*1.0,pi*1.8013661765) q[5];
    circ.add_RX_gate(0, np.pi*0.5*-1)                             # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 5)                                      # cx q[0],q[5];
    circ.add_RX_gate(0, np.pi*0.3512622903*-1)                    # rx(pi*0.3512622903) q[0];
    circ.add_RY_gate(5, np.pi*0.5*-1)                             # ry(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 0)                                      # cx q[5],q[0];
    circ.add_RX_gate(5, np.pi*-0.5*-1)                            # rx(pi*-0.5) q[5];
    circ.add_RZ_gate(5, np.pi*0.5*-1)                             # rz(pi*0.5) q[5];
    circ.add_CNOT_gate(0, 5)                                      # cx q[0],q[5];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0498961138, np.pi*1.0) # u3(pi*0.5,pi*1.0498961138,pi*1.0) q[0];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*1.0498961138, 0)         # u3(pi*0.5,pi*1.0498961138,0) q[5];
                                                                  #
    # Gate: ZZ**0.14873770971193984                               # // Gate: ZZ**0.14873770971193984
    circ.add_RZ_gate(1, np.pi*0.1487377097*-1)                    # rz(pi*0.1487377097) q[1];
    circ.add_RZ_gate(3, np.pi*0.1487377097*-1)                    # rz(pi*0.1487377097) q[3];
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*1.8013661765)         # u3(pi*0.5,0,pi*1.8013661765) q[1];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0, np.pi*1.8013661765) # u3(pi*0.5,pi*1.0,pi*1.8013661765) q[3];
    circ.add_RX_gate(1, np.pi*0.5*-1)                             # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 3)                                      # cx q[1],q[3];
    circ.add_RX_gate(1, np.pi*0.3512622903*-1)                    # rx(pi*0.3512622903) q[1];
    circ.add_RY_gate(3, np.pi*0.5*-1)                             # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 1)                                      # cx q[3],q[1];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                            # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                             # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(1, 3)                                      # cx q[1],q[3];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0498961138, np.pi*1.0) # u3(pi*0.5,pi*1.0498961138,pi*1.0) q[1];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0498961138, 0)         # u3(pi*0.5,pi*1.0498961138,0) q[3];
                                                                  #
    # Gate: ZZ**0.14873770971193984                               # // Gate: ZZ**0.14873770971193984
    circ.add_RZ_gate(2, np.pi*0.1487377097*-1)                    # rz(pi*0.1487377097) q[2];
    circ.add_RZ_gate(4, np.pi*0.1487377097*-1)                    # rz(pi*0.1487377097) q[4];
    circ.add_U3_gate(2, np.pi*0.5, 0, np.pi*1.8013661765)         # u3(pi*0.5,0,pi*1.8013661765) q[2];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*1.0, np.pi*1.8013661765) # u3(pi*0.5,pi*1.0,pi*1.8013661765) q[4];
    circ.add_RX_gate(2, np.pi*0.5*-1)                             # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 4)                                      # cx q[2],q[4];
    circ.add_RX_gate(2, np.pi*0.3512622903*-1)                    # rx(pi*0.3512622903) q[2];
    circ.add_RY_gate(4, np.pi*0.5*-1)                             # ry(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 2)                                      # cx q[4],q[2];
    circ.add_RX_gate(4, np.pi*-0.5*-1)                            # rx(pi*-0.5) q[4];
    circ.add_RZ_gate(4, np.pi*0.5*-1)                             # rz(pi*0.5) q[4];
    circ.add_CNOT_gate(2, 4)                                      # cx q[2],q[4];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0498961138, np.pi*1.0) # u3(pi*0.5,pi*1.0498961138,pi*1.0) q[2];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*1.0498961138, 0)         # u3(pi*0.5,pi*1.0498961138,0) q[4];
                                                                  #
    # Gate: ZZ**0.14873770971193984                               # // Gate: ZZ**0.14873770971193984
    circ.add_RZ_gate(4, np.pi*0.1487377097*-1)                    # rz(pi*0.1487377097) q[4];
    circ.add_RZ_gate(3, np.pi*0.1487377097*-1)                    # rz(pi*0.1487377097) q[3];
    circ.add_U3_gate(4, np.pi*0.5, 0, np.pi*1.8013661765)         # u3(pi*0.5,0,pi*1.8013661765) q[4];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0, np.pi*1.8013661765) # u3(pi*0.5,pi*1.0,pi*1.8013661765) q[3];
    circ.add_RX_gate(4, np.pi*0.5*-1)                             # rx(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 3)                                      # cx q[4],q[3];
    circ.add_RX_gate(4, np.pi*0.3512622903*-1)                    # rx(pi*0.3512622903) q[4];
    circ.add_RY_gate(3, np.pi*0.5*-1)                             # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 4)                                      # cx q[3],q[4];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                            # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                             # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(4, 3)                                      # cx q[4],q[3];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*1.0498961138, np.pi*1.0) # u3(pi*0.5,pi*1.0498961138,pi*1.0) q[4];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0498961138, 0)         # u3(pi*0.5,pi*1.0498961138,0) q[3];
                                                                  #
    circ.add_RX_gate(0, np.pi*-0.6710086873*-1)                   # rx(pi*-0.6710086873) q[0];
    circ.add_RX_gate(1, np.pi*-0.6710086873*-1)                   # rx(pi*-0.6710086873) q[1];
    circ.add_RX_gate(2, np.pi*-0.6710086873*-1)                   # rx(pi*-0.6710086873) q[2];
                                                                  #
    # Gate: ZZ**0.14873770971193984                               # // Gate: ZZ**0.14873770971193984
    circ.add_RZ_gate(4, np.pi*0.1487377097*-1)                    # rz(pi*0.1487377097) q[4];
    circ.add_RZ_gate(5, np.pi*0.1487377097*-1)                    # rz(pi*0.1487377097) q[5];
    circ.add_U3_gate(4, np.pi*0.5, 0, np.pi*1.8013661765)         # u3(pi*0.5,0,pi*1.8013661765) q[4];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*1.0, np.pi*1.8013661765) # u3(pi*0.5,pi*1.0,pi*1.8013661765) q[5];
    circ.add_RX_gate(4, np.pi*0.5*-1)                             # rx(pi*0.5) q[4];
    circ.add_CNOT_gate(4, 5)                                      # cx q[4],q[5];
    circ.add_RX_gate(4, np.pi*0.3512622903*-1)                    # rx(pi*0.3512622903) q[4];
    circ.add_RY_gate(5, np.pi*0.5*-1)                             # ry(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 4)                                      # cx q[5],q[4];
    circ.add_RX_gate(5, np.pi*-0.5*-1)                            # rx(pi*-0.5) q[5];
    circ.add_RZ_gate(5, np.pi*0.5*-1)                             # rz(pi*0.5) q[5];
    circ.add_CNOT_gate(4, 5)                                      # cx q[4],q[5];
    circ.add_U3_gate(4, np.pi*0.5, np.pi*1.0498961138, np.pi*1.0) # u3(pi*0.5,pi*1.0498961138,pi*1.0) q[4];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*1.0498961138, 0)         # u3(pi*0.5,pi*1.0498961138,0) q[5];
                                                                  #
    # Gate: ZZ**0.14873770971193984                               # // Gate: ZZ**0.14873770971193984
    circ.add_RZ_gate(3, np.pi*0.1487377097*-1)                    # rz(pi*0.1487377097) q[3];
    circ.add_RZ_gate(5, np.pi*0.1487377097*-1)                    # rz(pi*0.1487377097) q[5];
    circ.add_U3_gate(3, np.pi*0.5, 0, np.pi*1.8013661765)         # u3(pi*0.5,0,pi*1.8013661765) q[3];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*1.0, np.pi*1.8013661765) # u3(pi*0.5,pi*1.0,pi*1.8013661765) q[5];
    circ.add_RX_gate(3, np.pi*0.5*-1)                             # rx(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 5)                                      # cx q[3],q[5];
    circ.add_RX_gate(3, np.pi*0.3512622903*-1)                    # rx(pi*0.3512622903) q[3];
    circ.add_RY_gate(5, np.pi*0.5*-1)                             # ry(pi*0.5) q[5];
    circ.add_CNOT_gate(5, 3)                                      # cx q[5],q[3];
    circ.add_RX_gate(5, np.pi*-0.5*-1)                            # rx(pi*-0.5) q[5];
    circ.add_RZ_gate(5, np.pi*0.5*-1)                             # rz(pi*0.5) q[5];
    circ.add_CNOT_gate(3, 5)                                      # cx q[3],q[5];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0498961138, np.pi*1.0) # u3(pi*0.5,pi*1.0498961138,pi*1.0) q[3];
    circ.add_U3_gate(5, np.pi*0.5, np.pi*1.0498961138, 0)         # u3(pi*0.5,pi*1.0498961138,0) q[5];
                                                                  #
    circ.add_RX_gate(4, np.pi*-0.6710086873*-1)                   # rx(pi*-0.6710086873) q[4];
    circ.add_RX_gate(3, np.pi*-0.6710086873*-1)                   # rx(pi*-0.6710086873) q[3];
    circ.add_RX_gate(5, np.pi*-0.6710086873*-1)                   # rx(pi*-0.6710086873) q[5];

    circ.update_quantum_state(st)
#    with open('Qulacs.log', 'w') as f:
#        print(np.around(st.get_vector(),14)+0, file=f)
