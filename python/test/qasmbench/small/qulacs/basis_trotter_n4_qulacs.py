# Generated from Cirq v0.8.0

import numpy as np
from qulacs import QuantumCircuit, QuantumState
def func(st, circ):
    # Qubits: [0,  1,  2,  3
    #nqubits=4
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
    #st = QuantumState(nqubits)                                     # qreg q[4];
    #circ = QuantumCircuit(nqubits)

    circ.add_Z_gate(0)                                             # z q[0];
    circ.add_Z_gate(1)                                             # z q[1];
    circ.add_Z_gate(2)                                             # z q[2];
    circ.add_Z_gate(3)                                             # z q[3];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    # Gate: PhasedISWAP**0.08130614625631793                       # // Gate: PhasedISWAP**0.08130614625631793
    circ.add_RZ_gate(0, np.pi*0.25*-1)                             # rz(pi*0.25) q[0];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_H_gate(0)                                             # h q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(0, np.pi*0.0406530731*-1)                     # rz(pi*0.0406530731) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(0, np.pi*-0.0406530731*-1)                    # rz(pi*-0.0406530731) q[0];
    circ.add_H_gate(0)                                             # h q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(0, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[0];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    # Gate: PhasedISWAP**-0.08130614625631793                      # // Gate: PhasedISWAP**-0.08130614625631793
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(3, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(2, np.pi*-0.0406530731*-1)                    # rz(pi*-0.0406530731) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(2, np.pi*0.0406530731*-1)                     # rz(pi*0.0406530731) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(3, np.pi*0.25*-1)                             # rz(pi*0.25) q[3];
    #
    circ.add_RZ_gate(0, np.pi*0.1123177385*-1)                     # rz(pi*0.1123177385) q[0];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    circ.add_RZ_gate(1, np.pi*0.1123177385*-1)                     # rz(pi*0.1123177385) q[1];
    circ.add_RZ_gate(3, np.pi*0.0564909955*-1)                     # rz(pi*0.0564909955) q[3];
    circ.add_RZ_gate(2, np.pi*0.0564909955*-1)                     # rz(pi*0.0564909955) q[2];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    # Gate: PhasedISWAP**-0.05102950815299322                      # // Gate: PhasedISWAP**-0.05102950815299322
    circ.add_RZ_gate(0, np.pi*0.25*-1)                             # rz(pi*0.25) q[0];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_H_gate(0)                                             # h q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(0, np.pi*-0.0255147541*-1)                    # rz(pi*-0.0255147541) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(0, np.pi*0.0255147541*-1)                     # rz(pi*0.0255147541) q[0];
    circ.add_H_gate(0)                                             # h q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(0, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[0];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    # Gate: PhasedISWAP**0.05102950815299322                       # // Gate: PhasedISWAP**0.05102950815299322
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(3, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(2, np.pi*0.0255147541*-1)                     # rz(pi*0.0255147541) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(2, np.pi*-0.0255147541*-1)                    # rz(pi*-0.0255147541) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(3, np.pi*0.25*-1)                             # rz(pi*0.25) q[3];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    # Gate: CZ**-0.048279591094340914                              # // Gate: CZ**-0.048279591094340914
    circ.add_U3_gate(0, np.pi*0.5, 0, np.pi*0.5)                   # u3(pi*0.5,0,pi*0.5) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.0)           # u3(pi*0.5,pi*1.0,pi*1.0) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                              # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.4758602045*-1)                     # rx(pi*0.4758602045) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                              # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.4758602045, np.pi*1.0)  # u3(pi*0.5,pi*0.4758602045,pi*1.0) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.9758602045, 0)          # u3(pi*0.5,pi*1.9758602045,0) q[1];
    #
    circ.add_SWAP_gate(0, 1)                                       # swap q[0],q[1];
    #
    # Gate: CZ**-0.022156912718971442                              # // Gate: CZ**-0.022156912718971442
    circ.add_U3_gate(2, np.pi*0.5, 0, np.pi*1.75)                  # u3(pi*0.5,0,pi*1.75) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0, np.pi*1.25)          # u3(pi*0.5,pi*1.0,pi*1.25) q[3];
    circ.add_RX_gate(2, np.pi*0.5*-1)                              # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*0.4889215436*-1)                     # rx(pi*0.4889215436) q[2];
    circ.add_RY_gate(3, np.pi*0.5*-1)                              # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                              # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.2389215436, np.pi*1.0)  # u3(pi*0.5,pi*1.2389215436,pi*1.0) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.7389215436, 0)          # u3(pi*0.5,pi*1.7389215436,0) q[3];
    #
    circ.add_SWAP_gate(2, 3)                                       # swap q[2],q[3];
    #
    # Gate: CZ**-0.03270667647415345                               # // Gate: CZ**-0.03270667647415345
    circ.add_U3_gate(1, np.pi*0.5, 0, 0)                           # u3(pi*0.5,0,0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.5)           # u3(pi*0.5,pi*1.0,pi*1.5) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                              # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.4836466618*-1)                     # rx(pi*0.4836466618) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                              # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.9836466618, np.pi*1.0)  # u3(pi*0.5,pi*0.9836466618,pi*1.0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.4836466618, 0)          # u3(pi*0.5,pi*1.4836466618,0) q[2];
    #
    circ.add_SWAP_gate(1, 2)                                       # swap q[1],q[2];
    #
    # Gate: CZ**-0.03270667647415345                               # // Gate: CZ**-0.03270667647415345
    circ.add_U3_gate(0, np.pi*0.5, 0, 0)                           # u3(pi*0.5,0,0) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.5)           # u3(pi*0.5,pi*1.0,pi*1.5) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                              # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.4836466618*-1)                     # rx(pi*0.4836466618) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                              # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.9836466618, np.pi*1.0)  # u3(pi*0.5,pi*0.9836466618,pi*1.0) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.4836466618, 0)          # u3(pi*0.5,pi*1.4836466618,0) q[1];
    #
    # Gate: CZ**-0.03270667647415345                               # // Gate: CZ**-0.03270667647415345
    circ.add_U3_gate(2, np.pi*0.5, 0, 0)                           # u3(pi*0.5,0,0) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0, np.pi*1.5)           # u3(pi*0.5,pi*1.0,pi*1.5) q[3];
    circ.add_RX_gate(2, np.pi*0.5*-1)                              # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*0.4836466618*-1)                     # rx(pi*0.4836466618) q[2];
    circ.add_RY_gate(3, np.pi*0.5*-1)                              # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                              # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*0.9836466618, np.pi*1.0)  # u3(pi*0.5,pi*0.9836466618,pi*1.0) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.4836466618, 0)          # u3(pi*0.5,pi*1.4836466618,0) q[3];
    #
    circ.add_SWAP_gate(0, 1)                                       # swap q[0],q[1];
    circ.add_SWAP_gate(2, 3)                                       # swap q[2],q[3];
    #
    # Gate: CZ**-0.03270667647415345                               # // Gate: CZ**-0.03270667647415345
    circ.add_U3_gate(1, np.pi*0.5, 0, 0)                           # u3(pi*0.5,0,0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.5)           # u3(pi*0.5,pi*1.0,pi*1.5) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                              # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.4836466618*-1)                     # rx(pi*0.4836466618) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                              # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.9836466618, np.pi*1.0)  # u3(pi*0.5,pi*0.9836466618,pi*1.0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.4836466618, 0)          # u3(pi*0.5,pi*1.4836466618,0) q[2];
    #
    circ.add_RZ_gate(3, np.pi*-0.0241397955*-1)                    # rz(pi*-0.0241397955) q[3];
    circ.add_RZ_gate(0, np.pi*-0.0110784564*-1)                    # rz(pi*-0.0110784564) q[0];
    circ.add_SWAP_gate(1, 2)                                       # swap q[1],q[2];
    circ.add_RZ_gate(2, np.pi*-0.0241397955*-1)                    # rz(pi*-0.0241397955) q[2];
    circ.add_RZ_gate(1, np.pi*-0.0110784564*-1)                    # rz(pi*-0.0110784564) q[1];
    circ.add_Z_gate(2)                                             # z q[2];
    circ.add_Z_gate(1)                                             # z q[1];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    # Gate: PhasedISWAP**-0.9500630905158097                       # // Gate: PhasedISWAP**-0.9500630905158097
    circ.add_RZ_gate(3, np.pi*0.25*-1)                             # rz(pi*0.25) q[3];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_H_gate(3)                                             # h q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(3, np.pi*-0.4750315453*-1)                    # rz(pi*-0.4750315453) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(3, np.pi*0.4750315453*-1)                     # rz(pi*0.4750315453) q[3];
    circ.add_H_gate(3)                                             # h q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(3, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[3];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    # Gate: PhasedISWAP**0.9500630905158097                        # // Gate: PhasedISWAP**0.9500630905158097
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(0, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(1, np.pi*0.4750315453*-1)                     # rz(pi*0.4750315453) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(1, np.pi*-0.4750315453*-1)                    # rz(pi*-0.4750315453) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(0, np.pi*0.25*-1)                             # rz(pi*0.25) q[0];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
                                                                   #
    # Gate: CZ**-0.013654184706660842                              # // Gate: CZ**-0.013654184706660842
    circ.add_U3_gate(3, np.pi*0.5, 0, np.pi*1.5)                   # u3(pi*0.5,0,pi*1.5) q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.0)           # u3(pi*0.5,pi*1.0,pi*1.0) q[2];
    circ.add_RX_gate(3, np.pi*0.5*-1)                              # rx(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*0.4931729076*-1)                     # rx(pi*0.4931729076) q[3];
    circ.add_RY_gate(2, np.pi*0.5*-1)                              # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.4931729076, np.pi*1.0)  # u3(pi*0.5,pi*1.4931729076,pi*1.0) q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.9931729076, 0)          # u3(pi*0.5,pi*1.9931729076,0) q[2];
                                                                   #
    circ.add_SWAP_gate(3, 2)                                       # swap q[3],q[2];
                                                                   #
    # Gate: CZ**-0.006328040119021747                              # // Gate: CZ**-0.006328040119021747
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*1.4961253835)          # u3(pi*0.5,0,pi*1.4961253835) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0, np.pi*1.9961253835)  # u3(pi*0.5,pi*1.0,pi*1.9961253835) q[0];
    circ.add_RX_gate(1, np.pi*0.5*-1)                              # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*0.4968359799*-1)                     # rx(pi*0.4968359799) q[1];
    circ.add_RY_gate(0, np.pi*0.5*-1)                              # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                              # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.5007105964, np.pi*1.0)  # u3(pi*0.5,pi*1.5007105964,pi*1.0) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0007105964, 0)          # u3(pi*0.5,pi*1.0007105964,0) q[0];
                                                                   #
    circ.add_SWAP_gate(1, 0)                                       # swap q[1],q[0];
                                                                   #
    # Gate: CZ**0.009295387491454189                               # // Gate: CZ**0.009295387491454189
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.0820521548)  # u3(pi*0.5,pi*1.0,pi*1.0820521548) q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.5820521548)  # u3(pi*0.5,pi*1.0,pi*1.5820521548) q[1];
    circ.add_RX_gate(2, np.pi*0.5*-1)                              # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*0.4953523063*-1)                     # rx(pi*0.4953523063) q[2];
    circ.add_RY_gate(1, np.pi*0.5*-1)                              # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.9225955389, 0)          # u3(pi*0.5,pi*1.9225955389,0) q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.4225955389, 0)          # u3(pi*0.5,pi*1.4225955389,0) q[1];
                                                                   #
    circ.add_SWAP_gate(2, 1)                                       # swap q[2],q[1];
                                                                   #
    # Gate: CZ**0.009295387491454189                               # // Gate: CZ**0.009295387491454189
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0, np.pi*1.0820521548)  # u3(pi*0.5,pi*1.0,pi*1.0820521548) q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.5820521548)  # u3(pi*0.5,pi*1.0,pi*1.5820521548) q[2];
    circ.add_RX_gate(3, np.pi*0.5*-1)                              # rx(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*0.4953523063*-1)                     # rx(pi*0.4953523063) q[3];
    circ.add_RY_gate(2, np.pi*0.5*-1)                              # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.9225955389, 0)          # u3(pi*0.5,pi*1.9225955389,0) q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.4225955389, 0)          # u3(pi*0.5,pi*1.4225955389,0) q[2];
                                                                   #
    # Gate: CZ**0.009295387491454189                               # // Gate: CZ**0.009295387491454189
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.0820521548)  # u3(pi*0.5,pi*1.0,pi*1.0820521548) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0, np.pi*1.5820521548)  # u3(pi*0.5,pi*1.0,pi*1.5820521548) q[0];
    circ.add_RX_gate(1, np.pi*0.5*-1)                              # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*0.4953523063*-1)                     # rx(pi*0.4953523063) q[1];
    circ.add_RY_gate(0, np.pi*0.5*-1)                              # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                              # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.9225955389, 0)          # u3(pi*0.5,pi*1.9225955389,0) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.4225955389, 0)          # u3(pi*0.5,pi*1.4225955389,0) q[0];
                                                                   #
    circ.add_SWAP_gate(3, 2)                                       # swap q[3],q[2];
    circ.add_SWAP_gate(1, 0)                                       # swap q[1],q[0];
                                                                   #
    # Gate: CZ**0.009295387491454189                               # // Gate: CZ**0.009295387491454189
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.0820521548)  # u3(pi*0.5,pi*1.0,pi*1.0820521548) q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.5820521548)  # u3(pi*0.5,pi*1.0,pi*1.5820521548) q[1];
    circ.add_RX_gate(2, np.pi*0.5*-1)                              # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*0.4953523063*-1)                     # rx(pi*0.4953523063) q[2];
    circ.add_RY_gate(1, np.pi*0.5*-1)                              # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.9225955389, 0)          # u3(pi*0.5,pi*1.9225955389,0) q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.4225955389, 0)          # u3(pi*0.5,pi*1.4225955389,0) q[1];
                                                                   #
    circ.add_RZ_gate(0, np.pi*-0.0068270924*-1)                    # rz(pi*-0.0068270924) q[0];
    circ.add_RZ_gate(3, np.pi*-0.0031640201*-1)                    # rz(pi*-0.0031640201) q[3];
    circ.add_SWAP_gate(2, 1)                                       # swap q[2],q[1];
    circ.add_Z_gate(0)                                             # z q[0];
    circ.add_Z_gate(3)                                             # z q[3];
    circ.add_RZ_gate(1, np.pi*-0.0068270924*-1)                    # rz(pi*-0.0068270924) q[1];
    circ.add_RZ_gate(2, np.pi*-0.0031640201*-1)                    # rz(pi*-0.0031640201) q[2];
                                                                   #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
                                                                   #
    # Gate: PhasedISWAP**-0.5017530508495694                       # // Gate: PhasedISWAP**-0.5017530508495694
    circ.add_RZ_gate(0, np.pi*0.25*-1)                             # rz(pi*0.25) q[0];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_H_gate(0)                                             # h q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(0, np.pi*-0.2508765254*-1)                    # rz(pi*-0.2508765254) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(0, np.pi*0.2508765254*-1)                     # rz(pi*0.2508765254) q[0];
    circ.add_H_gate(0)                                             # h q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(0, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[0];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
                                                                   #
    # Gate: PhasedISWAP**0.5017530508495694                        # // Gate: PhasedISWAP**0.5017530508495694
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(3, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(2, np.pi*0.2508765254*-1)                     # rz(pi*0.2508765254) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(2, np.pi*-0.2508765254*-1)                    # rz(pi*-0.2508765254) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(3, np.pi*0.25*-1)                             # rz(pi*0.25) q[3];
                                                                   #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
                                                                   #
    # Gate: CZ**-0.00046375097365492423                            # // Gate: CZ**-0.00046375097365492423
    circ.add_U3_gate(0, np.pi*0.5, 0, np.pi*1.5001274262)          # u3(pi*0.5,0,pi*1.5001274262) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.0001274262)  # u3(pi*0.5,pi*1.0,pi*1.0001274262) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                              # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.4997681245*-1)                     # rx(pi*0.4997681245) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                              # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.4996406983, np.pi*1.0)  # u3(pi*0.5,pi*1.4996406983,pi*1.0) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.9996406983, 0)          # u3(pi*0.5,pi*1.9996406983,0) q[1];
    #
    circ.add_SWAP_gate(0, 1)                                       # swap q[0],q[1];
    #
    # Gate: CZ**-0.0004129506013584246                             # // Gate: CZ**-0.0004129506013584246
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.4998373235)  # u3(pi*0.5,pi*1.0,pi*1.4998373235) q[2];
    circ.add_U3_gate(3, np.pi*0.5, 0, np.pi*1.9998373235)          # u3(pi*0.5,0,pi*1.9998373235) q[3];
    circ.add_RX_gate(2, np.pi*0.5*-1)                              # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*0.4997935247*-1)                     # rx(pi*0.4997935247) q[2];
    circ.add_RY_gate(3, np.pi*0.5*-1)                              # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                              # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.4999562012, 0)          # u3(pi*0.5,pi*1.4999562012,0) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.9999562012, np.pi*1.0)  # u3(pi*0.5,pi*0.9999562012,pi*1.0) q[3];
    #
    circ.add_SWAP_gate(2, 3)                                       # swap q[2],q[3];
    #
    # Gate: CZ**0.00043761426330885954                             # // Gate: CZ**0.00043761426330885954
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*1.9993457511)          # u3(pi*0.5,0,pi*1.9993457511) q[1];
    circ.add_U3_gate(2, np.pi*0.5, 0, np.pi*1.4993457511)          # u3(pi*0.5,0,pi*1.4993457511) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                              # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.4997811929*-1)                     # rx(pi*0.4997811929) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                              # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.0008730561,pi*1.0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.5008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.5008730561,pi*1.0) q[2];
    #
    circ.add_SWAP_gate(1, 2)                                       # swap q[1],q[2];
    #
    # Gate: CZ**0.00043761426330885954                             # // Gate: CZ**0.00043761426330885954
    circ.add_U3_gate(0, np.pi*0.5, 0, np.pi*1.9993457511)          # u3(pi*0.5,0,pi*1.9993457511) q[0];
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*1.4993457511)          # u3(pi*0.5,0,pi*1.4993457511) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                              # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.4997811929*-1)                     # rx(pi*0.4997811929) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                              # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.0008730561,pi*1.0) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.5008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.5008730561,pi*1.0) q[1];
    #
    # Gate: CZ**0.00043761426330885954                             # // Gate: CZ**0.00043761426330885954
    circ.add_U3_gate(2, np.pi*0.5, 0, np.pi*1.9993457511)          # u3(pi*0.5,0,pi*1.9993457511) q[2];
    circ.add_U3_gate(3, np.pi*0.5, 0, np.pi*1.4993457511)          # u3(pi*0.5,0,pi*1.4993457511) q[3];
    circ.add_RX_gate(2, np.pi*0.5*-1)                              # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*0.4997811929*-1)                     # rx(pi*0.4997811929) q[2];
    circ.add_RY_gate(3, np.pi*0.5*-1)                              # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                              # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.0008730561,pi*1.0) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.5008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.5008730561,pi*1.0) q[3];
    #
    circ.add_SWAP_gate(0, 1)                                       # swap q[0],q[1];
    circ.add_SWAP_gate(2, 3)                                       # swap q[2],q[3];
    #
    # Gate: CZ**0.00043761426330885954                             # // Gate: CZ**0.00043761426330885954
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*1.9993457511)          # u3(pi*0.5,0,pi*1.9993457511) q[1];
    circ.add_U3_gate(2, np.pi*0.5, 0, np.pi*1.4993457511)          # u3(pi*0.5,0,pi*1.4993457511) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                              # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.4997811929*-1)                     # rx(pi*0.4997811929) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                              # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.0008730561,pi*1.0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.5008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.5008730561,pi*1.0) q[2];
    #
    circ.add_RZ_gate(3, np.pi*-0.0002318755*-1)                    # rz(pi*-0.0002318755) q[3];
    circ.add_RZ_gate(0, np.pi*-0.0002064753*-1)                    # rz(pi*-0.0002064753) q[0];
    circ.add_SWAP_gate(1, 2)                                       # swap q[1],q[2];
    circ.add_Z_gate(3)                                             # z q[3];
    circ.add_Z_gate(0)                                             # z q[0];
    circ.add_RZ_gate(2, np.pi*-0.0002318755*-1)                    # rz(pi*-0.0002318755) q[2];
    circ.add_RZ_gate(1, np.pi*-0.0002064753*-1)                    # rz(pi*-0.0002064753) q[1];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    # Gate: PhasedISWAP**-0.4158482042253096                       # // Gate: PhasedISWAP**-0.4158482042253096
    circ.add_RZ_gate(3, np.pi*0.25*-1)                             # rz(pi*0.25) q[3];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_H_gate(3)                                             # h q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(3, np.pi*-0.2079241021*-1)                    # rz(pi*-0.2079241021) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(3, np.pi*0.2079241021*-1)                     # rz(pi*0.2079241021) q[3];
    circ.add_H_gate(3)                                             # h q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(3, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[3];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    # Gate: PhasedISWAP**0.4158482042253096                        # // Gate: PhasedISWAP**0.4158482042253096
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(0, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(1, np.pi*0.2079241021*-1)                     # rz(pi*0.2079241021) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(1, np.pi*-0.2079241021*-1)                    # rz(pi*-0.2079241021) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(0, np.pi*0.25*-1)                             # rz(pi*0.25) q[0];
    #
    circ.add_Z_gate(3)                                             # z q[3];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    circ.add_Z_gate(2)                                             # z q[2];
    circ.add_Z_gate(0)                                             # z q[0];
    circ.add_Z_gate(1)                                             # z q[1];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    # Gate: PhasedISWAP**0.08130614625631793                       # // Gate: PhasedISWAP**0.08130614625631793
    circ.add_RZ_gate(3, np.pi*0.25*-1)                             # rz(pi*0.25) q[3];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_H_gate(3)                                             # h q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(3, np.pi*0.0406530731*-1)                     # rz(pi*0.0406530731) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(3, np.pi*-0.0406530731*-1)                    # rz(pi*-0.0406530731) q[3];
    circ.add_H_gate(3)                                             # h q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(3, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[3];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    # Gate: PhasedISWAP**-0.08130614625631793                      # // Gate: PhasedISWAP**-0.08130614625631793
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(0, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(1, np.pi*-0.0406530731*-1)                    # rz(pi*-0.0406530731) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(1, np.pi*0.0406530731*-1)                     # rz(pi*0.0406530731) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(0, np.pi*0.25*-1)                             # rz(pi*0.25) q[0];
    #
    circ.add_RZ_gate(3, np.pi*0.1123177385*-1)                     # rz(pi*0.1123177385) q[3];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    circ.add_RZ_gate(2, np.pi*0.1123177385*-1)                     # rz(pi*0.1123177385) q[2];
    circ.add_RZ_gate(0, np.pi*0.0564909955*-1)                     # rz(pi*0.0564909955) q[0];
    circ.add_RZ_gate(1, np.pi*0.0564909955*-1)                     # rz(pi*0.0564909955) q[1];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    # Gate: PhasedISWAP**-0.05102950815299322                      # // Gate: PhasedISWAP**-0.05102950815299322
    circ.add_RZ_gate(3, np.pi*0.25*-1)                             # rz(pi*0.25) q[3];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_H_gate(3)                                             # h q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(3, np.pi*-0.0255147541*-1)                    # rz(pi*-0.0255147541) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(3, np.pi*0.0255147541*-1)                     # rz(pi*0.0255147541) q[3];
    circ.add_H_gate(3)                                             # h q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(3, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[3];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    # Gate: PhasedISWAP**0.05102950815299322                       # // Gate: PhasedISWAP**0.05102950815299322
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(0, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(1, np.pi*0.0255147541*-1)                     # rz(pi*0.0255147541) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(1, np.pi*-0.0255147541*-1)                    # rz(pi*-0.0255147541) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(0, np.pi*0.25*-1)                             # rz(pi*0.25) q[0];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    # Gate: CZ**-0.048279591094340914                              # // Gate: CZ**-0.048279591094340914
    circ.add_U3_gate(3, np.pi*0.5, 0, np.pi*0.5)                   # u3(pi*0.5,0,pi*0.5) q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.0)           # u3(pi*0.5,pi*1.0,pi*1.0) q[2];
    circ.add_RX_gate(3, np.pi*0.5*-1)                              # rx(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*0.4758602045*-1)                     # rx(pi*0.4758602045) q[3];
    circ.add_RY_gate(2, np.pi*0.5*-1)                              # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.4758602045, np.pi*1.0)  # u3(pi*0.5,pi*0.4758602045,pi*1.0) q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.9758602045, 0)          # u3(pi*0.5,pi*1.9758602045,0) q[2];
    #
    circ.add_SWAP_gate(3, 2)                                       # swap q[3],q[2];
    #
    # Gate: CZ**-0.022156912718971442                              # // Gate: CZ**-0.022156912718971442
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*1.75)                  # u3(pi*0.5,0,pi*1.75) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0, np.pi*1.25)          # u3(pi*0.5,pi*1.0,pi*1.25) q[0];
    circ.add_RX_gate(1, np.pi*0.5*-1)                              # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*0.4889215436*-1)                     # rx(pi*0.4889215436) q[1];
    circ.add_RY_gate(0, np.pi*0.5*-1)                              # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                              # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.2389215436, np.pi*1.0)  # u3(pi*0.5,pi*1.2389215436,pi*1.0) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.7389215436, 0)          # u3(pi*0.5,pi*1.7389215436,0) q[0];
    #
    circ.add_SWAP_gate(1, 0)                                       # swap q[1],q[0];
    #
    # Gate: CZ**-0.03270667647415345                               # // Gate: CZ**-0.03270667647415345
    circ.add_U3_gate(2, np.pi*0.5, 0, 0)                           # u3(pi*0.5,0,0) q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.5)           # u3(pi*0.5,pi*1.0,pi*1.5) q[1];
    circ.add_RX_gate(2, np.pi*0.5*-1)                              # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*0.4836466618*-1)                     # rx(pi*0.4836466618) q[2];
    circ.add_RY_gate(1, np.pi*0.5*-1)                              # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*0.9836466618, np.pi*1.0)  # u3(pi*0.5,pi*0.9836466618,pi*1.0) q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.4836466618, 0)          # u3(pi*0.5,pi*1.4836466618,0) q[1];
    #
    circ.add_SWAP_gate(2, 1)                                       # swap q[2],q[1];
    #
    # Gate: CZ**-0.03270667647415345                               # // Gate: CZ**-0.03270667647415345
    circ.add_U3_gate(3, np.pi*0.5, 0, 0)                           # u3(pi*0.5,0,0) q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.5)           # u3(pi*0.5,pi*1.0,pi*1.5) q[2];
    circ.add_RX_gate(3, np.pi*0.5*-1)                              # rx(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*0.4836466618*-1)                     # rx(pi*0.4836466618) q[3];
    circ.add_RY_gate(2, np.pi*0.5*-1)                              # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.9836466618, np.pi*1.0)  # u3(pi*0.5,pi*0.9836466618,pi*1.0) q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.4836466618, 0)          # u3(pi*0.5,pi*1.4836466618,0) q[2];
    #
    # Gate: CZ**-0.03270667647415345                               # // Gate: CZ**-0.03270667647415345
    circ.add_U3_gate(1, np.pi*0.5, 0, 0)                           # u3(pi*0.5,0,0) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0, np.pi*1.5)           # u3(pi*0.5,pi*1.0,pi*1.5) q[0];
    circ.add_RX_gate(1, np.pi*0.5*-1)                              # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*0.4836466618*-1)                     # rx(pi*0.4836466618) q[1];
    circ.add_RY_gate(0, np.pi*0.5*-1)                              # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                              # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.9836466618, np.pi*1.0)  # u3(pi*0.5,pi*0.9836466618,pi*1.0) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.4836466618, 0)          # u3(pi*0.5,pi*1.4836466618,0) q[0];
    #
    circ.add_SWAP_gate(3, 2)                                       # swap q[3],q[2];
    circ.add_SWAP_gate(1, 0)                                       # swap q[1],q[0];
    #
    # Gate: CZ**-0.03270667647415345                               # // Gate: CZ**-0.03270667647415345
    circ.add_U3_gate(2, np.pi*0.5, 0, 0)                           # u3(pi*0.5,0,0) q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.5)           # u3(pi*0.5,pi*1.0,pi*1.5) q[1];
    circ.add_RX_gate(2, np.pi*0.5*-1)                              # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*0.4836466618*-1)                     # rx(pi*0.4836466618) q[2];
    circ.add_RY_gate(1, np.pi*0.5*-1)                              # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*0.9836466618, np.pi*1.0)  # u3(pi*0.5,pi*0.9836466618,pi*1.0) q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.4836466618, 0)          # u3(pi*0.5,pi*1.4836466618,0) q[1];
    #
    circ.add_RZ_gate(0, np.pi*-0.0241397955*-1)                    # rz(pi*-0.0241397955) q[0];
    circ.add_RZ_gate(3, np.pi*-0.0110784564*-1)                    # rz(pi*-0.0110784564) q[3];
    circ.add_SWAP_gate(2, 1)                                       # swap q[2],q[1];
    circ.add_RZ_gate(1, np.pi*-0.0241397955*-1)                    # rz(pi*-0.0241397955) q[1];
    circ.add_RZ_gate(2, np.pi*-0.0110784564*-1)                    # rz(pi*-0.0110784564) q[2];
    circ.add_Z_gate(1)                                             # z q[1];
    circ.add_Z_gate(2)                                             # z q[2];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    # Gate: PhasedISWAP**-0.9500630905158097                       # // Gate: PhasedISWAP**-0.9500630905158097
    circ.add_RZ_gate(0, np.pi*0.25*-1)                             # rz(pi*0.25) q[0];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_H_gate(0)                                             # h q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(0, np.pi*-0.4750315453*-1)                    # rz(pi*-0.4750315453) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(0, np.pi*0.4750315453*-1)                     # rz(pi*0.4750315453) q[0];
    circ.add_H_gate(0)                                             # h q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(0, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[0];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    # Gate: PhasedISWAP**0.9500630905158097                        # // Gate: PhasedISWAP**0.9500630905158097
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(3, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(2, np.pi*0.4750315453*-1)                     # rz(pi*0.4750315453) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(2, np.pi*-0.4750315453*-1)                    # rz(pi*-0.4750315453) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(3, np.pi*0.25*-1)                             # rz(pi*0.25) q[3];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    # Gate: CZ**-0.013654184706660842                              # // Gate: CZ**-0.013654184706660842
    circ.add_U3_gate(0, np.pi*0.5, 0, np.pi*1.5)                   # u3(pi*0.5,0,pi*1.5) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.0)           # u3(pi*0.5,pi*1.0,pi*1.0) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                              # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.4931729076*-1)                     # rx(pi*0.4931729076) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                              # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.4931729076, np.pi*1.0)  # u3(pi*0.5,pi*1.4931729076,pi*1.0) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.9931729076, 0)          # u3(pi*0.5,pi*1.9931729076,0) q[1];
    #
    circ.add_SWAP_gate(0, 1)                                       # swap q[0],q[1];
    #
    # Gate: CZ**-0.006328040119021747                              # // Gate: CZ**-0.006328040119021747
    circ.add_U3_gate(2, np.pi*0.5, 0, np.pi*1.4961253835)          # u3(pi*0.5,0,pi*1.4961253835) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0, np.pi*1.9961253835)  # u3(pi*0.5,pi*1.0,pi*1.9961253835) q[3];
    circ.add_RX_gate(2, np.pi*0.5*-1)                              # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*0.4968359799*-1)                     # rx(pi*0.4968359799) q[2];
    circ.add_RY_gate(3, np.pi*0.5*-1)                              # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                              # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.5007105964, np.pi*1.0)  # u3(pi*0.5,pi*1.5007105964,pi*1.0) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0007105964, 0)          # u3(pi*0.5,pi*1.0007105964,0) q[3];
    #
    circ.add_SWAP_gate(2, 3)                                       # swap q[2],q[3];
    #
    # Gate: CZ**0.009295387491454189                               # // Gate: CZ**0.009295387491454189
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.0820521548)  # u3(pi*0.5,pi*1.0,pi*1.0820521548) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.5820521548)  # u3(pi*0.5,pi*1.0,pi*1.5820521548) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                              # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.4953523063*-1)                     # rx(pi*0.4953523063) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                              # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.9225955389, 0)          # u3(pi*0.5,pi*1.9225955389,0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.4225955389, 0)          # u3(pi*0.5,pi*1.4225955389,0) q[2];
    #
    circ.add_SWAP_gate(1, 2)                                       # swap q[1],q[2];
    #
    # Gate: CZ**0.009295387491454189                               # // Gate: CZ**0.009295387491454189
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0, np.pi*1.0820521548)  # u3(pi*0.5,pi*1.0,pi*1.0820521548) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.5820521548)  # u3(pi*0.5,pi*1.0,pi*1.5820521548) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                              # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.4953523063*-1)                     # rx(pi*0.4953523063) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                              # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.9225955389, 0)          # u3(pi*0.5,pi*1.9225955389,0) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.4225955389, 0)          # u3(pi*0.5,pi*1.4225955389,0) q[1];
    #
    # Gate: CZ**0.009295387491454189                               # // Gate: CZ**0.009295387491454189
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.0820521548)  # u3(pi*0.5,pi*1.0,pi*1.0820521548) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0, np.pi*1.5820521548)  # u3(pi*0.5,pi*1.0,pi*1.5820521548) q[3];
    circ.add_RX_gate(2, np.pi*0.5*-1)                              # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*0.4953523063*-1)                     # rx(pi*0.4953523063) q[2];
    circ.add_RY_gate(3, np.pi*0.5*-1)                              # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                              # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.9225955389, 0)          # u3(pi*0.5,pi*1.9225955389,0) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.4225955389, 0)          # u3(pi*0.5,pi*1.4225955389,0) q[3];
    #
    circ.add_SWAP_gate(0, 1)                                       # swap q[0],q[1];
    circ.add_SWAP_gate(2, 3)                                       # swap q[2],q[3];
    #
    # Gate: CZ**0.009295387491454189                               # // Gate: CZ**0.009295387491454189
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.0820521548)  # u3(pi*0.5,pi*1.0,pi*1.0820521548) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.5820521548)  # u3(pi*0.5,pi*1.0,pi*1.5820521548) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                              # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.4953523063*-1)                     # rx(pi*0.4953523063) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                              # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.9225955389, 0)          # u3(pi*0.5,pi*1.9225955389,0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.4225955389, 0)          # u3(pi*0.5,pi*1.4225955389,0) q[2];
    #
    circ.add_RZ_gate(3, np.pi*-0.0068270924*-1)                    # rz(pi*-0.0068270924) q[3];
    circ.add_RZ_gate(0, np.pi*-0.0031640201*-1)                    # rz(pi*-0.0031640201) q[0];
    circ.add_SWAP_gate(1, 2)                                       # swap q[1],q[2];
    circ.add_Z_gate(3)                                             # z q[3];
    circ.add_Z_gate(0)                                             # z q[0];
    circ.add_RZ_gate(2, np.pi*-0.0068270924*-1)                    # rz(pi*-0.0068270924) q[2];
    circ.add_RZ_gate(1, np.pi*-0.0031640201*-1)                    # rz(pi*-0.0031640201) q[1];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    # Gate: PhasedISWAP**-0.5017530508495694                       # // Gate: PhasedISWAP**-0.5017530508495694
    circ.add_RZ_gate(3, np.pi*0.25*-1)                             # rz(pi*0.25) q[3];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_H_gate(3)                                             # h q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(3, np.pi*-0.2508765254*-1)                    # rz(pi*-0.2508765254) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(3, np.pi*0.2508765254*-1)                     # rz(pi*0.2508765254) q[3];
    circ.add_H_gate(3)                                             # h q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(3, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[3];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    # Gate: PhasedISWAP**0.5017530508495694                        # // Gate: PhasedISWAP**0.5017530508495694
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(0, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(1, np.pi*0.2508765254*-1)                     # rz(pi*0.2508765254) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(1, np.pi*-0.2508765254*-1)                    # rz(pi*-0.2508765254) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(0, np.pi*0.25*-1)                             # rz(pi*0.25) q[0];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    # Gate: CZ**-0.00046375097365492423                            # // Gate: CZ**-0.00046375097365492423
    circ.add_U3_gate(3, np.pi*0.5, 0, np.pi*1.5001274262)          # u3(pi*0.5,0,pi*1.5001274262) q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.0001274262)  # u3(pi*0.5,pi*1.0,pi*1.0001274262) q[2];
    circ.add_RX_gate(3, np.pi*0.5*-1)                              # rx(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*0.4997681245*-1)                     # rx(pi*0.4997681245) q[3];
    circ.add_RY_gate(2, np.pi*0.5*-1)                              # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.4996406983, np.pi*1.0)  # u3(pi*0.5,pi*1.4996406983,pi*1.0) q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.9996406983, 0)          # u3(pi*0.5,pi*1.9996406983,0) q[2];
    #
    circ.add_SWAP_gate(3, 2)                                       # swap q[3],q[2];
    #
    # Gate: CZ**-0.0004129506013584246                             # // Gate: CZ**-0.0004129506013584246
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.4998373235)  # u3(pi*0.5,pi*1.0,pi*1.4998373235) q[1];
    circ.add_U3_gate(0, np.pi*0.5, 0, np.pi*1.9998373235)          # u3(pi*0.5,0,pi*1.9998373235) q[0];
    circ.add_RX_gate(1, np.pi*0.5*-1)                              # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*0.4997935247*-1)                     # rx(pi*0.4997935247) q[1];
    circ.add_RY_gate(0, np.pi*0.5*-1)                              # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                              # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.4999562012, 0)          # u3(pi*0.5,pi*1.4999562012,0) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.9999562012, np.pi*1.0)  # u3(pi*0.5,pi*0.9999562012,pi*1.0) q[0];
    #
    circ.add_SWAP_gate(1, 0)                                       # swap q[1],q[0];
    #
    # Gate: CZ**0.00043761426330885954                             # // Gate: CZ**0.00043761426330885954
    circ.add_U3_gate(2, np.pi*0.5, 0, np.pi*1.9993457511)          # u3(pi*0.5,0,pi*1.9993457511) q[2];
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*1.4993457511)          # u3(pi*0.5,0,pi*1.4993457511) q[1];
    circ.add_RX_gate(2, np.pi*0.5*-1)                              # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*0.4997811929*-1)                     # rx(pi*0.4997811929) q[2];
    circ.add_RY_gate(1, np.pi*0.5*-1)                              # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.0008730561,pi*1.0) q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.5008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.5008730561,pi*1.0) q[1];
    #
    circ.add_SWAP_gate(2, 1)                                       # swap q[2],q[1];
    #
    # Gate: CZ**0.00043761426330885954                             # // Gate: CZ**0.00043761426330885954
    circ.add_U3_gate(3, np.pi*0.5, 0, np.pi*1.9993457511)          # u3(pi*0.5,0,pi*1.9993457511) q[3];
    circ.add_U3_gate(2, np.pi*0.5, 0, np.pi*1.4993457511)          # u3(pi*0.5,0,pi*1.4993457511) q[2];
    circ.add_RX_gate(3, np.pi*0.5*-1)                              # rx(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*0.4997811929*-1)                     # rx(pi*0.4997811929) q[3];
    circ.add_RY_gate(2, np.pi*0.5*-1)                              # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.0008730561,pi*1.0) q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.5008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.5008730561,pi*1.0) q[2];
    #
    # Gate: CZ**0.00043761426330885954                             # // Gate: CZ**0.00043761426330885954
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*1.9993457511)          # u3(pi*0.5,0,pi*1.9993457511) q[1];
    circ.add_U3_gate(0, np.pi*0.5, 0, np.pi*1.4993457511)          # u3(pi*0.5,0,pi*1.4993457511) q[0];
    circ.add_RX_gate(1, np.pi*0.5*-1)                              # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*0.4997811929*-1)                     # rx(pi*0.4997811929) q[1];
    circ.add_RY_gate(0, np.pi*0.5*-1)                              # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                              # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.0008730561,pi*1.0) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.5008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.5008730561,pi*1.0) q[0];
    #
    circ.add_SWAP_gate(3, 2)                                       # swap q[3],q[2];
    circ.add_SWAP_gate(1, 0)                                       # swap q[1],q[0];
    #
    # Gate: CZ**0.00043761426330885954                             # // Gate: CZ**0.00043761426330885954
    circ.add_U3_gate(2, np.pi*0.5, 0, np.pi*1.9993457511)          # u3(pi*0.5,0,pi*1.9993457511) q[2];
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*1.4993457511)          # u3(pi*0.5,0,pi*1.4993457511) q[1];
    circ.add_RX_gate(2, np.pi*0.5*-1)                              # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*0.4997811929*-1)                     # rx(pi*0.4997811929) q[2];
    circ.add_RY_gate(1, np.pi*0.5*-1)                              # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.0008730561,pi*1.0) q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.5008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.5008730561,pi*1.0) q[1];
    #
    circ.add_RZ_gate(0, np.pi*-0.0002318755*-1)                    # rz(pi*-0.0002318755) q[0];
    circ.add_RZ_gate(3, np.pi*-0.0002064753*-1)                    # rz(pi*-0.0002064753) q[3];
    circ.add_SWAP_gate(2, 1)                                       # swap q[2],q[1];
    circ.add_Z_gate(0)                                             # z q[0];
    circ.add_Z_gate(3)                                             # z q[3];
    circ.add_RZ_gate(1, np.pi*-0.0002318755*-1)                    # rz(pi*-0.0002318755) q[1];
    circ.add_RZ_gate(2, np.pi*-0.0002064753*-1)                    # rz(pi*-0.0002064753) q[2];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    # Gate: PhasedISWAP**-0.4158482042253096                       # // Gate: PhasedISWAP**-0.4158482042253096
    circ.add_RZ_gate(0, np.pi*0.25*-1)                             # rz(pi*0.25) q[0];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_H_gate(0)                                             # h q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(0, np.pi*-0.2079241021*-1)                    # rz(pi*-0.2079241021) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(0, np.pi*0.2079241021*-1)                     # rz(pi*0.2079241021) q[0];
    circ.add_H_gate(0)                                             # h q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(0, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[0];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    # Gate: PhasedISWAP**0.4158482042253096                        # // Gate: PhasedISWAP**0.4158482042253096
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(3, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(2, np.pi*0.2079241021*-1)                     # rz(pi*0.2079241021) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(2, np.pi*-0.2079241021*-1)                    # rz(pi*-0.2079241021) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(3, np.pi*0.25*-1)                             # rz(pi*0.25) q[3];
    #
    circ.add_Z_gate(0)                                             # z q[0];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    circ.add_Z_gate(1)                                             # z q[1];
    circ.add_Z_gate(3)                                             # z q[3];
    circ.add_Z_gate(2)                                             # z q[2];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    # Gate: PhasedISWAP**0.08130614625631793                       # // Gate: PhasedISWAP**0.08130614625631793
    circ.add_RZ_gate(0, np.pi*0.25*-1)                             # rz(pi*0.25) q[0];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_H_gate(0)                                             # h q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(0, np.pi*0.0406530731*-1)                     # rz(pi*0.0406530731) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(0, np.pi*-0.0406530731*-1)                    # rz(pi*-0.0406530731) q[0];
    circ.add_H_gate(0)                                             # h q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(0, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[0];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    # Gate: PhasedISWAP**-0.08130614625631793                      # // Gate: PhasedISWAP**-0.08130614625631793
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(3, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(2, np.pi*-0.0406530731*-1)                    # rz(pi*-0.0406530731) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(2, np.pi*0.0406530731*-1)                     # rz(pi*0.0406530731) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(3, np.pi*0.25*-1)                             # rz(pi*0.25) q[3];
    #
    circ.add_RZ_gate(0, np.pi*0.1123177385*-1)                     # rz(pi*0.1123177385) q[0];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    circ.add_RZ_gate(1, np.pi*0.1123177385*-1)                     # rz(pi*0.1123177385) q[1];
    circ.add_RZ_gate(3, np.pi*0.0564909955*-1)                     # rz(pi*0.0564909955) q[3];
    circ.add_RZ_gate(2, np.pi*0.0564909955*-1)                     # rz(pi*0.0564909955) q[2];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    # Gate: PhasedISWAP**-0.05102950815299322                      # // Gate: PhasedISWAP**-0.05102950815299322
    circ.add_RZ_gate(0, np.pi*0.25*-1)                             # rz(pi*0.25) q[0];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_H_gate(0)                                             # h q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(0, np.pi*-0.0255147541*-1)                    # rz(pi*-0.0255147541) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(0, np.pi*0.0255147541*-1)                     # rz(pi*0.0255147541) q[0];
    circ.add_H_gate(0)                                             # h q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(0, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[0];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    # Gate: PhasedISWAP**0.05102950815299322                       # // Gate: PhasedISWAP**0.05102950815299322
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(3, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(2, np.pi*0.0255147541*-1)                     # rz(pi*0.0255147541) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(2, np.pi*-0.0255147541*-1)                    # rz(pi*-0.0255147541) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(3, np.pi*0.25*-1)                             # rz(pi*0.25) q[3];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    # Gate: CZ**-0.048279591094340914                              # // Gate: CZ**-0.048279591094340914
    circ.add_U3_gate(0, np.pi*0.5, 0, np.pi*0.5)                   # u3(pi*0.5,0,pi*0.5) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.0)           # u3(pi*0.5,pi*1.0,pi*1.0) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                              # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.4758602045*-1)                     # rx(pi*0.4758602045) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                              # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.4758602045, np.pi*1.0)  # u3(pi*0.5,pi*0.4758602045,pi*1.0) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.9758602045, 0)          # u3(pi*0.5,pi*1.9758602045,0) q[1];
    #
    circ.add_SWAP_gate(0, 1)                                       # swap q[0],q[1];
    #
    # Gate: CZ**-0.022156912718971442                              # // Gate: CZ**-0.022156912718971442
    circ.add_U3_gate(2, np.pi*0.5, 0, np.pi*1.75)                  # u3(pi*0.5,0,pi*1.75) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0, np.pi*1.25)          # u3(pi*0.5,pi*1.0,pi*1.25) q[3];
    circ.add_RX_gate(2, np.pi*0.5*-1)                              # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*0.4889215436*-1)                     # rx(pi*0.4889215436) q[2];
    circ.add_RY_gate(3, np.pi*0.5*-1)                              # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                              # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.2389215436, np.pi*1.0)  # u3(pi*0.5,pi*1.2389215436,pi*1.0) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.7389215436, 0)          # u3(pi*0.5,pi*1.7389215436,0) q[3];
    #
    circ.add_SWAP_gate(2, 3)                                       # swap q[2],q[3];
    #
    # Gate: CZ**-0.03270667647415345                               # // Gate: CZ**-0.03270667647415345
    circ.add_U3_gate(1, np.pi*0.5, 0, 0)                           # u3(pi*0.5,0,0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.5)           # u3(pi*0.5,pi*1.0,pi*1.5) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                              # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.4836466618*-1)                     # rx(pi*0.4836466618) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                              # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.9836466618, np.pi*1.0)  # u3(pi*0.5,pi*0.9836466618,pi*1.0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.4836466618, 0)          # u3(pi*0.5,pi*1.4836466618,0) q[2];
    #
    circ.add_SWAP_gate(1, 2)                                       # swap q[1],q[2];
    #
    # Gate: CZ**-0.03270667647415345                               # // Gate: CZ**-0.03270667647415345
    circ.add_U3_gate(0, np.pi*0.5, 0, 0)                           # u3(pi*0.5,0,0) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.5)           # u3(pi*0.5,pi*1.0,pi*1.5) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                              # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.4836466618*-1)                     # rx(pi*0.4836466618) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                              # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.9836466618, np.pi*1.0)  # u3(pi*0.5,pi*0.9836466618,pi*1.0) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.4836466618, 0)          # u3(pi*0.5,pi*1.4836466618,0) q[1];
    #
    # Gate: CZ**-0.03270667647415345                               # // Gate: CZ**-0.03270667647415345
    circ.add_U3_gate(2, np.pi*0.5, 0, 0)                           # u3(pi*0.5,0,0) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0, np.pi*1.5)           # u3(pi*0.5,pi*1.0,pi*1.5) q[3];
    circ.add_RX_gate(2, np.pi*0.5*-1)                              # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*0.4836466618*-1)                     # rx(pi*0.4836466618) q[2];
    circ.add_RY_gate(3, np.pi*0.5*-1)                              # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                              # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*0.9836466618, np.pi*1.0)  # u3(pi*0.5,pi*0.9836466618,pi*1.0) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.4836466618, 0)          # u3(pi*0.5,pi*1.4836466618,0) q[3];
    #
    circ.add_SWAP_gate(0, 1)                                       # swap q[0],q[1];
    circ.add_SWAP_gate(2, 3)                                       # swap q[2],q[3];
    #
    # Gate: CZ**-0.03270667647415345                               # // Gate: CZ**-0.03270667647415345
    circ.add_U3_gate(1, np.pi*0.5, 0, 0)                           # u3(pi*0.5,0,0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.5)           # u3(pi*0.5,pi*1.0,pi*1.5) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                              # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.4836466618*-1)                     # rx(pi*0.4836466618) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                              # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.9836466618, np.pi*1.0)  # u3(pi*0.5,pi*0.9836466618,pi*1.0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.4836466618, 0)          # u3(pi*0.5,pi*1.4836466618,0) q[2];
    #
    circ.add_RZ_gate(3, np.pi*-0.0241397955*-1)                    # rz(pi*-0.0241397955) q[3];
    circ.add_RZ_gate(0, np.pi*-0.0110784564*-1)                    # rz(pi*-0.0110784564) q[0];
    circ.add_SWAP_gate(1, 2)                                       # swap q[1],q[2];
    circ.add_RZ_gate(2, np.pi*-0.0241397955*-1)                    # rz(pi*-0.0241397955) q[2];
    circ.add_RZ_gate(1, np.pi*-0.0110784564*-1)                    # rz(pi*-0.0110784564) q[1];
    circ.add_Z_gate(2)                                             # z q[2];
    circ.add_Z_gate(1)                                             # z q[1];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    # Gate: PhasedISWAP**-0.9500630905158097                       # // Gate: PhasedISWAP**-0.9500630905158097
    circ.add_RZ_gate(3, np.pi*0.25*-1)                             # rz(pi*0.25) q[3];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_H_gate(3)                                             # h q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(3, np.pi*-0.4750315453*-1)                    # rz(pi*-0.4750315453) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(3, np.pi*0.4750315453*-1)                     # rz(pi*0.4750315453) q[3];
    circ.add_H_gate(3)                                             # h q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(3, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[3];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    # Gate: PhasedISWAP**0.9500630905158097                        # // Gate: PhasedISWAP**0.9500630905158097
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(0, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(1, np.pi*0.4750315453*-1)                     # rz(pi*0.4750315453) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(1, np.pi*-0.4750315453*-1)                    # rz(pi*-0.4750315453) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(0, np.pi*0.25*-1)                             # rz(pi*0.25) q[0];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    # Gate: CZ**-0.013654184706660842                              # // Gate: CZ**-0.013654184706660842
    circ.add_U3_gate(3, np.pi*0.5, 0, np.pi*1.5)                   # u3(pi*0.5,0,pi*1.5) q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.0)           # u3(pi*0.5,pi*1.0,pi*1.0) q[2];
    circ.add_RX_gate(3, np.pi*0.5*-1)                              # rx(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*0.4931729076*-1)                     # rx(pi*0.4931729076) q[3];
    circ.add_RY_gate(2, np.pi*0.5*-1)                              # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.4931729076, np.pi*1.0)  # u3(pi*0.5,pi*1.4931729076,pi*1.0) q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.9931729076, 0)          # u3(pi*0.5,pi*1.9931729076,0) q[2];
    #
    circ.add_SWAP_gate(3, 2)                                       # swap q[3],q[2];
    #
    # Gate: CZ**-0.006328040119021747                              # // Gate: CZ**-0.006328040119021747
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*1.4961253835)          # u3(pi*0.5,0,pi*1.4961253835) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0, np.pi*1.9961253835)  # u3(pi*0.5,pi*1.0,pi*1.9961253835) q[0];
    circ.add_RX_gate(1, np.pi*0.5*-1)                              # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*0.4968359799*-1)                     # rx(pi*0.4968359799) q[1];
    circ.add_RY_gate(0, np.pi*0.5*-1)                              # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                              # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.5007105964, np.pi*1.0)  # u3(pi*0.5,pi*1.5007105964,pi*1.0) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0007105964, 0)          # u3(pi*0.5,pi*1.0007105964,0) q[0];
    #
    circ.add_SWAP_gate(1, 0)                                       # swap q[1],q[0];
    #
    # Gate: CZ**0.009295387491454189                               # // Gate: CZ**0.009295387491454189
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.0820521548)  # u3(pi*0.5,pi*1.0,pi*1.0820521548) q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.5820521548)  # u3(pi*0.5,pi*1.0,pi*1.5820521548) q[1];
    circ.add_RX_gate(2, np.pi*0.5*-1)                              # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*0.4953523063*-1)                     # rx(pi*0.4953523063) q[2];
    circ.add_RY_gate(1, np.pi*0.5*-1)                              # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.9225955389, 0)          # u3(pi*0.5,pi*1.9225955389,0) q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.4225955389, 0)          # u3(pi*0.5,pi*1.4225955389,0) q[1];
    #
    circ.add_SWAP_gate(2, 1)                                       # swap q[2],q[1];
    #
    # Gate: CZ**0.009295387491454189                               # // Gate: CZ**0.009295387491454189
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.0, np.pi*1.0820521548)  # u3(pi*0.5,pi*1.0,pi*1.0820521548) q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.5820521548)  # u3(pi*0.5,pi*1.0,pi*1.5820521548) q[2];
    circ.add_RX_gate(3, np.pi*0.5*-1)                              # rx(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*0.4953523063*-1)                     # rx(pi*0.4953523063) q[3];
    circ.add_RY_gate(2, np.pi*0.5*-1)                              # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.9225955389, 0)          # u3(pi*0.5,pi*1.9225955389,0) q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.4225955389, 0)          # u3(pi*0.5,pi*1.4225955389,0) q[2];
    #
    # Gate: CZ**0.009295387491454189                               # // Gate: CZ**0.009295387491454189
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.0820521548)  # u3(pi*0.5,pi*1.0,pi*1.0820521548) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0, np.pi*1.5820521548)  # u3(pi*0.5,pi*1.0,pi*1.5820521548) q[0];
    circ.add_RX_gate(1, np.pi*0.5*-1)                              # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*0.4953523063*-1)                     # rx(pi*0.4953523063) q[1];
    circ.add_RY_gate(0, np.pi*0.5*-1)                              # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                              # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.9225955389, 0)          # u3(pi*0.5,pi*1.9225955389,0) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.4225955389, 0)          # u3(pi*0.5,pi*1.4225955389,0) q[0];
    #
    circ.add_SWAP_gate(3, 2)                                       # swap q[3],q[2];
    circ.add_SWAP_gate(1, 0)                                       # swap q[1],q[0];
    #
    # Gate: CZ**0.009295387491454189                               # // Gate: CZ**0.009295387491454189
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.0820521548)  # u3(pi*0.5,pi*1.0,pi*1.0820521548) q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.5820521548)  # u3(pi*0.5,pi*1.0,pi*1.5820521548) q[1];
    circ.add_RX_gate(2, np.pi*0.5*-1)                              # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*0.4953523063*-1)                     # rx(pi*0.4953523063) q[2];
    circ.add_RY_gate(1, np.pi*0.5*-1)                              # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.9225955389, 0)          # u3(pi*0.5,pi*1.9225955389,0) q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.4225955389, 0)          # u3(pi*0.5,pi*1.4225955389,0) q[1];
    #
    circ.add_RZ_gate(0, np.pi*-0.0068270924*-1)                    # rz(pi*-0.0068270924) q[0];
    circ.add_RZ_gate(3, np.pi*-0.0031640201*-1)                    # rz(pi*-0.0031640201) q[3];
    circ.add_SWAP_gate(2, 1)                                       # swap q[2],q[1];
    circ.add_Z_gate(0)                                             # z q[0];
    circ.add_Z_gate(3)                                             # z q[3];
    circ.add_RZ_gate(1, np.pi*-0.0068270924*-1)                    # rz(pi*-0.0068270924) q[1];
    circ.add_RZ_gate(2, np.pi*-0.0031640201*-1)                    # rz(pi*-0.0031640201) q[2];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    # Gate: PhasedISWAP**-0.5017530508495694                       # // Gate: PhasedISWAP**-0.5017530508495694
    circ.add_RZ_gate(0, np.pi*0.25*-1)                             # rz(pi*0.25) q[0];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_H_gate(0)                                             # h q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(0, np.pi*-0.2508765254*-1)                    # rz(pi*-0.2508765254) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(0, np.pi*0.2508765254*-1)                     # rz(pi*0.2508765254) q[0];
    circ.add_H_gate(0)                                             # h q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(0, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[0];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    # Gate: PhasedISWAP**0.5017530508495694                        # // Gate: PhasedISWAP**0.5017530508495694
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(3, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(2, np.pi*0.2508765254*-1)                     # rz(pi*0.2508765254) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(2, np.pi*-0.2508765254*-1)                    # rz(pi*-0.2508765254) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(3, np.pi*0.25*-1)                             # rz(pi*0.25) q[3];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    # Gate: CZ**-0.00046375097365492423                            # // Gate: CZ**-0.00046375097365492423
    circ.add_U3_gate(0, np.pi*0.5, 0, np.pi*1.5001274262)          # u3(pi*0.5,0,pi*1.5001274262) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0, np.pi*1.0001274262)  # u3(pi*0.5,pi*1.0,pi*1.0001274262) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                              # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.4997681245*-1)                     # rx(pi*0.4997681245) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                              # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.4996406983, np.pi*1.0)  # u3(pi*0.5,pi*1.4996406983,pi*1.0) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.9996406983, 0)          # u3(pi*0.5,pi*1.9996406983,0) q[1];
    #
    circ.add_SWAP_gate(0, 1)                                       # swap q[0],q[1];
    #
    # Gate: CZ**-0.0004129506013584246                             # // Gate: CZ**-0.0004129506013584246
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.4998373235)  # u3(pi*0.5,pi*1.0,pi*1.4998373235) q[2];
    circ.add_U3_gate(3, np.pi*0.5, 0, np.pi*1.9998373235)          # u3(pi*0.5,0,pi*1.9998373235) q[3];
    circ.add_RX_gate(2, np.pi*0.5*-1)                              # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*0.4997935247*-1)                     # rx(pi*0.4997935247) q[2];
    circ.add_RY_gate(3, np.pi*0.5*-1)                              # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                              # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.4999562012, 0)          # u3(pi*0.5,pi*1.4999562012,0) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.9999562012, np.pi*1.0)  # u3(pi*0.5,pi*0.9999562012,pi*1.0) q[3];
    #
    circ.add_SWAP_gate(2, 3)                                       # swap q[2],q[3];
    #
    # Gate: CZ**0.00043761426330885954                             # // Gate: CZ**0.00043761426330885954
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*1.9993457511)          # u3(pi*0.5,0,pi*1.9993457511) q[1];
    circ.add_U3_gate(2, np.pi*0.5, 0, np.pi*1.4993457511)          # u3(pi*0.5,0,pi*1.4993457511) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                              # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.4997811929*-1)                     # rx(pi*0.4997811929) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                              # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.0008730561,pi*1.0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.5008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.5008730561,pi*1.0) q[2];
    #
    circ.add_SWAP_gate(1, 2)                                       # swap q[1],q[2];
    #
    # Gate: CZ**0.00043761426330885954                             # // Gate: CZ**0.00043761426330885954
    circ.add_U3_gate(0, np.pi*0.5, 0, np.pi*1.9993457511)          # u3(pi*0.5,0,pi*1.9993457511) q[0];
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*1.4993457511)          # u3(pi*0.5,0,pi*1.4993457511) q[1];
    circ.add_RX_gate(0, np.pi*0.5*-1)                              # rx(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*0.4997811929*-1)                     # rx(pi*0.4997811929) q[0];
    circ.add_RY_gate(1, np.pi*0.5*-1)                              # ry(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[1];
    circ.add_RZ_gate(1, np.pi*0.5*-1)                              # rz(pi*0.5) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.0008730561,pi*1.0) q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.5008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.5008730561,pi*1.0) q[1];
    #
    # Gate: CZ**0.00043761426330885954                             # // Gate: CZ**0.00043761426330885954
    circ.add_U3_gate(2, np.pi*0.5, 0, np.pi*1.9993457511)          # u3(pi*0.5,0,pi*1.9993457511) q[2];
    circ.add_U3_gate(3, np.pi*0.5, 0, np.pi*1.4993457511)          # u3(pi*0.5,0,pi*1.4993457511) q[3];
    circ.add_RX_gate(2, np.pi*0.5*-1)                              # rx(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*0.4997811929*-1)                     # rx(pi*0.4997811929) q[2];
    circ.add_RY_gate(3, np.pi*0.5*-1)                              # ry(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[3];
    circ.add_RZ_gate(3, np.pi*0.5*-1)                              # rz(pi*0.5) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.0008730561,pi*1.0) q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*1.5008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.5008730561,pi*1.0) q[3];
    #
    circ.add_SWAP_gate(0, 1)                                       # swap q[0],q[1];
    circ.add_SWAP_gate(2, 3)                                       # swap q[2],q[3];
    #
    # Gate: CZ**0.00043761426330885954                             # // Gate: CZ**0.00043761426330885954
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*1.9993457511)          # u3(pi*0.5,0,pi*1.9993457511) q[1];
    circ.add_U3_gate(2, np.pi*0.5, 0, np.pi*1.4993457511)          # u3(pi*0.5,0,pi*1.4993457511) q[2];
    circ.add_RX_gate(1, np.pi*0.5*-1)                              # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RX_gate(1, np.pi*0.4997811929*-1)                     # rx(pi*0.4997811929) q[1];
    circ.add_RY_gate(2, np.pi*0.5*-1)                              # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                             # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.0008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.0008730561,pi*1.0) q[1];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.5008730561, np.pi*1.0)  # u3(pi*0.5,pi*1.5008730561,pi*1.0) q[2];
    #
    circ.add_RZ_gate(3, np.pi*-0.0002318755*-1)                    # rz(pi*-0.0002318755) q[3];
    circ.add_RZ_gate(0, np.pi*-0.0002064753*-1)                    # rz(pi*-0.0002064753) q[0];
    circ.add_SWAP_gate(1, 2)                                       # swap q[1],q[2];
    circ.add_Z_gate(3)                                             # z q[3];
    circ.add_Z_gate(0)                                             # z q[0];
    circ.add_RZ_gate(2, np.pi*-0.0002318755*-1)                    # rz(pi*-0.0002318755) q[2];
    circ.add_RZ_gate(1, np.pi*-0.0002064753*-1)                    # rz(pi*-0.0002064753) q[1];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    # Gate: PhasedISWAP**-0.4158482042253096                       # // Gate: PhasedISWAP**-0.4158482042253096
    circ.add_RZ_gate(3, np.pi*0.25*-1)                             # rz(pi*0.25) q[3];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_H_gate(3)                                             # h q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(3, np.pi*-0.2079241021*-1)                    # rz(pi*-0.2079241021) q[3];
    circ.add_CNOT_gate(2, 3)                                       # cx q[2],q[3];
    circ.add_RZ_gate(3, np.pi*0.2079241021*-1)                     # rz(pi*0.2079241021) q[3];
    circ.add_H_gate(3)                                             # h q[3];
    circ.add_CNOT_gate(3, 2)                                       # cx q[3],q[2];
    circ.add_RZ_gate(3, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[3];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    #
    # Gate: PhasedISWAP**0.4158482042253096                        # // Gate: PhasedISWAP**0.4158482042253096
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    circ.add_RZ_gate(0, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[0];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(1, np.pi*0.2079241021*-1)                     # rz(pi*0.2079241021) q[1];
    circ.add_CNOT_gate(0, 1)                                       # cx q[0],q[1];
    circ.add_RZ_gate(1, np.pi*-0.2079241021*-1)                    # rz(pi*-0.2079241021) q[1];
    circ.add_H_gate(1)                                             # h q[1];
    circ.add_CNOT_gate(1, 0)                                       # cx q[1],q[0];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(0, np.pi*0.25*-1)                             # rz(pi*0.25) q[0];
    #
    # Gate: PhasedISWAP**-1.0                                      # // Gate: PhasedISWAP**-1.0
    circ.add_RZ_gate(2, np.pi*0.25*-1)                             # rz(pi*0.25) q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*-0.5*-1)                             # rz(pi*-0.5) q[2];
    circ.add_CNOT_gate(1, 2)                                       # cx q[1],q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                              # rz(pi*0.5) q[2];
    circ.add_H_gate(2)                                             # h q[2];
    circ.add_CNOT_gate(2, 1)                                       # cx q[2],q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                            # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                             # rz(pi*0.25) q[1];
    #
    circ.add_SWAP_gate(3, 2)                                       # swap q[3],q[2];
    circ.add_SWAP_gate(1, 0)                                       # swap q[1],q[0];
    circ.add_SWAP_gate(2, 1)                                       # swap q[2],q[1];
    circ.add_SWAP_gate(3, 2)                                       # swap q[3],q[2];
    circ.add_SWAP_gate(1, 0)                                       # swap q[1],q[0];
    circ.add_SWAP_gate(2, 1)                                       # swap q[2],q[1];


    circ.update_quantum_state(st)
    #with open('Qulacs.log', 'w') as f:
    #    print(np.around(st.get_vector(),14)+0, file=f)
