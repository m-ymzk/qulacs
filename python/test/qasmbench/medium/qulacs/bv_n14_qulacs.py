#@author Raymond Harry Rudy rudyhar@jp.ibm.com
#Bernstein-Vazirani with 14 qubits.
#Hidden string is 1111111111111

import numpy as np
from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import Identity, U1

def func(st, circ):
#    nqubits=14
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)                         # qreg qr[14];
#    circ = QuantumCircuit(nqubits)                     #
                                                       #
    circ.add_H_gate(0)                                 # h qr[0];
    circ.add_H_gate(1)                                 # h qr[1];
    circ.add_H_gate(2)                                 # h qr[2];
    circ.add_H_gate(3)                                 # h qr[3];
    circ.add_H_gate(4)                                 # h qr[4];
    circ.add_H_gate(5)                                 # h qr[5];
    circ.add_H_gate(6)                                 # h qr[6];
    circ.add_H_gate(7)                                 # h qr[7];
    circ.add_H_gate(8)                                 # h qr[8];
    circ.add_H_gate(9)                                 # h qr[9];
    circ.add_H_gate(10)                                # h qr[10];
    circ.add_H_gate(11)                                # h qr[11];
    circ.add_H_gate(12)                                # h qr[12];
    circ.add_X_gate(13)                                # x qr[13];
    circ.add_H_gate(13)                                # h qr[13];
                                                       # barrier qr[0],qr[1],qr[2],qr[3],qr[4],qr[5],qr[6],qr[7],qr[8],qr[9],qr[10],qr[11],qr[12],qr[13];
    circ.add_CNOT_gate(0, 13)                          # cx qr[0],qr[13];
    circ.add_CNOT_gate(1, 13)                          # cx qr[1],qr[13];
    circ.add_CNOT_gate(2, 13)                          # cx qr[2],qr[13];
    circ.add_CNOT_gate(3, 13)                          # cx qr[3],qr[13];
    circ.add_CNOT_gate(4, 13)                          # cx qr[4],qr[13];
    circ.add_CNOT_gate(5, 13)                          # cx qr[5],qr[13];
    circ.add_CNOT_gate(6, 13)                          # cx qr[6],qr[13];
    circ.add_CNOT_gate(7, 13)                          # cx qr[7],qr[13];
    circ.add_CNOT_gate(8, 13)                          # cx qr[8],qr[13];
    circ.add_CNOT_gate(9, 13)                          # cx qr[9],qr[13];
    circ.add_CNOT_gate(10, 13)                         # cx qr[10],qr[13];
    circ.add_CNOT_gate(11, 13)                         # cx qr[11],qr[13];
    circ.add_CNOT_gate(12, 13)                         # cx qr[12],qr[13];
                                                       # barrier qr[0],qr[1],qr[2],qr[3],qr[4],qr[5],qr[6],qr[7],qr[8],qr[9],qr[10],qr[11],qr[12],qr[13];
    circ.add_H_gate(0)                                 # h qr[0];
    circ.add_H_gate(1)                                 # h qr[1];
    circ.add_H_gate(2)                                 # h qr[2];
    circ.add_H_gate(3)                                 # h qr[3];
    circ.add_H_gate(4)                                 # h qr[4];
    circ.add_H_gate(5)                                 # h qr[5];
    circ.add_H_gate(6)                                 # h qr[6];
    circ.add_H_gate(7)                                 # h qr[7];
    circ.add_H_gate(8)                                 # h qr[8];
    circ.add_H_gate(9)                                 # h qr[9];
    circ.add_H_gate(10)                                # h qr[10];
    circ.add_H_gate(11)                                # h qr[11];
    circ.add_H_gate(12)                                # h qr[12];
    circ.update_quantum_state(st)
#    with open('Qulacs.log', 'w') as f:
#        print(np.around(st.get_vector(),14)+0, file=f)

