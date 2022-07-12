import numpy as np
from qulacs import QuantumCircuit, QuantumState

def func(st, circ):
#    nqubits=4
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)                         # qreg reg[4];
#    circ = QuantumCircuit(nqubits)                     #
    circ.add_H_gate(3)                                 # h reg[3];
    circ.add_H_gate(2)                                 # h reg[2];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_RZ_gate(0, 2.151746e+00*-1)               # rz(2.151746e+00) reg[0];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_H_gate(3)                                 # h reg[3];
    circ.add_H_gate(2)                                 # h reg[2];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_Y_gate(3)                                 # y reg[3];
    circ.add_Y_gate(2)                                 # y reg[2];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_RZ_gate(0, 2.151746e+00*-1)               # rz(2.151746e+00) reg[0];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_Y_gate(3)                                 # y reg[3];
    circ.add_Y_gate(2)                                 # y reg[2];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_H_gate(3)                                 # h reg[3];
    circ.add_Y_gate(2)                                 # y reg[2];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_RZ_gate(0, 2.151746e+00*-1)               # rz(2.151746e+00) reg[0];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_H_gate(3)                                 # h reg[3];
    circ.add_Y_gate(2)                                 # y reg[2];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_Y_gate(3)                                 # y reg[3];
    circ.add_H_gate(2)                                 # h reg[2];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_RZ_gate(0, 2.151746e+00*-1)               # rz(2.151746e+00) reg[0];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_Y_gate(3)                                 # y reg[3];
    circ.add_H_gate(2)                                 # h reg[2];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_Y_gate(3)                                 # y reg[3];
    circ.add_Y_gate(2)                                 # y reg[2];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_RZ_gate(0, 2.151746e+00*-1)               # rz(2.151746e+00) reg[0];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_Y_gate(3)                                 # y reg[3];
    circ.add_Y_gate(2)                                 # y reg[2];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_H_gate(3)                                 # h reg[3];
    circ.add_H_gate(2)                                 # h reg[2];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_RZ_gate(0, 2.151746e+00*-1)               # rz(2.151746e+00) reg[0];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_H_gate(3)                                 # h reg[3];
    circ.add_H_gate(2)                                 # h reg[2];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_Y_gate(3)                                 # y reg[3];
    circ.add_H_gate(2)                                 # h reg[2];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_RZ_gate(0, 2.151746e+00*-1)               # rz(2.151746e+00) reg[0];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_Y_gate(3)                                 # y reg[3];
    circ.add_H_gate(2)                                 # h reg[2];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_H_gate(3)                                 # h reg[3];
    circ.add_Y_gate(2)                                 # y reg[2];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_RZ_gate(0, 2.151746e+00*-1)               # rz(2.151746e+00) reg[0];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_H_gate(3)                                 # h reg[3];
    circ.add_Y_gate(2)                                 # y reg[2];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_H_gate(1)                                 # h reg[1];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_RZ_gate(0, 1.995482e+00*-1)               # rz(1.995482e+00) reg[0];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_H_gate(1)                                 # h reg[1];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_Y_gate(1)                                 # y reg[1];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_RZ_gate(0, 1.995482e+00*-1)               # rz(1.995482e+00) reg[0];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_Y_gate(1)                                 # y reg[1];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_H_gate(2)                                 # h reg[2];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_RZ_gate(0, 4.332582e+00*-1)               # rz(4.332582e+00) reg[0];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_H_gate(2)                                 # h reg[2];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_Y_gate(2)                                 # y reg[2];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_RZ_gate(0, 4.332582e+00*-1)               # rz(4.332582e+00) reg[0];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_Y_gate(2)                                 # y reg[2];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_H_gate(2)                                 # h reg[2];
    circ.add_H_gate(1)                                 # h reg[1];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_RZ_gate(1, 4.619220e-01*-1)               # rz(4.619220e-01) reg[1];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_H_gate(2)                                 # h reg[2];
    circ.add_H_gate(1)                                 # h reg[1];
    circ.add_Y_gate(2)                                 # y reg[2];
    circ.add_Y_gate(1)                                 # y reg[1];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_RZ_gate(1, 4.619220e-01*-1)               # rz(4.619220e-01) reg[1];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_Y_gate(2)                                 # y reg[2];
    circ.add_Y_gate(1)                                 # y reg[1];
    circ.add_H_gate(3)                                 # h reg[3];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_RZ_gate(0, 1.086976e+00*-1)               # rz(1.086976e+00) reg[0];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_H_gate(3)                                 # h reg[3];
    circ.add_H_gate(0)                                 # h reg[0];
    circ.add_Y_gate(3)                                 # y reg[3];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_RZ_gate(0, 1.086976e+00*-1)               # rz(1.086976e+00) reg[0];
    circ.add_CNOT_gate(1, 0)                           # cx reg[1],reg[0];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_Y_gate(3)                                 # y reg[3];
    circ.add_Y_gate(0)                                 # y reg[0];
    circ.add_H_gate(3)                                 # h reg[3];
    circ.add_H_gate(1)                                 # h reg[1];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_RZ_gate(1, 2.258394e+00*-1)               # rz(2.258394e+00) reg[1];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_H_gate(3)                                 # h reg[3];
    circ.add_H_gate(1)                                 # h reg[1];
    circ.add_Y_gate(3)                                 # y reg[3];
    circ.add_Y_gate(1)                                 # y reg[1];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_RZ_gate(1, 2.258394e+00*-1)               # rz(2.258394e+00) reg[1];
    circ.add_CNOT_gate(2, 1)                           # cx reg[2],reg[1];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_Y_gate(3)                                 # y reg[3];
    circ.add_Y_gate(1)                                 # y reg[1];
    circ.add_H_gate(3)                                 # h reg[3];
    circ.add_H_gate(2)                                 # h reg[2];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_RZ_gate(2, 1.228531e+00*-1)               # rz(1.228531e+00) reg[2];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_H_gate(3)                                 # h reg[3];
    circ.add_H_gate(2)                                 # h reg[2];
    circ.add_Y_gate(3)                                 # y reg[3];
    circ.add_Y_gate(2)                                 # y reg[2];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_RZ_gate(2, 1.228531e+00*-1)               # rz(1.228531e+00) reg[2];
    circ.add_CNOT_gate(3, 2)                           # cx reg[3],reg[2];
    circ.add_Y_gate(3)                                 # y reg[3];
    circ.add_Y_gate(2)                                 # y reg[2];

    circ.update_quantum_state(st)
#    with open('Qulacs.log', 'w') as f:
#        print(np.around(st.get_vector(),14)+0, file=f)
