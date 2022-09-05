# Error correction: distance-three 5-qubit code,  from the paper "Benchmarking gate-based quantum computers" by K. Michielsen et al.

import numpy as np
from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import Identity

def func(st, circ):
#    nqubits=5
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)                         # qreg q[5];
#    circ = QuantumCircuit(nqubits)                     # creg c[5];
                                                       #
    circ.add_H_gate(0)                                 # h q[0];
    circ.add_H_gate(1)                                 # h q[1];
    circ.add_gate(Identity(2))                         # id q[2];
    circ.add_H_gate(3)                                 # h q[3];
    circ.add_H_gate(4)                                 # h q[4];
    circ.add_CNOT_gate(1, 2)                           # cx q[1],q[2];
    circ.add_H_gate(1)                                 # h q[1];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_CNOT_gate(1, 2)                           # cx q[1],q[2];
    circ.add_H_gate(1)                                 # h q[1];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_CNOT_gate(1, 2)                           # cx q[1],q[2];
    circ.add_CNOT_gate(4, 2)                           # cx q[4],q[2];
    circ.add_CNOT_gate(1, 2)                           # cx q[1],q[2];
    circ.add_H_gate(1)                                 # h q[1];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_CNOT_gate(1, 2)                           # cx q[1],q[2];
    circ.add_H_gate(1)                                 # h q[1];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_CNOT_gate(1, 2)                           # cx q[1],q[2];
    circ.add_Sdag_gate(4)                              # sdg q[4];
    circ.add_CNOT_gate(4, 2)                           # cx q[4],q[2];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_CNOT_gate(4, 2)                           # cx q[4],q[2];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_CNOT_gate(0, 2)                           # cx q[0],q[2];
    circ.add_H_gate(0)                                 # h q[0];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_CNOT_gate(0, 2)                           # cx q[0],q[2];
    circ.add_H_gate(0)                                 # h q[0];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_CNOT_gate(0, 2)                           # cx q[0],q[2];
    circ.add_CNOT_gate(3, 2)                           # cx q[3],q[2];
    circ.add_CNOT_gate(0, 2)                           # cx q[0],q[2];
    circ.add_H_gate(0)                                 # h q[0];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_CNOT_gate(0, 2)                           # cx q[0],q[2];
    circ.add_H_gate(0)                                 # h q[0];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_CNOT_gate(0, 2)                           # cx q[0],q[2];
    circ.add_CNOT_gate(1, 2)                           # cx q[1],q[2];
    circ.add_H_gate(1)                                 # h q[1];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_CNOT_gate(1, 2)                           # cx q[1],q[2];
    circ.add_H_gate(1)                                 # h q[1];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_CNOT_gate(1, 2)                           # cx q[1],q[2];
    circ.add_CNOT_gate(3, 2)                           # cx q[3],q[2];
    circ.add_CNOT_gate(1, 2)                           # cx q[1],q[2];
    circ.add_H_gate(1)                                 # h q[1];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_CNOT_gate(1, 2)                           # cx q[1],q[2];
    circ.add_H_gate(1)                                 # h q[1];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_CNOT_gate(1, 2)                           # cx q[1],q[2];
    circ.add_CNOT_gate(3, 2)                           # cx q[3],q[2];
    circ.add_CNOT_gate(0, 2)                           # cx q[0],q[2];
    circ.add_H_gate(3)                                 # h q[3];
    circ.add_H_gate(4)                                 # h q[4];
    circ.add_CNOT_gate(3, 2)                           # cx q[3],q[2];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_H_gate(3)                                 # h q[3];
    circ.add_CNOT_gate(3, 2)                           # cx q[3],q[2];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_H_gate(3)                                 # h q[3];
    circ.add_CNOT_gate(3, 2)                           # cx q[3],q[2];
    circ.add_CNOT_gate(0, 2)                           # cx q[0],q[2];
    circ.add_CNOT_gate(3, 2)                           # cx q[3],q[2];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_H_gate(3)                                 # h q[3];
    circ.add_CNOT_gate(3, 2)                           # cx q[3],q[2];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_H_gate(3)                                 # h q[3];
    circ.add_CNOT_gate(3, 2)                           # cx q[3],q[2];
    circ.add_CNOT_gate(4, 2)                           # cx q[4],q[2];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_H_gate(4)                                 # h q[4];
    circ.add_CNOT_gate(4, 2)                           # cx q[4],q[2];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_H_gate(4)                                 # h q[4];
    circ.add_CNOT_gate(4, 2)                           # cx q[4],q[2];
    circ.add_CNOT_gate(0, 2)                           # cx q[0],q[2];
    circ.add_CNOT_gate(4, 2)                           # cx q[4],q[2];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_H_gate(4)                                 # h q[4];
    circ.add_CNOT_gate(4, 2)                           # cx q[4],q[2];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_H_gate(4)                                 # h q[4];
    circ.add_CNOT_gate(4, 2)                           # cx q[4],q[2];
    circ.add_CNOT_gate(1, 2 )                          # cx q[1],q[2] ;
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_CNOT_gate(1, 2)                           # cx q[1],q[2];
    circ.add_H_gate(1)                                 # h q[1];
    circ.add_CNOT_gate(4, 2)                           # cx q[4],q[2];
    circ.add_CNOT_gate(0, 1)                           # cx q[0],q[1];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_H_gate(4)                                 # h q[4];
    circ.add_H_gate(1)                                 # h q[1];
    circ.add_CNOT_gate(4, 2)                           # cx q[4],q[2];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_H_gate(4)                                 # h q[4];
    circ.add_CNOT_gate(4, 2)                           # cx q[4],q[2];
    circ.add_CNOT_gate(1, 2)                           # cx q[1],q[2];
    circ.add_CNOT_gate(4, 2)                           # cx q[4],q[2];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_H_gate(4)                                 # h q[4];
    circ.add_CNOT_gate(4, 2)                           # cx q[4],q[2];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_H_gate(4)                                 # h q[4];
    circ.add_CNOT_gate(4, 2)                           # cx q[4],q[2];
    circ.add_Sdag_gate(1)                              # sdg q[1];
    circ.add_H_gate(2)                                 # h q[2];
    circ.add_H_gate(3)                                 # h q[3];
    circ.add_H_gate(4)                                 # h q[4];

    circ.update_quantum_state(st)
#    with open('Qulacs.log', 'w') as f:
#        print(np.around(st.get_vector(),14)+0, file=f)

