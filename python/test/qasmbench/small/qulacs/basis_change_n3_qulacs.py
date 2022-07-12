# Generated from Cirq v0.8.0

import numpy as np
from qulacs import QuantumCircuit, QuantumState

def func(st, circ):
    # Qubits: [0, 1, 2]
    #nqubits=3
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
    #st = QuantumState(nqubits)     # qreg q[3];
    #circ = QuantumCircuit(nqubits)

    circ.add_U3_gate(2, np.pi*0.5, 0, np.pi*0.0564006755) # u3(pi*0.5,0,pi*0.0564006755) q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.5, np.pi*0.2945501109) # u3(pi*0.5,pi*1.5,pi*0.2945501109) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.5, np.pi*1.5) # u3(pi*0.5,pi*1.5,pi*1.5) q[0];
    circ.add_CZ_gate(1, 2) # cz q[1],q[2];
    circ.add_U3_gate(2, np.pi*0.1242949803, 0, 0) #u3(pi*0.1242949803,0,0) q[2];
    circ.add_U3_gate(1, np.pi*0.1242949803, np.pi*0.5, np.pi*1.5) #u3(pi*0.1242949803,pi*0.5,pi*1.5) q[1];
    circ.add_CZ_gate(1, 2) #cz q[1],q[2];
    circ.add_U3_gate(2, np.pi*0.0298311566, np.pi*1.5, np.pi*0.5) #u3(pi*0.0298311566,pi*1.5,pi*0.5) q[2];
    circ.add_U3_gate(1, np.pi*0.7273849664, np.pi*1.5, np.pi*1.0) #u3(pi*0.7273849664,pi*1.5,pi*1.0) q[1];
    circ.add_CZ_gate(0, 1) #cz q[0],q[1];
    circ.add_U3_gate(1, np.pi*0.328242091, 0, 0) #u3(pi*0.328242091,0,0) q[1];
    circ.add_U3_gate(0, np.pi*0.328242091, np.pi*0.5, np.pi*1.5) #u3(pi*0.328242091,pi*0.5,pi*1.5) q[0];
    circ.add_CZ_gate(0, 1) #cz q[0],q[1];
    circ.add_U3_gate(1, np.pi*0.1374475291, np.pi*2.0, np.pi*1.5) #u3(pi*0.1374475291,pi*2.0,pi*1.5) q[1];
    circ.add_U3_gate(0, np.pi*0.9766098537, 0, 0) #u3(pi*0.9766098537,0,0) q[0];
    circ.add_CZ_gate(1, 2) #cz q[1],q[2];
    circ.add_U3_gate(2, np.pi*0.2326621647, 0, 0) #u3(pi*0.2326621647,0,0) q[2];
    circ.add_U3_gate(1, np.pi*0.2326621647, np.pi*0.5, np.pi*1.5) #u3(pi*0.2326621647,pi*0.5,pi*1.5) q[1];
    circ.add_CZ_gate(1, 2) #cz q[1],q[2];
    circ.add_U3_gate(2, np.pi*0.5780153762, np.pi*0.5, np.pi*0.5) #u3(pi*0.5780153762,pi*0.5,pi*0.5) q[2];
    circ.add_U3_gate(1, np.pi*0.6257049652, np.pi*0.5, 0) #u3(pi*0.6257049652,pi*0.5,0) q[1];
    circ.add_CZ_gate(0, 1) #cz q[0],q[1];
    circ.add_U3_gate(1, np.pi*0.328242091, 0, 0) #u3(pi*0.328242091,0,0) q[1];
    circ.add_U3_gate(0, np.pi*0.328242091, np.pi*0.5, np.pi*1.5) #u3(pi*0.328242091,pi*0.5,pi*1.5) q[0];
    circ.add_CZ_gate(0, 1) #cz q[0],q[1];
    circ.add_U3_gate(1, np.pi*0.6817377913, 0, np.pi*0.5) #u3(pi*0.6817377913,0,pi*0.5) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*0.3593182384, np.pi*1.5) #u3(pi*0.5,pi*0.3593182384,pi*1.5) q[0];
    circ.add_CZ_gate(1, 2) #cz q[1],q[2];
    circ.add_U3_gate(2, np.pi*0.1242949803, 0, 0) #u3(pi*0.1242949803,0,0) q[2];
    circ.add_U3_gate(1, np.pi*0.1242949803, np.pi*0.5, np.pi*1.5) #u3(pi*0.1242949803,pi*0.5,pi*1.5) q[1];
    circ.add_CZ_gate(1, 2) #cz q[1],q[2];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.3937948052, 0) #u3(pi*0.5,pi*1.3937948052,0) q[2];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*1.1556453697, np.pi*0.5) #u3(pi*0.5,pi*1.1556453697,pi*0.5) q[1];
    circ.update_quantum_state(st)
    #with open('Qulacs.log', 'w') as f:
    #    print(np.around(st.get_vector(),14)+0, file=f)
