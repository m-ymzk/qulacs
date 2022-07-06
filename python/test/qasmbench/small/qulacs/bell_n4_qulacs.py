# Generated from Cirq v0.8.0

import numpy as np
from qulacs import QuantumCircuit, QuantumState

def func(st, circ):
    # Qubits: [(0,  0),  (0,  1),  (1,  0),  (1,  1)
#    nqubits=4
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)                           # qreg q[4];
#    circ = QuantumCircuit(nqubits)                       #
                                                         #
    circ.add_H_gate(0)                                   # h q[0];
    circ.add_H_gate(1)                                   # h q[1];
    circ.add_H_gate(3)                                   # h q[3];
    circ.add_CNOT_gate(0, 2)                             # cx q[0],q[2];
    circ.add_RX_gate(0, np.pi*-0.25*-1)                  # rx(pi*-0.25) q[0];
                                                         #
    # Gate: CNOT**0.5                                    # // Gate: CNOT**0.5
    circ.add_RY_gate(2, np.pi*-0.5*-1)                   # ry(pi*-0.5) q[2];
    circ.add_U3_gate(3, np.pi*0.5, 0, np.pi*0.75)        # u3(pi*0.5,0,pi*0.75) q[3];
    circ.add_U3_gate(2, np.pi*0.5, 0, np.pi*0.25)        # u3(pi*0.5,0,pi*0.25) q[2];
    circ.add_RX_gate(3, np.pi*0.5*-1)                    # rx(pi*0.5) q[3];
    circ.add_CNOT_gate(3, 2)                             # cx q[3],q[2];
    circ.add_RX_gate(3, np.pi*0.25*-1)                   # rx(pi*0.25) q[3];
    circ.add_RY_gate(2, np.pi*0.5*-1)                    # ry(pi*0.5) q[2];
    circ.add_CNOT_gate(2, 3)                             # cx q[2],q[3];
    circ.add_RX_gate(2, np.pi*-0.5*-1)                   # rx(pi*-0.5) q[2];
    circ.add_RZ_gate(2, np.pi*0.5*-1)                    # rz(pi*0.5) q[2];
    circ.add_CNOT_gate(3, 2)                             # cx q[3],q[2];
    circ.add_U3_gate(3, np.pi*0.5, np.pi*0.5, np.pi*1.0) # u3(pi*0.5,pi*0.5,pi*1.0) q[3];
    circ.add_U3_gate(2, np.pi*0.5, np.pi*1.0, np.pi*1.0) # u3(pi*0.5,pi*1.0,pi*1.0) q[2];
    circ.add_RY_gate(2, np.pi*0.5*-1)                    # ry(pi*0.5) q[2];
                                                         #
    # Gate: CNOT**0.5                                    # // Gate: CNOT**0.5
    circ.add_RY_gate(0, np.pi*-0.5*-1)                   # ry(pi*-0.5) q[0];
    circ.add_U3_gate(1, np.pi*0.5, 0, np.pi*0.75)        # u3(pi*0.5,0,pi*0.75) q[1];
    circ.add_U3_gate(0, np.pi*0.5, 0, np.pi*0.25)        # u3(pi*0.5,0,pi*0.25) q[0];
    circ.add_RX_gate(1, np.pi*0.5*-1)                    # rx(pi*0.5) q[1];
    circ.add_CNOT_gate(1, 0)                             # cx q[1],q[0];
    circ.add_RX_gate(1, np.pi*0.25*-1)                   # rx(pi*0.25) q[1];
    circ.add_RY_gate(0, np.pi*0.5*-1)                    # ry(pi*0.5) q[0];
    circ.add_CNOT_gate(0, 1)                             # cx q[0],q[1];
    circ.add_RX_gate(0, np.pi*-0.5*-1)                   # rx(pi*-0.5) q[0];
    circ.add_RZ_gate(0, np.pi*0.5*-1)                    # rz(pi*0.5) q[0];
    circ.add_CNOT_gate(1, 0)                             # cx q[1],q[0];
    circ.add_U3_gate(1, np.pi*0.5, np.pi*0.5, np.pi*1.0) # u3(pi*0.5,pi*0.5,pi*1.0) q[1];
    circ.add_U3_gate(0, np.pi*0.5, np.pi*1.0, np.pi*1.0) # u3(pi*0.5,pi*1.0,pi*1.0) q[0];
    circ.add_RY_gate(0, np.pi*0.5*-1)                    # ry(pi*0.5) q[0];

    circ.update_quantum_state(st)
    #with open('Qulacs.log', 'w') as f:
    #    print(np.around(st.get_vector(),14)+0, file=f)
