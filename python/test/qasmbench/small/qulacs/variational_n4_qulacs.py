# Generated from Cirq v0.8.0

import numpy as np
from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import Identity, U1

def func(st, circ):
    # Qubits: [0,  1,  2,  3
#    nqubits=4
    #st = QuantumState(nqubits, use_multi_cpu=True)  // for mpiQulacs
#    st = QuantumState(nqubits)                          # qreg q[4];
#    circ = QuantumCircuit(nqubits)

    circ.add_X_gate(0)                                  # x q[0];
    circ.add_X_gate(1)                                  # x q[1];

    # Gate: PhasedISWAP**0.9951774602384953
    circ.add_RZ_gate(1, np.pi*0.25*-1)                  # rz(pi*0.25) q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                 # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(1, 2)                            # cx q[1],q[2];
    circ.add_H_gate(1)                                  # h q[1];
    circ.add_CNOT_gate(2, 1)                            # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*0.4975887301*-1)          # rz(pi*0.4975887301) q[1];
    circ.add_CNOT_gate(2, 1)                            # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*-0.4975887301*-1)         # rz(pi*-0.4975887301) q[1];
    circ.add_H_gate(1)                                  # h q[1];
    circ.add_CNOT_gate(1, 2)                            # cx q[1],q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                 # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                  # rz(pi*0.25) q[2];

    circ.add_RZ_gate(2, 0*-1)                           # rz(0) q[2];

    # Gate: PhasedISWAP**-0.5024296754026449
    circ.add_RZ_gate(0, np.pi*0.25*-1)                  # rz(pi*0.25) q[0];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                 # rz(pi*-0.25) q[1];
    circ.add_CNOT_gate(0, 1)                            # cx q[0],q[1];
    circ.add_H_gate(0)                                  # h q[0];
    circ.add_CNOT_gate(1, 0)                            # cx q[1],q[0];
    circ.add_RZ_gate(0, np.pi*-0.2512148377*-1)         # rz(pi*-0.2512148377) q[0];
    circ.add_CNOT_gate(1, 0)                            # cx q[1],q[0];
    circ.add_RZ_gate(0, np.pi*0.2512148377*-1)          # rz(pi*0.2512148377) q[0];
    circ.add_H_gate(0)                                  # h q[0];
    circ.add_CNOT_gate(0, 1)                            # cx q[0],q[1];
    circ.add_RZ_gate(0, np.pi*-0.25*-1)                 # rz(pi*-0.25) q[0];
    circ.add_RZ_gate(1, np.pi*0.25*-1)                  # rz(pi*0.25) q[1];

    circ.add_RZ_gate(1, 0*-1)                           # rz(0) q[1];

    # Gate: PhasedISWAP**-0.49760685888033646
    circ.add_RZ_gate(2, np.pi*0.25*-1)                  # rz(pi*0.25) q[2];
    circ.add_RZ_gate(3, np.pi*-0.25*-1)                 # rz(pi*-0.25) q[3];
    circ.add_CNOT_gate(2, 3)                            # cx q[2],q[3];
    circ.add_H_gate(2)                                  # h q[2];
    circ.add_CNOT_gate(3, 2)                            # cx q[3],q[2];
    circ.add_RZ_gate(2, np.pi*-0.2488034294*-1)         # rz(pi*-0.2488034294) q[2];
    circ.add_CNOT_gate(3, 2)                            # cx q[3],q[2];
    circ.add_RZ_gate(2, np.pi*0.2488034294*-1)          # rz(pi*0.2488034294) q[2];
    circ.add_H_gate(2)                                  # h q[2];
    circ.add_CNOT_gate(2, 3)                            # cx q[2],q[3];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                 # rz(pi*-0.25) q[2];
    circ.add_RZ_gate(3, np.pi*0.25*-1)                  # rz(pi*0.25) q[3];

    circ.add_RZ_gate(3, 0*-1)                           # rz(0) q[3];

    # Gate: PhasedISWAP**0.004822678143889672
    circ.add_RZ_gate(1, np.pi*0.25*-1)                  # rz(pi*0.25) q[1];
    circ.add_RZ_gate(2, np.pi*-0.25*-1)                 # rz(pi*-0.25) q[2];
    circ.add_CNOT_gate(1, 2)                            # cx q[1],q[2];
    circ.add_H_gate(1)                                  # h q[1];
    circ.add_CNOT_gate(2, 1)                            # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*0.0024113391*-1)          # rz(pi*0.0024113391) q[1];
    circ.add_CNOT_gate(2, 1)                            # cx q[2],q[1];
    circ.add_RZ_gate(1, np.pi*-0.0024113391*-1)         # rz(pi*-0.0024113391) q[1];
    circ.add_H_gate(1)                                  # h q[1];
    circ.add_CNOT_gate(1, 2)                            # cx q[1],q[2];
    circ.add_RZ_gate(1, np.pi*-0.25*-1)                 # rz(pi*-0.25) q[1];
    circ.add_RZ_gate(2, np.pi*0.25*-1)                  # rz(pi*0.25) q[2];

    circ.add_RZ_gate(2, 0*-1)                           # rz(0) q[2];

    circ.update_quantum_state(st)
#    with open('Qulacs.log', 'w') as f:
#        print(np.around(st.get_vector(),14)+0, file=f)
