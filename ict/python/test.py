from argparse import ArgumentParser
import pytest
import numpy as np
from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import X, T, H, CNOT, ParametricRZ, ParametricRX, DenseMatrix
from qulacs.circuit import QuantumCircuitOptimizer as QCO
from time import time

def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-n', '--nqubits', type=int,
            default=4, help='Number of qbits')
    argparser.add_argument('-o', '--opt', type=int,
            default=-1, help='Enable QuantumCircuitOptimizer: 0 is light, 1-4 is opt, 5 is merge_full')
    return argparser.parse_args()

def build_circuit(nqubits, depth, pairs):
    circuit = QuantumCircuit(nqubits)
    #first_rotation
    for k in range(nqubits):
        circuit.add_RX_gate(k, np.random.rand())
        circuit.add_RZ_gate(k, np.random.rand())
    #entangler
    for a, b in pairs:
        circuit.add_CNOT_gate(a, b)
    for _ in range(depth):
        #mid_rotation
        for k in range(nqubits):
            circuit.add_RZ_gate(k, np.random.rand())
            circuit.add_RX_gate(k, np.random.rand())
            circuit.add_RZ_gate(k, np.random.rand())
        #entangler
        for a, b in pairs:
            circuit.add_CNOT_gate(a, b)

    #last_rotation
    for k in range(nqubits):
        circuit.add_RZ_gate(k, np.random.rand())
        circuit.add_RX_gate(k, np.random.rand())
    return circuit

if __name__ == '__main__':
    args = get_option()
    tstart = time()
    nqubits=args.nqubits
    pairs = [(i, (i + 1) % nqubits) for i in range(nqubits)]
    circuit = build_circuit(nqubits, 9, pairs)

    np.random.seed(seed=32)

    #if args.opt:
    if True:
        t1=time()
        #st = QuantumState(nqubits)
        st = QuantumState(nqubits, use_multi_cpu=True)
        circuit = build_circuit(nqubits, 9, pairs)
        if args.opt>=0:
            qco = QCO()
            if args.opt==0:
                qco.optimize_light(circuit)
                print("# opt circuit(light) info\n", circuit)
            elif 1 <= args.opt <= 4:
                qco.optimize(circuit, args.opt)
                print("# opt circuit(",args.opt,") info\n", circuit)
            else:
                qco.merge_all(circuit)
                print("# opt circuit(merge_all) info\n", circuit)
        t2=time()
        circuit.update_quantum_state(st)
        t3=time()
        #if comm.rank==0:
        if True:
            print("# time update :", t2-t1, t3-t2)
            print("# circuit ", circuit.to_string())
            print("# state ")
            print(st.to_string())
            print("# device ", st.get_device_name())

    else:
        t1=time()
        st = QuantumState(nqubits)
        circuit = build_circuit(nqubits, 9, pairs)
        t2=time()
        circuit.update_quantum_state(st)
        t3=time()
        print("# w/o opt circuit info", circuit)
        print("# time update :", t2-t1, t3-t2)

        #print("# circuit(opt) info", circuit)
        #print("# time optimize:", t3-t2)
        #print("# time update with opt:", t4-t3)

    #if args.opt:
    #    qco = QCO()
    #st = QuantumState(nqubits)
    #if args.opt:
    #    qco.optimize_light(circuit)
    #circuit.update_quantum_state(st)

    #tend = time()
    #state = QuantumState(nqubits)
    #circuit.update_quantum_state(state)

    #print("Time[s]: ", tend - tstart)
    #print("circuit:", circuit)
    #print("state:", st)
    
#EOF
