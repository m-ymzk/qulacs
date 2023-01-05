from argparse import ArgumentParser
#import pytest
import numpy as np
from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import X, T, H, RX, CNOT, ParametricRZ, ParametricRX, DenseMatrix, merge
#from qulacs.circuit import QuantumCircuitOptimizer as QCO
import time
from mpi4py import MPI

def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-n', '--nqubits', type=int,
            default=4, help='Number of qbits')
    argparser.add_argument('-r', '--repeats', type=int,
            default=6, help='Number of repeats')
    argparser.add_argument('-g', '--gate', type=str,
            default="RX", help='Target gate : RX, RZ, H, RU, CNOT or SWAP')
    argparser.add_argument('-s', '--step', type=int,
            default=4, help='Number of step')
    argparser.add_argument('-o', '--opt', type=int,
            default=-1, help='Enable QuantumCircuitOptimizer: 0 is light, 1-4 is opt, 5 is merge_full')
    return argparser.parse_args()

def build_circuit(nqubits, tgt, depth, gate, step):
    circuit = QuantumCircuit(nqubits)
    tgt2 = (tgt + step) % nqubits
    tgt3 = (tgt + 2*step) % nqubits
    tgt4 = (tgt + 3*step) % nqubits
    tgt5 = (tgt + 4*step) % nqubits

    for j in range(depth):
        if gate == "RX":
            circuit.add_RX_gate(tgt, np.random.rand())
        elif gate == "RX2":
            g1 = RX(tgt, np.random.rand())
            g2 = RX(tgt2, np.random.rand())
            circuit.add_gate(merge(g1, g2))
        elif gate == "RX3":
            g1 = RX(tgt, np.random.rand())
            g2 = RX(tgt2, np.random.rand())
            g3 = RX(tgt3, np.random.rand())
            g = merge(g1, g2)
            circuit.add_gate(merge(g, g3))
        elif gate == "RX3r":
            g1 = RX(tgt, np.random.rand())
            g2 = RX(tgt2, np.random.rand())
            g3 = RX(tgt3, np.random.rand())
            g = merge(g3, g2)
            circuit.add_gate(merge(g, g1))
        elif gate == "RX4":
            g1 = RX(tgt, np.random.rand())
            g2 = RX(tgt2, np.random.rand())
            g3 = RX(tgt3, np.random.rand())
            g4 = RX(tgt4, np.random.rand())
            g = merge(g1, g2)
            g = merge(g, g3)
            circuit.add_gate(merge(g, g4))
        elif gate == "RX5":
            g1 = RX(tgt, np.random.rand())
            g2 = RX(tgt2, np.random.rand())
            g3 = RX(tgt3, np.random.rand())
            g4 = RX(tgt4, np.random.rand())
            g5 = RX(tgt5, np.random.rand())
            g = merge(g1, g2)
            g = merge(g, g3)
            g = merge(g, g4)
            circuit.add_gate(merge(g, g5))
        elif gate == "RZ":
            circuit.add_RZ_gate(tgt, np.random.rand())
        elif gate == "H":
            circuit.add_H_gate(tgt)
        elif gate == "RU":
            circuit.add_random_unitary_gate(tgt, tgt2)
        elif gate == "DDM":
            circuit.add_gate(merge(X(tgt), H(tgt2)))
        elif gate == "CNOT":
            circuit.add_CNOT_gate(tgt, tgt2)
        elif gate == "SWAP":
            circuit.add_SWAP_gate(tgt, tgt2)
        else:
            print("# not defined!")
            exit(1)

    #if rank==0:
        #print(circuit)
    return circuit

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = get_option()
    nqubits=args.nqubits
    numRepeats = args.repeats

    np.random.seed(seed=32)
    mode = args.gate + "gate"

    #if rank==0:
        #print('[ROI], mode, #qubits, avg of last 5 runs, std of last 5 runs, runtimes of 6 runs')
    simTimes = np.zeros(numRepeats)
    st = QuantumState(nqubits, use_multi_cpu=True)
    #for tgt in range(nqubits - 10, nqubits):
    for tgt in range(nqubits):
        constStart = time.perf_counter()
        circuit = build_circuit(nqubits, tgt, 1, args.gate, args.step)
        constTime = time.perf_counter() - constStart

        comm.Barrier()
        for i in range(numRepeats):
            comm.Barrier()
            simStart = time.perf_counter()
            circuit.update_quantum_state(st)
            comm.Barrier()
            simTimes[i] = time.perf_counter() - simStart

        del circuit
        #del st

        if rank==0:
            if numRepeats > 1:
                print('[qulacs] {}, size {}, {} qubits, target0 {}, const= {}, sim= {} +- {}'.format(
                    mode, size, nqubits, tgt,
                    constTime,
                    np.average(simTimes[1:]), np.std(simTimes[1:])), simTimes)
            else:
                print('[qulacs] {}, size {}, {} qubits, target0 {}, const= {}, sim= {}'.format(
                    mode, size, nqubits, tgt,
                    constTime,
                    np.average(simTimes)))

    #print('[qulacs construction] {}, {} qubits, {}, {}, {}'.format(
    #    mode, n, np.average(constTimes[1:]), np.std(constTimes[1:]), constTimes))
    #print('[qulacs simulation] {}, {} qubits, {}, {}, {}'.format(
    #    mode, n, np.average(simTimes[1:]), np.std(simTimes[1:]), simTimes))
    #print('[qulacs total] {}, {} qubits, {}, {}, {}'.format(
    #    mode, n, np.average(constTimes[1:]+simTimes[1:]),
    #    np.std(constTimes[1:]+simTimes[1:]), constTimes+simTimes))

#EOF
