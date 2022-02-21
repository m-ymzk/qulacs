from argparse import ArgumentParser
#import pytest
import numpy as np
from qulacs import QuantumCircuit, QuantumState
#from qulacs.gate import X, T, H, CNOT, ParametricRZ, ParametricRX, DenseMatrix
#from qulacs.circuit import QuantumCircuitOptimizer as QCO
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-n', '--nqubits', type=int,
            default=4, help='Number of qbits')
    argparser.add_argument('-o', '--opt', type=int,
            default=-1, help='Enable QuantumCircuitOptimizer: 0 is light, 1-4 is opt, 5 is merge_full')
    return argparser.parse_args()

def entangler(circuit, nqubits, pairs):
    for a, b in pairs:
        circuit.add_CNOT_gate(a, b)

def build_circuit(nqubits, depth, pairs):
    outer_qc=int(np.log2(size))
    inner_qc=nqubits - outer_qc
    if rank==0:
        print("#inner, outer=", inner_qc, outer_qc)

    circuit = QuantumCircuit(nqubits)
    ### without BSWAP
    for _ in range(depth):
        for i in range(nqubits):
            circuit.add_H_gate(i)

    ### with BSWAP
    #for _ in range(depth):
    #    for i in range(inner_qc):
    #        circuit.add_H_gate(i)
    #    if outer_qc != 0:
    #        circuit.add_BSWAP_gate(inner_qc - outer_qc, inner_qc, outer_qc)
    #    for i in range(outer_qc):
    #        circuit.add_H_gate((inner_qc + i) - outer_qc)
    #    #circuit.add_BSWAP_gate(inner_qc - outer_qc, inner_qc, outer_qc)
    #    if (outer_qc != 0) & (depth % 2 == 1):
    #        circuit.add_BSWAP_gate(inner_qc - outer_qc, inner_qc, outer_qc)

    return circuit

if __name__ == '__main__':

    args = get_option()
    n=args.nqubits
    pairs = [(i, (i + 1) % n) for i in range(n)]
    numRepeats = 1

    np.random.seed(seed=32)
    mode = "hbench w/o BSWAP"

    #if rank==0:
        #print('[ROI], mode, #qubits, avg of last 5 runs, std of last 5 runs, runtimes of 6 runs')
    constTimes = np.zeros(numRepeats)
    simTimes = np.zeros(numRepeats)
    for i in range(numRepeats):
        constStart = time.perf_counter()
        st = QuantumState(n, use_multi_cpu=True)
        circuit = build_circuit(n, 11, pairs)
        #if rank==0:
        #    print(circuit)
        constTimes[i] = time.perf_counter() - constStart

        simStart = time.perf_counter()
        circuit.update_quantum_state(st)
        simTimes[i] = time.perf_counter() - simStart

        del circuit
        del st

    if rank==0:
        print('[qulacs] {}, {} qubits, const= {} +- {}, sim= {} +- {}'.format(
            mode, n,
            np.average(constTimes), np.std(constTimes), 
            np.average(simTimes), np.std(simTimes)))

    #print('[qulacs construction] {}, {} qubits, {}, {}, {}'.format(
    #    mode, n, np.average(constTimes[1:]), np.std(constTimes[1:]), constTimes))
    #print('[qulacs simulation] {}, {} qubits, {}, {}, {}'.format(
    #    mode, n, np.average(simTimes[1:]), np.std(simTimes[1:]), simTimes))
    #print('[qulacs total] {}, {} qubits, {}, {}, {}'.format(
    #    mode, n, np.average(constTimes[1:]+simTimes[1:]),
    #    np.std(constTimes[1:]+simTimes[1:]), constTimes+simTimes))

#EOF
