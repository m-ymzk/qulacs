from argparse import ArgumentParser
import numpy as np
from qulacs import QuantumCircuit, QuantumState
import time

np.random.seed(seed=32)

def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-n', '--nqubits', type=int,
            default=4, help='Number of qbits')
    argparser.add_argument('-d', '--depth', type=int,
            default=10, help='Number of Depth')
    argparser.add_argument('-r', '--repeats', type=int,
            default=5, help='Repeat times to measure')
    argparser.add_argument('-v', '--verbose', type=int,
            default=0, help='Define Output level. 0: time only, 1: Circuit info and time,2: All Gates') 
    return argparser.parse_args()

def build_circuit(args, mpisize, pairs):
    depth = args.depth
    vb = args.verbose
    nqubits = args.nqubits
    outer_qc = int(np.log2(mpisize))
    inner_qc = nqubits - outer_qc
    #if rank==0: print("#inner, outer, depth=", inner_qc, outer_qc, depth)

    circuit = QuantumCircuit(nqubits)
    USE_BSWAP=True
    if USE_BSWAP:
        for _ in range(depth):
            for i in range(inner_qc):
                circuit.add_H_gate(i)
            if outer_qc != 0:
                circuit.add_BSWAP_gate(inner_qc - outer_qc, inner_qc, outer_qc)
            for i in range(outer_qc):
                circuit.add_H_gate(inner_qc - outer_qc + i)
            #if (outer_qc!=0):
            #circuit.add_BSWAP_gate(inner_qc - outer_qc, inner_qc, outer_qc)
        if (outer_qc != 0) & (depth % 2 == 1):
            circuit.add_BSWAP_gate(inner_qc - outer_qc, inner_qc, outer_qc)

    else:
        for _ in range(depth):
            for i in range(nqubits):
                circuit.add_H_gate(i)

    return circuit

if __name__ == '__main__':
    args = get_option()
    n = args.nqubits
    pairs = [(i, (i + 1) % n) for i in range(n)]
    repeats = args.repeats

    mode = "hbench"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpisize = comm.Get_size()

    #if rank==0:
        #print('[ROI], mode, #qubits, avg of last 5 runs, std of last 5 runs, runtimes of 6 runs')
    constTimes = np.zeros(repeats)
    simTimes = np.zeros(repeats)
    st = QuantumState(n, use_multi_cpu=True)
    for i in range(repeats):
        constStart = time.perf_counter()
        circuit = build_circuit(args, mpisize, pairs)
        constTimes[i] = time.perf_counter() - constStart

        comm.Barrier()
        simStart = time.perf_counter()
        circuit.update_quantum_state(st)
        comm.Barrier()
        simTimes[i] = time.perf_counter() - simStart
        del circuit
    del st

    if rank==0:
        if repeats > 1:
            ctime_avg = np.average(constTimes[1:])
            ctime_std = np.std(constTimes[1:])
            stime_avg = np.average(simTimes[1:])
            stime_std = np.std(simTimes[1:])
        else:
            ctime_avg = constTimes[0]
            ctime_std = 0.
            stime_avg = simTimes[0]
            stime_std = 0.

        print('[qulacs] {}, mpisize {}, {} qubits, const= {} +- {}, sim= {} +- {}'.format(
            mode, mpisize, n, ctime_avg, ctime_std, stime_avg, stime_std))

#EOF
