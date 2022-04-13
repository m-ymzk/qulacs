from argparse import ArgumentParser
import numpy as np
from qulacs import QuantumCircuit, QuantumState
import time
from mpi4py import MPI
mpicomm = MPI.COMM_WORLD
mpirank = mpicomm.Get_rank()
mpisize = mpicomm.Get_size()

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

def build_circuit(args, pairs):
    depth = args.depth
    vb = args.verbose
    nqubits = args.nqubits
    global_qc = int(np.log2(mpisize))
    local_qc = nqubits - global_qc
    #if mpirank==0: print("#local, global, depth=", local_qc, global_qc, depth)

    circuit = QuantumCircuit(nqubits)
    USE_FusedSWAP=True
    if USE_FusedSWAP:
        for _ in range(depth):
            for i in range(local_qc):
                circuit.add_H_gate(i)
            if global_qc != 0:
                circuit.add_FusedSWAP_gate(local_qc - global_qc, local_qc, global_qc)
            for i in range(global_qc):
                circuit.add_H_gate(local_qc - global_qc + i)
            #if (global_qc!=0):
            #circuit.add_FusedSWAP_gate(local_qc - global_qc, local_qc, global_qc)
        if (global_qc != 0) & (depth % 2 == 1):
            circuit.add_FusedSWAP_gate(local_qc - global_qc, local_qc, global_qc)

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

    #if mpirank==0:
        #print('[ROI], mode, #qubits, avg of last 5 runs, std of last 5 runs, runtimes of 6 runs')
    constTimes = np.zeros(repeats)
    simTimes = np.zeros(repeats)
    st = QuantumState(n, use_multi_cpu=True)
    for i in range(repeats):
        constStart = time.perf_counter()
        circuit = build_circuit(args, pairs)
        constTimes[i] = time.perf_counter() - constStart

        mpicomm.Barrier()
        simStart = time.perf_counter()
        circuit.update_quantum_state(st)
        mpicomm.Barrier()
        simTimes[i] = time.perf_counter() - simStart
        del circuit
    del st

    if mpirank==0:
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
            mode, mpisize, n, ctime_avg, ctime_std, stime_avg, stime_std), simTimes)

#EOF
