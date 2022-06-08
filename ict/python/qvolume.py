from argparse import ArgumentParser
import numpy as np
from qulacs import QuantumCircuit, QuantumState
from circuits import get_circuit
import time
from mpi4py import MPI
mpicomm = MPI.COMM_WORLD
mpirank = mpicomm.Get_rank()
mpisize = mpicomm.Get_size()

#np.random.seed(seed=32)
rng = np.random.default_rng(seed=2022)

def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-n', '--nqubits', type=int,
            default=4, help='Number of qubits')
    argparser.add_argument('-d', '--depth', type=int,
            default=10, help='Number of Depth')
    argparser.add_argument('-r', '--repeats', type=int,
            default=5, help='Repeat times to measure')
    argparser.add_argument('-v', '--verbose',
            action='store_true',
            help='Verbose switch, Output circuit infomations')
    return argparser.parse_args()

if __name__ == '__main__':
    args = get_option()
    n = args.nqubits
    repeats = args.repeats

    mode = "QuantumVolume"

    constTimes = np.zeros(repeats)
    simTimes = np.zeros(repeats)
    st = QuantumState(n, use_multi_cpu=True)
    for i in range(repeats):
        constStart = time.perf_counter()
        circuit = get_circuit("quantumvolume",
                nqubits=args.nqubits,
                global_nqubits=int(np.log2(mpisize)),
                depth=args.depth,
                verbose=(args.verbose and (mpirank == 0)),
                random_gen=rng)
        constTimes[i] = time.perf_counter() - constStart

        mpicomm.Barrier()
        simStart = time.perf_counter()
        circuit.update_quantum_state(st)
        mpicomm.Barrier()
        simTimes[i] = time.perf_counter() - simStart

        del circuit
    del st

    if mpirank == 0:
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

        print('[qulacs] {}, size {}, {} qubits, const= {} +- {}, sim= {} +- {}'.format(
            mode, mpisize, n, ctime_avg, ctime_std, stime_avg, stime_std), simTimes)

#EOF
