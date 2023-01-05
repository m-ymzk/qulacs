from argparse import ArgumentParser
import numpy as np
from qulacs import QuantumCircuit, QuantumState
from qulacs.circuit import QuantumCircuitOptimizer as QCO
from circuits import get_circuit
import time
import random
from mpi4py import MPI
mpicomm = MPI.COMM_WORLD
mpirank = mpicomm.Get_rank()
mpisize = mpicomm.Get_size()

def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-n', '--nqubits', type=int,
            default=4, help='Number of qubits')
    argparser.add_argument('-d', '--depth', type=int,
            default=10, help='Number of Depth')
    argparser.add_argument('-r', '--repeats', type=int,
            default=6, help='Repeat times to measure')
    argparser.add_argument('-o', '--opt', type=int,
            default=-1, help='Enable QuantumCircuitOptimizer: 99 is to use opt_light, 0-6 is to use opt')
    argparser.add_argument('-f', '--fused', type=int,
            default=-1, help='Enable QuantumCircuitOptimizer: 0 is not to use, 1-2 is to use Fused-swap opt')
    argparser.add_argument('-v', '--verbose',
            action='store_true',
            help='Verbose switch, Output circuit infomations')
    argparser.add_argument('-p', '--printcircuit',
            action='store_true',
            help='Print Circuit')
    argparser.add_argument('-s', '--seed', type=int,
            default=-1, help='Random Seed')
    argparser.add_argument('-c', '--check', action='store_true', help='Check each values of the state vector between with and without multi-cpu')
    return argparser.parse_args()

def print_circuit(circuit):
    print(circuit)
    print('### name, target, control, matrix')
    for i in range(circuit.get_gate_count()):
        g = circuit.get_gate(i)
        print('{}, {}, {}'.format(g.get_name(), g.get_target_index_list(), g.get_control_index_list()), g.get_matrix())
    print('###')

if __name__ == '__main__':
    args = get_option()
    n = args.nqubits
    repeats = args.repeats
    global_nqubits = 0
    if args.opt >= 0:
        qco = QCO()
    if args.fused == -1:
        global_nqubits=int(np.log2(mpisize))

    mode = "QuantumVolume" + str(args.depth)

    simTimes = np.zeros(repeats)
    st = QuantumState(n, use_multi_cpu=True)
    if args.check:
        st_ref = QuantumState(n)

    #for i in range(repeats):
    #
    if args.seed >= 0:
        random.seed(args.seed)
        rng = np.random.default_rng(seed=args.seed)

        # to fix random generator in qulacs
        dmy_state = QuantumState(1)
        #dmy_state.set_Haar_random_state(args.seed)
        dmy_circuit = QuantumCircuit(1)
        dmy_circuit.update_quantum_state(dmy_state, args.seed)
        del dmy_state
        del dmy_circuit
    else: rng = np.random.default_rng()

    constStart = time.perf_counter()
    circuit = get_circuit("quantumvolume",
            nqubits=args.nqubits,
            global_nqubits=global_nqubits,
            depth=args.depth,
            verbose=(args.verbose and (mpirank == 0)),
            random_gen=rng)
    #mpicomm.bcast(circuit, root=0)
    if args.check:
        circuit_ref = circuit.copy()
    if args.printcircuit and args.opt == -1 and mpirank == 0: print_circuit(circuit)
    if args.printcircuit and mpirank == 0: print(circuit)
    if args.opt == 99:
        if args.fused == -1:
            qco.optimize_light(circuit)
        else:
            qco.optimize_light(circuit, args.fused)
        if args.printcircuit and mpirank == 0: print_circuit(circuit)
    elif args.opt >= 0:
        if args.fused == -1:
            qco.optimize(circuit, args.opt)
        else:
            qco.optimize(circuit, args.opt, args.fused)
        if args.printcircuit and mpirank == 0: print_circuit(circuit)
    constTimes = time.perf_counter() - constStart
    #

    mpicomm.Barrier()
    for i in range(repeats):
        simStart = time.perf_counter()
        circuit.update_quantum_state(st)

        if args.check:
            circuit_ref.update_quantum_state(st_ref)
            vec = st.get_vector()
            vec_ref = st_ref.get_vector()
            offs = (1 << (args.nqubits - int(np.log2(mpisize)))) * mpirank
            if np.allclose(vec, vec_ref[offs:offs+len(vec)]):
                print("# check OK!")
            else:
                print("# check NG! ", mpirank, ": ", vec, ", " , vec_ref[offs:offs+len(vec)])
                exit()
        mpicomm.Barrier()
        simTimes[i] = time.perf_counter() - simStart

        #del circuit
    del st

    if mpirank == 0:
        if repeats > 1:
            ctime_avg = constTimes
            ctime_std = 0.
            stime_avg = np.average(simTimes[1:])
            stime_std = np.std(simTimes[1:])
        else:
            ctime_avg = constTimes
            ctime_std = 0.
            stime_avg = simTimes[0]
            stime_std = 0.

        print('[qulacs] {}, size {}, {} qubits, build/opt= {} +- {}, sim= {} +- {}'.format(
            mode, mpisize, n, ctime_avg, ctime_std, stime_avg, stime_std), simTimes)

#EOF
