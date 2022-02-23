from argparse import ArgumentParser
import numpy as np
from qulacs import QuantumCircuit, QuantumState
import time
from mpi4py import MPI

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
    argparser.add_argument('-v', '--verbose', type=int,
            default=0, help='Define Output level. 0: time only, 1: Circuit info and time, 2: All Gates')
    return argparser.parse_args()

def simple_swap(p, q, array):
    tmp = array[p]
    array[p] = array[q]
    array[q] = tmp

def local_swap(p, q, done_ug, qubit_table):
    simple_swap(p, q, done_ug)
    simple_swap(p, q, qubit_table)
    #simple_swap(p, q, master_table)

def block_swap(p, q, bs, done_ug, qubit_table):
    for t in range(bs):
        simple_swap(p + t, q + t, qubit_table)
        #simple_swap(p + t, q + t, master_table)

def phys_idx(idx, qubit_table):
    while(1):
        a = qubit_table[idx]
        if a < inner_qc:
            return a

def build_circuit(args, size):
    depth = args.depth
    vb = args.verbose
    nqubits = args.nqubits
    outer_qc = int(np.log2(size))
    inner_qc = nqubits - outer_qc
    ter_table = list(range(nqubits))
    perm_0 = list(range(nqubits))

    circuit = QuantumCircuit(nqubits)

    for d in range(depth):
        qubit_table = list(range(nqubits))
        perm = rng.permutation(perm_0)
        pend_pair=[]
        done_ug = [0]*nqubits

        # add random_unitary_gate for inner_qcs, first
        for w in range(nqubits//2):
            physical_qubits = [int(perm[2 * w]), int(perm[2 * w + 1])]
            if physical_qubits[0] < inner_qc and physical_qubits[1] < inner_qc:
                if vb > 1 and rank == 0: print("#1: circuit.add_random_unitary_gate(",physical_qubits,")")
                circuit.add_random_unitary_gate(physical_qubits)
                done_ug[physical_qubits[0]] = 1
                done_ug[physical_qubits[1]] = 1
            else:
                pend_pair.append(physical_qubits)

        # add SWAP gate for BSWAP
        work_qubit = inner_qc - outer_qc
        for s in range(outer_qc):
            if done_ug[work_qubit + s] == 0:
                for t in range(work_qubit):
                    if done_ug[work_qubit - t - 1] == 1:
                        p = work_qubit + s
                        q = work_qubit - t - 1
                        local_swap(p, q, done_ug, qubit_table)
                        if vb > 1 and rank == 0: print("#2: circuit.add_SWAP_gate(", p, ", ", q, ")")
                        circuit.add_SWAP_gate(p, q)
                        break

        if vb > 1 and rank == 0: print("#3 block_swap(", work_qubit,", ", inner_qc,", ", outer_qc, ")")
        block_swap(work_qubit, inner_qc, outer_qc, done_ug, qubit_table)
        if vb > 1 and rank == 0: print("#: qubit_table=", qubit_table)

        # add random_unitary_gate for qubits that were originally outside.
        for pair in pend_pair:
            unitary_pair = [qubit_table.index(pair[0]), qubit_table.index(pair[1])]
            if vb > 1 and rank == 0: print("#4: circuit.add_random_unitary_gate(", unitary_pair, ")")
            circuit.add_random_unitary_gate(unitary_pair)
            done_ug[unitary_pair[0]] = 1
            done_ug[unitary_pair[1]] = 1

    if vb > 0 and rank == 0: print("rank=", rank, ", circuit=",circuit)

    return circuit

if __name__ == '__main__':
    args = get_option()
    n = args.nqubits
    repeats = args.repeats

    mode = "QuantumVolume"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    constTimes = np.zeros(repeats)
    simTimes = np.zeros(repeats)
    st = QuantumState(n, use_multi_cpu=True)
    for i in range(repeats):
        constStart = time.perf_counter()
        circuit = build_circuit(args, size)
        constTimes[i] = time.perf_counter() - constStart

        comm.Barrier()
        simStart = time.perf_counter()
        circuit.update_quantum_state(st)
        simTimes[i] = time.perf_counter() - simStart

        del circuit
    del st

    if rank == 0:
        if repeats > 1:
            ctime_avg = np.average(constTimes[1:])
            ctime_std = np.std(constTimes[1:]) 
            stime_avg = np.average(simTimes[1:])
            stime_std = np.std(simTimes[1:])
        else:
            ctime_avg = constTimes
            ctime_std = 0.
            stime_avg = simTimes
            stime_std = 0.

        print('[qulacs] {}, {} qubits, const= {} +- {}, sim= {} +- {}'.format(
            mode, n, ctime_avg, ctime_std, stime_avg, stime_std))

#EOF
