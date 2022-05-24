from argparse import ArgumentParser
import numpy as np
from qulacs import QuantumCircuit, QuantumState
import time

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

def local_swap(p, q, done_ug, qubit_table, master_table):
    simple_swap(p, q, done_ug)
    simple_swap(p, q, qubit_table)
    simple_swap(p, q, master_table)

def block_swap(p, q, bs, done_ug, qubit_table, master_table):
    for t in range(bs):
        simple_swap(p + t, q + t, done_ug)
        simple_swap(p + t, q + t, qubit_table)
        simple_swap(p + t, q + t, master_table)

def phys_idx(idx, qubit_table):
    while(1):
        a = qubit_table[idx]
        if a < inner_qc:
            return a

def build_circuit(args, size, master_table):
    depth = args.depth
    vb = args.verbose
    nqubits = args.nqubits
    outer_qc = int(np.log2(size))
    inner_qc = nqubits - outer_qc
    perm_0 = list(range(nqubits))

    rng = np.random.default_rng(seed=2022)
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
                        local_swap(p, q, done_ug, qubit_table, master_table)
                        circuit.add_SWAP_gate(p, q)
                        if vb > 1 and rank == 0: print("#2: circuit.add_SWAP_gate(", p, ", ", q, ")")
                        break

        if vb > 1 and rank == 0: print("#3 block_swap(", work_qubit,", ", inner_qc,", ", outer_qc, ")")
        if outer_qc > 0:
            block_swap(work_qubit, inner_qc, outer_qc, done_ug, qubit_table, master_table)
            circuit.add_BSWAP_gate(work_qubit, inner_qc, outer_qc)
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

    ## build
    st = QuantumState(n, use_multi_cpu=True)
    st_single = QuantumState(n, use_multi_cpu=False)

    master_table = list(range(n))
    circuit_single = build_circuit(args, 1, master_table)
    master_table = list(range(n))
    circuit = build_circuit(args, size, master_table)

    ## Eval
    circuit_single.update_quantum_state(st_single)
    comm.Barrier()
    circuit.update_quantum_state(st)
    comm.Barrier()

    ## Sort
    BSortFlag=True
    sortcircuit = QuantumCircuit(n)

    while BSortFlag:
        BSortFlag=False
        for i in range(n-1):
            if master_table[i] > master_table[i+1]:
                sortcircuit.add_SWAP_gate(master_table[i], master_table[i+1])
                simple_swap(i, i+1, master_table)
                BSortFlag=True
                
    sortcircuit.update_quantum_state(st)
    #sortcircuit.update_quantum_state(st_single)

    ## Value checking
    state = st.get_vector();
    state_ref = st_single.get_vector();
    numElements = (2**n) // size
    offset = numElements * rank
    checkFlag = True
    _EPS = 1.0e-15
    for idx in range(numElements):
        val = state[idx]
        val_ref = state_ref[offset + idx]
        if val_ref.real == 0.0 or val_ref.imag == 0:
            if abs(val.real - val_ref.real) > _EPS or abs(val.imag - val_ref.imag) > _EPS:
                print( "#rank: ", rank, " , idx: ", offset+idx, "ref val: ", val_ref, " , val: ", val)
                checkFlag = False
                break

        else:
            if (abs(val.real - val_ref.real)/abs(val_ref.real)) > _EPS or (abs(val.imag - val_ref.imag)/abs(val_ref.imag)) > _EPS:
                print( "#rank: ", rank, " , idx: ", offset+idx, "ref val: ", val_ref, " , val: ", val)
                checkFlag = False
                break

    if checkFlag is True:
        print("Value check: PASSED (rank: ", rank, ")")
    else:
        print("Value check: Failed (rank: ", rank, ")")
    comm.Barrier()

    if rank == 0:
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
            mode, size, n, ctime_avg, ctime_std, stime_avg, stime_std))

#EOF
