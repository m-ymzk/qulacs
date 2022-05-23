from argparse import ArgumentParser
import pytest
import numpy as np
from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import X, T, H, CNOT, ParametricRZ, ParametricRX, DenseMatrix
from qulacs.circuit import QuantumCircuitOptimizer as QCO
import time
from mpi4py import MPI

use_bswap = True
debug = False
rank = MPI.COMM_WORLD.Get_rank()

# mutable
swapped = False

def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-n', '--nqubits', type=int,
            default=4, help='Number of qbits')
    argparser.add_argument('-o', '--opt', type=int,
            default=-1, help='Enable QuantumCircuitOptimizer: 0 is light, 1-4 is opt, 5 is merge_full')
    return argparser.parse_args()

def get_act_idx(i, inner_qc, outer_qc):
    if swapped:
        if i >= inner_qc:
            return i - outer_qc
        elif i >= inner_qc - outer_qc:
            return i + outer_qc
        else:
            return i
    else:
        return i

def first_rotation(circuit, nqubits, inner_qc, outer_qc):
    global swapped

    inner_qubits = list(filter(lambda i: get_act_idx(i, inner_qc, outer_qc) <  inner_qc, range(nqubits)))
    outer_qubits = list(filter(lambda i: get_act_idx(i, inner_qc, outer_qc) >= inner_qc, range(nqubits)))

    for k in inner_qubits:
        k_phy = get_act_idx(k, inner_qc, outer_qc)
        if debug and rank == 0: print('RX/RZ {} ({})'.format(k, k_phy))
        circuit.add_RX_gate(k_phy, np.random.rand())
        circuit.add_RZ_gate(k_phy, np.random.rand())

    if use_bswap and outer_qc > 0:
        if debug and rank == 0: print('BSWAP {} {} {}'.format(inner_qc - outer_qc, inner_qc, outer_qc))
        circuit.add_BSWAP_gate(inner_qc - outer_qc, inner_qc, outer_qc)
        swapped = not swapped

    for k in outer_qubits:
        k_phy = get_act_idx(k, inner_qc, outer_qc)
        if debug and rank == 0: print('RX/RZ {} ({})'.format(k, k_phy))
        circuit.add_RX_gate(k_phy, np.random.rand())
        circuit.add_RZ_gate(k_phy, np.random.rand())

    if use_bswap and outer_qc > 0:
        if debug and rank == 0: print('BSWAP {} {} {}'.format(inner_qc - outer_qc, inner_qc, outer_qc))
        circuit.add_BSWAP_gate(inner_qc - outer_qc, inner_qc, outer_qc)
        swapped = not swapped

def mid_rotation(circuit, nqubits, inner_qc, outer_qc):
    global swapped

    inner_qubits = list(filter(lambda i: get_act_idx(i, inner_qc, outer_qc) <  inner_qc, range(nqubits)))
    outer_qubits = list(filter(lambda i: get_act_idx(i, inner_qc, outer_qc) >= inner_qc, range(nqubits)))

    for k in inner_qubits:
        k_phy = get_act_idx(k, inner_qc, outer_qc)
        if debug and rank == 0: print('RZ/RX/RZ {} ({})'.format(k, k_phy))
        circuit.add_RZ_gate(k_phy, np.random.rand())
        circuit.add_RX_gate(k_phy, np.random.rand())
        circuit.add_RZ_gate(k_phy, np.random.rand())

    if use_bswap and outer_qc > 0:
        if debug and rank == 0: print('BSWAP {} {} {}'.format(inner_qc - outer_qc, inner_qc, outer_qc))
        circuit.add_BSWAP_gate(inner_qc - outer_qc, inner_qc, outer_qc)
        swapped = not swapped

    for k in outer_qubits:
        k_phy = get_act_idx(k, inner_qc, outer_qc)
        if debug and rank == 0: print('RZ/RX/RZ {} ({})'.format(k, k_phy))
        circuit.add_RZ_gate(k_phy, np.random.rand())
        circuit.add_RX_gate(k_phy, np.random.rand())
        circuit.add_RZ_gate(k_phy, np.random.rand())

    if use_bswap and outer_qc > 0:
        if debug and rank == 0: print('BSWAP {} {} {}'.format(inner_qc - outer_qc, inner_qc, outer_qc))
        circuit.add_BSWAP_gate(inner_qc - outer_qc, inner_qc, outer_qc)
        swapped = not swapped


def last_rotation(circuit, nqubits, inner_qc, outer_qc):
    global swapped

    inner_qubits = list(filter(lambda i: get_act_idx(i, inner_qc, outer_qc) <  inner_qc, range(nqubits)))
    outer_qubits = list(filter(lambda i: get_act_idx(i, inner_qc, outer_qc) >= inner_qc, range(nqubits)))

    for k in inner_qubits:
        k_phy = get_act_idx(k, inner_qc, outer_qc)
        if debug and rank == 0: print('RZ/RX {} ({})'.format(k, k_phy))
        circuit.add_RZ_gate(k_phy, np.random.rand())
        circuit.add_RX_gate(k_phy, np.random.rand())

    if use_bswap and outer_qc > 0:
        if debug and rank == 0: print('BSWAP {} {} {}'.format(inner_qc - outer_qc, inner_qc, outer_qc))
        circuit.add_BSWAP_gate(inner_qc - outer_qc, inner_qc, outer_qc)
        swapped = not swapped

    for k in outer_qubits:
        k_phy = get_act_idx(k, inner_qc, outer_qc)
        if debug and rank == 0: print('RZ/RX {} ({})'.format(k, k_phy))
        circuit.add_RZ_gate(k_phy, np.random.rand())
        circuit.add_RX_gate(k_phy, np.random.rand())

    if use_bswap and outer_qc > 0:
        if debug and rank == 0: print('BSWAP {} {} {}'.format(inner_qc - outer_qc, inner_qc, outer_qc))
        circuit.add_BSWAP_gate(inner_qc - outer_qc, inner_qc, outer_qc)
        swapped = not swapped


def entangler(circuit, nqubits, pairs, inner_qc, outer_qc):
    global swapped

    for a, b in pairs:
        if use_bswap and get_act_idx(b, inner_qc, outer_qc) >= inner_qc:
            if debug and rank == 0: print('BSWAP {} {} {}'.format(inner_qc - outer_qc, inner_qc, outer_qc))
            circuit.add_BSWAP_gate(inner_qc - outer_qc, inner_qc, outer_qc)
            swapped = not swapped

        a_phy = get_act_idx(a, inner_qc, outer_qc)
        b_phy = get_act_idx(b, inner_qc, outer_qc)
        if debug and rank == 0: print('CNOT {} {} ({} {})'.format(a, b, a_phy, b_phy))
        circuit.add_CNOT_gate(a_phy, b_phy)

        if use_bswap and swapped:
            if debug and rank == 0: print('BSWAP {} {} {}'.format(inner_qc - outer_qc, inner_qc, outer_qc))
            circuit.add_BSWAP_gate(inner_qc - outer_qc, inner_qc, outer_qc)
            swapped = not swapped

def build_circuit(nqubits, depth, pairs, commsize):
    global swapped

    swapped = False

    np.random.seed(seed=32)
    outer_qc = int(np.log2(commsize))
    inner_qc = nqubits - outer_qc

    circuit = QuantumCircuit(nqubits)
    first_rotation(circuit, nqubits, inner_qc, outer_qc)
    entangler(circuit, nqubits, pairs, inner_qc, outer_qc)
    for k in range(depth):
        mid_rotation(circuit, nqubits, inner_qc, outer_qc)
        entangler(circuit, nqubits, pairs, inner_qc, outer_qc)
    last_rotation(circuit, nqubits, inner_qc, outer_qc)

    # recover if swapped
    if use_bswap and outer_qc > 0 and swapped:
        if debug and rank == 0: print('BSWAP {} {} {}'.format(inner_qc - outer_qc, inner_qc, outer_qc))
        circuit.add_BSWAP_gate(inner_qc - outer_qc, inner_qc, outer_qc)
        swapped = not swapped

    assert swapped is False

    return circuit

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = get_option()
    n=args.nqubits
    pairs = [(i, (i + 1) % n) for i in range(n)]
    numRepeats = 5

    np.random.seed(seed=32)
    mode = "qulacsbench"

    #if rank==0:
        #print('[ROI], mode, #qubits, avg of last 5 runs, std of last 5 runs, runtimes of 6 runs')
    constTimes = np.zeros(numRepeats)
    simTimes = np.zeros(numRepeats)
    st = QuantumState(n, use_multi_cpu=True)
    st_single = QuantumState(n, use_multi_cpu=False)
    for i in range(numRepeats):
        constStart = time.perf_counter()
        circuit = build_circuit(n, 1, pairs, size)
        circuit_single = build_circuit(n, 1, pairs, 1)
        constTimes[i] = time.perf_counter() - constStart

        comm.Barrier()
        simStart = time.perf_counter()
        circuit.update_quantum_state(st)
        comm.Barrier()
        circuit_single.update_quantum_state(st_single)
        simTimes[i] = time.perf_counter() - simStart

        del circuit

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

    if rank==0:
        print("ref:")
        print(st_single.to_string())
        print("val:")

    comm.Barrier()
    for i in range(size):
        if rank==i:
            print(st.to_string())
            comm.Barrier()
            

    if rank==0:
        if numRepeats == 1:
            print('[qulacs] {}, size {}, {} qubits, const= {} +- {}, sim= {} +- {}'.format(
                mode, size, n,
                np.average(constTimes), np.std(constTimes),
                np.average(simTimes), np.std(simTimes)))
        else:
            print('[qulacs] {}, size {}, {} qubits, const= {} +- {}, sim= {} +- {}'.format(
                mode, size, n,
                np.average(constTimes[1:]), np.std(constTimes[1:]),
                np.average(simTimes[1:]), np.std(simTimes[1:])), simTimes)

    #print('[qulacs construction] {}, {} qubits, {}, {}, {}'.format(
    #    mode, n, np.average(constTimes[1:]), np.std(constTimes[1:]), constTimes))
    #print('[qulacs simulation] {}, {} qubits, {}, {}, {}'.format(
    #    mode, n, np.average(simTimes[1:]), np.std(simTimes[1:]), simTimes))
    #print('[qulacs total] {}, {} qubits, {}, {}, {}'.format(
    #    mode, n, np.average(constTimes[1:]+simTimes[1:]),
    #    np.std(constTimes[1:]+simTimes[1:]), constTimes+simTimes))

#EOF
