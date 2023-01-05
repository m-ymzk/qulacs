from argparse import ArgumentParser
import pytest
import numpy as np
from qulacs import QuantumCircuit, QuantumState
from qulacs.gate import X, T, H, CNOT, ParametricRZ, ParametricRX, DenseMatrix
from qulacs.circuit import QuantumCircuitOptimizer as QCO
import time
from mpi4py import MPI

use_bswap = False
debug = False
mpicomm = MPI.COMM_WORLD
mpirank = mpicomm.Get_rank()
mpisize = mpicomm.Get_size()

# mutable
swapped = False

def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-n', '--nqubits', type=int,
            default=4, help='Number of qbits')
    argparser.add_argument('-r', '--repeats', type=int,
            default=6, help='Repeat times to measure')
    argparser.add_argument('-o', '--opt', type=int,
            default=-1, help='Enable QuantumCircuitOptimizer: 99 is to use opt_light, 0-6 is to use opt')
    argparser.add_argument('-f', '--fused', type=int,
            default=-1, help='Enable QuantumCircuitOptimizer: 0 is not to use, 1-2 is to use Fused-swap opt')
    argparser.add_argument('-p', '--printcircuit',
            action='store_true',
            help='Print Circuit')
    return argparser.parse_args()


def print_circuit(circuit):
    print(circuit)
    print('### name, target, control')
    for i in range(circuit.get_gate_count()):
        g = circuit.get_gate(i)
        print('{}, {}, {}'.format(g.get_name(), g.get_target_index_list(), g.get_control_index_list()))
    print('###')

def get_act_idx(i, local_qc, global_qc):
    if swapped:
        if i >= local_qc:
            return i - global_qc
        elif i >= local_qc - global_qc:
            return i + global_qc
        else:
            return i
    else:
        return i

def first_rotation(circuit, nqubits, local_qc, global_qc):
    global swapped

    local_qubits = list(filter(lambda i: get_act_idx(i, local_qc, global_qc) <  local_qc, range(nqubits)))
    global_qubits = list(filter(lambda i: get_act_idx(i, local_qc, global_qc) >= local_qc, range(nqubits)))

    for k in local_qubits:
        k_phy = get_act_idx(k, local_qc, global_qc)
        if debug and mpirank == 0: print('RX/RZ {} ({})'.format(k, k_phy))
        circuit.add_RX_gate(k_phy, np.random.rand())
        circuit.add_RZ_gate(k_phy, np.random.rand())

    if use_bswap and global_qc > 0:
        if debug and mpirank == 0: print('FusedSWAP {} {} {}'.format(local_qc - global_qc, local_qc, global_qc))
        circuit.add_FusedSWAP_gate(local_qc - global_qc, local_qc, global_qc)
        swapped = not swapped

    for k in global_qubits:
        k_phy = get_act_idx(k, local_qc, global_qc)
        if debug and mpirank == 0: print('RX/RZ {} ({})'.format(k, k_phy))
        circuit.add_RX_gate(k_phy, np.random.rand())
        circuit.add_RZ_gate(k_phy, np.random.rand())

def mid_rotation(circuit, nqubits, local_qc, global_qc):
    global swapped

    local_qubits = list(filter(lambda i: get_act_idx(i, local_qc, global_qc) <  local_qc, range(nqubits)))
    global_qubits = list(filter(lambda i: get_act_idx(i, local_qc, global_qc) >= local_qc, range(nqubits)))

    for k in local_qubits:
        k_phy = get_act_idx(k, local_qc, global_qc)
        if debug and mpirank == 0: print('RZ/RX/RZ {} ({})'.format(k, k_phy))
        circuit.add_RZ_gate(k_phy, np.random.rand())
        circuit.add_RX_gate(k_phy, np.random.rand())
        circuit.add_RZ_gate(k_phy, np.random.rand())

    if use_bswap and global_qc > 0:
        if debug and mpirank == 0: print('FusedSWAP {} {} {}'.format(local_qc - global_qc, local_qc, global_qc))
        circuit.add_FusedSWAP_gate(local_qc - global_qc, local_qc, global_qc)
        swapped = not swapped

    for k in global_qubits:
        k_phy = get_act_idx(k, local_qc, global_qc)
        if debug and mpirank == 0: print('RZ/RX/RZ {} ({})'.format(k, k_phy))
        circuit.add_RZ_gate(k_phy, np.random.rand())
        circuit.add_RX_gate(k_phy, np.random.rand())
        circuit.add_RZ_gate(k_phy, np.random.rand())

def last_rotation(circuit, nqubits, local_qc, global_qc):
    global swapped

    local_qubits = list(filter(lambda i: get_act_idx(i, local_qc, global_qc) <  local_qc, range(nqubits)))
    global_qubits = list(filter(lambda i: get_act_idx(i, local_qc, global_qc) >= local_qc, range(nqubits)))

    for k in local_qubits:
        k_phy = get_act_idx(k, local_qc, global_qc)
        if debug and mpirank == 0: print('RZ/RX {} ({})'.format(k, k_phy))
        circuit.add_RZ_gate(k_phy, np.random.rand())
        circuit.add_RX_gate(k_phy, np.random.rand())

    if use_bswap and global_qc > 0:
        if debug and mpirank == 0: print('FusedSWAP {} {} {}'.format(local_qc - global_qc, local_qc, global_qc))
        circuit.add_FusedSWAP_gate(local_qc - global_qc, local_qc, global_qc)
        swapped = not swapped

    for k in global_qubits:
        k_phy = get_act_idx(k, local_qc, global_qc)
        if debug and mpirank == 0: print('RZ/RX {} ({})'.format(k, k_phy))
        circuit.add_RZ_gate(k_phy, np.random.rand())
        circuit.add_RX_gate(k_phy, np.random.rand())

def entangler(circuit, nqubits, pairs, local_qc, global_qc):
    global swapped

    for a, b in pairs:
        if use_bswap and get_act_idx(b, local_qc, global_qc) >= local_qc:
            if debug and mpirank == 0: print('FusedSWAP {} {} {}'.format(local_qc - global_qc, local_qc, global_qc))
            circuit.add_FusedSWAP_gate(local_qc - global_qc, local_qc, global_qc)
            swapped = not swapped

        a_phy = get_act_idx(a, local_qc, global_qc)
        b_phy = get_act_idx(b, local_qc, global_qc)
        if debug and mpirank == 0: print('CNOT {} {} ({} {})'.format(a, b, a_phy, b_phy))
        circuit.add_CNOT_gate(a_phy, b_phy)

def build_circuit(nqubits, depth, pairs):
    global swapped

    swapped = False

    global_qc = int(np.log2(mpisize))
    local_qc = nqubits - global_qc

    circuit = QuantumCircuit(nqubits)
    first_rotation(circuit, nqubits, local_qc, global_qc)
    entangler(circuit, nqubits, pairs, local_qc, global_qc)
    for k in range(depth):
        mid_rotation(circuit, nqubits, local_qc, global_qc)
        entangler(circuit, nqubits, pairs, local_qc, global_qc)
    last_rotation(circuit, nqubits, local_qc, global_qc)

    # recover if swapped
    if use_bswap and global_qc > 0 and swapped:
        if debug and mpirank == 0: print('FusedSWAP {} {} {}'.format(local_qc - global_qc, local_qc, global_qc))
        circuit.add_FusedSWAP_gate(local_qc - global_qc, local_qc, global_qc)
        swapped = not swapped

    assert swapped is False

    return circuit

if __name__ == '__main__':
    args = get_option()
    n=args.nqubits
    pairs = [(i, (i + 1) % n) for i in range(n)]
    numRepeats = args.repeats
    if args.opt >= 0: qco = QCO()

    np.random.seed(seed=32)
    mode = "qulacsbench"

    #if mpirank==0:
        #print('[ROI], mode, #qubits, avg of last 5 runs, std of last 5 runs, runtimes of 6 runs')
    constTimes = np.zeros(numRepeats)
    simTimes = np.zeros(numRepeats)
    st = QuantumState(n, use_multi_cpu=True)
    for i in range(numRepeats):
        constStart = time.perf_counter()
        circuit = build_circuit(n, 9, pairs)
        if args.printcircuit and args.opt == -1 and mpirank == 0: print_circuit(circuit)
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
        constTimes[i] = time.perf_counter() - constStart

        mpicomm.Barrier()
        simStart = time.perf_counter()
        circuit.update_quantum_state(st)
        mpicomm.Barrier()
        simTimes[i] = time.perf_counter() - simStart

        del circuit
    del st

    if mpirank==0:
        if numRepeats == 1:
            print('[qulacs] {}, size {}, {} qubits, build/opt= {} +- {}, sim= {} +- {}'.format(
                mode, mpisize, n,
                np.average(constTimes), np.std(constTimes),
                np.average(simTimes), np.std(simTimes)))
        else:
            print('[qulacs] {}, size {}, {} qubits, const= {} +- {}, sim= {} +- {}'.format(
                mode, mpisize, n,
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
