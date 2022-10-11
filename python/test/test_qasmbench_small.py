
# set library dir
import pytest
import qulacs
import unittest
import numpy as np
import time
from mpi4py import MPI

# set the qulacs code
from qasmbench.small.qulacs import adder_n4_qulacs, basis_change_n3_qulacs, basis_trotter_n4_qulacs, \
    bell_n4_qulacs, cat_state_n4_qulacs, deutsch_n2_qulacs, dnn_n2_qulacs, \
    error_correctiond3_n5_qulacs, fredkin_n3_qulacs, grover_n2_qulacs, hs4_n4_qulacs, \
    iswap_n2_qulacs, linearsolver_n3_qulacs, lpn_n5_qulacs, qaoa_n3_qulacs, qec_en_n5_qulacs, \
    qrng_n4_qulacs, quantumwalks_n2_qulacs, teleportation_n3_qulacs, toffoli_n3_qulacs, \
    variational_n4_qulacs, vqe_uccsd_n4_qulacs

# set the reference data
from qasmbench.small.reference import adder_n4_ref, basis_change_n3_ref, basis_trotter_n4_ref, \
    bell_n4_ref, cat_state_n4_ref, deutsch_n2_ref, dnn_n2_ref, \
    error_correctiond3_n5_ref, fredkin_n3_ref, grover_n2_ref, hs4_n4_ref, \
    iswap_n2_ref, linearsolver_n3_ref, lpn_n5_ref, qaoa_n3_ref, qec_en_n5_ref, \
    qrng_n4_ref, quantumwalks_n2_ref, teleportation_n3_ref, toffoli_n3_ref, \
    variational_n4_ref, vqe_uccsd_n4_ref

comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()

class TestQASMBenchSmall(unittest.TestCase):
    def tearDown(self):
        del self.state
        del self.circuit
        del self.statev
        if mpirank == 0:
            del self.statev_ref

    def get_state_vector(self, state):
        """
        Gets the entire state vector from the given state.
        
        Args:
             state (qulacs.QuantumState): a quantum state
        Return:
             vector: a state vector
        """
        if state.get_device_name() == 'multi-cpu':
            vec_part = state.get_vector()
            len_part = len(vec_part)
            vector_len = len_part * mpisize
            vector = np.zeros(vector_len, dtype=np.complex128)
            comm.Allgather([vec_part, MPI.C_DOUBLE_COMPLEX],
                           [vector, MPI.C_DOUBLE_COMPLEX])
            return vector
        else:
            return state.get_vector()

    def setParameter(self, nqubit, name):
        self.n = nqubit
        self.dim = 2**nqubit
        self.state = qulacs.QuantumState(self.n, True)
        self.circuit = qulacs.QuantumCircuit(self.n)
        self.name = name
        self.func = name + "_qulacs.func"
        self.ref = name + "_ref.get_ref"

    def postProcess(self):
        self.statev_ref = eval(self.ref)()
        # Differences in sign of 0 are acceptable.
        # -0.0 becomes +0.0 when 0 is added. Compare the results with each other.
        np.testing.assert_allclose(np.around(self.statev,14)+0, np.around(self.statev_ref,14)+0)
        print('\n[Convert QASMbench to Qulacs] {}, sim(s) = {}'.format(self.name.rjust(21), self.simTimes))

    def test_adder_n4(self):
        self.setParameter(4, "adder_n4")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_basis_change_n3(self):
        self.setParameter(3, "basis_change_n3")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_basis_trotter_n4(self):
        self.setParameter(4, "basis_trotter_n4")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_bell_n4(self):
        self.setParameter(4, "bell_n4")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_cat_state_n4(self):
        self.setParameter(4, "cat_state_n4")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_deutsch_n2(self):
        self.setParameter(2, "deutsch_n2")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_dnn_n2(self):
        self.setParameter(2, "dnn_n2")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_error_correctiond3_n5(self):
        self.setParameter(5, "error_correctiond3_n5")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_fredkin_n3(self):
        self.setParameter(3, "fredkin_n3")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_grover_n2(self):
        self.setParameter(2, "grover_n2")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_hs4_n4(self):
        self.setParameter(4, "hs4_n4")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_iswap_n2(self):
        self.setParameter(2, "iswap_n2")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_linearsolver_n3(self):
        self.setParameter(3, "linearsolver_n3")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_lpn_n5(self):
        self.setParameter(5, "lpn_n5")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_qaoa_n3(self):
        self.setParameter(3, "qaoa_n3")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_qec_en_n5(self):
        self.setParameter(5, "qec_en_n5")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_qrng_n4(self):
        self.setParameter(4, "qrng_n4")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_quantumwalks_n2(self):
        self.setParameter(2, "quantumwalks_n2")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_teleportation_n3(self):
        self.setParameter(3, "teleportation_n3")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_toffoli_n3(self):
        self.setParameter(3, "toffoli_n3")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_variational_n4(self):
        self.setParameter(4, "variational_n4")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_vqe_uccsd_n4(self):
        self.setParameter(4, "vqe_uccsd_n4")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

if __name__ == "__main__":
    unittest.main()
