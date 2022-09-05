
# set library dir
import pytest
import qulacs
import unittest
import numpy as np
import time
from mpi4py import MPI

# set the qulacs code
from qasmbench.medium.qulacs import bb84_n8_qulacs, bv_n14_qulacs,  \
    dnn_n8_qulacs, ising_n10_qulacs, multiplier_n15_qulacs, multiply_n13_qulacs, \
    qaoa_n6_qulacs, qft_n15_qulacs, seca_n11_qulacs, simon_n6_qulacs, vqe_uccsd_n8_qulacs

# set the reference data
from qasmbench.medium.reference import bb84_n8_ref, bv_n14_ref,  \
    dnn_n8_ref, ising_n10_ref, multiplier_n15_ref, multiply_n13_ref, \
    qaoa_n6_ref, qft_n15_ref, seca_n11_ref, simon_n6_ref, vqe_uccsd_n8_ref

comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()

class TestQASMBenchMedium(unittest.TestCase):
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
            comm.Allgather([vec_part, MPI.DOUBLE_COMPLEX],
                              [vector, MPI.DOUBLE_COMPLEX])
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

    def test_bb84_n8(self):
        self.setParameter(8, "bb84_n8")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_bv_n14(self):
        self.setParameter(14, "bv_n14")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_dnn_n8(self):
        self.setParameter(8, "dnn_n8")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_ising_n10(self):
        self.setParameter(10, "ising_n10")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    @pytest.mark.skip(reason='not implemented')
    def _test_multiplier_n15(self): # If implemented function name _test -> test
        self.setParameter(15, "multiplier_n15")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    @pytest.mark.skip(reason='not implemented')
    def _test_multiply_n13(self): # If implemented function name _test -> test
        self.setParameter(13, "multiply_n13")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()
        
    def test_qaoa_n6(self):
        self.setParameter(6, "qaoa_n6")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_qft_n15(self):
        self.setParameter(15, "qft_n15")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    @pytest.mark.skip(reason='not implemented')
    def _test_seca_n11(self): # If implemented function name _test -> test
        self.setParameter(11, "seca_n11")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()
        
    @pytest.mark.skip(reason='not implemented')
    def _test_simon_n6(self): # If implemented function name _test -> test
        self.setParameter(6, "simon_n6")
        comm.Barrier()
        simStart = time.perf_counter()
        eval(self.func)(self.state, self.circuit)
        comm.Barrier()
        self.simTimes = time.perf_counter() - simStart
        self.statev = self.get_state_vector(self.state)
        if mpirank == 0:
            self.postProcess()

    def test_vqe_uccsd_n8(self):
        self.setParameter(8, "vqe_uccsd_n8")
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
