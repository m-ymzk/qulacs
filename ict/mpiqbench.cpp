#include <iostream>
#include <iomanip>
#include <time.h>
#include "mpi.h"
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/circuit_optimizer.hpp>

double get_realtime(void)
{
    struct timespec t;
    //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    clock_gettime(CLOCK_REALTIME, &t);
    return t.tv_sec + (double)t.tv_nsec*1e-9;
}

void bench(int nqubits, int depth, int lv_opt, int num_dmy, int num_meas, int rank, int size) {
    double dtp, dts;
    double psum = 0;
    double psum2 = 0;
    double ssum = 0;
    double ssum2 = 0;

    for (int i = 0; i < (num_dmy + num_meas); ++i) {
        dtp = -1. * get_realtime();

        QuantumState state(nqubits, 1);
    
        // Build Circuit
        QuantumCircuit circuit(nqubits);
        for (int i=0; i<nqubits; ++i){
            circuit.add_RX_gate(i, std::rand());
            circuit.add_RZ_gate(i, std::rand());
        }
        for (int i=0; i<nqubits; ++i)
            circuit.add_CNOT_gate(i, (i+1)%nqubits);
        for (int j=0; j<depth; ++j){
            for (int i=0; i<nqubits; ++i){
                circuit.add_RZ_gate(i, std::rand());
                circuit.add_RX_gate(i, std::rand());
                circuit.add_RZ_gate(i, std::rand());
                circuit.add_CNOT_gate(i, (i+1)%nqubits);
            }
        }
        for (int i=0; i<nqubits; ++i){
            circuit.add_RZ_gate(i, std::rand());
            circuit.add_RX_gate(i, std::rand());
        }
    
        // Output information of quantum circuit
        // std::cout << circuit << std::endl;
    
        // Circuit Optimize
        if (lv_opt >= 0){
            QuantumCircuitOptimizer opt;
            if (lv_opt == 0) opt.optimize_light(&circuit);
            else {
                opt.optimize(&circuit, lv_opt);
            }
        }
        dtp += get_realtime();
        dts = -1. * get_realtime();
    
        // Update State
        circuit.update_quantum_state(&state);
    
        dts += get_realtime();
        if (i >= num_dmy){
            psum += dtp;
            psum2 += dtp * dtp;
            ssum += dts;
            ssum2 += dts * dts;
        }
		//circuit.~QuantumCircuit();
		//state.~QuantumState();
    }
    double pavg = psum / num_meas;
    double pstd = sqrt(psum2 / num_meas - pavg * pavg);
    double savg = ssum / num_meas;
    double sstd = sqrt(ssum2 / num_meas - savg * savg);

    if (rank == 0)
        std::cout << "mpisize " << size << ", opt= " << lv_opt << ", q= " << nqubits
                  << ", prep_time[s] " << pavg << " +- " << pstd
                  << ", sim_time[s] " << savg << " +- " << sstd << std::endl;
    return;
}

int main(int argc, char *argv[]){

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int num_dmy = 1;
    int num_meas = 5;
    int st_nq = 10;
    int ed_nq = 20;
    int depth = 9;
    //int depth = 99;
    if (argc > 1) st_nq = atoi(argv[1]);
    if (argc > 2) ed_nq = atoi(argv[2]);

	/*
    if (rank == 0) {
        std::cout << "# nqubits: " << st_nq << " ~ " << ed_nq << std::endl;
        std::cout << "# measure: dummy " << num_dmy << ", meas " << num_meas << std::endl;
        std::cout << "# MPI_COMM_WORLD: rank " << rank << ", size " << size << std::endl;
	}
	*/

    std::cout << std::scientific << std::setprecision(9);
    int i = std::max(st_nq, ed_nq);
    //std::cout << "# dummy";
    //bench(20, 9, -1, 0, 3, rank);

    //std::cout << "# start meas" << std::endl;
    int stp=1;
    if (st_nq > ed_nq) stp=-1;
    ed_nq+=stp;
    //for (int lv_opt=-1; lv_opt<5; ++lv_opt){
    for (int lv_opt=-1; lv_opt<0; ++lv_opt){
        for (int i=st_nq; i!=ed_nq; i+=stp){
            bench(i, depth, lv_opt, num_dmy, num_meas, rank, size);
        }
    }

    MPI_Finalize();
}

