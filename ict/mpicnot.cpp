#include <iostream>
#include <sys/types.h> // for debug
#include <unistd.h> // for debug
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>

double get_realtime(void) {
    struct timespec t;
    //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    clock_gettime(CLOCK_REALTIME, &t);
    return t.tv_sec + (double)t.tv_nsec*1e-9;
}

void print_state_in_rank_order(QuantumState* state, int rank, int size) {
    std::cout << std::flush;
    for (int i=0; i < rank + 1; i++){
        MPI_Barrier(MPI_COMM_WORLD);
    }
    std::cout << state->to_string() << std::endl;
    for (int i=0; i < (size - rank); i++){
        MPI_Barrier(MPI_COMM_WORLD);
    }
    //std::cout << std::flush;
}

int main(int argc, char *argv[]) {
    double t[100];

    //int provided;
    //MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
		if (rank==0) {
            printf("USAGE: %s [debug-flag] [n-qubits]\n", argv[0]);
            printf("  debug-flag: n-th rank is waiting before barrier.(-1: w/o waiting)\n");
		}
        exit(1);
    }
    //std::cout << "Rank " << rank << ", PID " << getpid() << ", provided=" << provided << std::endl << std::flush;
    std::cout << "Rank " << rank << ", PID " << getpid() << std::endl << std::flush;
    int waitrank = atoi(argv[1]);
    while (waitrank == rank) sleep(1); // for mpi debug
    MPI_Barrier(MPI_COMM_WORLD);

    int nq = atoi(argv[2]);

    QuantumState state(nq, true);

    t[0] = get_realtime();

    QuantumCircuit circuit_init(nq);

	for (int i=0; i<nq; ++i) {
    	circuit_init.add_RX_gate(i, 0.1 * i);
	}
    for (int i=0; i<3; ++i) {
        circuit_init.update_quantum_state(&state);
		MPI_Barrier(MPI_COMM_WORLD);
		t[i+1] = get_realtime();
	}

    QuantumCircuit circuit_cnot(nq);
    t[4] = get_realtime();

	for (int i=0; i<nq; ++i) {
    	circuit_cnot.add_CNOT_gate(i, (i+1)%nq);
	}

    for (int i=0; i<3; ++i) {
		circuit_cnot.update_quantum_state(&state);
		MPI_Barrier(MPI_COMM_WORLD);
		t[i+5] = get_realtime();
    }

    std::cout << "#rank, time: " << rank << ", ";
    for (int i=0; i<7; ++i) {
        std::cout << t[i+1] - t[i] << " ";
    }
    std::cout << std::endl << std::flush;

	//print_state_in_rank_order(&state, rank, size);

    sleep(1);
    MPI_Finalize();

    return 0;

}
