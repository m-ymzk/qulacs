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
    double tstart, tpre, tsim;
    int rank, size;

    //int provided;
    //MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if ((argc != 4) && (argc != 5)) {
        if (rank==0) {
            printf("USAGE: %s [debug-flag] [n-qubits] [target-qubit] [numLoops]\n", argv[0]);
            printf("  debug-flag: n-th rank is waiting before barrier.(-1: w/o waiting)\n");
            printf("  numLoops  : Number of simulation runs (default: 10)\n");
        }
        exit(1);
    }
    
    std::cout << "Rank " << rank << ", PID " << getpid() << std::endl << std::flush;
    int r = atoi(argv[1]);
    while (r == rank) sleep(1); // for mpi debug
    MPI_Barrier(MPI_COMM_WORLD);

    int nqubits = atoi(argv[2]);
    int target = atoi(argv[3]);
    int numLoops = argc != 5 ? 10 : atoi(argv[4]);

    tstart = get_realtime();

    /* 
     * Prepare: 
     */
    QuantumState state(nqubits, true); // # of qubits, distirbution flag
    QuantumCircuit circuit(nqubits);

    //circuit.add_X_gate(target);
    circuit.add_RX_gate(target, 0.5);

    tpre = get_realtime() - tstart;

    /*
     * Simulate: 
     */
    tstart = get_realtime();
    for (int i=0; i<numLoops; ++i) {
      circuit.update_quantum_state(&state);
    }
    tsim = get_realtime() - tstart;

    std::cout << "#rank, " << rank 
              << ", pre time[sec], " << tpre 
              << ", avg. sim time[sec], " << tsim/numLoops
              << std::endl << std::flush;

#if 0
    // sampling
    //   1st param. is number of sampling.
    //   2nd param. is random-seed.
    // You must call state.sampling on every mpi-ranks.
    std::vector<ITYPE> sample = state.sampling(50, 2021);
    if (rank==0) {
        std::cout << "#result_state.sampling: ";
        for (const auto& e : sample) std::cout << e << " ";
        std::cout << std::endl << std::flush;
    }
#endif

#if 0
    //print_state_in_rank_order(&state, rank, size);
    /*
    QuantumState state_in(nqubits);
    state_in.load(&state);

    if (rank == 0) std::cout << state_in.to_string() << std::endl;
    */
    std::cout << state.to_string() << std::endl;
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    sleep(1);
    MPI_Finalize();

    return 0;
}
