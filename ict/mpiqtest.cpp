#include <iostream>
#include <sys/types.h> // for debug
#include <unistd.h> // for debug
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include "mpi.h"

double get_realtime(void) {
    struct timespec t;
    //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    clock_gettime(CLOCK_REALTIME, &t);
    return t.tv_sec + (double)t.tv_nsec*1e-9;
}

void print_state_in_rank_order(QuantumState* state) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << std::flush;
    for (int i=0; i < rank + 1; i++){
        MPI_Barrier(MPI_COMM_WORLD);
    }
    std::cout << state->to_string() << std::endl;
    for (int i=0; i < (size - rank); i++){
        MPI_Barrier(MPI_COMM_WORLD);
    }
    std::cout << std::flush;
}

int main(int argc, char *argv[]) {
    double dt;
    int _rank, _size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &_size);
    std::cout << "Rank " << _rank << ", PID " << getpid() << std::endl << std::flush;
    int i = 0;
    //while (i == _rank) sleep(1); // for debug
    MPI_Barrier(MPI_COMM_WORLD);

    int nqubits = atoi(argv[1]);

    dt = -1*get_realtime();
    QuantumState state(nqubits, MPI_COMM_WORLD);
    //QuantumState state1(nqubits, (MPI_Comm)((intptr_t)MPI_COMM_WORLD+1)); // MPI_Comm warning check.
    //std::cout << state.to_string() << std::endl;

    state.set_Haar_random_state(1+_rank);

    print_state_in_rank_order(&state);
    /*
    for (int i=0; i<_rank; i++){
        MPI_Barrier(MPI_COMM_WORLD);
    }
    std::cout << state.to_string() << std::endl;
    for (int i=0; i<(_size - _rank); i++){
        MPI_Barrier(MPI_COMM_WORLD);
    }
    */

    dt += get_realtime();
    std::cout << "#rank, time: " << _rank << ", " << dt << std::endl << std::flush;

    QuantumCircuit circuit(nqubits);

    circuit.add_X_gate(0);
    circuit.add_X_gate(1);
    circuit.add_X_gate(nqubits - 2);
    circuit.add_X_gate(nqubits - 1);
    circuit.add_T_gate(0);
    circuit.add_T_gate(1);
    circuit.add_T_gate(nqubits - 2);
    circuit.add_T_gate(nqubits - 1);
    circuit.update_quantum_state(&state);

    print_state_in_rank_order(&state);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;

//    circuit.add_X_gate(0);
//    auto merged_gate = gate::merge(gate::CNOT(0,1),gate::Y(1));
//    circuit.add_gate(merged_gate);
//    circuit.add_RX_gate(1,0.5);
//    circuit.update_quantum_state(&state);
//
//    Observable observable(3);
//    observable.add_operator(2.0, "X 2 Y 1 Z 0");
//    observable.add_operator(-3.0, "Z 2");
//    auto value = observable.get_expectation_value(&state);
//    std::cout << value << std::endl;
//    return 0;
}
