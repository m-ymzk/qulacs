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
    //std::cout << std::flush;
}

int main(int argc, char *argv[]) {
    double dt;
    int _rank, _size;
    if (argc != 3) {
        printf("USAGE: %s [debug-flag] [n-qubits]\n", argv[0]);
        printf("  debug-flag: n-th rank is waiting before barrier.(-1: w/o waiting)\n");
        exit(1);
    }
    MPI_Init(&argc, &argv);
    //int provided;
    //MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &_size);
    //std::cout << "Rank " << _rank << ", PID " << getpid() << ", provided=" << provided << std::endl << std::flush;
    std::cout << "Rank " << _rank << ", PID " << getpid() << std::endl << std::flush;
    int i = atoi(argv[1]);
    while (i == _rank) sleep(1); // for debug
    MPI_Barrier(MPI_COMM_WORLD);

    int nqubits = atoi(argv[2]);

    QuantumState state(nqubits, MPI_COMM_WORLD);
    //QuantumState state2(nqubits);
    //QuantumState state1(nqubits, (MPI_Comm)((intptr_t)MPI_COMM_WORLD+1)); // MPI_Comm warning check.
    //std::cout << state.to_string() << std::endl;

    //state.set_Haar_random_state();
    //state2.set_Haar_random_state();
    state.set_Haar_random_state(1);
    //state2.set_Haar_random_state(1);
    //state.set_computational_basis(0b00111);
    state.set_computational_basis(0b0000);

    //print_state_in_rank_order(&state);
    /*
    for (int i=0; i<_rank; i++){
        MPI_Barrier(MPI_COMM_WORLD);
    }
    std::cout << state.to_string() << std::endl;
    for (int i=0; i<(_size - _rank); i++){
        MPI_Barrier(MPI_COMM_WORLD);
    }
    */

    dt = -1*get_realtime();

    QuantumCircuit circuit(nqubits);

    //circuit.add_X_gate(0);
    //for (int i=0; i<100; ++i) {
    //    int index = atoi(argv[2]);
    //    circuit.add_X_gate(index);
    //}
    circuit.add_H_gate(0);
    circuit.add_H_gate(1);
    circuit.add_H_gate(2);
    circuit.add_H_gate(3);
    circuit.add_RX_gate(0, 0.5);
    circuit.add_RX_gate(1, 0.25);
    circuit.add_RX_gate(2, 0.125);
    circuit.add_RX_gate(3, 0.375);
    //circuit.add_H_gate(nqubits - 2);
    //circuit.add_H_gate(nqubits - 1);
    circuit.add_X_gate(0);
    circuit.add_X_gate(1);
    circuit.add_X_gate(2);
    circuit.add_X_gate(3);
    circuit.add_S_gate(0);
    circuit.add_S_gate(1);
    circuit.add_S_gate(2);
    circuit.add_S_gate(3);
    circuit.add_CNOT_gate(0, 3);
    circuit.add_CNOT_gate(1, 0);
    gate::Identity(0)->update_quantum_state(&state);
    gate::Identity(1)->update_quantum_state(&state);
    gate::Identity(2)->update_quantum_state(&state);
    gate::Identity(3)->update_quantum_state(&state);
    circuit.add_RY_gate(0, 1.5);
    circuit.add_RY_gate(1, 1.25);
    circuit.add_RY_gate(2, 1.125);
    circuit.add_RY_gate(3, 1.325);
    //circuit.add_RY_gate(nqubits - 2, 0.125);
    //circuit.add_RY_gate(nqubits - 1, 0.125);
    circuit.add_T_gate(0);
    circuit.add_T_gate(1);
    circuit.add_T_gate(2);
    circuit.add_T_gate(3);
    circuit.add_CNOT_gate(1, 3);
    circuit.add_CNOT_gate(2, 0);
    circuit.add_RZ_gate(0, 2.5);
    circuit.add_RZ_gate(1, 2.25);
    circuit.add_RZ_gate(2, 2.125);
    circuit.add_RZ_gate(3, 2.325);
    //circuit.add_RZ_gate(nqubits - 2, 0.125);
    //circuit.add_RZ_gate(nqubits - 1, 0.125);
    circuit.add_CNOT_gate(0, 1);
    circuit.add_CNOT_gate(3, 2);
    circuit.add_Tdag_gate(0);
    circuit.add_Tdag_gate(1);
    circuit.add_Tdag_gate(2);
    circuit.add_Tdag_gate(3);
    circuit.add_CNOT_gate(2, 3);
    circuit.add_CNOT_gate(3, 0);
    circuit.add_Sdag_gate(0);
    circuit.add_Sdag_gate(1);
    circuit.add_Sdag_gate(2);
    circuit.add_Sdag_gate(3);
    circuit.add_CNOT_gate(0, 2);
    circuit.add_CNOT_gate(1, 2);
    //circuit.add_T_gate(nqubits - 2);
    //circuit.add_T_gate(nqubits - 1);
    circuit.add_RX_gate(0, -0.5);
    circuit.add_RX_gate(1, -0.25);
    circuit.add_RX_gate(2, -0.125);
    circuit.add_RX_gate(3, -0.375);
    circuit.add_CNOT_gate(2, 1);
    circuit.add_CNOT_gate(3, 1);
    //circuit.add_CNOT_gate(0, nqubits - 1);
    //circuit.add_CNOT_gate(nqubits - 1, 0);
    //circuit.add_CNOT_gate(nqubits - 2, 1);
    //circuit.add_CNOT_gate(nqubits - 2, nqubits - 1);
    //circuit.add_CNOT_gate(nqubits - 1, nqubits - 2);
    //auto merged_gate = gate::merge(gate::CNOT(0,1),gate::Y(1));
    //auto merged_gate = gate::merge(
    //        gate::Identity(0),
    //        gate::Identity(0));
    //auto merged_gate = gate::merge(gate::X(0),gate::Identity(0));
    //circuit.add_gate(merged_gate);
    //circuit.add_RX_gate(1,0.5);

    circuit.update_quantum_state(&state);
    dt += get_realtime();
    std::cout << "#rank, time: " << _rank << ", " << dt << std::endl << std::flush;

    //delete merged_gate;

    print_state_in_rank_order(&state);
    /*
    QuantumState state_in(nqubits);
    state_in.load(&state);

    if (_rank == 0) std::cout << state_in.to_string() << std::endl;
    */

    MPI_Barrier(MPI_COMM_WORLD);
    sleep(1);
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
