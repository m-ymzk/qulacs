#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>

double get_realtime(void)
{
    struct timespec t;
    //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    clock_gettime(CLOCK_REALTIME, &t);
    return t.tv_sec + (double)t.tv_nsec*1e-9;
}

int main(int argc, char *argv[]){
    double dt;
    int _rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
    int nqubits = atoi(argv[1]);

    dt = -1*get_realtime();
    QuantumState state(nqubits, MPI_COMM_WORLD);
    //std::cout << state.to_string() << std::endl;

    state.set_Haar_random_state(1+_rank);
    //std::cout << state.to_string() << std::endl;

    dt += get_realtime();
    std::cout << "#rank, time: " << _rank << ", " << dt << std::endl;

    MPI_Finalize();
    return 0;

    QuantumCircuit circuit(3);
    circuit.add_X_gate(0);
    auto merged_gate = gate::merge(gate::CNOT(0,1),gate::Y(1));
    circuit.add_gate(merged_gate);
    circuit.add_RX_gate(1,0.5);
    circuit.update_quantum_state(&state);

    Observable observable(3);
    observable.add_operator(2.0, "X 2 Y 1 Z 0");
    observable.add_operator(-3.0, "Z 2");
    auto value = observable.get_expectation_value(&state);
    std::cout << value << std::endl;
    return 0;
}
