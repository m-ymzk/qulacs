#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>

int main(int argc, char *argv[]){
    int nqubits = 20;
    int depth = 9;
    if (argc > 1) nqubits = atoi(argv[1]);
    std::cout << "# argc=" << argc << std::endl;
    std::cout << "# nqubits=" << nqubits << std::endl;

    QuantumState state(nqubits);
    state.set_Haar_random_state();

    //build_curcuit
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
    circuit.update_quantum_state(&state);
    /*
    circuit.add_X_gate(0);
    auto merged_gate = gate::merge(gate::CNOT(0,1),gate::Y(1));
    circuit.add_gate(merged_gate);

    observable.add_operator(2.0, "X 2 Y 1 Z 0");
    observable.add_operator(-3.0, "Z 2");
    Observable observable(nqubits);
    auto value = observable.get_expectation_value(&state);
    std::cout << value << std::endl;
    */
    return 0;
}
