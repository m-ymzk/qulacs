#include <gtest/gtest.h>
#include "../util/util.h"

#include <cmath>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/circuit_optimizer.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/utility.hpp>
#include <csim/update_ops.h>
#include <functional>
#include <algorithm>

void _ApplyOptimizer(QuantumCircuit* circuit_ref, int opt_lv, UINT swap_lv, UINT num_exp_outer_swaps) {
    const UINT n = circuit_ref->qubit_count;
    const ITYPE dim = 1ULL << n;
    double eps = _EPS;

    QuantumState state_ref(n);
    QuantumState state(n, 1);

    MPIutil m = get_mpiutil();
    const ITYPE inner_dim = dim >> state.outer_qc;
    const ITYPE offs = inner_dim * m->get_rank();

    {
        state_ref.set_Haar_random_state(2022);
        for (ITYPE i = 0; i < inner_dim; ++i)
            state.data_cpp()[i] = state_ref.data_cpp()[(i + offs) % dim];

        QuantumCircuit* circuit = circuit_ref->copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(circuit, opt_lv, swap_lv);

        circuit->update_quantum_state(&state);
        circuit_ref->update_quantum_state(&state_ref);

#if 0
        if (m->get_rank() == 0) {
            for (auto& gate : circuit->gate_list) {
                auto t_index_list = gate->get_target_index_list();
                auto c_index_list = gate->get_control_index_list();
                std::cout << gate->get_name() << "(t:{" ;
                for (auto idx : t_index_list) {
                    std::cout << idx << ",";
                }
                std::cout << "}, c:{";
                for (auto idx : c_index_list) {
                    std::cout << idx << ",";
                }
                std::cout<< "})" << std::endl;
            }
        }
#endif

        // check if all target qubits are inner except for SWAP and BSWAP gate
        for (auto& gate : circuit->gate_list) {
            if (gate->get_name() == "SWAP" || gate->get_name() == "BSWAP") {
                continue;
            }
            auto t_index_list = gate->get_target_index_list();
            for (auto idx : t_index_list) {
                ASSERT_TRUE(idx < state.inner_qc);
            }
        }

        // check the number of SWAP and BSWAP is the same with expected.
        UINT num_outer_swap_gates = 0;
        for (auto& gate : circuit->gate_list) {
            if (gate->get_name() == "SWAP" || gate->get_name() == "BSWAP") {
                auto t_index_list = gate->get_target_index_list();
                for (auto idx : t_index_list) {
                    if (idx >= state.inner_qc) {
                        num_outer_swap_gates++;
                        break;
                    }
                }
            }
        }
        ASSERT_TRUE(num_outer_swap_gates <= num_exp_outer_swaps)
            << "# outer swaps is " << num_outer_swap_gates << ", but expected is " << num_exp_outer_swaps;


        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(abs(state.data_cpp()[i] -
                            state_ref.data_cpp()[(i + offs) % dim]),
                        0, eps) << "[rank:" << m->get_rank() << "] Optimizer diff at " << i;

        delete circuit;
    }
}

TEST(CircuitTest_multicpu, FSWAPOptimizer_6qubits) {
    UINT n = 6;

    MPIutil m = get_mpiutil();
    const UINT outer_qc = std::log2(m->get_size());
    const UINT inner_qc = n - outer_qc;
    std::cout << "inner_qc="<<inner_qc<<",outer_qc="<<outer_qc<<std::endl;

    Random random;
    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        circuit.add_RZ_gate(0, random.uniform()*3.14159);
        circuit.add_RZ_gate(n-1, random.uniform()*3.14159);
        _ApplyOptimizer(&circuit, 0, 1, 2);
    }

    if(inner_qc >= outer_qc * 2){
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT rep = 0; rep < 2; rep++) {
            for (UINT i = 0; i < n; i++) {
                circuit.add_H_gate(i);
            }
        }
        // TODO gate順序変更に対応したら2回に変更
        _ApplyOptimizer(&circuit, 0, 1, 4);
    }

    if(inner_qc >= outer_qc * 2){
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT rep = 0; rep < 2; rep++) {
            for (UINT i = 0; i < n; i++) {
                circuit.add_CNOT_gate(i, (i+1)%n);
            }
        }
        // TODO gate順序変更に対応したら2回に変更
        _ApplyOptimizer(&circuit, 0, 1, 4);
    }

    if(outer_qc <= n/2){
        random.set_seed(2022);
        QuantumCircuit circuit(n);

        // outer
        for (UINT rep = 0; rep < n*2; rep++) {
            circuit.add_RZ_gate(inner_qc+(rep%outer_qc), random.uniform()*3.14159);
        }
        // inner
        for (UINT rep = 0; rep < n*2; rep++) {
            circuit.add_RZ_gate((rep%inner_qc), random.uniform()*3.14159);
        }
        // outer
        for (UINT rep = 0; rep < n*2; rep++) {
            circuit.add_RZ_gate(inner_qc+(rep%outer_qc), random.uniform()*3.14159);
        }

        _ApplyOptimizer(&circuit, 0, 1, 4);
    }

    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);

        for (UINT rep = 0; rep < n*4; rep++) {
            circuit.add_RZ_gate(((UINT)(random.uniform()*n))%n, random.uniform()*3.14159);
        }

        _ApplyOptimizer(&circuit, 0, 1, n*8);
    }

    if (inner_qc >= 3 && outer_qc >= 3) {
        random.set_seed(2022);
        QuantumCircuit circuit(n);

        circuit.add_H_gate(n-1);
        circuit.add_H_gate(n-3);

        _ApplyOptimizer(&circuit, 0, 1, 2);
    }
}
