#include <gtest/gtest.h>

//#define _USE_MATH_DEFINES
//#include <cmath>
#include <csim/constant.h>

#include <cppsim/circuit.hpp>
#include <cppsim/circuit_optimizer.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/state.hpp>
#include <cppsim/type.hpp>
#include <cppsim/utility.hpp>
#include <unsupported/Eigen/MatrixFunctions>
#include <utility>

#include "../util/util.h"

TEST(CircuitTest_multicpu, CircuitBasic) {
    Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2), H(2, 2),
        S(2, 2), T(2, 2), sqrtX(2, 2), sqrtY(2, 2), P0(2, 2), P1(2, 2);

    Identity << 1, 0, 0, 1;
    X << 0, 1, 1, 0;
    Y << 0, -1.i, 1.i, 0;
    Z << 1, 0, 0, -1;
    H << 1, 1, 1, -1;
    H /= sqrt(2.);
    S << 1, 0, 0, 1.i;
    T << 1, 0, 0, (1. + 1.i) / sqrt(2.);
    sqrtX << 0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i;
    sqrtY << 0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i;
    P0 << 1, 0, 0, 0;
    P1 << 0, 0, 0, 1;

    const UINT n = 4;
    const UINT dim = 1ULL << n;
    double eps = 1e-14;
    Random random;
    random.set_seed(2022);

    QuantumState state_ref(n, 0);
    QuantumState state(n, 1);
    ComplexVector state_eigen(dim);

    MPIutil m = get_mpiutil();
    const ITYPE inner_dim = dim >> state.outer_qc;
    UINT offs = 0;
    if (state.outer_qc > 0) offs = inner_dim * m->get_rank();
    // std::cout << "#test_circuit_multicpu " << m->get_rank() << ": " << dim <<
    // ", " << inner_dim << ", " << offs << std::endl;

    state_ref.set_Haar_random_state(2022);
    for (ITYPE i = 0; i < dim; ++i) state_eigen[i] = state_ref.data_cpp()[i];
    for (ITYPE i = 0; i < inner_dim; ++i)
        state.data_cpp()[i] = state_ref.data_cpp()[i + offs];

    QuantumCircuit circuit(n);
    UINT target, target_sub;
    double angle;

    target = random.int32() % n;
    circuit.add_X_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(
                      target, get_eigen_matrix_single_Pauli(1), n) *
                  state_eigen;

    target = random.int32() % n;
    circuit.add_Y_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(
                      target, get_eigen_matrix_single_Pauli(2), n) *
                  state_eigen;

    target = random.int32() % n;
    circuit.add_Z_gate(target);
    state_eigen = get_expanded_eigen_matrix_with_identity(
                      target, get_eigen_matrix_single_Pauli(3), n) *
                  state_eigen;

    target = random.int32() % n;
    circuit.add_H_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, H, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_S_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, S, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_Sdag_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, S.adjoint(), n) *
        state_eigen;

    target = random.int32() % n;
    circuit.add_T_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, T, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_Tdag_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, T.adjoint(), n) *
        state_eigen;

    target = random.int32() % n;
    circuit.add_sqrtX_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, sqrtX, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_sqrtXdag_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, sqrtX.adjoint(), n) *
        state_eigen;

    target = random.int32() % n;
    circuit.add_sqrtY_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, sqrtY, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_sqrtYdag_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, sqrtY.adjoint(), n) *
        state_eigen;

    target = random.int32() % n;
    circuit.add_P0_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, P0, n) * state_eigen;

    target = (target + 1) % n;
    circuit.add_P1_gate(target);
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, P1, n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_RX_gate(target, angle);
    state_eigen = get_expanded_eigen_matrix_with_identity(target,
                      cos(angle / 2) * Identity + 1.i * sin(angle / 2) * X, n) *
                  state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_RY_gate(target, angle);
    state_eigen = get_expanded_eigen_matrix_with_identity(target,
                      cos(angle / 2) * Identity + 1.i * sin(angle / 2) * Y, n) *
                  state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_RZ_gate(target, angle);
    state_eigen = get_expanded_eigen_matrix_with_identity(target,
                      cos(angle / 2) * Identity + 1.i * sin(angle / 2) * Z, n) *
                  state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_CNOT_gate(target, target_sub);
    state_eigen =
        get_eigen_matrix_full_qubit_CNOT(target, target_sub, n) * state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_CZ_gate(target, target_sub);
    state_eigen =
        get_eigen_matrix_full_qubit_CZ(target, target_sub, n) * state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_SWAP_gate(target, target_sub);
    state_eigen =
        get_eigen_matrix_full_qubit_SWAP(target, target_sub, n) * state_eigen;

    circuit.update_quantum_state(&state);
    for (ITYPE i = 0; i < inner_dim; ++i)
        ASSERT_NEAR(abs(state_eigen[i + offs] - state.data_cpp()[i]), 0, eps)
            << ", i=" << i << " rank=" << m->get_rank();
}

TEST(CircuitTest_multicpu, CircuitOptimize) {
    const UINT n = 4;
    const UINT dim = 1ULL << n;
    double eps = 1e-14;

    MPIutil m = get_mpiutil();
    QuantumState dummy_state(n, 1);
    const ITYPE inner_dim = dim >> dummy_state.outer_qc;
    const ITYPE offs = (dummy_state.outer_qc != 0) * inner_dim * m->get_rank();

#if 0  // block_size > 1
    {
        // merge successive gates
        QuantumState state(n, 0), test_state(n, 1);
        state.set_Haar_random_state(2022);
        //test_state.set_Haar_random_state(2022);
        //for (ITYPE i = 0; i < inner_dim; ++i) state.data_cpp()[i] = test_state.data_cpp()[i + offs];
        test_state.load(&state);
        //state.load(&test_state);
        QuantumCircuit circuit(n);

        circuit.add_X_gate(0);
        circuit.add_Y_gate(0);
        UINT block_size = 2;
        UINT expected_depth = 1;
        UINT expected_gate_count = 1;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit,block_size);
        //circuit.update_quantum_state(&test_state);
        //copy_circuit->update_quantum_state(&state);
        circuit.update_quantum_state(&state);
        copy_circuit->update_quantum_state(&test_state);
        //std::cout << circuit << std::endl << copy_circuit << std::endl;
        //for (UINT i = 0; i < inner_dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i + offs] - test_state.data_cpp()[i]), 0, eps);
        for (UINT i = 0; i < inner_dim; ++i) ASSERT_NEAR(real(state.data_cpp()[i]), real(test_state.data_cpp()[i + offs]), eps) << ", i=" << i << " rank=" << m->get_rank();
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // tensor product, merged
        QuantumState state(n, 0), test_state(n, 1);
        state.set_Haar_random_state(2022);
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_X_gate(0);
        circuit.add_Y_gate(1);
        UINT block_size = 2;
        UINT expected_depth = 1;
        UINT expected_gate_count = 1;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        //std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < inner_dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i + offs] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }
#endif

    {
        // do not take tensor product
        QuantumState state(n, 0), test_state(n, 1);
        state.set_Haar_random_state(2022);
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_X_gate(0);
        circuit.add_Y_gate(1);
        UINT block_size = 1;
        UINT expected_depth = 1;
        UINT expected_gate_count = 2;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        // std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i + offs] - test_state.data_cpp()[i]), 0,
                eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, control does not commute with X
        QuantumState state(n, 0), test_state(n, 1);
        state.set_Haar_random_state(2022);
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_X_gate(0);
        circuit.add_CNOT_gate(0, 1);
        circuit.add_Y_gate(0);
        UINT block_size = 1;
        UINT expected_depth = 3;
        UINT expected_gate_count = 3;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        // std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i + offs] - test_state.data_cpp()[i]), 0,
                eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, control does not commute with Z
        QuantumState state(n, 0), test_state(n, 1);
        state.set_Haar_random_state(2022);
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_X_gate(0);
        circuit.add_CNOT_gate(0, 1);
        circuit.add_Z_gate(0);
        UINT block_size = 1;
        UINT expected_depth = 2;
        UINT expected_gate_count = 2;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        // std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i + offs] - test_state.data_cpp()[i]), 0,
                eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, control commute with Z
        QuantumState state(n, 0), test_state(n, 1);
        state.set_Haar_random_state(2022);
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_Z_gate(0);
        circuit.add_CNOT_gate(0, 1);
        circuit.add_Z_gate(0);
        UINT block_size = 1;
        UINT expected_depth = 2;
        UINT expected_gate_count = 2;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        // std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i + offs] - test_state.data_cpp()[i]), 0,
                eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, target commute with X
        QuantumState state(n, 0), test_state(n, 1);
        state.set_Haar_random_state(2022);
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_X_gate(1);
        circuit.add_CNOT_gate(0, 1);
        circuit.add_X_gate(1);
        UINT block_size = 1;
        UINT expected_depth = 2;
        UINT expected_gate_count = 2;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        // std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i + offs] - test_state.data_cpp()[i]), 0,
                eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, target commute with X
        QuantumState state(n, 0), test_state(n, 1);
        state.set_Haar_random_state(2022);
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_Z_gate(1);
        circuit.add_CNOT_gate(0, 1);
        circuit.add_X_gate(1);
        UINT block_size = 1;
        UINT expected_depth = 2;
        UINT expected_gate_count = 2;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        // std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i + offs] - test_state.data_cpp()[i]), 0,
                eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, target commute with X
        QuantumState state(n, 0), test_state(n, 1);
        state.set_Haar_random_state(2022);
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_X_gate(1);
        circuit.add_CNOT_gate(0, 1);
        circuit.add_Z_gate(1);
        UINT block_size = 1;
        UINT expected_depth = 2;
        UINT expected_gate_count = 2;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        // std::cout << circuit << std::endl << copy_circuit << std::endl;
        // std::cout << state << std::endl << test_state << std::endl;
        for (UINT i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i + offs] - test_state.data_cpp()[i]), 0,
                eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, target commute with X
        QuantumState state(n, 0), test_state(n, 1);
        state.set_Haar_random_state(2022);
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_Z_gate(1);
        circuit.add_CNOT_gate(0, 1);
        circuit.add_Z_gate(1);
        UINT block_size = 1;
        UINT expected_depth = 3;
        UINT expected_gate_count = 3;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        // std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(
                abs(state.data_cpp()[i + offs] - test_state.data_cpp()[i]), 0,
                eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

#if 0  // block_size > 1
    {
        // CNOT, target commute with X
        QuantumState state(n, 0), test_state(n, 1);
        state.set_Haar_random_state(2022);
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_Z_gate(1);
        circuit.add_CNOT_gate(0, 1);
        circuit.add_Z_gate(1);
        UINT block_size = 2;
        UINT expected_depth = 1;
        UINT expected_gate_count = 1;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        //std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < inner_dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i + offs] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, target commute with X
        QuantumState state(n, 0), test_state(n, 1);
        state.set_Haar_random_state(2022);
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_Z_gate(0);
        circuit.add_gate( gate::merge(gate::CNOT(0,1), gate::Y(2)));
        circuit.add_gate( gate::merge(gate::CNOT(1,0), gate::Y(2)));
        circuit.add_Z_gate(1);
        UINT block_size = 2;
        UINT expected_depth = 3;
        UINT expected_gate_count = 3;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        //std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < inner_dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i + offs] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }

    {
        // CNOT, target commute with X
        QuantumState state(n, 0), test_state(n, 1);
        state.set_Haar_random_state(2022);
        test_state.load(&state);
        QuantumCircuit circuit(n);

        circuit.add_Z_gate(0);
        circuit.add_gate(gate::merge(gate::CNOT(0, 1), gate::Y(2)));
        circuit.add_gate(gate::merge(gate::CNOT(1, 0), gate::Y(2)));
        circuit.add_Z_gate(1);
        UINT block_size = 3;
        UINT expected_depth = 1;
        UINT expected_gate_count = 1;

        QuantumCircuit* copy_circuit = circuit.copy();
        QuantumCircuitOptimizer qco;
        qco.optimize(copy_circuit, block_size);
        circuit.update_quantum_state(&test_state);
        copy_circuit->update_quantum_state(&state);
        //std::cout << circuit << std::endl << copy_circuit << std::endl;
        for (UINT i = 0; i < inner_dim; ++i) ASSERT_NEAR(abs(state.data_cpp()[i + offs] - test_state.data_cpp()[i]), 0, eps);
        ASSERT_EQ(copy_circuit->calculate_depth(), expected_depth);
        ASSERT_EQ(copy_circuit->gate_list.size(), expected_gate_count);
        delete copy_circuit;
    }
#endif
}

TEST(CircuitTest_multicpu, RandomCircuitOptimize) {
    const UINT n = 5;
    const UINT dim = 1ULL << n;
    const UINT depth = 5;
    Random random;
    random.set_seed(2022);
    double eps = 1e-14;
    UINT max_repeat = 3;
    UINT max_block_size =
        1;  // The maximum block size is one when using multi-cpu.

    MPIutil m = get_mpiutil();
    QuantumState dummy_state(n, 1);
    const ITYPE inner_dim = dim >> dummy_state.outer_qc;
    const ITYPE offs = (dummy_state.outer_qc != 0) * inner_dim * m->get_rank();

    for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
        QuantumState state(n, 1), org_state(n, 1), test_state(n, 1);
        state.set_Haar_random_state(2022);
        org_state.load(&state);
        QuantumCircuit circuit(n);

        for (UINT d = 0; d < depth; ++d) {
            for (UINT i = 0; i < n; ++i) {
                UINT r = random.int32() % 5;
                if (r == 0)
                    circuit.add_sqrtX_gate(i);
                else if (r == 1)
                    circuit.add_sqrtY_gate(i);
                else if (r == 2)
                    circuit.add_T_gate(i);
                else if (r == 3) {
                    if (i + 1 < n) circuit.add_CNOT_gate(i, i + 1);
                } else if (r == 4) {
                    if (i + 1 < n) circuit.add_CZ_gate(i, i + 1);
                }
            }
        }

        test_state.load(&org_state);
        circuit.update_quantum_state(&test_state);
        // std::cout << circuit << std::endl;
        QuantumCircuitOptimizer qco;
        for (UINT block_size = 1; block_size <= max_block_size; ++block_size) {
            QuantumCircuit* copy_circuit = circuit.copy();
            qco.optimize(copy_circuit, block_size);
            state.load(&org_state);  // TODO cpu to multicpu function
            copy_circuit->update_quantum_state(&state);
            // std::cout << copy_circuit << std::endl;
            // for (UINT i = 0; i < inner_dim; ++i)
            // ASSERT_NEAR(abs(state.data_cpp()[i + offs] -
            // test_state.data_cpp()[i]), 0, eps);
            for (UINT i = 0; i < inner_dim; ++i)
                ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]),
                    0, eps);
            delete copy_circuit;
        }
    }
}

TEST(CircuitTest_multicpu, RandomCircuitOptimize2) {
    const UINT n = 5;
    const UINT dim = 1ULL << n;
    const UINT depth = 10;
    Random random;
    random.set_seed(2022);
    double eps = 1e-14;
    UINT max_repeat = 3;
    UINT max_block_size =
        1;  // The maximum block size is one when using multi-cpu.

    MPIutil m = get_mpiutil();
    QuantumState dummy_state(n, 1);
    const ITYPE inner_dim = dim >> dummy_state.outer_qc;
    const ITYPE offs = (dummy_state.outer_qc != 0) * inner_dim * m->get_rank();

    for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
        QuantumState state(n, 1), org_state(n, 1), test_state(n, 1);
        state.set_Haar_random_state(2022);
        org_state.load(&state);
        QuantumCircuit circuit(n);

        for (UINT d = 0; d < depth; ++d) {
            for (UINT i = 0; i < n; ++i) {
                UINT r = random.int32() % 6;
                if (r == 0)
                    circuit.add_sqrtX_gate(i);
                else if (r == 1)
                    circuit.add_sqrtY_gate(i);
                else if (r == 2)
                    circuit.add_T_gate(i);
                else if (r == 3) {
                    UINT r2 = random.int32() % n;
                    if (r2 == i) r2 = (r2 + 1) % n;
                    if (i + 1 < n) circuit.add_CNOT_gate(i, r2);
                } else if (r == 4) {
                    UINT r2 = random.int32() % n;
                    if (r2 == i) r2 = (r2 + 1) % n;
                    if (i + 1 < n) circuit.add_CZ_gate(i, r2);
                } else if (r == 5) {
                    UINT r2 = random.int32() % n;
                    if (r2 == i) r2 = (r2 + 1) % n;
                    if (i + 1 < n) circuit.add_SWAP_gate(i, r2);
                }
            }
        }

        test_state.load(&org_state);
        circuit.update_quantum_state(&test_state);
        // std::cout << circuit << std::endl;
        QuantumCircuitOptimizer qco;
        for (UINT block_size = 1; block_size <= max_block_size; ++block_size) {
            QuantumCircuit* copy_circuit = circuit.copy();
            qco.optimize(copy_circuit, block_size);
            state.load(&org_state);
            copy_circuit->update_quantum_state(&state);
            // std::cout << copy_circuit << std::endl;
            // for (UINT i = 0; i < inner_dim; ++i)
            // ASSERT_NEAR(abs(state.data_cpp()[i + offs] -
            // test_state.data_cpp()[i]), 0, eps);
            for (UINT i = 0; i < inner_dim; ++i)
                ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]),
                    0, eps);
            delete copy_circuit;
        }
    }
}

/* This test uses multi_qubit_dense_matrix_gate.
TEST(CircuitTest_multicpu, RandomCircuitOptimize3) {
        const UINT n = 5;
        const UINT dim = 1ULL << n;
        const UINT depth = 10*n;
        Random random;
        random.set_seed(2022);
        double eps = 1e-14;
        UINT max_repeat = 3;
    UINT max_block_size = 1; // The maximum block size is one when using
multi-cpu.

        MPIutil m = get_mpiutil();
        QuantumState dummy_state(n, 1);
        const ITYPE inner_dim = dim >> dummy_state.outer_qc;
        const ITYPE offs = (dummy_state.outer_qc != 0) * inner_dim *
m->get_rank();

        std::vector<UINT> qubit_list;
        for (int i = 0; i < n; ++i) qubit_list.push_back(i);

        for (UINT repeat = 0; repeat < max_repeat; ++repeat) {
                QuantumState state(n, 1), org_state(n, 1), test_state(n, 1);
                state.set_Haar_random_state(2022);
                org_state.load(&state);
                QuantumCircuit circuit(n);

                for (UINT d = 0; d < depth; ++d) {
                        std::random_shuffle(qubit_list.begin(),
qubit_list.end()); std::vector<UINT> mylist; mylist.push_back(qubit_list[0]);
                        mylist.push_back(qubit_list[1]);
                        circuit.add_random_unitary_gate(mylist);
                }

                test_state.load(&org_state);
                circuit.update_quantum_state(&test_state);
                //std::cout << circuit << std::endl;
                QuantumCircuitOptimizer qco;
                for (UINT block_size = 1; block_size <= max_block_size;
++block_size) { QuantumCircuit* copy_circuit = circuit.copy();
                        qco.optimize(copy_circuit, block_size);
                        state.load(&org_state);
                        copy_circuit->update_quantum_state(&state);
                        //std::cout << copy_circuit << std::endl;
            //for (UINT i = 0; i < inner_dim; ++i)
ASSERT_NEAR(abs(state.data_cpp()[i + offs] - test_state.data_cpp()[i]), 0, eps);
                        for (UINT i = 0; i < inner_dim; ++i)
ASSERT_NEAR(abs(state.data_cpp()[i] - test_state.data_cpp()[i]), 0, eps); delete
copy_circuit;
                }
        }
}
*/

TEST(CircuitTest_multicpu, SimpleExpansionZ_6qubit) {
    const UINT n = 6;
    UINT num_repeats;
    const UINT dim = 1ULL << n;
    const double eps = 1e-14;

    double angle;
    std::vector<double> coef;

    const UINT seed = 2022;
    Random random;
    random.set_seed(seed);

    CPPCTYPE res;
    CPPCTYPE res_ref;

    Observable observable(n);

    QuantumState state(n, true);
    QuantumState state_ref(n, false);
    const ITYPE inner_dim = dim >> state.outer_qc;

    for (ITYPE i = 0; i < n; ++i) {
        coef.push_back(-random.uniform());
        // coef.push_back(-1.);
    }

    // Z only
    observable.add_operator(coef[0], "Z 0 I 1");
    observable.add_operator(coef[1], "Z 1 I 0");
    observable.add_operator(coef[2], "Z 2 I 1");
    observable.add_operator(coef[3], "Z 3 I 1");
    observable.add_operator(coef[4], "Z 4 I 1");
    observable.add_operator(coef[5], "Z 5 I 1");

    state_ref.set_Haar_random_state(2022);
    state.load(&state_ref);

    res = observable.get_expectation_value(&state);
    res_ref = observable.get_expectation_value(&state_ref);

    ASSERT_NEAR(abs(res_ref.real() - res.real()) / res_ref.real(), 0, eps)
    << "ref(real): " << res_ref.real() << " value(real): " << res.real() << std::endl;
}

TEST(CircuitTest_multicpu, SimpleExpansionXYZ_6qubit) {
    const UINT n = 6;
    UINT num_repeats;
    const UINT dim = 1ULL << n;
    const double eps = 1e-14;

    double angle;
    std::vector<double> coef;

    const UINT seed = 2022;
    Random random;
    random.set_seed(seed);

    CPPCTYPE res;
    CPPCTYPE res_ref;

    Observable observable(n);

    QuantumState state(n, true);
    QuantumState state_ref(n, false);
    const ITYPE inner_dim = dim >> state.outer_qc;

    for (ITYPE i = 0; i < n; ++i) {
        coef.push_back(-random.uniform());
        // coef.push_back(-1.);
    }

    observable.add_operator(coef[0], "Z 0 Y 1");
    observable.add_operator(coef[1], "Z 1 I 0");
    observable.add_operator(coef[2], "X 2 Y 0");
    observable.add_operator(coef[3], "Z 3 I 1");
    observable.add_operator(coef[4], "Z 4 I 1");
    observable.add_operator(coef[5], "X 5 I 1");

    state_ref.set_Haar_random_state(2022);
    state.load(&state_ref);

    res = observable.get_expectation_value(&state);
    res_ref = observable.get_expectation_value(&state_ref);

    ASSERT_NEAR(abs(res_ref.real() - res.real()) / res_ref.real(), 0, eps);
}


TEST(CircuitTest_multicpu, SpecialGatesToString) {
    QuantumState state(1, 1);
    QuantumCircuit c(1);
    c.add_gate(gate::DepolarizingNoise(0, 0));
    c.update_quantum_state(&state);
    std::string s = c.to_string();
    std::cout << "#s =" << std::endl << s;
}

void _ApplyOptimizer(QuantumCircuit* circuit_ref, int opt_lv, UINT swap_lv,
    UINT num_exp_outer_swaps) {
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
        if (opt_lv >= 0) {
            qco.optimize(circuit, opt_lv, swap_lv);
        } else {
            qco.optimize_light(circuit, swap_lv);
        }

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

        // check if all target qubits are inner except for SWAP, FusedSWAP, CZ
        // gate
        for (auto& gate : circuit->gate_list) {
            auto gate_name = gate->get_name();
            if (gate_name == "SWAP" || gate_name == "FusedSWAP" ||
                gate_name == "I" || gate_name == "Z" ||
                gate_name == "Z-rotation" || gate_name == "CZ" ||
                gate_name == "Projection-0" || gate_name == "Projection-1" ||
                gate_name == "S" || gate_name == "Sdag" || gate_name == "T" ||
                gate_name == "Tdag" || gate_name == "DiagonalMatrix") {
                continue;
            }
            auto t_index_list = gate->get_target_index_list();
            for (auto idx : t_index_list) {
                ASSERT_TRUE(idx < state.inner_qc);
            }
        }

        // check the number of SWAP and FusedSWAP is the same with expected.
        UINT num_outer_swap_gates = 0;
        for (auto& gate : circuit->gate_list) {
            if (gate->get_name() == "SWAP" || gate->get_name() == "FusedSWAP") {
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
            << "# outer swaps is " << num_outer_swap_gates
            << ", but expected is " << num_exp_outer_swaps;

        for (ITYPE i = 0; i < inner_dim; ++i)
            ASSERT_NEAR(abs(state.data_cpp()[i] -
                            state_ref.data_cpp()[(i + offs) % dim]),
                0, eps)
                << "[rank:" << m->get_rank() << "] Optimizer diff at " << i;

        delete circuit;
    }
}

TEST(CircuitTest_multicpu, FSWAPOptimizer_6qubits) {
    UINT n = 6;

    MPIutil m = get_mpiutil();
    const UINT outer_qc = std::log2(m->get_size());
    const UINT inner_qc = n - outer_qc;

    if (outer_qc < 1) {
        return;
    }

    Random random;
    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        circuit.add_RX_gate(0, random.uniform() * 3.14159);
        circuit.add_RX_gate(n - 1, random.uniform() * 3.14159);
        _ApplyOptimizer(&circuit, 0, 1, 2);
    }

    if (inner_qc >= outer_qc * 2) {
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

    if (inner_qc >= outer_qc * 2) {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT rep = 0; rep < 2; rep++) {
            for (UINT i = 0; i < n; i++) {
                circuit.add_CNOT_gate(i, (i + 1) % n);
            }
        }
        // TODO gate順序変更に対応したら2回に変更
        _ApplyOptimizer(&circuit, 0, 1, 4);
    }

    if (outer_qc <= n / 2) {
        random.set_seed(2022);
        QuantumCircuit circuit(n);

        // outer
        for (UINT rep = 0; rep < n * 2; rep++) {
            circuit.add_RX_gate(
                inner_qc + (rep % outer_qc), random.uniform() * 3.14159);
        }
        // inner
        for (UINT rep = 0; rep < n * 2; rep++) {
            circuit.add_RX_gate((rep % inner_qc), random.uniform() * 3.14159);
        }
        // outer
        for (UINT rep = 0; rep < n * 2; rep++) {
            circuit.add_RX_gate(
                inner_qc + (rep % outer_qc), random.uniform() * 3.14159);
        }

        _ApplyOptimizer(&circuit, 0, 1, 4);
    }

    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);

        for (UINT rep = 0; rep < n * 4; rep++) {
            circuit.add_RX_gate(
                ((UINT)(random.uniform() * n)) % n, random.uniform() * 3.14159);
        }

        _ApplyOptimizer(&circuit, 0, 1, n * 8);
    }

    if (inner_qc >= 3 && outer_qc >= 3) {
        random.set_seed(2022);
        QuantumCircuit circuit(n);

        circuit.add_H_gate(n - 1);
        circuit.add_H_gate(n - 3);

        _ApplyOptimizer(&circuit, 0, 1, 2);
    }
}

TEST(CircuitTest_multicpu, FSWAPOptimizer_nocomm_6qubits) {
    UINT n = 6;

    MPIutil m = get_mpiutil();
    const UINT outer_qc = std::log2(m->get_size());
    const UINT inner_qc = n - outer_qc;

    if (outer_qc < 1) {
        return;
    }

    Random random;

    // X, Yゲートはouter qubitを使わなければFSWAP不要
    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT i = 0; i < inner_qc; i++) {
            circuit.add_X_gate(i);
            circuit.add_Y_gate(i);
        }
        _ApplyOptimizer(&circuit, 0, 1, 0);
    }
    // CNOTゲートはtargetでouter qubitを使わなければFSWAP不要
    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT i = 0; i < n; i++) {
            if ((i + 1) % n < inner_qc) {
                circuit.add_CNOT_gate(i, (i + 1) % n);
            }
        }
        _ApplyOptimizer(&circuit, 0, 1, 0);
    }
    // Hゲートはouter qubitを使わなければFSWAP不要
    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT i = 0; i < inner_qc; i++) {
            circuit.add_H_gate(i);
        }
        _ApplyOptimizer(&circuit, 0, 1, 0);
    }
    // RX, RYゲートはouter qubitを使わなければFSWAP不要
    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT i = 0; i < inner_qc; i++) {
            circuit.add_RX_gate(i, random.uniform() * 3.14159);
            circuit.add_RY_gate(i, random.uniform() * 3.14159);
        }
        _ApplyOptimizer(&circuit, 0, 1, 0);
    }
    // SqrtX, SqrtXdag, SqrtY, SqrtYdagゲートはtargetでouter
    // qubitを使わなければFSWAP不要
    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT i = 0; i < inner_qc; i++) {
            circuit.add_sqrtX_gate(i);
            circuit.add_sqrtXdag_gate(i);
            circuit.add_sqrtY_gate(i);
            circuit.add_sqrtYdag_gate(i);
        }
        _ApplyOptimizer(&circuit, 0, 1, 0);
    }
    // U1, U2, U3ゲートはtargetでouter qubitを使わなければFSWAP不要
    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT i = 0; i < inner_qc; i++) {
            circuit.add_U1_gate(i, random.uniform() * 3.14159);
            circuit.add_U2_gate(
                i, random.uniform() * 3.14159, random.uniform() * 3.14159);
            circuit.add_U3_gate(i, random.uniform() * 3.14159,
                random.uniform() * 3.14159, random.uniform() * 3.14159);
        }
        _ApplyOptimizer(&circuit, 0, 1, 0);
    }
    // DenseMatrixゲートはtargetでouter qubitを使わなければFSWAP不要
    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT i = 0; i < inner_qc; i++) {
            std::vector<UINT> target{i};
            ComplexMatrix mat = get_eigen_matrix_random_single_qubit_unitary();
            auto DenseMatrix_gate = gate::DenseMatrix(target, mat);
            circuit.add_gate(DenseMatrix_gate);
        }
        _ApplyOptimizer(&circuit, 0, 1, 0);
    }

    // Zゲートは全てのパターンでFSWAP不要
    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT i = 0; i < n; i++) {
            circuit.add_Z_gate(i);
        }
        _ApplyOptimizer(&circuit, 0, 1, 0);
    }
    // CZゲートは全てのパターンでFSWAP不要
    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT i = 0; i < n; i++) {
            circuit.add_CZ_gate(i, (i + 1) % n);
        }
        _ApplyOptimizer(&circuit, 0, 1, 0);
    }
    // Iゲートは全てのパターンでFSWAP不要
    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT i = 0; i < n; i++) {
            auto I_gate = gate::Identity(i);
            circuit.add_gate(I_gate);
        }
        _ApplyOptimizer(&circuit, 0, 1, 0);
    }
    // P0, P1ゲートは全てのパターンでFSWAP不要
    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT i = 0; i < n; i++) {
            circuit.add_P0_gate(i);
            circuit.add_P1_gate(i);
        }
        _ApplyOptimizer(&circuit, 0, 1, 0);
    }
    // RZゲートは全てのパターンでFSWAP不要
    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT i = 0; i < n; i++) {
            circuit.add_RZ_gate(i, random.uniform() * 3.14159);
        }
        _ApplyOptimizer(&circuit, 0, 1, 0);
    }
    // S, Sdag, T, Tdagゲートは全てのパターンでFSWAP不要
    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT i = 0; i < n; i++) {
            circuit.add_S_gate(i);
            circuit.add_Sdag_gate(i);
            circuit.add_T_gate(i);
            circuit.add_Tdag_gate(i);
        }
        _ApplyOptimizer(&circuit, 0, 1, 0);
    }
    // DiagonalMatrixゲートは全てのパターンでFSWAP不要
    {
        random.set_seed(2022);
        QuantumCircuit circuit(n);
        for (UINT i = 0; i < n; i++) {
            std::vector<UINT> target{i};
            ComplexVector diag =
                get_eigen_diagonal_matrix_random_multi_qubit_unitary(1);
            auto DiagonalMatrix_gate = gate::DiagonalMatrix(target, diag);
            circuit.add_gate(DiagonalMatrix_gate);
        }
        _ApplyOptimizer(&circuit, 0, 1, 0);
    }
}

// TEST(CircuitTest_multicpu, FSWAPOptimizerLight_6qubits) {
//    UINT n = 6;
//
//    MPIutil m = get_mpiutil();
//    const UINT outer_qc = std::log2(m->get_size());
//    const UINT inner_qc = n - outer_qc;
//
//    if (outer_qc < 2) {
//        return;
//    }
//
//    Random random;
//    {
//        random.set_seed(2022);
//        QuantumCircuit circuit(n);
//        circuit.add_RX_gate(0, random.uniform()*3.14159);
//        circuit.add_RX_gate(n-1, random.uniform()*3.14159);
//        _ApplyOptimizer(&circuit, -1, 1, 2);
//    }
//
//    if(inner_qc >= outer_qc * 2){
//        random.set_seed(2022);
//        QuantumCircuit circuit(n);
//        for (UINT rep = 0; rep < 2; rep++) {
//            for (UINT i = 0; i < n; i++) {
//                circuit.add_H_gate(i);
//            }
//        }
//        // TODO gate順序変更に対応したら2回に変更
//        _ApplyOptimizer(&circuit, -1, 1, 4);
//    }
//}
