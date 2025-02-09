#ifdef _USE_MPI
#include <gtest/gtest.h>

#include <cppsim/state.hpp>
#include <cppsim/utility.hpp>
#include <csim/MPIutil.hpp>

#include "../util/util.hpp"

TEST(StateTest_multicpu, GenerateAndRelease) {
    UINT n = 10;
    UINT mpirank, mpisize, global_qubit, local_qubit;
    ITYPE part_dim;

    QuantumState state_multicpu(n, true);
    if (state_multicpu.get_device_name() == "multi-cpu") {
        MPIutil mpiutil = get_mpiutil();
        mpirank = mpiutil->get_rank();
        mpisize = mpiutil->get_size();
        global_qubit = std::log2(mpisize);
        local_qubit = n - global_qubit;
        part_dim = (1ULL << n) / mpisize;
    } else {
        mpirank = 0;
        mpisize = 1;
        global_qubit = 0;
        local_qubit = n;
        part_dim = 1ULL << n;
    }

    ASSERT_EQ(state_multicpu.qubit_count, n);
    ASSERT_EQ(state_multicpu.dim, part_dim);
    state_multicpu.set_zero_state();
    for (UINT i = 0; i < state_multicpu.dim; ++i) {
        if (i == 0 && mpirank == 0)
            ASSERT_NEAR(abs(state_multicpu.data_cpp()[i] - 1.), 0, eps);
        else
            ASSERT_NEAR(abs(state_multicpu.data_cpp()[i]), 0, eps);
    }
    Random random;
    for (UINT repeat = 0; repeat < 10; ++repeat) {
        ITYPE basis = random.int64() % state_multicpu.dim;
        state_multicpu.set_computational_basis(basis);
        for (UINT i = 0; i < state_multicpu.dim; ++i) {
            if (i == (basis % (1ULL << local_qubit)) &&
                basis >> local_qubit == mpirank)
                ASSERT_NEAR(abs(state_multicpu.data_cpp()[i] - 1.), 0, eps);
            else
                ASSERT_NEAR(abs(state_multicpu.data_cpp()[i]), 0, eps);
        }
    }
    for (UINT repeat = 0; repeat < 10; ++repeat) {
        state_multicpu.set_Haar_random_state();
        ASSERT_NEAR(state_multicpu.get_squared_norm(), 1., eps);
    }
}

TEST(StateTest_multicpu, SamplingComputationalBasis) {
    const UINT n = 10;
    const UINT nshot = 1024;
    QuantumState state(n, true);
    state.set_computational_basis(100);
    auto res = state.sampling(nshot);
    for (UINT i = 0; i < nshot; ++i) {
        ASSERT_TRUE(res[i] == 100);
    }
}

TEST(StateTest_multicpu, SamplingSuperpositionState) {
    const UINT n = 10;
    const UINT nshot = 1024;
    QuantumState state(n, true);
    state.set_computational_basis(0);
    for (ITYPE i = 1; i <= 4; ++i) {
        QuantumState tmp_state(n);
        tmp_state.set_computational_basis(i);
        state.add_state_with_coef_single_thread(1 << i, &tmp_state);
    }
    state.normalize_single_thread(state.get_squared_norm_single_thread());
    auto res = state.sampling(nshot);
    std::array<UINT, 5> cnt = {};
    for (UINT i = 0; i < nshot; ++i) {
        ASSERT_GE(res[i], 0);
        ASSERT_LE(res[i], 4);
        cnt[res[i]] += 1;
    }
    for (UINT i = 0; i < 4; i++) {
        ASSERT_GT(cnt[i + 1], cnt[i]);
    }
}
#endif
