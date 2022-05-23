
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constant.h"
#include "update_ops.h"
#include "utility.h"

#ifdef _USE_SIMD
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

void P0_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim, 13);
    P0_gate_parallel(target_qubit_index, state, dim);
    omputil->reset_qulacs_num_threads();
#else
    P0_gate_single(target_qubit_index, state, dim);
#endif
}

void P1_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim, 13);
    P1_gate_parallel(target_qubit_index, state, dim);
    omputil->reset_qulacs_num_threads();
#else
    P1_gate_single(target_qubit_index, state, dim);
#endif
}

void P0_gate_single(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE low_mask = mask - 1;
    const ITYPE high_mask = ~low_mask;

    ITYPE state_index;
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE temp_index =
            (state_index & low_mask) + ((state_index & high_mask) << 1) + mask;
        state[temp_index] = 0;
    }
}

void P1_gate_single(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE low_mask = mask - 1;
    const ITYPE high_mask = ~low_mask;

    ITYPE state_index;
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE temp_index =
            (state_index & low_mask) + ((state_index & high_mask) << 1);
        state[temp_index] = 0;
    }
}

#ifdef _OPENMP
void P0_gate_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE low_mask = mask - 1;
    const ITYPE high_mask = ~low_mask;

    ITYPE state_index;
#pragma omp parallel for
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE temp_index =
            (state_index & low_mask) + ((state_index & high_mask) << 1) + mask;
        state[temp_index] = 0;
    }
}

void P1_gate_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE low_mask = mask - 1;
    const ITYPE high_mask = ~low_mask;

    ITYPE state_index;
#pragma omp parallel for
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE temp_index =
            (state_index & low_mask) + ((state_index & high_mask) << 1);
        state[temp_index] = 0;
    }
}
#endif

#ifdef _USE_MPI
void P0_gate_mpi(
    UINT target_qubit_index, CTYPE *state, ITYPE dim, UINT inner_qc) {
    if (target_qubit_index < inner_qc) {
        P0_gate(target_qubit_index, state, dim);
    } else {
        const MPIutil m = get_mpiutil();
        const int rank = m->get_rank();
        const int pair_rank_bit = 1 << (target_qubit_index - inner_qc);
        if ((rank & pair_rank_bit) != 0) {
#ifdef _OPENMP
            OMPutil omputil = get_omputil();
            omputil->set_qulacs_num_threads(dim, 13);
#pragma omp parallel for
#endif
            for (ITYPE iter = 0; iter < dim; ++iter) {
                state[iter] = 0;
            }
#ifdef _OPENMP
            omputil->reset_qulacs_num_threads();
#endif
        }  // else nothing to do.
    }
}

void P1_gate_mpi(
    UINT target_qubit_index, CTYPE *state, ITYPE dim, UINT inner_qc) {
    if (target_qubit_index < inner_qc) {
        P1_gate(target_qubit_index, state, dim);
    } else {
        const MPIutil m = get_mpiutil();
        const int rank = m->get_rank();
        const int pair_rank_bit = 1 << (target_qubit_index - inner_qc);
        if ((rank & pair_rank_bit) == 0) {
#ifdef _OPENMP
            OMPutil omputil = get_omputil();
            omputil->set_qulacs_num_threads(dim, 13);
#pragma omp parallel for
#endif
            for (ITYPE iter = 0; iter < dim; ++iter) {
                state[iter] = 0;
            }
#ifdef _OPENMP
            omputil->reset_qulacs_num_threads();
#endif
        }  // else nothing to do.
    }
}
#endif
