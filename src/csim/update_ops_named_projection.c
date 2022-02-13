
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constant.h"
#include "update_ops.h"
#include "utility.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _USE_MPI
#include "MPIutil.h"
#endif

#ifdef _USE_SIMD
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

void P0_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
#ifdef _OPENMP
    UINT threshold = 13;
    if (dim < (((ITYPE)1) << threshold)) {
        P0_gate_single(target_qubit_index, state, dim);
    } else {
        P0_gate_parallel(target_qubit_index, state, dim);
    }
#else
    P0_gate_single(target_qubit_index, state, dim);
#endif
}

void P1_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
#ifdef _OPENMP
    UINT threshold = 13;
    if (dim < (((ITYPE)1) << threshold)) {
        P1_gate_single(target_qubit_index, state, dim);
    } else {
        P1_gate_parallel(target_qubit_index, state, dim);
    }
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
        const UINT threshold = 13;
        if ((rank & pair_rank_bit) != 0) {
            if (dim < (((ITYPE)1) << threshold)) {
                for (ITYPE iter = 0; iter < dim; ++iter) {
                    state[iter] = 0;
                }
            } else {
#pragma omp parallel for
                for (ITYPE iter = 0; iter < dim; ++iter) {
                    state[iter] = 0;
                }
            }
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
        const UINT threshold = 13;
        if ((rank & pair_rank_bit) == 0) {
            if (dim < (((ITYPE)1) << threshold)) {
                for (ITYPE iter = 0; iter < dim; ++iter) {
                    state[iter] = 0;
                }
            } else {
#pragma omp parallel for
                for (ITYPE iter = 0; iter < dim; ++iter) {
                    state[iter] = 0;
                }
            }
        }  // else nothing to do.
    }
}
#endif
