
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constant.h"
#include "update_ops.h"
#include "utility.h"
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

void single_qubit_phase_gate(
    UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim) {
    // single_qubit_phase_gate_single(target_qubit_index, phase, state, dim);
    // single_qubit_phase_gate_single_unroll(target_qubit_index, phase, state,
    // dim);
    // single_qubit_phase_gate_single_simd(target_qubit_index, phase, state,
    // dim); single_qubit_phase_gate_parallel_simd(target_qubit_index, phase,
    // state, dim);

#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim, 12);
#endif

#ifdef _USE_SIMD
#ifdef _OPENMP
    single_qubit_phase_gate_parallel_simd(
        target_qubit_index, phase, state, dim);
#else
    single_qubit_phase_gate_single_simd(target_qubit_index, phase, state, dim);
#endif
#else
#ifdef _OPENMP
    single_qubit_phase_gate_parallel_unroll(
        target_qubit_index, phase, state, dim);
#else
    single_qubit_phase_gate_single_unroll(
        target_qubit_index, phase, state, dim);
#endif
#endif

#ifdef _OPENMP
    omputil->reset_qulacs_num_threads();
#endif
}

void single_qubit_phase_gate_single_unroll(
    UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim) {
    // target tmask
    const ITYPE mask = 1ULL << target_qubit_index;
    const ITYPE low_mask = mask - 1;
    const ITYPE high_mask = ~low_mask;

    // loop varaibles
    const ITYPE loop_dim = dim / 2;
    if (target_qubit_index == IS_OUTER_QB) {
        ITYPE state_index;
        for (state_index = 0; state_index < dim; state_index++) {
            state[state_index] *= phase;
        }
    } else if (target_qubit_index == 0) {
        ITYPE state_index;
        for (state_index = 1; state_index < dim; state_index += 2) {
            state[state_index] *= phase;
        }
    } else {
        ITYPE state_index;
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis = (state_index & low_mask) +
                          ((state_index & high_mask) << 1) + mask;
            state[basis] *= phase;
            state[basis + 1] *= phase;
        }
    }
}

#ifdef _OPENMP
void single_qubit_phase_gate_parallel_unroll(
    UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim) {
    // target tmask
    const ITYPE mask = 1ULL << target_qubit_index;
    const ITYPE low_mask = mask - 1;
    const ITYPE high_mask = ~low_mask;

    // loop varaibles
    const ITYPE loop_dim = dim / 2;
    if (target_qubit_index == IS_OUTER_QB) {
        ITYPE state_index;
#pragma omp parallel for
        for (state_index = 0; state_index < dim; state_index++) {
            state[state_index] *= phase;
        }
    } else if (target_qubit_index == 0) {
        ITYPE state_index;
#pragma omp parallel for
        for (state_index = 1; state_index < dim; state_index += 2) {
            state[state_index] *= phase;
        }
    } else {
        ITYPE state_index;
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis = (state_index & low_mask) +
                          ((state_index & high_mask) << 1) + mask;
            state[basis] *= phase;
            state[basis + 1] *= phase;
        }
    }
}
#endif

#ifdef _USE_SIMD
void single_qubit_phase_gate_single_simd(
    UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim) {
    // target tmask
    const ITYPE mask = 1ULL << target_qubit_index;
    const ITYPE low_mask = mask - 1;
    const ITYPE high_mask = ~low_mask;

    // loop varaibles
    const ITYPE loop_dim = dim / 2;
    if (target_qubit_index == IS_OUTER_QB) {
        ITYPE state_index;
        for (state_index = 0; state_index < dim; state_index++) {
            state[state_index] *= phase;
        }
    } else if (target_qubit_index == 0) {
        ITYPE state_index;
        for (state_index = 1; state_index < dim; state_index += 2) {
            state[state_index] *= phase;
        }
    } else {
        ITYPE state_index;
        __m256d mv0 = _mm256_set_pd(
            -cimag(phase), creal(phase), -cimag(phase), creal(phase));
        __m256d mv1 = _mm256_set_pd(
            creal(phase), cimag(phase), creal(phase), cimag(phase));
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis = (state_index & low_mask) +
                          ((state_index & high_mask) << 1) + mask;
            double *ptr = (double *)(state + basis);
            __m256d data = _mm256_loadu_pd(ptr);
            __m256d data0 = _mm256_mul_pd(data, mv0);
            __m256d data1 = _mm256_mul_pd(data, mv1);
            data = _mm256_hadd_pd(data0, data1);
            _mm256_storeu_pd(ptr, data);
        }
    }
}

#ifdef _OPENMP
void single_qubit_phase_gate_parallel_simd(
    UINT target_qubit_index, CTYPE phase, CTYPE *state, ITYPE dim) {
    // target tmask
    const ITYPE mask = 1ULL << target_qubit_index;
    const ITYPE low_mask = mask - 1;
    const ITYPE high_mask = ~low_mask;

    // loop varaibles
    const ITYPE loop_dim = dim / 2;
    if (target_qubit_index == IS_OUTER_QB) {
        ITYPE state_index;
#pragma omp parallel for
        for (state_index = 0; state_index < dim; state_index++) {
            state[state_index] *= phase;
        }
    } else if (target_qubit_index == 0) {
        ITYPE state_index;
#pragma omp parallel for
        for (state_index = 1; state_index < dim; state_index += 2) {
            state[state_index] *= phase;
        }
    } else {
        ITYPE state_index;
        __m256d mv0 = _mm256_set_pd(
            -cimag(phase), creal(phase), -cimag(phase), creal(phase));
        __m256d mv1 = _mm256_set_pd(
            creal(phase), cimag(phase), creal(phase), cimag(phase));
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis = (state_index & low_mask) +
                          ((state_index & high_mask) << 1) + mask;
            double *ptr = (double *)(state + basis);
            __m256d data = _mm256_loadu_pd(ptr);
            __m256d data0 = _mm256_mul_pd(data, mv0);
            __m256d data1 = _mm256_mul_pd(data, mv1);
            data = _mm256_hadd_pd(data0, data1);
            _mm256_storeu_pd(ptr, data);
        }
    }
}
#endif
#endif

#ifdef _USE_MPI
void single_qubit_phase_gate_mpi(UINT target_qubit_index, CTYPE phase,
    CTYPE *state, ITYPE dim, UINT inner_qc) {
    if (target_qubit_index < inner_qc) {
        single_qubit_phase_gate(target_qubit_index, phase, state, dim);
    } else {
        int target_rank_bit = 1 << (target_qubit_index - inner_qc);
        MPIutil m = get_mpiutil();
        int rank = m->get_rank();
        if (rank & target_rank_bit) {
            single_qubit_phase_gate(IS_OUTER_QB, phase, state, dim);
        }  // if else, nothing to do.
    }
}
#endif
