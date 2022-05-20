
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

void Z_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
#ifdef _OPENMP
	OMPutil omputil = get_omputil();
	omputil->set_qulacs_num_threads(dim, 13);
#endif

#ifdef _USE_SIMD
#ifdef _OPENMP
    Z_gate_single_simd(target_qubit_index, state, dim);
#else
    Z_gate_single_simd(target_qubit_index, state, dim);
#endif
#else
#ifdef _OPENMP
    Z_gate_single_unroll(target_qubit_index, state, dim);
#else
    Z_gate_single_unroll(target_qubit_index, state, dim);
#endif
#endif

#ifdef _OPENMP
	omputil->reset_qulacs_num_threads();
#endif
}

void Z_gate_single_unroll(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    ITYPE state_index = 0;
    if (target_qubit_index == 0) {
        for (state_index = 1; state_index < dim; state_index += 2) {
            state[state_index] *= -1;
        }
    } else if (target_qubit_index == IS_OUTER_QB) {
        for (state_index = 0; state_index < dim; ++state_index) {
            state[state_index] *= -1;
        }
    }
#if defined(__ARM_FEATURE_SVE)
#ifdef __aarch64__
    else if (4 <= target_qubit_index && target_qubit_index <= 11) {
        for (state_index = 0; state_index < loop_dim; state_index += 4) {
            ITYPE basis_index = (state_index & mask_low) +
                                ((state_index & mask_high) << 1) + mask;
            ETYPE *restrict state_tmp = (ETYPE *)&state[basis_index];
            // L1 prefetch
#undef _PRF_L1_ITR
#define _PRF_L1_ITR 4
            ITYPE basis_pf_l1 =
                ((state_index + 4 * _PRF_L1_ITR) & mask_low) +
                (((state_index + 4 * _PRF_L1_ITR) & mask_high) << 1) + mask;

            ETYPE *restrict state_l1pf = (ETYPE *)&state[basis_pf_l1];
            __builtin_prefetch(&state_l1pf[0], 1, 3);
            __builtin_prefetch(&state_l1pf[7], 1, 3);
            // L2 prefetch
#undef _PRF_L2_ITR
#define _PRF_L2_ITR 64
            ITYPE basis_pf_l2 =
                ((state_index + 4 * _PRF_L2_ITR) & mask_low) +
                (((state_index + 4 * _PRF_L2_ITR) & mask_high) << 1) + mask;

            ETYPE *restrict state_l2pf = (ETYPE *)&state[basis_pf_l2];
            __builtin_prefetch(&state_l2pf[0], 1, 2);
            __builtin_prefetch(&state_l2pf[7], 1, 2);
#pragma omp simd
            for (ITYPE i = 0; i < 8; ++i) {
                state_tmp[i] *= -1;
            }
        }
    } else if (target_qubit_index >= 2) {
        for (state_index = 0; state_index < loop_dim; state_index += 4) {
            ITYPE basis_index = (state_index & mask_low) +
                                ((state_index & mask_high) << 1) + mask;
            ETYPE *restrict state_tmp = (ETYPE *)&state[basis_index];
#pragma omp simd
            for (ITYPE i = 0; i < 8; ++i) {
                state_tmp[i] *= -1;
            }
        }
    }
#endif  // #ifdef __aarch64__
#endif  // #if defined(__ARM_FEATURE_SVE)
    else {
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index = (state_index & mask_low) +
                                ((state_index & mask_high) << 1) + mask;
            state[basis_index] *= -1;
            state[basis_index + 1] *= -1;
        }
    }
}

#ifdef _OPENMP
void Z_gate_parallel_unroll(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    ITYPE state_index = 0;
    if (target_qubit_index == 0) {
#pragma omp parallel for
        for (state_index = 1; state_index < dim; state_index += 2) {
            state[state_index] *= -1;
        }
    } else if (target_qubit_index == IS_OUTER_QB) {
#pragma omp parallel for
        for (state_index = 0; state_index < dim; ++state_index) {
            state[state_index] *= -1;
        }
    }
#if defined(__ARM_FEATURE_SVE)
#ifdef __aarch64__
    else if (4 <= target_qubit_index && target_qubit_index <= 11) {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 4) {
            ITYPE basis_index = (state_index & mask_low) +
                                ((state_index & mask_high) << 1) + mask;
            ETYPE *restrict state_tmp = (ETYPE *)&state[basis_index];
            // L1 prefetch
#undef _PRF_L1_ITR
#define _PRF_L1_ITR 4
            ITYPE basis_pf_l1 =
                ((state_index + 4 * _PRF_L1_ITR) & mask_low) +
                (((state_index + 4 * _PRF_L1_ITR) & mask_high) << 1) + mask;

            ETYPE *restrict state_l1pf = (ETYPE *)&state[basis_pf_l1];
            __builtin_prefetch(&state_l1pf[0], 1, 3);
            __builtin_prefetch(&state_l1pf[7], 1, 3);
            // L2 prefetch
#undef _PRF_L2_ITR
#define _PRF_L2_ITR 64
            ITYPE basis_pf_l2 =
                ((state_index + 4 * _PRF_L2_ITR) & mask_low) +
                (((state_index + 4 * _PRF_L2_ITR) & mask_high) << 1) + mask;

            ETYPE *restrict state_l2pf = (ETYPE *)&state[basis_pf_l2];
            __builtin_prefetch(&state_l2pf[0], 1, 2);
            __builtin_prefetch(&state_l2pf[7], 1, 2);
#pragma omp simd
            for (ITYPE i = 0; i < 8; ++i) {
                state_tmp[i] *= -1;
            }
        }
    } else if (target_qubit_index >= 2) {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 4) {
            ITYPE basis_index = (state_index & mask_low) +
                                ((state_index & mask_high) << 1) + mask;
            ETYPE *restrict state_tmp = (ETYPE *)&state[basis_index];
#pragma omp simd
            for (ITYPE i = 0; i < 8; ++i) {
                state_tmp[i] *= -1;
            }
        }
    }
#endif  // #ifdef __aarch64__
#endif  // #if defined(__ARM_FEATURE_SVE)
    else {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index = (state_index & mask_low) +
                                ((state_index & mask_high) << 1) + mask;
            state[basis_index] *= -1;
            state[basis_index + 1] *= -1;
        }
    }
}
#endif

#ifdef _USE_SIMD
void Z_gate_single_simd(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    ITYPE state_index = 0;
    __m256d minus_one = _mm256_set_pd(-1, -1, -1, -1);
    if (target_qubit_index == 0) {
        for (state_index = 1; state_index < dim; state_index += 2) {
            state[state_index] *= -1;
        }
    } else if (target_qubit_index == IS_OUTER_QB) {
        for (state_index = 0; state_index < dim; ++state_index) {
            state[state_index] *= -1;
        }
    } else {
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index = (state_index & mask_low) +
                                ((state_index & mask_high) << 1) + mask;
            double *ptr0 = (double *)(state + basis_index);
            __m256d data0 = _mm256_loadu_pd(ptr0);
            data0 = _mm256_mul_pd(data0, minus_one);
            _mm256_storeu_pd(ptr0, data0);
        }
    }
}

#ifdef _OPENMP
void Z_gate_parallel_simd(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    ITYPE state_index = 0;
    __m256d minus_one = _mm256_set_pd(-1, -1, -1, -1);
    if (target_qubit_index == 0) {
#pragma omp parallel for
        for (state_index = 1; state_index < dim; state_index += 2) {
            state[state_index] *= -1;
        }
    } else if (target_qubit_index == IS_OUTER_QB) {
#pragma omp parallel for
        for (state_index = 0; state_index < dim; ++state_index) {
            state[state_index] *= -1;
        }
    } else {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index = (state_index & mask_low) +
                                ((state_index & mask_high) << 1) + mask;
            double *ptr0 = (double *)(state + basis_index);
            __m256d data0 = _mm256_loadu_pd(ptr0);
            data0 = _mm256_mul_pd(data0, minus_one);
            _mm256_storeu_pd(ptr0, data0);
        }
    }
}
#endif
#endif

#ifdef _USE_MPI
void Z_gate_mpi(
    UINT target_qubit_index, CTYPE *state, ITYPE dim, UINT inner_qc) {
    if (target_qubit_index < inner_qc) {
        Z_gate(target_qubit_index, state, dim);
    } else {
        const MPIutil m = get_mpiutil();
        const int rank = m->get_rank();
        const int pair_rank_bit = 1 << (target_qubit_index - inner_qc);
        if (rank & pair_rank_bit) {
            Z_gate(IS_OUTER_QB, state, dim);
        }
    }
}
#endif

/*


void Z_gate_old_single(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
        const ITYPE loop_dim = dim / 2;
        ITYPE state_index;
        ITYPE mask = (1ULL << target_qubit_index);
        for (state_index = 0; state_index < loop_dim; ++state_index) {
                ITYPE temp_index = insert_zero_to_basis_index(state_index, mask,
target_qubit_index) ^ mask; state[temp_index] *= -1;
        }
}

void Z_gate_old_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
        const ITYPE loop_dim = dim / 2;
        ITYPE state_index;
        ITYPE mask = (1ULL << target_qubit_index);
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; ++state_index) {
                ITYPE temp_index = insert_zero_to_basis_index(state_index, mask,
target_qubit_index) ^ mask; state[temp_index] *= -1;
        }
}

void Z_gate_single(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
        const ITYPE loop_dim = dim / 2;
        const ITYPE mask = (1ULL << target_qubit_index);
        const ITYPE mask_low = mask - 1;
        const ITYPE mask_high = ~mask_low;
        ITYPE state_index = 0;
        for (state_index = 0; state_index < loop_dim; ++state_index) {
                ITYPE basis_index = (state_index&mask_low) +
((state_index&mask_high) << 1) + mask; state[basis_index] *= -1;
        }
}

#ifdef _OPENMP
void Z_gate_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
        const ITYPE loop_dim = dim / 2;
        const ITYPE mask = (1ULL << target_qubit_index);
        const ITYPE mask_low = mask - 1;
        const ITYPE mask_high = ~mask_low;
        ITYPE state_index = 0;
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
                ITYPE basis_index = (state_index&mask_low) +
((state_index&mask_high) << 1) + mask; state[basis_index] *= -1;
        }
}
#endif

*/
