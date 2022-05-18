
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "constant.h"
#include "memory_ops.h"
#include "update_ops.h"
#include "utility.h"

#ifdef _USE_SIMD
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

// void SWAP_gate_old_single(UINT target_qubit_index_0, UINT
// target_qubit_index_1, CTYPE *state, ITYPE dim); void
// SWAP_gate_old_parallel(UINT target_qubit_index_0, UINT target_qubit_index_1,
// CTYPE *state, ITYPE dim); void SWAP_gate_single(UINT target_qubit_index_0,
// UINT target_qubit_index_1, CTYPE *state, ITYPE dim); void
// SWAP_gate_parallel(UINT target_qubit_index_0, UINT target_qubit_index_1,
// CTYPE *state, ITYPE dim);

void SWAP_gate(UINT target_qubit_index_0, UINT target_qubit_index_1,
    CTYPE* state, ITYPE dim) {
    // SWAP_gate_old_single(target_qubit_index_0, target_qubit_index_1, state,
    // dim); SWAP_gate_old_parallel(target_qubit_index_0, target_qubit_index_1,
    // state, dim); SWAP_gate_single(target_qubit_index_0, target_qubit_index_1,
    // state, dim); SWAP_gate_single_unroll(target_qubit_index_0,
    // target_qubit_index_1, state, dim);
    // SWAP_gate_single_simd(target_qubit_index_0, target_qubit_index_1, state,
    // dim); SWAP_gate_parallel(target_qubit_index_0, target_qubit_index_1,
    // state, dim); return;

#ifdef _USE_SIMD
#ifdef _OPENMP
    UINT threshold = 13;
    if (dim < (((ITYPE)1) << threshold)) {
        SWAP_gate_single_simd(
            target_qubit_index_0, target_qubit_index_1, state, dim);
    } else {
        SWAP_gate_parallel_simd(
            target_qubit_index_0, target_qubit_index_1, state, dim);
    }
#else
    SWAP_gate_single_simd(
        target_qubit_index_0, target_qubit_index_1, state, dim);
#endif
#else
#ifdef _OPENMP
    UINT threshold = 13;
    if (dim < (((ITYPE)1) << threshold)) {
        SWAP_gate_single_unroll(
            target_qubit_index_0, target_qubit_index_1, state, dim);
    } else {
        SWAP_gate_parallel_unroll(
            target_qubit_index_0, target_qubit_index_1, state, dim);
    }
#else
    SWAP_gate_single_unroll(
        target_qubit_index_0, target_qubit_index_1, state, dim);
#endif
#endif
}

void SWAP_gate_single_unroll(UINT target_qubit_index_0,
    UINT target_qubit_index_1, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE mask_0 = 1ULL << target_qubit_index_0;
    const ITYPE mask_1 = 1ULL << target_qubit_index_1;
    const ITYPE mask = mask_0 + mask_1;

    const UINT min_qubit_index =
        get_min_ui(target_qubit_index_0, target_qubit_index_1);
    const UINT max_qubit_index =
        get_max_ui(target_qubit_index_0, target_qubit_index_1);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    ITYPE state_index = 0;
    if (target_qubit_index_0 == 0 || target_qubit_index_1 == 0) {
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 = (state_index & low_mask) +
                                  ((state_index & mid_mask) << 1) +
                                  ((state_index & high_mask) << 2) + mask_0;
            ITYPE basis_index_1 = basis_index_0 ^ mask;
            CTYPE temp = state[basis_index_0];
            state[basis_index_0] = state[basis_index_1];
            state[basis_index_1] = temp;
        }
    } else {
        // a,a+1 is swapped to a^m, a^m+1, respectively
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 = (state_index & low_mask) +
                                  ((state_index & mid_mask) << 1) +
                                  ((state_index & high_mask) << 2) + mask_0;
            ITYPE basis_index_1 = basis_index_0 ^ mask;
            CTYPE temp0 = state[basis_index_0];
            CTYPE temp1 = state[basis_index_0 + 1];
            state[basis_index_0] = state[basis_index_1];
            state[basis_index_0 + 1] = state[basis_index_1 + 1];
            state[basis_index_1] = temp0;
            state[basis_index_1 + 1] = temp1;
        }
    }
}

#ifdef _OPENMP
void SWAP_gate_parallel_unroll(UINT target_qubit_index_0,
    UINT target_qubit_index_1, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE mask_0 = 1ULL << target_qubit_index_0;
    const ITYPE mask_1 = 1ULL << target_qubit_index_1;
    const ITYPE mask = mask_0 + mask_1;

    const UINT min_qubit_index =
        get_min_ui(target_qubit_index_0, target_qubit_index_1);
    const UINT max_qubit_index =
        get_max_ui(target_qubit_index_0, target_qubit_index_1);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    ITYPE state_index = 0;
    if (target_qubit_index_0 == 0 || target_qubit_index_1 == 0) {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 = (state_index & low_mask) +
                                  ((state_index & mid_mask) << 1) +
                                  ((state_index & high_mask) << 2) + mask_0;
            ITYPE basis_index_1 = basis_index_0 ^ mask;
            CTYPE temp = state[basis_index_0];
            state[basis_index_0] = state[basis_index_1];
            state[basis_index_1] = temp;
        }
    } else {
        // a,a+1 is swapped to a^m, a^m+1, respectively
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 = (state_index & low_mask) +
                                  ((state_index & mid_mask) << 1) +
                                  ((state_index & high_mask) << 2) + mask_0;
            ITYPE basis_index_1 = basis_index_0 ^ mask;
            CTYPE temp0 = state[basis_index_0];
            CTYPE temp1 = state[basis_index_0 + 1];
            state[basis_index_0] = state[basis_index_1];
            state[basis_index_0 + 1] = state[basis_index_1 + 1];
            state[basis_index_1] = temp0;
            state[basis_index_1 + 1] = temp1;
        }
    }
}
#endif

#ifdef _USE_SIMD
void SWAP_gate_single_simd(UINT target_qubit_index_0, UINT target_qubit_index_1,
    CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE mask_0 = 1ULL << target_qubit_index_0;
    const ITYPE mask_1 = 1ULL << target_qubit_index_1;
    const ITYPE mask = mask_0 + mask_1;

    const UINT min_qubit_index =
        get_min_ui(target_qubit_index_0, target_qubit_index_1);
    const UINT max_qubit_index =
        get_max_ui(target_qubit_index_0, target_qubit_index_1);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    ITYPE state_index = 0;
    if (target_qubit_index_0 == 0 || target_qubit_index_1 == 0) {
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 = (state_index & low_mask) +
                                  ((state_index & mid_mask) << 1) +
                                  ((state_index & high_mask) << 2) + mask_0;
            ITYPE basis_index_1 = basis_index_0 ^ mask;
            CTYPE temp = state[basis_index_0];
            state[basis_index_0] = state[basis_index_1];
            state[basis_index_1] = temp;
        }
    } else {
        // a,a+1 is swapped to a^m, a^m+1, respectively
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 = (state_index & low_mask) +
                                  ((state_index & mid_mask) << 1) +
                                  ((state_index & high_mask) << 2) + mask_0;
            ITYPE basis_index_1 = basis_index_0 ^ mask;
            double* ptr0 = (double*)(state + basis_index_0);
            double* ptr1 = (double*)(state + basis_index_1);
            __m256d data0 = _mm256_loadu_pd(ptr0);
            __m256d data1 = _mm256_loadu_pd(ptr1);
            _mm256_storeu_pd(ptr1, data0);
            _mm256_storeu_pd(ptr0, data1);
        }
    }
}

#ifdef _OPENMP
void SWAP_gate_parallel_simd(UINT target_qubit_index_0,
    UINT target_qubit_index_1, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE mask_0 = 1ULL << target_qubit_index_0;
    const ITYPE mask_1 = 1ULL << target_qubit_index_1;
    const ITYPE mask = mask_0 + mask_1;

    const UINT min_qubit_index =
        get_min_ui(target_qubit_index_0, target_qubit_index_1);
    const UINT max_qubit_index =
        get_max_ui(target_qubit_index_0, target_qubit_index_1);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    ITYPE state_index = 0;
    if (target_qubit_index_0 == 0 || target_qubit_index_1 == 0) {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 = (state_index & low_mask) +
                                  ((state_index & mid_mask) << 1) +
                                  ((state_index & high_mask) << 2) + mask_0;
            ITYPE basis_index_1 = basis_index_0 ^ mask;
            CTYPE temp = state[basis_index_0];
            state[basis_index_0] = state[basis_index_1];
            state[basis_index_1] = temp;
        }
    } else {
        // a,a+1 is swapped to a^m, a^m+1, respectively
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 = (state_index & low_mask) +
                                  ((state_index & mid_mask) << 1) +
                                  ((state_index & high_mask) << 2) + mask_0;
            ITYPE basis_index_1 = basis_index_0 ^ mask;
            double* ptr0 = (double*)(state + basis_index_0);
            double* ptr1 = (double*)(state + basis_index_1);
            __m256d data0 = _mm256_loadu_pd(ptr0);
            __m256d data1 = _mm256_loadu_pd(ptr1);
            _mm256_storeu_pd(ptr1, data0);
            _mm256_storeu_pd(ptr0, data1);
        }
    }
}
#endif
#endif

#ifdef _USE_MPI
void SWAP_gate_mpi(UINT target_qubit_index_0, UINT target_qubit_index_1,
    CTYPE* state, ITYPE dim, UINT inner_qc) {
    // UINT _inner_qc = count_population(dim - 1);
    // printf("#enter SWAP, %d, %d, %d\n", target_qubit_index_0,
    // target_qubit_index_1, inner_qc);
    UINT left_qubit, right_qubit;
    if (target_qubit_index_0 > target_qubit_index_1) {
        left_qubit = target_qubit_index_0;
        right_qubit = target_qubit_index_1;
    } else {
        left_qubit = target_qubit_index_1;
        right_qubit = target_qubit_index_0;
    }

    if (left_qubit < inner_qc) {  // both qubits are inner
        // printf("#SWAP both targets are inner, %d, %d, %d\n", left_qubit,
        // right_qubit, inner_qc);
        SWAP_gate(target_qubit_index_0, target_qubit_index_1, state, dim);
    } else if (right_qubit < inner_qc) {  // one target is outer
        // printf("#SWAP one target is outer, %d, %d, %d\n", left_qubit,
        // right_qubit, inner_qc);
        const MPIutil m = get_mpiutil();
        const UINT rank = m->get_rank();
        ITYPE dim_work = dim;
        ITYPE num_work = 0;
        CTYPE* t = m->get_workarea(&dim_work, &num_work);
        const ITYPE tgt_rank_bit = 1 << (left_qubit - inner_qc);
        const ITYPE rtgt_blk_dim = 1 << right_qubit;
        const int pair_rank = rank ^ tgt_rank_bit;

        ITYPE rtgt_offset = 0;
        if ((rank & tgt_rank_bit) == 0) rtgt_offset = rtgt_blk_dim;

        if (rtgt_blk_dim < dim_work) {
            // printf("#SWAP rtgt_blk_dim < dim_work, %lld, %lld, %d\n",
            // tgt_rank_bit, rtgt_blk_dim, pair_rank);
            dim_work >>= 1;  // 1/2: for send, 1/2: for recv
            CTYPE* t_send = t;
            CTYPE* t_recv = t + dim_work;
            const ITYPE num_rtgt_block = (dim / dim_work) >> 1;
            const ITYPE num_elem_block = dim_work >> right_qubit;

            CTYPE* si0 = state + rtgt_offset;
            for (ITYPE i = 0; i < num_rtgt_block; ++i) {
                // gather
                CTYPE* si = si0;
                CTYPE* ti = t_send;
                for (ITYPE k = 0; k < num_elem_block; ++k) {
                    memcpy(ti, si, rtgt_blk_dim * sizeof(CTYPE));
                    si += (rtgt_blk_dim << 1);
                    ti += rtgt_blk_dim;
                }

                // sendrecv
                m->m_DC_sendrecv(t_send, t_recv, dim_work, pair_rank);

                // scatter
                si = t_recv;
                ti = si0;
                for (ITYPE k = 0; k < num_elem_block; ++k) {
                    memcpy(ti, si, rtgt_blk_dim * sizeof(CTYPE));
                    si += rtgt_blk_dim;
                    ti += (rtgt_blk_dim << 1);
                }
                si0 += (dim_work << 1);
            }

        } else {  // rtgt_blk_dim >= dim_work
            // printf("#SWAP rtgt_blk_dim >= dim_work, %lld, %lld, %d\n",
            // tgt_rank_bit, rtgt_blk_dim, pair_rank);
            const ITYPE num_rtgt_block = dim >> (right_qubit + 1);
            const ITYPE num_work_block = rtgt_blk_dim / dim_work;

            CTYPE* si = state + rtgt_offset;
            for (ITYPE i = 0; i < num_rtgt_block; ++i) {
                for (ITYPE j = 0; j < num_work_block; ++j) {
                    m->m_DC_sendrecv(si, t, dim_work, pair_rank);
#if defined(__ARM_FEATURE_SVE)
                    memcpy_sve((double*)si, (double*)t, dim_work * 2);
#else
                    memcpy(si, t, dim_work * sizeof(CTYPE));
#endif
                    si += dim_work;
                }
                si += rtgt_blk_dim;
            }
        }
    } else {  // both targets are outer
        // printf("#SWAP both target is outer, %d, %d, %d\n", left_qubit,
        // right_qubit, inner_qc);
        const MPIutil m = get_mpiutil();
        const UINT rank = m->get_rank();
        ITYPE dim_work = dim;
        ITYPE num_work = 0;
        CTYPE* t = m->get_workarea(&dim_work, &num_work);
        const UINT tgt0_rank_bit = 1 << (left_qubit - inner_qc);
        const UINT tgt1_rank_bit = 1 << (right_qubit - inner_qc);
        const UINT tgt_rank_bit = tgt0_rank_bit + tgt1_rank_bit;

        const int pair_rank = rank ^ tgt_rank_bit;
        const int not_zerozero = ((rank & tgt_rank_bit) != 0);
        const int with_zero =
            (((rank & tgt0_rank_bit) * (rank & tgt1_rank_bit)) == 0);

        CTYPE* si = state;
        for (ITYPE i = 0; i < num_work; ++i) {
            if (not_zerozero && with_zero) {  // 01 or 10
                m->m_DC_sendrecv(si, t, dim_work, pair_rank);
#if defined(__ARM_FEATURE_SVE)
                memcpy_sve((double*)si, (double*)t, dim_work * 2);
#else
                memcpy(si, t, dim_work * sizeof(CTYPE));
#endif
                si += dim_work;
            } else {
                m->get_tag();  // dummy to count up tag
            }
        }
    }
}
#endif

/*
#ifdef _OPENMP
void SWAP_gate_parallel(UINT target_qubit_index_0, UINT target_qubit_index_1,
CTYPE *state, ITYPE dim) { const ITYPE loop_dim = dim / 4;

        const ITYPE mask_0 = 1ULL << target_qubit_index_0;
        const ITYPE mask_1 = 1ULL << target_qubit_index_1;
        const ITYPE mask = mask_0 + mask_1;

        const UINT min_qubit_index = get_min_ui(target_qubit_index_0,
target_qubit_index_1); const UINT max_qubit_index =
get_max_ui(target_qubit_index_0, target_qubit_index_1); const ITYPE
min_qubit_mask = 1ULL << min_qubit_index; const ITYPE max_qubit_mask = 1ULL <<
(max_qubit_index - 1); const ITYPE low_mask = min_qubit_mask - 1; const ITYPE
mid_mask = (max_qubit_mask - 1) ^ low_mask; const ITYPE high_mask =
~(max_qubit_mask - 1);

        ITYPE state_index = 0;
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
                ITYPE basis_index_0 = (state_index&low_mask)
                        + ((state_index&mid_mask) << 1)
                        + ((state_index&high_mask) << 2)
                        + mask_0;
                ITYPE basis_index_1 = basis_index_0 ^ mask;
                CTYPE temp = state[basis_index_0];
                state[basis_index_0] = state[basis_index_1];
                state[basis_index_1] = temp;
        }
}
#endif


void SWAP_gate_old_single(UINT target_qubit_index_0, UINT target_qubit_index_1,
CTYPE *state, ITYPE dim) { const ITYPE loop_dim = dim / 4; const UINT
min_qubit_index = get_min_ui(target_qubit_index_0, target_qubit_index_1); const
UINT max_qubit_index = get_max_ui(target_qubit_index_0, target_qubit_index_1);
        const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
        const ITYPE max_qubit_mask = 1ULL << max_qubit_index;
        const ITYPE target_mask_0 = 1ULL << target_qubit_index_0;
        const ITYPE target_mask_1 = 1ULL << target_qubit_index_1;
        ITYPE state_index;
        for (state_index = 0; state_index < loop_dim; ++state_index) {
                ITYPE basis_insert_only_min =
insert_zero_to_basis_index(state_index, min_qubit_mask, min_qubit_index); ITYPE
basis_00 = insert_zero_to_basis_index(basis_insert_only_min, max_qubit_mask,
max_qubit_index); ITYPE basis_01 = basis_00 ^ target_mask_0; ITYPE basis_10 =
basis_00 ^ target_mask_1; swap_amplitude(state, basis_01, basis_10);
        }
}

#ifdef _OPENMP
void SWAP_gate_old_parallel(UINT target_qubit_index_0, UINT
target_qubit_index_1, CTYPE *state, ITYPE dim) { const ITYPE loop_dim = dim / 4;
        const UINT min_qubit_index = get_min_ui(target_qubit_index_0,
target_qubit_index_1); const UINT max_qubit_index =
get_max_ui(target_qubit_index_0, target_qubit_index_1); const ITYPE
min_qubit_mask = 1ULL << min_qubit_index; const ITYPE max_qubit_mask = 1ULL <<
max_qubit_index; const ITYPE target_mask_0 = 1ULL << target_qubit_index_0; const
ITYPE target_mask_1 = 1ULL << target_qubit_index_1; ITYPE state_index; #pragma
omp parallel for for (state_index = 0; state_index < loop_dim; ++state_index) {
                ITYPE basis_insert_only_min =
insert_zero_to_basis_index(state_index, min_qubit_mask, min_qubit_index); ITYPE
basis_00 = insert_zero_to_basis_index(basis_insert_only_min, max_qubit_mask,
max_qubit_index); ITYPE basis_01 = basis_00 ^ target_mask_0; ITYPE basis_10 =
basis_00 ^ target_mask_1; swap_amplitude(state, basis_01, basis_10);
        }
}
#endif


void SWAP_gate_single(UINT target_qubit_index_0, UINT target_qubit_index_1,
CTYPE *state, ITYPE dim) { const ITYPE loop_dim = dim / 4;

        const ITYPE mask_0 = 1ULL << target_qubit_index_0;
        const ITYPE mask_1 = 1ULL << target_qubit_index_1;
        const ITYPE mask = mask_0 + mask_1;

        const UINT min_qubit_index = get_min_ui(target_qubit_index_0,
target_qubit_index_1); const UINT max_qubit_index =
get_max_ui(target_qubit_index_0, target_qubit_index_1); const ITYPE
min_qubit_mask = 1ULL << min_qubit_index; const ITYPE max_qubit_mask = 1ULL <<
(max_qubit_index-1); const ITYPE low_mask = min_qubit_mask-1; const ITYPE
mid_mask = (max_qubit_mask-1)^low_mask; const ITYPE high_mask = ~(max_qubit_mask
- 1);

        ITYPE state_index = 0;
        for (state_index = 0; state_index < loop_dim; ++state_index) {
                ITYPE basis_index_0 = (state_index&low_mask)
                        + ((state_index&mid_mask) << 1)
                        + ((state_index&high_mask) << 2)
                        + mask_0;
                ITYPE basis_index_1 = basis_index_0 ^ mask;
                CTYPE temp = state[basis_index_0];
                state[basis_index_0] = state[basis_index_1];
                state[basis_index_1] = temp;
        }
}
*/
