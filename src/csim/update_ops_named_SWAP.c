
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "constant.h"
#include "memory_ops.h"
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

void SWAP_gate(UINT target_qubit_index_0, UINT target_qubit_index_1,
    CTYPE* state, ITYPE dim) {
#if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)

#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim, 13);
#endif  // ifdef _OPENMP

    SWAP_gate_sve(target_qubit_index_0, target_qubit_index_1, state, dim);

#ifdef _OPENMP
    omputil->reset_qulacs_num_threads();
#endif

#else  // if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)

#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim, 13);
#endif

#ifdef _USE_SIMD
    SWAP_gate_simd(target_qubit_index_0, target_qubit_index_1, state, dim);
#else
    SWAP_gate_unroll(target_qubit_index_0, target_qubit_index_1, state, dim);
#endif

#ifdef _OPENMP
    omputil->reset_qulacs_num_threads();
#endif
#endif  // if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
}

void SWAP_gate_unroll(UINT target_qubit_index_0, UINT target_qubit_index_1,
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
#ifdef _OPENMP
#pragma omp parallel for
#endif
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

#ifdef _USE_SIMD
void SWAP_gate_simd(UINT target_qubit_index_0, UINT target_qubit_index_1,
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
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
#ifdef _OPENMP
#pragma omp parallel for
#endif
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

#if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
void SWAP_gate_sve(UINT target_qubit_index_0, UINT target_qubit_index_1,
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

    // In case of process parallel.
    // Get # of elements in SVE registers
    // note: # of complex numbers is halved.
    ITYPE vec_len = getVecLength();

    if (dim >= vec_len) {
        // Create an all 1's predicate variable
        SV_PRED pg = Svptrue();

        ITYPE vec_step = (vec_len >> 1);
        if (min_qubit_mask >= vec_step) {
#pragma omp parallel for
            for (state_index = 0; state_index < loop_dim;
                 state_index += vec_step) {
                // Calculate indices
                ITYPE basis_0 = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + mask_0;
                ITYPE basis_1 = basis_0 ^ mask;

                /* PREFETCH */
                if ((4 <= target_qubit_index_0 && target_qubit_index_0 <= 11) ||
                    (4 <= target_qubit_index_1 && target_qubit_index_1 <= 11)) {
                    // L1 prefetch
#undef _PRF_L1_ITR
#define _PRF_L1_ITR 8
                    ITYPE basis_index_l1pf0 =
                        ((state_index + vec_step * _PRF_L1_ITR) & low_mask) +
                        (((state_index + vec_step * _PRF_L1_ITR) & mid_mask)
                            << 1) +
                        (((state_index + vec_step * _PRF_L1_ITR) & high_mask)
                            << 2) +
                        mask_0;
                    ITYPE basis_index_l1pf1 = basis_index_l1pf0 ^ mask;
                    __builtin_prefetch(&state[basis_index_l1pf0], 1, 3);
                    __builtin_prefetch(&state[basis_index_l1pf1], 1, 3);

                    // L2 prefetch
#undef _PRF_L2_ITR
#define _PRF_L2_ITR 64
                    ITYPE basis_index_l2pf0 =
                        ((state_index + vec_step * _PRF_L2_ITR) & low_mask) +
                        (((state_index + vec_step * _PRF_L2_ITR) & mid_mask)
                            << 1) +
                        (((state_index + vec_step * _PRF_L2_ITR) & high_mask)
                            << 2) +
                        mask_0;
                    ITYPE basis_index_l2pf1 = basis_index_l2pf0 ^ mask;
                    __builtin_prefetch(&state[basis_index_l2pf0], 1, 2);
                    __builtin_prefetch(&state[basis_index_l2pf1], 1, 2);
                }

                // Load values
                SV_FTYPE input0 = svld1(pg, (ETYPE*)&state[basis_0]);
                SV_FTYPE input1 = svld1(pg, (ETYPE*)&state[basis_1]);

                // Store values
                svst1(pg, (ETYPE*)&state[basis_0], input1);
                svst1(pg, (ETYPE*)&state[basis_1], input0);
            }
        } else {  // if (min_qubit_mask >= vec_step)
            if ((loop_dim % (vec_step * 4)) == 0) {
                SWAP_gate_sve_gather_scatter_unroll4(
                    target_qubit_index_0, target_qubit_index_1, state, dim);
            } else {
#pragma omp parallel for
                for (state_index = 0; state_index < loop_dim; ++state_index) {
                    ITYPE basis_index_0 = (state_index & low_mask) +
                                          ((state_index & mid_mask) << 1) +
                                          ((state_index & high_mask) << 2) +
                                          mask_0;
                    ITYPE basis_index_1 = basis_index_0 ^ mask;
                    CTYPE temp = state[basis_index_0];
                    state[basis_index_0] = state[basis_index_1];
                    state[basis_index_1] = temp;
                }
            }
        }     // if (min_qubit_mask >= vec_step)
    } else {  // if (dim >= vec_len)
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
    }  // if (dim >= vec_len)
}

void SWAP_gate_sve_gather_scatter_unroll4(UINT target_qubit_index_0,
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

    ITYPE vec_len =
        getVecLength();  // length of SVE registers (# of 64-bit elements)

    SV_PRED pg = Svptrue();  // this predicate register is all 1.

    SV_ITYPE sv_low_mask = SvdupI(low_mask);
    SV_ITYPE sv_mid_mask = SvdupI(mid_mask);
    SV_ITYPE sv_high_mask = SvdupI(high_mask);
    SV_ITYPE sv_mask0 = SvdupI(mask_0);
    SV_ITYPE sv_mask = SvdupI(mask);

    SV_ITYPE vec_index = SvindexI(0, 1);    // {0,1,2,3,4,..,7}
    vec_index = svlsr_z(pg, vec_index, 1);  // {0,0,1,1,2,..,3}

    UINT vec_step = (vec_len >> 1);
    UINT loop_step = vec_step * 4;  // unroll 4

    ITYPE state_index;

#pragma omp parallel for
    for (state_index = 0; state_index < loop_dim; state_index += loop_step) {
        SV_ITYPE sv_vec_index1 = SvdupI(state_index);
        SV_ITYPE sv_vec_index2 = SvdupI(state_index + vec_step);
        SV_ITYPE sv_vec_index3 = SvdupI(state_index + vec_step * 2);
        SV_ITYPE sv_vec_index4 = SvdupI(state_index + vec_step * 3);
        sv_vec_index1 = svadd_z(pg, sv_vec_index1, vec_index);
        sv_vec_index2 = svadd_z(pg, sv_vec_index2, vec_index);
        sv_vec_index3 = svadd_z(pg, sv_vec_index3, vec_index);
        sv_vec_index4 = svadd_z(pg, sv_vec_index4, vec_index);

        /* prefetch */
        if ((6 <= target_qubit_index_0 && target_qubit_index_0 <= 9) ||
            (6 <= target_qubit_index_1 && target_qubit_index_1 <= 9)) {
            // L1 prefetch
#undef _PRF_L1_ITR
#define _PRF_L1_ITR 4
            ITYPE basis_index_l1pf0 =
                ((state_index + loop_step * _PRF_L1_ITR) & low_mask) +
                (((state_index + loop_step * _PRF_L1_ITR) & mid_mask) << 1) +
                (((state_index + loop_step * _PRF_L1_ITR) & high_mask) << 2) +
                mask_0;
            ITYPE basis_index_l1pf1 = basis_index_l1pf0 + mask;
            ITYPE basis_index_l1pf2 =
                ((state_index + loop_step * _PRF_L1_ITR + vec_step) &
                    low_mask) +
                (((state_index + loop_step * _PRF_L1_ITR + vec_step) & mid_mask)
                    << 1) +
                (((state_index + loop_step * _PRF_L1_ITR + vec_step) &
                     high_mask)
                    << 2) +
                mask_0;
            ITYPE basis_index_l1pf3 = basis_index_l1pf2 + mask;
            ITYPE basis_index_l1pf4 =
                ((state_index + loop_step * _PRF_L1_ITR + (vec_step * 2)) &
                    low_mask) +
                (((state_index + loop_step * _PRF_L1_ITR + (vec_step * 2)) &
                     mid_mask)
                    << 1) +
                (((state_index + loop_step * _PRF_L1_ITR + (vec_step * 2)) &
                     high_mask)
                    << 2) +
                mask_0;
            ITYPE basis_index_l1pf5 = basis_index_l1pf4 + mask;
            ITYPE basis_index_l1pf6 =
                ((state_index + loop_step * _PRF_L1_ITR + (vec_step + 3)) &
                    low_mask) +
                (((state_index + loop_step * _PRF_L1_ITR + (vec_step + 3)) &
                     mid_mask)
                    << 1) +
                (((state_index + loop_step * _PRF_L1_ITR + (vec_step + 3)) &
                     high_mask)
                    << 2) +
                mask_0;
            ITYPE basis_index_l1pf7 = basis_index_l1pf6 + mask;

            __builtin_prefetch(&state[basis_index_l1pf0], 1, 3);
            __builtin_prefetch(&state[basis_index_l1pf1], 1, 3);
            __builtin_prefetch(&state[basis_index_l1pf2], 1, 3);
            __builtin_prefetch(&state[basis_index_l1pf3], 1, 3);
            __builtin_prefetch(&state[basis_index_l1pf4], 1, 3);
            __builtin_prefetch(&state[basis_index_l1pf5], 1, 3);
            __builtin_prefetch(&state[basis_index_l1pf6], 1, 3);
            __builtin_prefetch(&state[basis_index_l1pf7], 1, 3);
            // L2 prefetch
#undef _PRF_L2_ITR
#define _PRF_L2_ITR 32
            ITYPE basis_index_l2pf0 =
                ((state_index + loop_step * _PRF_L2_ITR) & low_mask) +
                (((state_index + loop_step * _PRF_L2_ITR) & mid_mask) << 1) +
                (((state_index + loop_step * _PRF_L2_ITR) & high_mask) << 2) +
                mask_0;
            ITYPE basis_index_l2pf1 = basis_index_l2pf0 + mask;
            ITYPE basis_index_l2pf2 =
                ((state_index + loop_step * _PRF_L2_ITR + vec_step) &
                    low_mask) +
                (((state_index + loop_step * _PRF_L2_ITR + vec_step) & mid_mask)
                    << 1) +
                (((state_index + loop_step * _PRF_L2_ITR + vec_step) &
                     high_mask)
                    << 2) +
                mask_0;
            ITYPE basis_index_l2pf3 = basis_index_l2pf2 + mask;
            ITYPE basis_index_l2pf4 =
                ((state_index + loop_step * _PRF_L2_ITR + (vec_step * 2)) &
                    low_mask) +
                (((state_index + loop_step * _PRF_L2_ITR + (vec_step * 2)) &
                     mid_mask)
                    << 1) +
                (((state_index + loop_step * _PRF_L2_ITR + (vec_step * 2)) &
                     high_mask)
                    << 2) +
                mask_0;
            ITYPE basis_index_l2pf5 = basis_index_l2pf4 + mask;
            ITYPE basis_index_l2pf6 =
                ((state_index + loop_step * _PRF_L2_ITR + (vec_step + 3)) &
                    low_mask) +
                (((state_index + loop_step * _PRF_L2_ITR + (vec_step + 3)) &
                     mid_mask)
                    << 1) +
                (((state_index + loop_step * _PRF_L2_ITR + (vec_step + 3)) &
                     high_mask)
                    << 2) +
                mask_0;
            ITYPE basis_index_l2pf7 = basis_index_l2pf6 + mask;

            __builtin_prefetch(&state[basis_index_l2pf0], 1, 2);
            __builtin_prefetch(&state[basis_index_l2pf1], 1, 2);
            __builtin_prefetch(&state[basis_index_l2pf2], 1, 2);
            __builtin_prefetch(&state[basis_index_l2pf3], 1, 2);
            __builtin_prefetch(&state[basis_index_l2pf4], 1, 2);
            __builtin_prefetch(&state[basis_index_l2pf5], 1, 2);
            __builtin_prefetch(&state[basis_index_l2pf6], 1, 2);
            __builtin_prefetch(&state[basis_index_l2pf7], 1, 2);
        }

        /* calclate the index */
        // (state_index & low_mask) 1/4
        SV_ITYPE sv_tmp_index1 = svand_z(pg, sv_vec_index1, sv_low_mask);
        // ((state_index & mid_mask) << 1) 1/4
        SV_ITYPE sv_tmp_index2 =
            svlsl_z(pg, svand_z(pg, sv_vec_index1, sv_mid_mask), SvdupI(1));
        // (state_index & high_mask) << 2 1/4
        SV_ITYPE sv_tmp_index3 =
            svlsl_z(pg, svand_z(pg, sv_vec_index1, sv_high_mask), SvdupI(2));
        // (state_index & low_mask) 2/4
        SV_ITYPE sv_tmp_index4 = svand_z(pg, sv_vec_index2, sv_low_mask);
        // ((state_index & mid_mask) << 1) 2/4
        SV_ITYPE sv_tmp_index5 =
            svlsl_z(pg, svand_z(pg, sv_vec_index2, sv_mid_mask), SvdupI(1));
        // (state_index & high_mask) << 2 2/4
        SV_ITYPE sv_tmp_index6 =
            svlsl_z(pg, svand_z(pg, sv_vec_index2, sv_high_mask), SvdupI(2));
        // (state_index & low_mask) 3/4
        SV_ITYPE sv_tmp_index7 = svand_z(pg, sv_vec_index3, sv_low_mask);
        // ((state_index & mid_mask) << 1) 3/4
        SV_ITYPE sv_tmp_index8 =
            svlsl_z(pg, svand_z(pg, sv_vec_index3, sv_mid_mask), SvdupI(1));
        // (state_index & high_mask) << 2 3/4
        SV_ITYPE sv_tmp_index9 =
            svlsl_z(pg, svand_z(pg, sv_vec_index3, sv_high_mask), SvdupI(2));
        // (state_index & low_mask) 4/4
        SV_ITYPE sv_tmp_index10 = svand_z(pg, sv_vec_index4, sv_low_mask);
        // ((state_index & mid_mask) << 1) 4/4
        SV_ITYPE sv_tmp_index11 =
            svlsl_z(pg, svand_z(pg, sv_vec_index4, sv_mid_mask), SvdupI(1));
        // (state_index & high_mask) << 2 4/4
        SV_ITYPE sv_tmp_index12 =
            svlsl_z(pg, svand_z(pg, sv_vec_index4, sv_high_mask), SvdupI(2));

        SV_ITYPE sv_basis_0, sv_basis_1;
        sv_basis_0 = svadd_z(pg, sv_tmp_index1, sv_tmp_index2);
        sv_basis_0 = svadd_z(pg, sv_basis_0, sv_tmp_index3);
        sv_basis_0 = svadd_z(pg, sv_basis_0, sv_mask0);
        sv_basis_1 = sveor_z(pg, sv_basis_0, sv_mask);
        SV_ITYPE sv_basis_2, sv_basis_3;
        sv_basis_2 = svadd_z(pg, sv_tmp_index4, sv_tmp_index5);
        sv_basis_2 = svadd_z(pg, sv_basis_2, sv_tmp_index6);
        sv_basis_2 = svadd_z(pg, sv_basis_2, sv_mask0);
        sv_basis_3 = sveor_z(pg, sv_basis_2, sv_mask);
        SV_ITYPE sv_basis_4, sv_basis_5;
        sv_basis_4 = svadd_z(pg, sv_tmp_index7, sv_tmp_index8);
        sv_basis_4 = svadd_z(pg, sv_basis_4, sv_tmp_index9);
        sv_basis_4 = svadd_z(pg, sv_basis_4, sv_mask0);
        sv_basis_5 = sveor_z(pg, sv_basis_4, sv_mask);
        SV_ITYPE sv_basis_6, sv_basis_7;
        sv_basis_6 = svadd_z(pg, sv_tmp_index10, sv_tmp_index11);
        sv_basis_6 = svadd_z(pg, sv_basis_6, sv_tmp_index12);
        sv_basis_6 = svadd_z(pg, sv_basis_6, sv_mask0);
        sv_basis_7 = sveor_z(pg, sv_basis_6, sv_mask);

        // complex -> double
        SV_ITYPE zero_one = svzip1(SvdupI(0), SvdupI(1));
        sv_basis_0 = svmul_z(pg, sv_basis_0, SvdupI(2));
        sv_basis_1 = svmul_z(pg, sv_basis_1, SvdupI(2));
        sv_basis_0 = svadd_z(pg, sv_basis_0, zero_one);
        sv_basis_1 = svadd_z(pg, sv_basis_1, zero_one);
        sv_basis_2 = svmul_z(pg, sv_basis_2, SvdupI(2));
        sv_basis_3 = svmul_z(pg, sv_basis_3, SvdupI(2));
        sv_basis_2 = svadd_z(pg, sv_basis_2, zero_one);
        sv_basis_3 = svadd_z(pg, sv_basis_3, zero_one);
        sv_basis_4 = svmul_z(pg, sv_basis_4, SvdupI(2));
        sv_basis_5 = svmul_z(pg, sv_basis_5, SvdupI(2));
        sv_basis_4 = svadd_z(pg, sv_basis_4, zero_one);
        sv_basis_5 = svadd_z(pg, sv_basis_5, zero_one);
        sv_basis_6 = svmul_z(pg, sv_basis_6, SvdupI(2));
        sv_basis_7 = svmul_z(pg, sv_basis_7, SvdupI(2));
        sv_basis_6 = svadd_z(pg, sv_basis_6, zero_one);
        sv_basis_7 = svadd_z(pg, sv_basis_7, zero_one);

        // Load values (Gather)
        ETYPE* ptr = (ETYPE*)&state[0];
        const SV_FTYPE input0 = svld1_gather_index(pg, ptr, sv_basis_0);
        const SV_FTYPE input1 = svld1_gather_index(pg, ptr, sv_basis_1);
        const SV_FTYPE input2 = svld1_gather_index(pg, ptr, sv_basis_2);
        const SV_FTYPE input3 = svld1_gather_index(pg, ptr, sv_basis_3);
        const SV_FTYPE input4 = svld1_gather_index(pg, ptr, sv_basis_4);
        const SV_FTYPE input5 = svld1_gather_index(pg, ptr, sv_basis_5);
        const SV_FTYPE input6 = svld1_gather_index(pg, ptr, sv_basis_6);
        const SV_FTYPE input7 = svld1_gather_index(pg, ptr, sv_basis_7);

        // Store values (Scatter)
        svst1_scatter_index(pg, ptr, sv_basis_0, input1);
        svst1_scatter_index(pg, ptr, sv_basis_1, input0);
        svst1_scatter_index(pg, ptr, sv_basis_2, input3);
        svst1_scatter_index(pg, ptr, sv_basis_3, input2);
        svst1_scatter_index(pg, ptr, sv_basis_4, input5);
        svst1_scatter_index(pg, ptr, sv_basis_5, input4);
        svst1_scatter_index(pg, ptr, sv_basis_6, input7);
        svst1_scatter_index(pg, ptr, sv_basis_7, input6);
    }
}
#endif  // #if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
