#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "constant.h"
#include "memory_ops.h"
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

void BSWAP_gate(UINT target_qubit_index_0, UINT target_qubit_index_1,
    UINT num_qubit, CTYPE* state, ITYPE dim) {
    for (UINT i = 0; i < num_qubit; ++i) {
        SWAP_gate(target_qubit_index_0 + i, target_qubit_index_1 + i, state, dim);
    }
}

#ifdef _USE_MPI

void BSWAP_gate_mpi(UINT target_qubit_index_0, UINT target_qubit_index_1,
    UINT num_qubit, CTYPE* state, ITYPE dim, UINT inner_qc) {
    for (UINT i = 0; i < num_qubit; i++){
        SWAP_gate_mpi(target_qubit_index_0+i, target_qubit_index_1+i, state, dim, inner_qc);
    }

#if 0
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
#endif
}
#endif

