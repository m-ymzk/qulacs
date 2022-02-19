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
        SWAP_gate(
            target_qubit_index_0 + i, target_qubit_index_1 + i, state, dim);
    }
}

#ifdef _USE_MPI

void BSWAP_gate_mpi(UINT target_qubit_index_0, UINT target_qubit_index_1,
    UINT num_qubit, CTYPE* state, ITYPE dim, UINT inner_qc) {
#if 0
    for (UINT i = 0; i < num_qubit; i++){
        SWAP_gate_mpi(target_qubit_index_0+i, target_qubit_index_1+i, state, dim, inner_qc);
    }
#else
    UINT left_qubit, right_qubit;
    if (target_qubit_index_0 > target_qubit_index_1) {
        left_qubit = target_qubit_index_0;
        right_qubit = target_qubit_index_1;
    } else {
        left_qubit = target_qubit_index_1;
        right_qubit = target_qubit_index_0;
    }
    assert(left_qubit > (right_qubit + num_qubit - 1));

    UINT act_bs = num_qubit;
    if ((left_qubit + num_qubit - 1) < inner_qc) {  // all swaps are in inner
        for (UINT i = 0; i < num_qubit; i++) {
            SWAP_gate(
                target_qubit_index_0 + i, target_qubit_index_1 + i, state, dim);
        }
        return;
    }

    if ((right_qubit + num_qubit - 1) >=
        inner_qc) {  // part of right qubit is in outer
        UINT num_outer_swap = right_qubit + num_qubit - inner_qc;
        for (UINT i = num_qubit - 1; i > num_qubit - 1 - num_outer_swap; i--) {
            SWAP_gate_mpi(target_qubit_index_0 + i, target_qubit_index_1 + i,
                state, dim, inner_qc);
        }
        act_bs -= num_outer_swap;
    }

    if (left_qubit < inner_qc) {  // part of left qubit is in inner
        UINT num_inner_swap = inner_qc - left_qubit;
        /* Do inner swap individualy*/
        for (UINT i = 0; i < num_inner_swap; i++) {
            SWAP_gate_mpi(target_qubit_index_0 + i, target_qubit_index_1 + i,
                state, dim, inner_qc);
        }
        act_bs -= num_inner_swap;
        left_qubit += num_inner_swap;
        right_qubit += num_inner_swap;
    }

    if (act_bs == 0) {
        return;
    }

    /*  BSWAP main */
    /* All remained swaps are pairs of inner and outer */

    const MPIutil m = get_mpiutil();
    const UINT rank = m->get_rank();
    ITYPE dim_work = dim;
    ITYPE num_work = 0;
    CTYPE* t = m->get_workarea(&dim_work, &num_work);
    const ITYPE rtgt_blk_dim = 1 << right_qubit;
    const UINT total_peer_procs = 1 << act_bs;
    const UINT tgt_outer_rank = left_qubit - inner_qc;

    dim_work = get_min_ll(dim_work, dim >> (act_bs - 1));

    if (rtgt_blk_dim < dim_work) {  // unit elems block smaller than worksize
        for (UINT step = 1; step < total_peer_procs; ++step) {
            const UINT peer_rank = rank ^ (step << tgt_outer_rank);
            UINT rtgt_offset = (rank ^ step) * rtgt_blk_dim;
            dim_work >>= 1;  // 1/2: for send, 1/2: for recv
            CTYPE* t_send = t;
            CTYPE* t_recv = t + dim_work;
            const ITYPE num_rtgt_block = (dim / dim_work) >> act_bs;
            const ITYPE num_elem_block = dim_work >> right_qubit;
            CTYPE* si0 = state + rtgt_offset;
            for (ITYPE i = 0; i < num_rtgt_block; ++i) {
                // gather
                CTYPE* si = si0;
                CTYPE* ti = t_send;
                for (ITYPE k = 0; k < num_elem_block; ++k) {
                    memcpy(ti, si, rtgt_blk_dim * sizeof(CTYPE));
                    si += (rtgt_blk_dim << act_bs);
                    ti += rtgt_blk_dim;
                }

                // sendrecv
                m->m_DC_sendrecv(t_send, t_recv, dim_work, peer_rank);

                // scatter
                si = t_recv;
                ti = si0;
                for (ITYPE k = 0; k < num_elem_block; ++k) {
                    memcpy(ti, si, rtgt_blk_dim * sizeof(CTYPE));
                    si += rtgt_blk_dim;
                    ti += (rtgt_blk_dim << act_bs);
                }
                si0 += (dim_work << act_bs);
            }
        }
    } else {  // rtgt_blk_dim >= dim_work
        printf("Not implmented yet.\n");
        return;
    }
#endif
}
#endif
