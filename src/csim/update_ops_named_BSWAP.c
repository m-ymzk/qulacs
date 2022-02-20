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
    UINT blk_qubits, CTYPE* state, ITYPE dim) {
   // printf("#call BSWAP_gate(%d, %d, %d)\n", target_qubit_index_0, target_qubit_index_1, blk_qubits);
    //fflush(stdout);
    for (UINT i = 0; i < blk_qubits; ++i) {
        SWAP_gate(
            target_qubit_index_0 + i, target_qubit_index_1 + i, state, dim);
    }
}

#ifdef _USE_MPI

void BSWAP_gate_mpi(UINT target_qubit_index_0, UINT target_qubit_index_1,
    UINT blk_qubits, CTYPE* state, ITYPE dim, UINT inner_qc) {
    //printf("#call BSWAP_gate_mpi(%d, %d, %d)\n", target_qubit_index_0, target_qubit_index_1, blk_qubits);
    //fflush(stdout);
    if (blk_qubits == 0) return;
#if 0
    for (UINT i = 0; i < blk_qubits; i++){
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
    assert(left_qubit > (right_qubit + blk_qubits - 1));

    UINT act_bs = blk_qubits;
    if ((left_qubit + blk_qubits - 1) < inner_qc) {  // all swaps are in inner
        for (UINT i = 0; i < blk_qubits; i++) {
            SWAP_gate(
                target_qubit_index_0 + i, target_qubit_index_1 + i, state, dim);
        }
        return;
    }

    if (right_qubit >= inner_qc) {  // all swaps are in outer
        for (UINT i = 0; i < blk_qubits; i++) {
            SWAP_gate_mpi(target_qubit_index_0 + i, target_qubit_index_1 + i,
                state, dim, inner_qc);
        }
        return;
    }

    if ((right_qubit + blk_qubits - 1) >=
        inner_qc) {  // part of right qubit is in outer
        UINT num_outer_swap = right_qubit + blk_qubits - inner_qc;
        for (UINT i = blk_qubits - 1; i > blk_qubits - 1 - num_outer_swap; i--) {
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
    const UINT tgt_outer_rank_gap = left_qubit - inner_qc;
    const UINT tgt_inner_rank_gap = inner_qc - right_qubit;

    dim_work = get_min_ll(dim_work, dim >> (act_bs - 1));

    if (rtgt_blk_dim < dim_work) {  // unit elems block smaller than worksize
        dim_work >>= 1;  // 1/2: for send, 1/2: for recv
        CTYPE* t_send = t;
        CTYPE* t_recv = t + dim_work;
        for (UINT step = 1; step < total_peer_procs; ++step) {
            const UINT peer_rank = rank ^ (step << tgt_outer_rank_gap);
            UINT rtgt_offset_index = ((rank>>tgt_outer_rank_gap) ^step);
            UINT offset_mask = (1 << tgt_inner_rank_gap) - 1;
            rtgt_offset_index &= offset_mask;
            //UINT rtgt_offset_index = ((rank>>act_bs) ^step);
            const ITYPE num_rtgt_block = (dim / dim_work) >> act_bs;
            const ITYPE num_elem_block = dim_work / rtgt_blk_dim;
            //printf("step=%d, num_rtgt_block=%lld, num_elem_block=%lld\n", step, num_rtgt_block, num_elem_block);
            //printf("rtgt_offset_index=%d, myrank=%d, peer_rank=%d\n", rtgt_offset_index, rank, peer_rank);
            for (UINT i = 0; i < num_rtgt_block; ++i) {
                // gather
                CTYPE* si;
                CTYPE* ti;
                for (UINT k = 0; k < num_elem_block; ++k) {
                    UINT iter = i * num_elem_block + k;
                    si = state + (rtgt_offset_index^(iter<<act_bs)) * rtgt_blk_dim ;
                    ti = t_send + k * rtgt_blk_dim;
#if defined(__ARM_FEATURE_SVE)
                    memcpy_sve((double*)ti, (double*)si, rtgt_blk_dim * 2);
#else
                    memcpy(ti, si, rtgt_blk_dim * sizeof(CTYPE));
#endif
                }

                // sendrecv
                m->m_DC_sendrecv(t_send, t_recv, dim_work, peer_rank);

                // scatter
                for (UINT k = 0; k < num_elem_block; ++k) {
                    UINT iter = i * num_elem_block + k;
                    ti = state + (rtgt_offset_index^(iter<<act_bs)) * rtgt_blk_dim ;
                    si = t_recv + k * rtgt_blk_dim;
#if defined(__ARM_FEATURE_SVE)
                    memcpy_sve((double*)ti, (double*)si, rtgt_blk_dim * 2);
#else
                    memcpy(ti, si, rtgt_blk_dim * sizeof(CTYPE));
#endif
                }
            }
        }
    } else {  // rtgt_blk_dim >= dim_work
        UINT TotalSizePerPairComm = dim >> act_bs;

        for (UINT step = 1; step < total_peer_procs; step++) { // pair communication
            const UINT peer_rank = rank ^ (step << tgt_outer_rank_gap);
            UINT rtgt_offset_index = ((rank>>tgt_outer_rank_gap) ^step);
            UINT offset_mask = (1 << tgt_inner_rank_gap) - 1;
            rtgt_offset_index &= offset_mask;

            assert((rtgt_blk_dim % dim_work) == 0);
            const ITYPE num_elem_block = TotalSizePerPairComm >> right_qubit;
            const ITYPE num_loop_per_block = rtgt_blk_dim / dim_work;

            for (ITYPE j = 0; j < num_elem_block; j++) {
                CTYPE* si = state + (rtgt_offset_index^(j<<act_bs)) * rtgt_blk_dim ;
                for(ITYPE k = 0; k < num_loop_per_block; k++){
                    m->m_DC_sendrecv(si, t, dim_work, peer_rank);
#if defined(__ARM_FEATURE_SVE)
                    memcpy_sve((double*)si, (double*)t, dim_work * 2);
#else
                    memcpy(si, t, dim_work * sizeof(CTYPE));
#endif
                    si += dim_work;
                }
            }
        }

    }
#endif
}
#endif
