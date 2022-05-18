#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constant.h"
#include "stat_ops.h"
#include "utility.h"

double expectation_value_multi_qubit_Pauli_operator_XZ_mask(ITYPE bit_flip_mask,
    ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, const CTYPE* state, ITYPE dim, ITYPE outer_qc,
    ITYPE inner_qc);
double expectation_value_multi_qubit_Pauli_operator_Z_mask(
    ITYPE phase_flip_mask, const CTYPE* state, ITYPE dim, int rank,
    ITYPE inner_qc);

// calculate expectation value of multi-qubit Pauli operator on qubits.
// bit-flip mask : the n-bit binary string of which the i-th element is 1 iff
// the i-th pauli operator is X or Y phase-flip mask : the n-bit binary string
// of which the i-th element is 1 iff the i-th pauli operator is Y or Z We
// assume bit-flip mask is nonzero, namely, there is at least one X or Y
// operator. the pivot qubit is any qubit index which has X or Y To generate
// bit-flip mask and phase-flip mask, see get_masks_*_list at utility.h
double expectation_value_multi_qubit_Pauli_operator_XZ_mask(ITYPE bit_flip_mask,
    ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, const CTYPE* state, ITYPE dim, ITYPE outer_qc,
    ITYPE inner_qc) {
    const ITYPE pivot_mask = 1ULL << pivot_qubit_index;
    ITYPE state_index;
    double sum = 0.;

    int comm_flag = 0;
    if (outer_qc > 0) {
        comm_flag = bit_flip_mask >> inner_qc;
    }

    if (comm_flag) {
        MPIutil m = get_mpiutil();
        int rank = m->get_rank();
        int pair_rank = rank ^ comm_flag;

        ITYPE dim_work = dim;
        ITYPE num_work = 0;
        CTYPE* recvptr = m->get_workarea(&dim_work, &num_work);
        ITYPE inner_mask = dim - 1;
        ITYPE i, j;

        state_index = 0;

        for (i = 0; i < num_work; ++i) {
            const CTYPE* sendptr = state + dim_work * i;

            if (rank < pair_rank) {
                // recv
                m->m_DC_recv(recvptr, dim_work, pair_rank);
#pragma omp parallel for reduction(+ : sum)
                for (j = 0; j < dim_work; ++j) {
                    ITYPE basis_1 = state_index + j + (pair_rank << inner_qc);
                    ITYPE basis_0 = basis_1 ^ bit_flip_mask;
                    UINT sign_0 =
                        count_population(basis_0 & phase_flip_mask) % 2;

                    sum += creal(
                        state[basis_0 & inner_mask] *
                        conj(recvptr[basis_1 & (dim_work - 1)]) *
                        PHASE_90ROT[(global_phase_90rot_count + sign_0 * 2) %
                                    4] *
                        2.0);
                }
                state_index += dim_work;
            } else {
                // send
                m->m_DC_send((void*)sendptr, dim_work, pair_rank);
            }
        }

    } else {
        const ITYPE loop_dim = dim / 2;
#pragma omp parallel for reduction(+ : sum)
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_0 = insert_zero_to_basis_index(
                state_index, pivot_mask, pivot_qubit_index);
            ITYPE basis_1 = basis_0 ^ bit_flip_mask;
            UINT sign_0 = count_population(basis_0 & phase_flip_mask) % 2;

            sum += creal(
                state[basis_0] * conj(state[basis_1]) *
                PHASE_90ROT[(global_phase_90rot_count + sign_0 * 2) % 4] * 2.0);
        }
    }
    return sum;
}

double expectation_value_multi_qubit_Pauli_operator_Z_mask(
    ITYPE phase_flip_mask, const CTYPE* state, ITYPE dim, int rank,
    ITYPE inner_qc) {
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    double sum = 0.;

#if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
    ITYPE vec_len = getVecLength(); // # of double elements in a vector

    if (loop_dim >= vec_len) {

#pragma omp parallel private(state_index) reduction(+: sum)
        {
            SV_PRED pg = Svptrue();
            SV_FTYPE sv_sum = SvdupF(0.0);
            SV_ITYPE sv_offset = SvindexI(0, 1);
            SV_ITYPE sv_phase_flip_mask = SvdupI(phase_flip_mask);

#pragma omp for
            for (state_index = 0; state_index < loop_dim; state_index += vec_len) {
                ITYPE global_index = state_index + (rank << inner_qc);
                // A
                SV_ITYPE svidx = svadd_z(pg, SvdupI(global_index), sv_offset);
                SV_ITYPE sv_bit_parity = svand_z(pg, svidx, sv_phase_flip_mask);
                sv_bit_parity = svcnt_z(pg, sv_bit_parity);
                sv_bit_parity = svand_z(pg, sv_bit_parity, SvdupI(1));
                // B
                SV_PRED pg_sign = svcmpeq(pg, sv_bit_parity, SvdupI(1));

                // C
                SV_FTYPE sv_val0 = svld1(pg, (ETYPE *)&state[state_index]);
                SV_FTYPE sv_val1 = svld1(pg, (ETYPE *)&state[state_index+(vec_len>>1)]);

                sv_val0 = svmul_z(pg, sv_val0, sv_val0);
                sv_val1 = svmul_z(pg, sv_val1, sv_val1);

                sv_val0 = svadd_z(pg, sv_val0, svext(sv_val0, sv_val0, 1));
                sv_val1 = svadd_z(pg, sv_val1, svext(sv_val1, sv_val1, 1));

                sv_val0 = svuzp1(sv_val0, sv_val1);
                sv_val0 = svneg_m(sv_val0, pg_sign, sv_val0);

                sv_sum = svadd_z(pg, sv_sum, sv_val0);
            }

            // TODO: supports 512-bit SVE engine only
            // reduction
            sv_sum = svadd_z(pg, sv_sum, svext(sv_sum, sv_sum, 4));
            sv_sum = svadd_z(pg, sv_sum, svext(sv_sum, sv_sum, 2));
            sv_sum = svadd_z(pg, sv_sum, svext(sv_sum, sv_sum, 1));
            sum = svlastb(svptrue_pat_b64(SV_VL1), sv_sum);

        }
    } else
#endif
    {
#pragma omp parallel for reduction(+ : sum)
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE global_index = state_index + (rank << inner_qc);
            // A
            int bit_parity = count_population(global_index & phase_flip_mask) % 2;
            // B
            int sign = 1 - 2 * bit_parity;
            // C
            sum += state[state_index] * conj(state[state_index]) * sign;
        }
    }
    return sum;
}

double expectation_value_multi_qubit_Pauli_operator_partial_list(
    const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list,
    UINT target_qubit_index_count, const CTYPE* state, ITYPE dim,
    ITYPE outer_qc, ITYPE inner_qc) {
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_partial_list(target_qubit_index_list,
        Pauli_operator_type_list, target_qubit_index_count, &bit_flip_mask,
        &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
    double result;
#ifdef _USE_MPI
    if (outer_qc > 0) {
        MPIutil m = get_mpiutil();
        if (bit_flip_mask == 0) {
            result = expectation_value_multi_qubit_Pauli_operator_Z_mask(
                phase_flip_mask, state, dim, m->get_rank(), inner_qc);
        } else {
            result = expectation_value_multi_qubit_Pauli_operator_XZ_mask(
                bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
                pivot_qubit_index, state, dim, outer_qc, inner_qc);
        }
        m->s_D_allreduce(&result);
    } else
#endif
    {
        if (bit_flip_mask == 0) {
            result = expectation_value_multi_qubit_Pauli_operator_Z_mask(
                phase_flip_mask, state, dim, 0 /*rank*/, inner_qc);
        } else {
            result = expectation_value_multi_qubit_Pauli_operator_XZ_mask(
                bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
                pivot_qubit_index, state, dim, outer_qc, inner_qc);
        }
    }
    return result;
}

double expectation_value_multi_qubit_Pauli_operator_whole_list(
    const UINT* Pauli_operator_type_list, UINT qubit_count, const CTYPE* state,
    ITYPE dim, ITYPE outer_qc, ITYPE inner_qc) {
    ITYPE bit_flip_mask = 0;
    ITYPE phase_flip_mask = 0;
    UINT global_phase_90rot_count = 0;
    UINT pivot_qubit_index = 0;
    get_Pauli_masks_whole_list(Pauli_operator_type_list, qubit_count,
        &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count,
        &pivot_qubit_index);
    double result;
    if (bit_flip_mask == 0) {
        result = expectation_value_multi_qubit_Pauli_operator_Z_mask(
            phase_flip_mask, state, dim, 0 /*rank*/, inner_qc);
    } else {
        result = expectation_value_multi_qubit_Pauli_operator_XZ_mask(
            bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
            pivot_qubit_index, state, dim, outer_qc, inner_qc);
    }
    return result;
}


