#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constant.h"
#include "stat_ops.h"
#include "utility.h"

#ifdef _USE_MPI
#include "csim/MPIutil.h"
#endif

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
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif
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
                m->m_DC_send(sendptr, dim_work, pair_rank);
            }
        }

    } else {
        const ITYPE loop_dim = dim / 2;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif
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
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE global_index = state_index + (rank << inner_qc);
        int bit_parity = count_population(global_index & phase_flip_mask) % 2;
        int sign = 1 - 2 * bit_parity;
        sum += pow(cabs(state[state_index]), 2) * sign;
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

/****
 * Single thread version of expectation value
 **/
// calculate expectation value of multi-qubit Pauli operator on qubits.
// bit-flip mask : the n-bit binary string of which the i-th element is 1 iff
// the i-th pauli operator is X or Y phase-flip mask : the n-bit binary string
// of which the i-th element is 1 iff the i-th pauli operator is Y or Z We
// assume bit-flip mask is nonzero, namely, there is at least one X or Y
// operator. the pivot qubit is any qubit index which has X or Y To generate
// bit-flip mask and phase-flip mask, see get_masks_*_list at utility.h
double expectation_value_multi_qubit_Pauli_operator_XZ_mask_single_thread(
    ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,
    UINT pivot_qubit_index, const CTYPE* state, ITYPE dim, ITYPE outer_qc,
    ITYPE inner_qc) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE pivot_mask = 1ULL << pivot_qubit_index;
    ITYPE state_index;
    double sum = 0.;
    UINT sign_0;
    ITYPE basis_0, basis_1;

    int comm_flag = 0;
    if (outer_qc > 0) {
        comm_flag = bit_flip_mask >> inner_qc;
    }

    if (comm_flag) {
        fprintf(stderr,
            "#ERROR: not implemented "
            "expectation_value_multi_qubit_Pauli_operator_XZ_mask"
            " with outer_qc (file: %s, line: %d)\n",
            __FILE__, __LINE__);
    } else {
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            basis_0 = insert_zero_to_basis_index(
                state_index, pivot_mask, pivot_qubit_index);
            basis_1 = basis_0 ^ bit_flip_mask;
            sign_0 = count_population(basis_0 & phase_flip_mask) % 2;
            sum += creal(
                state[basis_0] * conj(state[basis_1]) *
                PHASE_90ROT[(global_phase_90rot_count + sign_0 * 2) % 4] * 2.0);
        }
    }
    return sum;
}

double expectation_value_multi_qubit_Pauli_operator_Z_mask_single_thread(
    ITYPE phase_flip_mask, const CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    double sum = 0.;
    int bit_parity, sign;
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        bit_parity = count_population(state_index & phase_flip_mask) % 2;
        sign = 1 - 2 * bit_parity;
        sum += pow(cabs(state[state_index]), 2) * sign;
    }
    return sum;
}

double expectation_value_multi_qubit_Pauli_operator_partial_list_single_thread(
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
            result =
                expectation_value_multi_qubit_Pauli_operator_Z_mask_single_thread(
                    phase_flip_mask, state, dim);
        } else {
            result =
                expectation_value_multi_qubit_Pauli_operator_XZ_mask_single_thread(
                    bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
                    pivot_qubit_index, state, dim, outer_qc, inner_qc);
        }
        m->s_D_allreduce(&result);
    } else
#endif
    {
        if (bit_flip_mask == 0) {
            result =
                expectation_value_multi_qubit_Pauli_operator_Z_mask_single_thread(
                    phase_flip_mask, state, dim);
        } else {
            result =
                expectation_value_multi_qubit_Pauli_operator_XZ_mask_single_thread(
                    bit_flip_mask, phase_flip_mask, global_phase_90rot_count,
                    pivot_qubit_index, state, dim, outer_qc, inner_qc);
        }
    }
    return result;
}
