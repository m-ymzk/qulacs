
#include <stdio.h>

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

// void CZ_gate_old_single(UINT control_qubit_index, UINT target_qubit_index,
// CTYPE *state, ITYPE dim); void CZ_gate_old_parallel(UINT control_qubit_index,
// UINT target_qubit_index, CTYPE *state, ITYPE dim); void CZ_gate_single(UINT
// control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim); void
// CZ_gate_parallel(UINT control_qubit_index, UINT target_qubit_index, CTYPE
// *state, ITYPE dim);

void CZ_gate(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state,
    ITYPE dim) {
    // CZ_gate_old_single(control_qubit_index, target_qubit_index, state, dim);
    // CZ_gate_old_parallel(control_qubit_index, target_qubit_index, state,
    // dim); CZ_gate_single(control_qubit_index, target_qubit_index, state,
    // dim); CZ_gate_single_unroll(control_qubit_index, target_qubit_index,
    // state, dim); CZ_gate_single_simd(control_qubit_index, target_qubit_index,
    // state, dim); CZ_gate_parallel(control_qubit_index, target_qubit_index,
    // state, dim); return;

#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim, 13);
#endif

#ifdef _USE_SIMD
#ifdef _OPENMP
    CZ_gate_parallel_simd(control_qubit_index, target_qubit_index, state, dim);
#else
    CZ_gate_single_simd(control_qubit_index, target_qubit_index, state, dim);
#endif
#else
#ifdef _OPENMP
    CZ_gate_parallel_unroll(
        control_qubit_index, target_qubit_index, state, dim);
#else
    CZ_gate_single_unroll(control_qubit_index, target_qubit_index, state, dim);
#endif
#endif

#ifdef _OPENMP
    omputil->reset_qulacs_num_threads();
#endif
}

void CZ_gate_single_unroll(UINT control_qubit_index, UINT target_qubit_index,
    CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE target_mask = 1ULL << target_qubit_index;
    const ITYPE control_mask = 1ULL << control_qubit_index;

    const UINT min_qubit_index =
        get_min_ui(control_qubit_index, target_qubit_index);
    const UINT max_qubit_index =
        get_max_ui(control_qubit_index, target_qubit_index);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    const ITYPE mask = target_mask + control_mask;
    ITYPE state_index = 0;
    if (target_qubit_index == 0 || control_qubit_index == 0) {
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + mask;
            state[basis_index] *= -1;
        }
    } else {
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + mask;
            state[basis_index] *= -1;
            state[basis_index + 1] *= -1;
        }
    }
}

#ifdef _OPENMP
void CZ_gate_parallel_unroll(UINT control_qubit_index, UINT target_qubit_index,
    CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE target_mask = 1ULL << target_qubit_index;
    const ITYPE control_mask = 1ULL << control_qubit_index;

    const UINT min_qubit_index =
        get_min_ui(control_qubit_index, target_qubit_index);
    const UINT max_qubit_index =
        get_max_ui(control_qubit_index, target_qubit_index);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    const ITYPE mask = target_mask + control_mask;
    ITYPE state_index = 0;
    if (target_qubit_index == 0 || control_qubit_index == 0) {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + mask;
            state[basis_index] *= -1;
        }
    } else {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + mask;
            state[basis_index] *= -1;
            state[basis_index + 1] *= -1;
        }
    }
}
#endif

#ifdef _USE_SIMD
void CZ_gate_single_simd(UINT control_qubit_index, UINT target_qubit_index,
    CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE target_mask = 1ULL << target_qubit_index;
    const ITYPE control_mask = 1ULL << control_qubit_index;

    const UINT min_qubit_index =
        get_min_ui(control_qubit_index, target_qubit_index);
    const UINT max_qubit_index =
        get_max_ui(control_qubit_index, target_qubit_index);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    const ITYPE mask = target_mask + control_mask;
    ITYPE state_index = 0;
    if (target_qubit_index == 0 || control_qubit_index == 0) {
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + mask;
            state[basis_index] *= -1;
        }
    } else {
        __m256d minus_one = _mm256_set_pd(-1, -1, -1, -1);
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + mask;
            double *ptr = (double *)(state + basis_index);
            __m256d data = _mm256_loadu_pd(ptr);
            data = _mm256_mul_pd(data, minus_one);
            _mm256_storeu_pd(ptr, data);
        }
    }
}

#ifdef _OPENMP

void CZ_gate_parallel_simd(UINT control_qubit_index, UINT target_qubit_index,
    CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE target_mask = 1ULL << target_qubit_index;
    const ITYPE control_mask = 1ULL << control_qubit_index;

    const UINT min_qubit_index =
        get_min_ui(control_qubit_index, target_qubit_index);
    const UINT max_qubit_index =
        get_max_ui(control_qubit_index, target_qubit_index);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    const ITYPE mask = target_mask + control_mask;
    ITYPE state_index = 0;
    if (target_qubit_index == 0 || control_qubit_index == 0) {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + mask;
            state[basis_index] *= -1;
        }
    } else {
        __m256d minus_one = _mm256_set_pd(-1, -1, -1, -1);
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + mask;
            double *ptr = (double *)(state + basis_index);
            __m256d data = _mm256_loadu_pd(ptr);
            data = _mm256_mul_pd(data, minus_one);
            _mm256_storeu_pd(ptr, data);
        }
    }
}
#endif
#endif

#ifdef _USE_MPI
void CZ_gate_mpi(UINT control_qubit_index, UINT target_qubit_index,
    CTYPE *state, ITYPE dim, UINT inner_qc) {
    UINT left_qubit, right_qubit;
    if (control_qubit_index > target_qubit_index) {
        left_qubit = control_qubit_index;
        right_qubit = target_qubit_index;
    } else {
        left_qubit = target_qubit_index;
        right_qubit = control_qubit_index;
    }

    if (left_qubit < inner_qc) {
        CZ_gate(control_qubit_index, target_qubit_index, state, dim);
    } else if (right_qubit < inner_qc) {  // one quibit is outer
        // printf("#enter CZ gate, one qubit is outer\n");
        const MPIutil m = get_mpiutil();
        const UINT rank = m->get_rank();
        const UINT tgt_rank_bit = 1 << (left_qubit - inner_qc);
        if (rank & tgt_rank_bit) {
            Z_gate(right_qubit, state, dim);
        }     // if else, nothing to do.
    } else {  // both qubits are outer;
              // printf("#enter CZ gate, both is outer_qubit\n");
        const MPIutil m = get_mpiutil();
        const UINT rank = m->get_rank();
        const UINT tgt0_rank_bit = 1 << (left_qubit - inner_qc);
        const UINT tgt1_rank_bit = 1 << (right_qubit - inner_qc);
        if (rank & tgt0_rank_bit && rank & tgt1_rank_bit) {
            Z_gate(IS_OUTER_QB, state, dim);
        }  // if else, nothing to do.
    }
}
#endif

/*


void CZ_gate_old_single(UINT control_qubit_index, UINT target_qubit_index, CTYPE
*state, ITYPE dim) { const ITYPE loop_dim = dim / 4; const UINT min_qubit_index
= get_min_ui(control_qubit_index, target_qubit_index); const UINT
max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index); const
ITYPE min_qubit_mask = 1ULL << min_qubit_index; const ITYPE max_qubit_mask =
1ULL << max_qubit_index; const ITYPE control_mask = 1ULL << control_qubit_index;
        const ITYPE target_mask = 1ULL << target_qubit_index;
        ITYPE state_index;
        for (state_index = 0; state_index < loop_dim; ++state_index) {
                ITYPE basis_insert_only_min =
insert_zero_to_basis_index(state_index, min_qubit_mask, min_qubit_index); ITYPE
basis_c1t1 = insert_zero_to_basis_index(basis_insert_only_min, max_qubit_mask,
max_qubit_index) ^ control_mask ^ target_mask; state[basis_c1t1] *= -1;
        }
}


#ifdef _OPENMP
void CZ_gate_old_parallel(UINT control_qubit_index, UINT target_qubit_index,
CTYPE *state, ITYPE dim) { const ITYPE loop_dim = dim / 4; const UINT
min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index); const
UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
        const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
        const ITYPE max_qubit_mask = 1ULL << max_qubit_index;
        const ITYPE control_mask = 1ULL << control_qubit_index;
        const ITYPE target_mask = 1ULL << target_qubit_index;
        ITYPE state_index;
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
                ITYPE basis_insert_only_min =
insert_zero_to_basis_index(state_index, min_qubit_mask, min_qubit_index); ITYPE
basis_c1t1 = insert_zero_to_basis_index(basis_insert_only_min, max_qubit_mask,
max_qubit_index) ^ control_mask ^ target_mask; state[basis_c1t1] *= -1;
        }
}
#endif


void CZ_gate_single(UINT control_qubit_index, UINT target_qubit_index, CTYPE
*state, ITYPE dim) { const ITYPE loop_dim = dim / 4;

        const ITYPE target_mask = 1ULL << target_qubit_index;
        const ITYPE control_mask = 1ULL << control_qubit_index;

        const UINT min_qubit_index = get_min_ui(control_qubit_index,
target_qubit_index); const UINT max_qubit_index =
get_max_ui(control_qubit_index, target_qubit_index); const ITYPE min_qubit_mask
= 1ULL << min_qubit_index; const ITYPE max_qubit_mask = 1ULL << (max_qubit_index
- 1); const ITYPE low_mask = min_qubit_mask - 1; const ITYPE mid_mask =
(max_qubit_mask - 1) ^ low_mask; const ITYPE high_mask = ~(max_qubit_mask - 1);

        const ITYPE mask = target_mask + control_mask;

        ITYPE state_index = 0;
        for (state_index = 0; state_index < loop_dim; ++state_index) {
                ITYPE basis_index = (state_index&low_mask)
                        + ((state_index&mid_mask) << 1)
                        + ((state_index&high_mask) << 2)
                        + mask;
                state[basis_index] *= -1;
        }
}

#ifdef _OPENMP
void CZ_gate_parallel(UINT control_qubit_index, UINT target_qubit_index, CTYPE
*state, ITYPE dim) { const ITYPE loop_dim = dim / 4;

        const ITYPE target_mask = 1ULL << target_qubit_index;
        const ITYPE control_mask = 1ULL << control_qubit_index;

        const UINT min_qubit_index = get_min_ui(control_qubit_index,
target_qubit_index); const UINT max_qubit_index =
get_max_ui(control_qubit_index, target_qubit_index); const ITYPE min_qubit_mask
= 1ULL << min_qubit_index; const ITYPE max_qubit_mask = 1ULL << (max_qubit_index
- 1); const ITYPE low_mask = min_qubit_mask - 1; const ITYPE mid_mask =
(max_qubit_mask - 1) ^ low_mask; const ITYPE high_mask = ~(max_qubit_mask - 1);

        const ITYPE mask = target_mask + control_mask;

        ITYPE state_index = 0;
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
                ITYPE basis_index = (state_index&low_mask)
                        + ((state_index&mid_mask) << 1)
                        + ((state_index&high_mask) << 2)
                        + mask;
                state[basis_index] *= -1;
        }
}
#endif
*/
