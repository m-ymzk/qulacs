
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "MPIutil.hpp"
#include "constant.hpp"
#include "update_ops.hpp"
#include "utility.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _USE_SIMD
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

void single_qubit_dense_matrix_gate(
    UINT target_qubit_index, const CTYPE matrix[4], CTYPE* state, ITYPE dim) {
#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim, 13);
#endif

#ifdef _USE_SIMD
    single_qubit_dense_matrix_gate_parallel_simd(
        target_qubit_index, matrix, state, dim);
#else
    single_qubit_dense_matrix_gate_parallel(
        target_qubit_index, matrix, state, dim);
#endif

#ifdef _OPENMP
    omputil->reset_qulacs_num_threads();
#endif
}

void single_qubit_dense_matrix_gate_parallel(
    UINT target_qubit_index, const CTYPE matrix[4], CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;

    ITYPE state_index = 0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        ITYPE basis_0 =
            (state_index & mask_low) + ((state_index & mask_high) << 1);
        ITYPE basis_1 = basis_0 + mask;

        // fetch values
        CTYPE cval_0 = state[basis_0];
        CTYPE cval_1 = state[basis_1];

        // set values
        state[basis_0] = matrix[0] * cval_0 + matrix[1] * cval_1;
        state[basis_1] = matrix[2] * cval_0 + matrix[3] * cval_1;
    }
}

void single_qubit_dense_matrix_gate_parallel_unroll(
    UINT target_qubit_index, const CTYPE matrix[4], CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;

    if (target_qubit_index == 0) {
        ITYPE basis = 0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (basis = 0; basis < dim; basis += 2) {
            CTYPE val0a = state[basis];
            CTYPE val1a = state[basis + 1];
            CTYPE res0a = val0a * matrix[0] + val1a * matrix[1];
            CTYPE res1a = val0a * matrix[2] + val1a * matrix[3];
            state[basis] = res0a;
            state[basis + 1] = res1a;
        }
    } else {
        ITYPE state_index = 0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_0 =
                (state_index & mask_low) + ((state_index & mask_high) << 1);
            ITYPE basis_1 = basis_0 + mask;
            CTYPE val0a = state[basis_0];
            CTYPE val0b = state[basis_0 + 1];
            CTYPE val1a = state[basis_1];
            CTYPE val1b = state[basis_1 + 1];

            CTYPE res0a = val0a * matrix[0] + val1a * matrix[1];
            CTYPE res1b = val0b * matrix[2] + val1b * matrix[3];
            CTYPE res1a = val0a * matrix[2] + val1a * matrix[3];
            CTYPE res0b = val0b * matrix[0] + val1b * matrix[1];

            state[basis_0] = res0a;
            state[basis_0 + 1] = res0b;
            state[basis_1] = res1a;
            state[basis_1 + 1] = res1b;
        }
    }
}

#ifdef _USE_SIMD
void single_qubit_dense_matrix_gate_parallel_simd(
    UINT target_qubit_index, const CTYPE matrix[4], CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;

    if (target_qubit_index == 0) {
        ITYPE basis = 0;
        __m256d mv00 = _mm256_set_pd(-_cimag(matrix[1]), _creal(matrix[1]),
            -_cimag(matrix[0]), _creal(matrix[0]));
        __m256d mv01 = _mm256_set_pd(_creal(matrix[1]), _cimag(matrix[1]),
            _creal(matrix[0]), _cimag(matrix[0]));
        __m256d mv20 = _mm256_set_pd(-_cimag(matrix[3]), _creal(matrix[3]),
            -_cimag(matrix[2]), _creal(matrix[2]));
        __m256d mv21 = _mm256_set_pd(_creal(matrix[3]), _cimag(matrix[3]),
            _creal(matrix[2]), _cimag(matrix[2]));
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (basis = 0; basis < dim; basis += 2) {
            double* ptr = (double*)(state + basis);
            __m256d data = _mm256_loadu_pd(ptr);

            __m256d data_u0 = _mm256_mul_pd(data, mv00);
            __m256d data_u1 = _mm256_mul_pd(data, mv01);
            __m256d data_u2 = _mm256_hadd_pd(data_u0, data_u1);
            data_u2 = _mm256_permute4x64_pd(data_u2,
                216);  // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216

            __m256d data_d0 = _mm256_mul_pd(data, mv20);
            __m256d data_d1 = _mm256_mul_pd(data, mv21);
            __m256d data_d2 = _mm256_hadd_pd(data_d0, data_d1);
            data_d2 = _mm256_permute4x64_pd(data_d2,
                216);  // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216

            __m256d data_r = _mm256_hadd_pd(data_u2, data_d2);

            data_r = _mm256_permute4x64_pd(data_r,
                216);  // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216
            _mm256_storeu_pd(ptr, data_r);
        }
    } else {
        ITYPE state_index = 0;
        __m256d mv00 = _mm256_set_pd(-_cimag(matrix[0]), _creal(matrix[0]),
            -_cimag(matrix[0]), _creal(matrix[0]));
        __m256d mv01 = _mm256_set_pd(_creal(matrix[0]), _cimag(matrix[0]),
            _creal(matrix[0]), _cimag(matrix[0]));
        __m256d mv10 = _mm256_set_pd(-_cimag(matrix[1]), _creal(matrix[1]),
            -_cimag(matrix[1]), _creal(matrix[1]));
        __m256d mv11 = _mm256_set_pd(_creal(matrix[1]), _cimag(matrix[1]),
            _creal(matrix[1]), _cimag(matrix[1]));
        __m256d mv20 = _mm256_set_pd(-_cimag(matrix[2]), _creal(matrix[2]),
            -_cimag(matrix[2]), _creal(matrix[2]));
        __m256d mv21 = _mm256_set_pd(_creal(matrix[2]), _cimag(matrix[2]),
            _creal(matrix[2]), _cimag(matrix[2]));
        __m256d mv30 = _mm256_set_pd(-_cimag(matrix[3]), _creal(matrix[3]),
            -_cimag(matrix[3]), _creal(matrix[3]));
        __m256d mv31 = _mm256_set_pd(_creal(matrix[3]), _cimag(matrix[3]),
            _creal(matrix[3]), _cimag(matrix[3]));
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_0 =
                (state_index & mask_low) + ((state_index & mask_high) << 1);
            ITYPE basis_1 = basis_0 + mask;
            double* ptr0 = (double*)(state + basis_0);
            double* ptr1 = (double*)(state + basis_1);
            __m256d data0 = _mm256_loadu_pd(ptr0);
            __m256d data1 = _mm256_loadu_pd(ptr1);

            __m256d data_u2 = _mm256_mul_pd(data0, mv00);
            __m256d data_u3 = _mm256_mul_pd(data1, mv10);
            __m256d data_u4 = _mm256_mul_pd(data0, mv01);
            __m256d data_u5 = _mm256_mul_pd(data1, mv11);

            __m256d data_u6 = _mm256_hadd_pd(data_u2, data_u4);
            __m256d data_u7 = _mm256_hadd_pd(data_u3, data_u5);

            __m256d data_d2 = _mm256_mul_pd(data0, mv20);
            __m256d data_d3 = _mm256_mul_pd(data1, mv30);
            __m256d data_d4 = _mm256_mul_pd(data0, mv21);
            __m256d data_d5 = _mm256_mul_pd(data1, mv31);

            __m256d data_d6 = _mm256_hadd_pd(data_d2, data_d4);
            __m256d data_d7 = _mm256_hadd_pd(data_d3, data_d5);

            __m256d data_r0 = _mm256_add_pd(data_u6, data_u7);
            __m256d data_r1 = _mm256_add_pd(data_d6, data_d7);

            _mm256_storeu_pd(ptr0, data_r0);
            _mm256_storeu_pd(ptr1, data_r1);
        }
    }
}
#endif

#ifdef _USE_MPI
void single_qubit_dense_matrix_gate_partial(
    CTYPE* t, const CTYPE matrix[4], CTYPE* state, ITYPE dim, int flag);

void single_qubit_dense_matrix_gate_mpi(UINT target_qubit_index,
    const CTYPE matrix[4], CTYPE* state, ITYPE dim, UINT inner_qc) {
    if (target_qubit_index < inner_qc) {
        single_qubit_dense_matrix_gate(target_qubit_index, matrix, state, dim);
    } else {
        const MPIutil m = get_mpiutil();
        const int rank = m->get_rank();
        ITYPE dim_work = dim;
        ITYPE num_work = 0;
        CTYPE* ptr_pair = m->get_workarea(&dim_work, &num_work);
        assert(num_work > 0);
        const int pair_rank_bit = 1 << (target_qubit_index - inner_qc);
        const int pair_rank = rank ^ pair_rank_bit;
        CTYPE* ptr_state = state;

#ifdef _OPENMP
        OMPutil omputil = get_omputil();
        omputil->set_qulacs_num_threads(dim, 13);
#endif
        for (ITYPE iter = 0; iter < num_work; ++iter) {
            m->m_DC_sendrecv(ptr_state, ptr_pair, dim_work, pair_rank);

            single_qubit_dense_matrix_gate_partial(
                ptr_pair, matrix, ptr_state, dim_work, rank & pair_rank_bit);

            ptr_state += dim_work;
        }
#ifdef _OPENMP
        omputil->reset_qulacs_num_threads();
#endif
    }
}

void single_qubit_dense_matrix_gate_partial(
    CTYPE* t, const CTYPE matrix[4], CTYPE* state, ITYPE dim, int flag) {
    {
#pragma omp parallel for
        for (ITYPE state_index = 0; state_index < dim; ++state_index) {
            if (flag) {  // val=1
                // fetch values
                CTYPE cval_0 = t[state_index];
                CTYPE cval_1 = state[state_index];

                // set values
                state[state_index] = matrix[2] * cval_0 + matrix[3] * cval_1;
            } else {  // val=0
                // fetch values
                CTYPE cval_0 = state[state_index];
                CTYPE cval_1 = t[state_index];

                // set values
                state[state_index] = matrix[0] * cval_0 + matrix[1] * cval_1;
            }
        }
    }
}
#endif
