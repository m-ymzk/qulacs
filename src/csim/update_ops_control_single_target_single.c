
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constant.h"
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

void single_qubit_control_single_qubit_dense_matrix_gate(
    UINT control_qubit_index, UINT control_value, UINT target_qubit_index,
    const CTYPE matrix[4], CTYPE* state, ITYPE dim) {
#ifdef _USE_SIMD
#ifdef _OPENMP
    UINT threshold = 13;
    if (dim < (((ITYPE)1) << threshold)) {
        single_qubit_control_single_qubit_dense_matrix_gate_single_simd(
            control_qubit_index, control_value, target_qubit_index, matrix,
            state, dim);
    } else {
        single_qubit_control_single_qubit_dense_matrix_gate_parallel_simd(
            control_qubit_index, control_value, target_qubit_index, matrix,
            state, dim);
    }
#else
    single_qubit_control_single_qubit_dense_matrix_gate_single_simd(
        control_qubit_index, control_value, target_qubit_index, matrix, state,
        dim);
#endif
#else
#ifdef _OPENMP
    UINT threshold = 13;
    if (dim < (((ITYPE)1) << threshold)) {
        single_qubit_control_single_qubit_dense_matrix_gate_single_unroll(
            control_qubit_index, control_value, target_qubit_index, matrix,
            state, dim);
    } else {
        single_qubit_control_single_qubit_dense_matrix_gate_parallel_unroll(
            control_qubit_index, control_value, target_qubit_index, matrix,
            state, dim);
    }
#else
    single_qubit_control_single_qubit_dense_matrix_gate_single_unroll(
        control_qubit_index, control_value, target_qubit_index, matrix, state,
        dim);
#endif
#endif
}

void single_qubit_control_single_qubit_dense_matrix_gate_single_unroll(
    UINT control_qubit_index, UINT control_value, UINT target_qubit_index,
    const CTYPE matrix[4], CTYPE* state, ITYPE dim) {
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

    ITYPE state_index;
    if (target_qubit_index == 0) {
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask * control_value;

            // fetch values
            CTYPE cval0 = state[basis_index];
            CTYPE cval1 = state[basis_index + 1];

            // set values
            state[basis_index] = matrix[0] * cval0 + matrix[1] * cval1;
            state[basis_index + 1] = matrix[2] * cval0 + matrix[3] * cval1;
        }
    } else if (control_qubit_index == 0) {
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask * control_value;
            ITYPE basis_index_1 = basis_index_0 + target_mask;

            // fetch values
            CTYPE cval0 = state[basis_index_0];
            CTYPE cval1 = state[basis_index_1];

            // set values
            state[basis_index_0] = matrix[0] * cval0 + matrix[1] * cval1;
            state[basis_index_1] = matrix[2] * cval0 + matrix[3] * cval1;
        }
    } else {
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask * control_value;
            ITYPE basis_index_1 = basis_index_0 + target_mask;

            // fetch values
            CTYPE cval0 = state[basis_index_0];
            CTYPE cval1 = state[basis_index_1];
            CTYPE cval2 = state[basis_index_0 + 1];
            CTYPE cval3 = state[basis_index_1 + 1];

            // set values
            state[basis_index_0] = matrix[0] * cval0 + matrix[1] * cval1;
            state[basis_index_1] = matrix[2] * cval0 + matrix[3] * cval1;
            state[basis_index_0 + 1] = matrix[0] * cval2 + matrix[1] * cval3;
            state[basis_index_1 + 1] = matrix[2] * cval2 + matrix[3] * cval3;
        }
    }
}

#ifdef _OPENMP
void single_qubit_control_single_qubit_dense_matrix_gate_parallel_unroll(
    UINT control_qubit_index, UINT control_value, UINT target_qubit_index,
    const CTYPE matrix[4], CTYPE* state, ITYPE dim) {
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

    ITYPE state_index;
    if (target_qubit_index == 0) {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask * control_value;

            // fetch values
            CTYPE cval0 = state[basis_index];
            CTYPE cval1 = state[basis_index + 1];

            // set values
            state[basis_index] = matrix[0] * cval0 + matrix[1] * cval1;
            state[basis_index + 1] = matrix[2] * cval0 + matrix[3] * cval1;
        }
    } else if (control_qubit_index == 0) {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask * control_value;
            ITYPE basis_index_1 = basis_index_0 + target_mask;

            // fetch values
            CTYPE cval0 = state[basis_index_0];
            CTYPE cval1 = state[basis_index_1];

            // set values
            state[basis_index_0] = matrix[0] * cval0 + matrix[1] * cval1;
            state[basis_index_1] = matrix[2] * cval0 + matrix[3] * cval1;
        }
    } else {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask * control_value;
            ITYPE basis_index_1 = basis_index_0 + target_mask;

            // fetch values
            CTYPE cval0 = state[basis_index_0];
            CTYPE cval1 = state[basis_index_1];
            CTYPE cval2 = state[basis_index_0 + 1];
            CTYPE cval3 = state[basis_index_1 + 1];

            // set values
            state[basis_index_0] = matrix[0] * cval0 + matrix[1] * cval1;
            state[basis_index_1] = matrix[2] * cval0 + matrix[3] * cval1;
            state[basis_index_0 + 1] = matrix[0] * cval2 + matrix[1] * cval3;
            state[basis_index_1 + 1] = matrix[2] * cval2 + matrix[3] * cval3;
        }
    }
}
#endif

#ifdef _USE_SIMD
void single_qubit_control_single_qubit_dense_matrix_gate_single_simd(
    UINT control_qubit_index, UINT control_value, UINT target_qubit_index,
    const CTYPE matrix[4], CTYPE* state, ITYPE dim) {
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

    ITYPE state_index;
    if (target_qubit_index == 0) {
        __m256d mv00 = _mm256_set_pd(-cimag(matrix[1]), creal(matrix[1]),
            -cimag(matrix[0]), creal(matrix[0]));
        __m256d mv01 = _mm256_set_pd(creal(matrix[1]), cimag(matrix[1]),
            creal(matrix[0]), cimag(matrix[0]));
        __m256d mv20 = _mm256_set_pd(-cimag(matrix[3]), creal(matrix[3]),
            -cimag(matrix[2]), creal(matrix[2]));
        __m256d mv21 = _mm256_set_pd(creal(matrix[3]), cimag(matrix[3]),
            creal(matrix[2]), cimag(matrix[2]));
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask * control_value;
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
    } else if (control_qubit_index == 0) {
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask * control_value;
            ITYPE basis_index_1 = basis_index_0 + target_mask;

            // fetch values
            CTYPE cval0 = state[basis_index_0];
            CTYPE cval1 = state[basis_index_1];

            // set values
            state[basis_index_0] = matrix[0] * cval0 + matrix[1] * cval1;
            state[basis_index_1] = matrix[2] * cval0 + matrix[3] * cval1;
        }
    } else {
        __m256d mv00 = _mm256_set_pd(-cimag(matrix[0]), creal(matrix[0]),
            -cimag(matrix[0]), creal(matrix[0]));
        __m256d mv01 = _mm256_set_pd(creal(matrix[0]), cimag(matrix[0]),
            creal(matrix[0]), cimag(matrix[0]));
        __m256d mv10 = _mm256_set_pd(-cimag(matrix[1]), creal(matrix[1]),
            -cimag(matrix[1]), creal(matrix[1]));
        __m256d mv11 = _mm256_set_pd(creal(matrix[1]), cimag(matrix[1]),
            creal(matrix[1]), cimag(matrix[1]));
        __m256d mv20 = _mm256_set_pd(-cimag(matrix[2]), creal(matrix[2]),
            -cimag(matrix[2]), creal(matrix[2]));
        __m256d mv21 = _mm256_set_pd(creal(matrix[2]), cimag(matrix[2]),
            creal(matrix[2]), cimag(matrix[2]));
        __m256d mv30 = _mm256_set_pd(-cimag(matrix[3]), creal(matrix[3]),
            -cimag(matrix[3]), creal(matrix[3]));
        __m256d mv31 = _mm256_set_pd(creal(matrix[3]), cimag(matrix[3]),
            creal(matrix[3]), cimag(matrix[3]));
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask * control_value;
            ITYPE basis_1 = basis_0 + target_mask;

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

#ifdef _OPENMP
void single_qubit_control_single_qubit_dense_matrix_gate_parallel_simd(
    UINT control_qubit_index, UINT control_value, UINT target_qubit_index,
    const CTYPE matrix[4], CTYPE* state, ITYPE dim) {
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

    ITYPE state_index;
    if (target_qubit_index == 0) {
        __m256d mv00 = _mm256_set_pd(-cimag(matrix[1]), creal(matrix[1]),
            -cimag(matrix[0]), creal(matrix[0]));
        __m256d mv01 = _mm256_set_pd(creal(matrix[1]), cimag(matrix[1]),
            creal(matrix[0]), cimag(matrix[0]));
        __m256d mv20 = _mm256_set_pd(-cimag(matrix[3]), creal(matrix[3]),
            -cimag(matrix[2]), creal(matrix[2]));
        __m256d mv21 = _mm256_set_pd(creal(matrix[3]), cimag(matrix[3]),
            creal(matrix[2]), cimag(matrix[2]));
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask * control_value;
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
    } else if (control_qubit_index == 0) {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask * control_value;
            ITYPE basis_index_1 = basis_index_0 + target_mask;

            // fetch values
            CTYPE cval0 = state[basis_index_0];
            CTYPE cval1 = state[basis_index_1];

            // set values
            state[basis_index_0] = matrix[0] * cval0 + matrix[1] * cval1;
            state[basis_index_1] = matrix[2] * cval0 + matrix[3] * cval1;
        }
    } else {
        __m256d mv00 = _mm256_set_pd(-cimag(matrix[0]), creal(matrix[0]),
            -cimag(matrix[0]), creal(matrix[0]));
        __m256d mv01 = _mm256_set_pd(creal(matrix[0]), cimag(matrix[0]),
            creal(matrix[0]), cimag(matrix[0]));
        __m256d mv10 = _mm256_set_pd(-cimag(matrix[1]), creal(matrix[1]),
            -cimag(matrix[1]), creal(matrix[1]));
        __m256d mv11 = _mm256_set_pd(creal(matrix[1]), cimag(matrix[1]),
            creal(matrix[1]), cimag(matrix[1]));
        __m256d mv20 = _mm256_set_pd(-cimag(matrix[2]), creal(matrix[2]),
            -cimag(matrix[2]), creal(matrix[2]));
        __m256d mv21 = _mm256_set_pd(creal(matrix[2]), cimag(matrix[2]),
            creal(matrix[2]), cimag(matrix[2]));
        __m256d mv30 = _mm256_set_pd(-cimag(matrix[3]), creal(matrix[3]),
            -cimag(matrix[3]), creal(matrix[3]));
        __m256d mv31 = _mm256_set_pd(creal(matrix[3]), cimag(matrix[3]),
            creal(matrix[3]), cimag(matrix[3]));
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask * control_value;
            ITYPE basis_1 = basis_0 + target_mask;

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
#endif

#ifdef _USE_MPI
void single_qubit_control_single_qubit_dense_matrix_gate_mpi(
    UINT control_qubit_index, UINT control_value, UINT target_qubit_index,
    const CTYPE matrix[4], CTYPE* state, ITYPE dim, UINT inner_qc) {
    const MPIutil m = get_mpiutil();
    const UINT rank = m->get_rank();
    const UINT control_rank_bit = 1 << (control_qubit_index - inner_qc);
    ITYPE dim_work = dim;
    ITYPE num_work = 0;
    CTYPE* t = m->get_workarea(&dim_work, &num_work);
    assert(num_work > 0);
    const int pair_rank_bit = 1 << (target_qubit_index - inner_qc);
    const int pair_rank = rank ^ pair_rank_bit;
    CTYPE* si = state;

    if (control_qubit_index < inner_qc) {     // control_qubit_index is in inner
        if (target_qubit_index < inner_qc) {  // target_qubit_index is in inner

            single_qubit_control_single_qubit_dense_matrix_gate(
                control_qubit_index, control_value, target_qubit_index, matrix,
                state, dim);

        } else {  // target_qubit_index is outer

#ifdef _OPENMP
            UINT threshold = 13;
            UINT default_thread_count = omp_get_max_threads();
            if (dim < (((ITYPE)1) << threshold)) omp_set_num_threads(1);
#endif

            for (ITYPE iter = 0; iter < num_work; ++iter) {
                m->m_DC_sendrecv(si, t, dim_work, pair_rank);

                UINT index_offset = iter * dim_work;
                single_qubit_control_single_qubit_dense_matrix_gate_mpi_OI(
                    control_qubit_index, control_value, t, matrix, si, dim_work,
                    rank & pair_rank_bit, index_offset);

                si += dim_work;
            }

#ifdef _OPENMP
            omp_set_num_threads(default_thread_count);
#endif
        }
    } else {                                  // control_qubit_index is outer
        if (target_qubit_index < inner_qc) {  // target_qubit_index is in inner
            if (((rank & control_rank_bit) && (control_value == 1)) ||
                (!(rank & control_rank_bit) && (control_value == 0)))
                single_qubit_dense_matrix_gate(
                    target_qubit_index, matrix, state, dim);
        } else {  // target_qubit_index is outer

#ifdef _OPENMP
            UINT threshold = 13;
            UINT default_thread_count = omp_get_max_threads();
            if (dim < (((ITYPE)1) << threshold)) omp_set_num_threads(1);
#endif

            ITYPE dummy_flag =
                !(((rank & control_rank_bit) && (control_value == 1)) ||
                    (!(rank & control_rank_bit) && (control_value == 0)));
            for (ITYPE iter = 0; iter < num_work; ++iter) {
                if (dummy_flag) {  // only count up tag
                    m->get_tag();
                } else {
                    m->m_DC_sendrecv(si, t, dim_work, pair_rank);

                    single_qubit_control_single_qubit_dense_matrix_gate_mpi_OO(
                        t, matrix, si, dim_work, rank & pair_rank_bit);

                    si += dim_work;
                }
            }

#ifdef _OPENMP
            omp_set_num_threads(default_thread_count);
#endif
        }
    }
}

void single_qubit_control_single_qubit_dense_matrix_gate_mpi_OI(
    UINT control_qubit_index, UINT control_value, CTYPE* t,
    const CTYPE matrix[4], CTYPE* state, ITYPE dim, int flag,
    UINT index_offset) {
    UINT control_qubit_mask = 1ULL << control_qubit_index;

#pragma omp parallel for
    for (ITYPE state_index = 0; state_index < dim; ++state_index) {
        UINT skip_flag = (state_index + index_offset) & control_qubit_mask;
        skip_flag = skip_flag >> control_qubit_index;
        if (skip_flag != control_value) continue;

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

static inline void MatrixVectorProduct2x2Half(SV_PRED pg, SV_FTYPE mat0r,
    SV_FTYPE mat0i, SV_FTYPE mat1r, SV_FTYPE mat1i, SV_FTYPE input02r,
    SV_FTYPE input02i, SV_FTYPE input13r, SV_FTYPE input13i,
    SV_FTYPE* result01r, SV_FTYPE* result01i);

static inline void MatrixVectorProduct2x2Half(SV_PRED pg, SV_FTYPE mat0r,
    SV_FTYPE mat0i, SV_FTYPE mat1r, SV_FTYPE mat1i, SV_FTYPE input02r,
    SV_FTYPE input02i, SV_FTYPE input13r, SV_FTYPE input13i,
    SV_FTYPE* result01r, SV_FTYPE* result01i) {
    // perform matrix-vector product
    *result01r = svmul_x(pg, input02r, mat0r);
    *result01i = svmul_x(pg, input02i, mat0r);

    *result01r = svmsb_x(pg, input02i, mat0i, *result01r);
    *result01i = svmad_x(pg, input02r, mat0i, *result01i);

    *result01r = svmad_x(pg, input13r, mat1r, *result01r);
    *result01i = svmad_x(pg, input13r, mat1i, *result01i);

    *result01r = svmsb_x(pg, input13i, mat1i, *result01r);
    *result01i = svmad_x(pg, input13i, mat1r, *result01i);
}

void single_qubit_control_single_qubit_dense_matrix_gate_mpi_OO(
    CTYPE* t, const CTYPE matrix[4], CTYPE* state, ITYPE dim, int flag) {
#if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
    ITYPE vec_len =
        getVecLength();  // length of SVE registers (# of 64-bit elements)
    if (dim >= vec_len) {
        SV_PRED pg = Svptrue();  // this predicate register is all 1.

        if (flag) {
            // SVE registers for matrix[4]
            SV_FTYPE mat2r, mat2i, mat3r, mat3i;

            // load matrix elements
            mat2r = SvdupF(creal(matrix[2]));
            mat2i = SvdupF(cimag(matrix[2]));
            mat3r = SvdupF(creal(matrix[3]));
            mat3i = SvdupF(cimag(matrix[3]));

#pragma omp parallel for shared(pg, mat2r, mat2i, mat3r, mat3i)
            for (ITYPE state_index = 0; state_index < dim;
                 state_index += vec_len) {
                // fetch values
                SV_FTYPE input0 = svld1(pg, (ETYPE*)&t[state_index]);
                SV_FTYPE input1 = svld1(pg, (ETYPE*)&state[state_index]);
                SV_FTYPE input2 =
                    svld1(pg, (ETYPE*)&t[state_index + (vec_len >> 1)]);
                SV_FTYPE input3 =
                    svld1(pg, (ETYPE*)&state[state_index + (vec_len >> 1)]);

                // select odd or even elements from two vectors
                SV_FTYPE cval02r = svuzp1(input0, input2);
                SV_FTYPE cval02i = svuzp2(input0, input2);
                SV_FTYPE cval13r = svuzp1(input1, input3);
                SV_FTYPE cval13i = svuzp2(input1, input3);

                // perform matrix-vector product
                SV_FTYPE result01r, result01i;

                MatrixVectorProduct2x2Half(pg, mat2r, mat2i, mat3r, mat3i,
                    cval02r, cval02i, cval13r, cval13i, &result01r, &result01i);

                // interleave elements from low halves of two vectors
                SV_FTYPE output0 = svzip1(result01r, result01i);
                SV_FTYPE output1 = svzip2(result01r, result01i);

                // set values
                svst1(pg, (ETYPE*)&state[state_index], output0);
                svst1(
                    pg, (ETYPE*)&state[state_index + (vec_len >> 1)], output1);
            }
        } else {  // val=0
            // SVE registers for matrix[4]
            SV_FTYPE mat0r, mat0i, mat1r, mat1i;

            // load matrix elements
            mat0r = SvdupF(creal(matrix[0]));
            mat0i = SvdupF(cimag(matrix[0]));
            mat1r = SvdupF(creal(matrix[1]));
            mat1i = SvdupF(cimag(matrix[1]));

#pragma omp parallel for shared(pg, mat0r, mat0i, mat1r, mat1i)
            for (ITYPE state_index = 0; state_index < dim;
                 state_index += vec_len) {
                // fetch values
                SV_FTYPE input0 = svld1(pg, (ETYPE*)&state[state_index]);
                SV_FTYPE input1 = svld1(pg, (ETYPE*)&t[state_index]);
                SV_FTYPE input2 =
                    svld1(pg, (ETYPE*)&state[state_index + (vec_len >> 1)]);
                SV_FTYPE input3 =
                    svld1(pg, (ETYPE*)&t[state_index + (vec_len >> 1)]);

                // select odd or even elements from two vectors
                SV_FTYPE cval02r = svuzp1(input0, input2);
                SV_FTYPE cval02i = svuzp2(input0, input2);
                SV_FTYPE cval13r = svuzp1(input1, input3);
                SV_FTYPE cval13i = svuzp2(input1, input3);

                // perform matrix-vector product
                SV_FTYPE result01r, result01i;

                MatrixVectorProduct2x2Half(pg, mat0r, mat0i, mat1r, mat1i,
                    cval02r, cval02i, cval13r, cval13i, &result01r, &result01i);

                // interleave elements from low halves of two vectors
                SV_FTYPE output0 = svzip1(result01r, result01i);
                SV_FTYPE output1 = svzip2(result01r, result01i);

                // set values
                svst1(pg, (ETYPE*)&state[state_index], output0);
                svst1(
                    pg, (ETYPE*)&state[state_index + (vec_len >> 1)], output1);
            }
        }
    } else
#endif  // #if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
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
