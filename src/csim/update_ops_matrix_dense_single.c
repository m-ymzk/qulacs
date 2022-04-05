
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

void single_qubit_dense_matrix_gate(
    UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
#if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
// ARMV8.2-A + SVE
#ifdef _OPENMP
    UINT threshold = 13;
    UINT default_thread_count = omp_get_max_threads();
    if (dim < (((ITYPE)1) << threshold)) omp_set_num_threads(1);
#endif
    single_qubit_dense_matrix_gate_sve(target_qubit_index, matrix, state, dim);
#ifdef _OPENMP
    omp_set_num_threads(default_thread_count);
#endif

#elif _USE_SIMD

#ifdef _OPENMP
    UINT threshold = 13;
    if (dim < (((ITYPE)1) << threshold)) {
        single_qubit_dense_matrix_gate_single_simd(
            target_qubit_index, matrix, state, dim);
    } else {
        single_qubit_dense_matrix_gate_parallel_simd(
            target_qubit_index, matrix, state, dim);
    }
#else   // #ifdef _USE_SIMD
    single_qubit_dense_matrix_gate_single_simd(
        target_qubit_index, matrix, state, dim);
#endif  // #ifdef _OPENMP

#else  // #ifdef _USE_SIMD

#ifdef _OPENMP
    UINT threshold = 13;
    if (dim < (((ITYPE)1) << threshold)) {
        single_qubit_dense_matrix_gate_single(
            target_qubit_index, matrix, state, dim);
    } else {
        single_qubit_dense_matrix_gate_parallel(
            target_qubit_index, matrix, state, dim);
    }
#else   // #ifdef _OEPNMP
    single_qubit_dense_matrix_gate_single(
        target_qubit_index, matrix, state, dim);
#endif  // #ifdef _OPENMP
#endif  // #ifdef _USE_SIMD
}

void single_qubit_dense_matrix_gate_single(
    UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;

    ITYPE state_index = 0;
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

#ifdef _OPENMP
void single_qubit_dense_matrix_gate_parallel(
    UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;

    ITYPE state_index = 0;
#pragma omp parallel for
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
#endif

void single_qubit_dense_matrix_gate_single_unroll(
    UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;

    if (target_qubit_index == 0) {
        ITYPE basis = 0;
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

#ifdef _OPENMP
void single_qubit_dense_matrix_gate_parallel_unroll(
    UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;

    if (target_qubit_index == 0) {
        ITYPE basis = 0;
#pragma omp parallel for
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
#pragma omp parallel for
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
#endif

#if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)

static inline void MatrixVectorProduct2x2(SV_PRED pg, SV_FTYPE in00r,
    SV_FTYPE in00i, SV_FTYPE in11r, SV_FTYPE in11i, SV_FTYPE mat02r,
    SV_FTYPE mat02i, SV_FTYPE mat13r, SV_FTYPE mat13i, SV_FTYPE *out01r,
    SV_FTYPE *out01i);

// clang-format off
/*
 * This function performs multiplication of a 2x2 matrix and four vectors
 *
 *            2x2 matrix                           four vectors
 * [ x_00 + iy_00, x_01+iy_01]   [ a_0+ib_0 ][ c_0+id_0 ][ e_0+if_0 ][ g_0+ih_0 ]
 * [ x_10 + iy_10, x_11+iy_11] * [ a_1+ib_1 ][ c_1+id_1 ][ e_1+if_1 ][ g_1+ih_1 ]
 *
 * params
 * - pg: All 1's predecate register 
 * - in00r: An SVE register has the real part of the first component in each vector
 *   - e.g.) 512-bit SVE & FP64: [ a_0, c_0, e_0, g_0, a_0, c_0, e_0, g_0]
 * - in00i: An SVE register has the imag. part of the first component in each vector
 *   - e.g.) 512-bit SVE & FP64: [ b_0, d_0, f_0, h_0, b_0, d_0, f_0, h_0]
 * - in11r: An SVE register has the real part of the second component in each vector
 *   - e.g.) 512-bit SVE & FP64: [ a_1, c_1, e_1, g_1, a_1, c_1, e_1, g_1]
 * - in11i: An SVE register has the imag. part of the second component in each vector
 *   - e.g.) 512-bit SVE & FP64: [ b_1, d_1, f_1, h_1, b_1, d_1, f_1, h_1]
 * - mat02r: An SVE register has the real part of the first column of the matrix
 *   - e.g.) 512-bit SVE & FP64: [ x_00, x_00, x_00, x_00, x_10, x_10, x_10, x_10]
 * - mat02i: An SVE register has the imag. part of the first column of the matrix
 *   - e.g.) 512-bit SVE & FP64: [ y_00, y_00, y_00, y_00, y_10, y_10, y_10, y_10]
 * - mat13r: An SVE register has the real part of the second column of the matrix
 *   - e.g.) 512-bit SVE & FP64: [ x_01, x_01, x_01, x_01, x_11, x_11, x_11, x_11]
 * - mat13i: An SVE register has the imag. part of the second column of the matrix
 *   - e.g.) 512-bit SVE & FP64: [ y_01, y_01, y_01, y_01, y_11, y_11, y_11, y_11]
 * - out*: SVE register store results of matrix-vector products
 *
 */
// clang-format on

static inline void MatrixVectorProduct2x2(SV_PRED pg, SV_FTYPE in00r,
    SV_FTYPE in00i, SV_FTYPE in11r, SV_FTYPE in11i, SV_FTYPE mat02r,
    SV_FTYPE mat02i, SV_FTYPE mat13r, SV_FTYPE mat13i, SV_FTYPE *out01r,
    SV_FTYPE *out01i) {
    *out01r = svmul_x(pg, in00r, mat02r);
    *out01i = svmul_x(pg, in00i, mat02r);

    *out01r = svmsb_x(pg, in00i, mat02i, *out01r);
    *out01i = svmad_x(pg, in00r, mat02i, *out01i);

    *out01r = svmad_x(pg, in11r, mat13r, *out01r);
    *out01i = svmad_x(pg, in11r, mat13i, *out01i);

    *out01r = svmsb_x(pg, in11i, mat13i, *out01r);
    *out01i = svmad_x(pg, in11i, mat13r, *out01i);
}

void single_qubit_dense_matrix_gate_sve(
    UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;

    ITYPE state_index = 0;

    // Get # of elements in SVE registers
    // note: # of complex numbers is halved.
    ITYPE vec_len = getVecLength();

    if (dim >= vec_len) {
        // Create an all 1's predicate variable
        SV_PRED pg = Svptrue();

        SV_FTYPE mat02r, mat02i, mat13r, mat13i;

        // Load matrix elements to SVE variables
        // e.g.) mat02_real is [matrix[0].real, matrix[0].real, matrix[2].real,
        // matrix[2].real],
        //       if # of elements in SVE variables is four.
        mat02r = svuzp1(SvdupF(creal(matrix[0])), SvdupF(creal(matrix[2])));
        mat02i = svuzp1(SvdupF(cimag(matrix[0])), SvdupF(cimag(matrix[2])));
        mat13r = svuzp1(SvdupF(creal(matrix[1])), SvdupF(creal(matrix[3])));
        mat13i = svuzp1(SvdupF(cimag(matrix[1])), SvdupF(cimag(matrix[3])));

        if (mask >= (vec_len >> 1)) {
            // If the above condition is met, the continuous elements loaded
            // into SVE variables can be applied without reordering.

#pragma omp parallel for
            for (state_index = 0; state_index < loop_dim;
                 state_index += (vec_len >> 1)) {
                // Calculate indices
                ITYPE basis_0 =
                    (state_index & mask_low) + ((state_index & mask_high) << 1);
                ITYPE basis_1 = basis_0 + mask;

                // Load values
                SV_FTYPE input0 = svld1(pg, (ETYPE *)&state[basis_0]);
                SV_FTYPE input1 = svld1(pg, (ETYPE *)&state[basis_1]);

                // Select odd or even elements from two vectors
                SV_FTYPE cval00r = svuzp1(input0, input0);
                SV_FTYPE cval00i = svuzp2(input0, input0);
                SV_FTYPE cval11r = svuzp1(input1, input1);
                SV_FTYPE cval11i = svuzp2(input1, input1);

                // Perform matrix-vector products
                SV_FTYPE result01r, result01i;
                MatrixVectorProduct2x2(pg, cval00r, cval00i, cval11r, cval11i,
                    mat02r, mat02i, mat13r, mat13i, &result01r, &result01i);
                // Interleave elements from low or high halves of two vectors
                SV_FTYPE output0 = svzip1(result01r, result01i);
                SV_FTYPE output1 = svzip2(result01r, result01i);

                if (5 <= target_qubit_index && target_qubit_index <= 10) {
                    // L1 prefetch
                    __builtin_prefetch(&state[basis_0 + mask * 4], 1, 3);
                    __builtin_prefetch(&state[basis_1 + mask * 4], 1, 3);
                    // L2 prefetch
                    __builtin_prefetch(&state[basis_0 + mask * 8], 1, 2);
                    __builtin_prefetch(&state[basis_1 + mask * 8], 1, 2);
                }

                // Store values
                svst1(pg, (ETYPE *)&state[basis_0], output0);
                svst1(pg, (ETYPE *)&state[basis_1], output1);
            }
        } else {
            // In this case, the reordering between two SVE variables is
            // performed before and after the matrix-vector product.

            // Define a predicate variable for reordering
            SV_PRED select_flag;

            // Define SVE variables for reordering
            SV_ITYPE vec_shuffle_table, vec_index;

            // Prepare a table and a flag for reordering
            vec_index = SvindexI(0, 1);             // [0, 1, 2, 3, 4, .., 7]
            vec_index = svlsr_z(pg, vec_index, 1);  // [0, 0, 1, 1, 2, ..., 3]
            select_flag = svcmpne(pg, SvdupI(0),
                svand_z(pg, vec_index, SvdupI(1ULL << target_qubit_index)));
            vec_shuffle_table = sveor_z(
                pg, SvindexI(0, 1), SvdupI(1ULL << (target_qubit_index + 1)));

#pragma omp parallel for
            for (state_index = 0; state_index < dim; state_index += vec_len) {
                // fetch values
                SV_FTYPE input0 = svld1(pg, (ETYPE *)&state[state_index]);
                SV_FTYPE input1 =
                    svld1(pg, (ETYPE *)&state[state_index + (vec_len >> 1)]);

                // Reordering input vectors
                SV_FTYPE reordered0 = svsel(
                    select_flag, svtbl(input1, vec_shuffle_table), input0);
                SV_FTYPE reordered1 = svsel(
                    select_flag, input1, svtbl(input0, vec_shuffle_table));

                // Select odd or even elements from two vectors
                SV_FTYPE cval00r = svuzp1(reordered0, reordered0);
                SV_FTYPE cval00i = svuzp2(reordered0, reordered0);
                SV_FTYPE cval11r = svuzp1(reordered1, reordered1);
                SV_FTYPE cval11i = svuzp2(reordered1, reordered1);

                // Perform matrix-vector products
                SV_FTYPE result01r, result01i;
                MatrixVectorProduct2x2(pg, cval00r, cval00i, cval11r, cval11i,
                    mat02r, mat02i, mat13r, mat13i, &result01r, &result01i);

                // Interleave elements from low or high halves of two vectors
                reordered0 = svzip1(result01r, result01i);
                reordered1 = svzip2(result01r, result01i);

                // Reordering output vectors
                SV_FTYPE output0 = svsel(select_flag,
                    svtbl(reordered1, vec_shuffle_table), reordered0);
                SV_FTYPE output1 = svsel(select_flag, reordered1,
                    svtbl(reordered0, vec_shuffle_table));

                // Store values
                svst1(pg, (ETYPE *)&state[state_index], output0);
                svst1(
                    pg, (ETYPE *)&state[state_index + (vec_len >> 1)], output1);
            }
        }
    } else {
#pragma omp parallel for
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
}
#endif  // #if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)

#ifdef _USE_SIMD
void single_qubit_dense_matrix_gate_single_simd(
    UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;

    if (target_qubit_index == 0) {
        ITYPE basis = 0;
        __m256d mv00 = _mm256_set_pd(-cimag(matrix[1]), creal(matrix[1]),
            -cimag(matrix[0]), creal(matrix[0]));
        __m256d mv01 = _mm256_set_pd(creal(matrix[1]), cimag(matrix[1]),
            creal(matrix[0]), cimag(matrix[0]));
        __m256d mv20 = _mm256_set_pd(-cimag(matrix[3]), creal(matrix[3]),
            -cimag(matrix[2]), creal(matrix[2]));
        __m256d mv21 = _mm256_set_pd(creal(matrix[3]), cimag(matrix[3]),
            creal(matrix[2]), cimag(matrix[2]));
        for (basis = 0; basis < dim; basis += 2) {
            double *ptr = (double *)(state + basis);
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
                (state_index & mask_low) + ((state_index & mask_high) << 1);
            ITYPE basis_1 = basis_0 + mask;
            double *ptr0 = (double *)(state + basis_0);
            double *ptr1 = (double *)(state + basis_1);
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

void single_qubit_dense_matrix_gate_parallel_simd(
    UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;

    if (target_qubit_index == 0) {
        ITYPE basis = 0;
        __m256d mv00 = _mm256_set_pd(-cimag(matrix[1]), creal(matrix[1]),
            -cimag(matrix[0]), creal(matrix[0]));
        __m256d mv01 = _mm256_set_pd(creal(matrix[1]), cimag(matrix[1]),
            creal(matrix[0]), cimag(matrix[0]));
        __m256d mv20 = _mm256_set_pd(-cimag(matrix[3]), creal(matrix[3]),
            -cimag(matrix[2]), creal(matrix[2]));
        __m256d mv21 = _mm256_set_pd(creal(matrix[3]), cimag(matrix[3]),
            creal(matrix[2]), cimag(matrix[2]));
#pragma omp parallel for
        for (basis = 0; basis < dim; basis += 2) {
            double *ptr = (double *)(state + basis);
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
                (state_index & mask_low) + ((state_index & mask_high) << 1);
            ITYPE basis_1 = basis_0 + mask;
            double *ptr0 = (double *)(state + basis_0);
            double *ptr1 = (double *)(state + basis_1);
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
void single_qubit_dense_matrix_gate_mpi(UINT target_qubit_index,
    const CTYPE matrix[4], CTYPE *state, ITYPE dim, UINT inner_qc) {
    if (target_qubit_index < inner_qc) {
        single_qubit_dense_matrix_gate(target_qubit_index, matrix, state, dim);
    } else {
        const MPIutil m = get_mpiutil();
        const int rank = m->get_rank();
        ITYPE dim_work = dim;
        ITYPE num_work = 0;
        CTYPE *t = m->get_workarea(&dim_work, &num_work);
        assert(num_work > 0);
        const int pair_rank_bit = 1 << (target_qubit_index - inner_qc);
        const int pair_rank = rank ^ pair_rank_bit;
        CTYPE *si = state;

#ifdef _OPENMP
        UINT threshold = 13;
        UINT default_thread_count = omp_get_max_threads();
        if (dim < (((ITYPE)1) << threshold)) omp_set_num_threads(1);
#endif
        for (ITYPE iter = 0; iter < num_work; ++iter) {
            m->m_DC_sendrecv(si, t, dim_work, pair_rank);

            _single_qubit_dense_matrix_gate_mpi(
                t, matrix, si, dim_work, rank & pair_rank_bit);

            si += dim_work;
        }
#ifdef _OPENMP
        omp_set_num_threads(default_thread_count);
#endif
    }
}

#ifdef _OPENMP
void _single_qubit_dense_matrix_gate_mpi(
    CTYPE *t, const CTYPE matrix[4], CTYPE *state, ITYPE dim, int flag) {
#if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
    ITYPE vec_len =
        getVecLength();  // length of SVE registers (# of 64-bit elements)
    if (dim >= vec_len) {
        CTYPE *s0, *s1;
        s0 = (flag) ? t : state;
        s1 = (flag) ? state : t;

        SV_PRED pg = Svptrue();  // this predicate register is all 1.

        // SVE registers for matrix[4]
        SV_FTYPE mat0r, mat0i, mat1r, mat1i;

        // load matrix elements
        mat0r = SvdupF(creal(matrix[0 + (flag != 0) * 2]));
        mat0i = SvdupF(cimag(matrix[0 + (flag != 0) * 2]));
        mat1r = SvdupF(creal(matrix[1 + (flag != 0) * 2]));
        mat1i = SvdupF(cimag(matrix[1 + (flag != 0) * 2]));

#pragma omp parallel for shared(pg, mat0r, mat0i, mat1r, mat1i)
        for (ITYPE state_index = 0; state_index < dim; state_index += vec_len) {
            // fetch values
            SV_FTYPE input0 = svld1(pg, (ETYPE *)&s0[state_index]);
            SV_FTYPE input1 = svld1(pg, (ETYPE *)&s1[state_index]);
            SV_FTYPE input2 =
                svld1(pg, (ETYPE *)&s0[state_index + (vec_len >> 1)]);
            SV_FTYPE input3 =
                svld1(pg, (ETYPE *)&s1[state_index + (vec_len >> 1)]);

            // select odd or even elements from two vectors
            SV_FTYPE cval02r = svuzp1(input0, input2);
            SV_FTYPE cval02i = svuzp2(input0, input2);
            SV_FTYPE cval13r = svuzp1(input1, input3);
            SV_FTYPE cval13i = svuzp2(input1, input3);

            // perform matrix-vector product
            SV_FTYPE result01r = svmul_x(pg, cval02r, mat0r);
            SV_FTYPE result01i = svmul_x(pg, cval02i, mat0r);

            result01r = svmsb_x(pg, cval02i, mat0i, result01r);
            result01i = svmad_x(pg, cval02r, mat0i, result01i);

            result01r = svmad_x(pg, cval13r, mat1r, result01r);
            result01i = svmad_x(pg, cval13r, mat1i, result01i);

            result01r = svmsb_x(pg, cval13i, mat1i, result01r);
            result01i = svmad_x(pg, cval13i, mat1r, result01i);

            // interleave elements from low halves of two vectors
            SV_FTYPE output0 = svzip1(result01r, result01i);
            SV_FTYPE output1 = svzip2(result01r, result01i);

            // set values
            svst1(pg, (ETYPE *)&state[state_index], output0);
            svst1(pg, (ETYPE *)&state[state_index + (vec_len >> 1)], output1);
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
#endif  //#ifdef _USE_MPI
