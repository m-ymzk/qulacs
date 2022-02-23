
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

#ifdef _USE_SIMD
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

#ifdef _USE_SIMD
void double_qubit_dense_matrix_gate_simd_high(UINT target_qubit_index1,
    UINT target_qubit_index2, const CTYPE mat[16], CTYPE* vec, ITYPE dim);
void double_qubit_dense_matrix_gate_simd_middle(UINT target_qubit_index1,
    UINT target_qubit_index2, const CTYPE mat[16], CTYPE* vec, ITYPE dim);
void double_qubit_dense_matrix_gate_simd_low(UINT target_qubit_index1,
    UINT target_qubit_index2, const CTYPE mat[16], CTYPE* vec, ITYPE dim);
#endif

#if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
void double_qubit_dense_matrix_gate_sve_high(UINT target_qubit_index1,
    UINT target_qubit_index2, const CTYPE matrix[16], CTYPE* state, ITYPE dim);
void double_qubit_dense_matrix_gate_sve_middle(UINT target_qubit_index1,
    UINT target_qubit_index2, const CTYPE matrix[16], CTYPE* state, ITYPE dim);
void double_qubit_dense_matrix_gate_sve_low(UINT target_qubit_index1,
    UINT target_qubit_index2, const CTYPE matrix[16], CTYPE* state, ITYPE dim);
#endif  // #if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)

void double_qubit_dense_matrix_gate_c(UINT target_qubit_index1,
    UINT target_qubit_index2, const CTYPE matrix[16], CTYPE* state, ITYPE dim) {
#ifdef _OPENMP
    UINT threshold = 13;
    UINT default_thread_count = omp_get_max_threads();
    if (dim < (((ITYPE)1) << threshold)) omp_set_num_threads(1);
#endif

#ifdef _USE_SIMD
    double_qubit_dense_matrix_gate_simd(
        target_qubit_index1, target_qubit_index2, matrix, state, dim);
#elif defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
    double_qubit_dense_matrix_gate_sve(
        target_qubit_index1, target_qubit_index2, matrix, state, dim);
#else
    double_qubit_dense_matrix_gate_nosimd(
        target_qubit_index1, target_qubit_index2, matrix, state, dim);
#endif

#ifdef _OPENMP
    omp_set_num_threads(default_thread_count);
#endif
}

void double_qubit_dense_matrix_gate_nosimd(UINT target_qubit_index1,
    UINT target_qubit_index2, const CTYPE matrix[16], CTYPE* state, ITYPE dim) {
    const UINT min_qubit_index =
        get_min_ui(target_qubit_index1, target_qubit_index2);
    const UINT max_qubit_index =
        get_max_ui(target_qubit_index1, target_qubit_index2);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    const ITYPE target_mask1 = 1ULL << target_qubit_index1;
    const ITYPE target_mask2 = 1ULL << target_qubit_index2;

    // loop variables
    const ITYPE loop_dim = dim / 4;
    ITYPE state_index;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (state_index = 0; state_index < loop_dim; ++state_index) {
        // create index
        ITYPE basis_0 = (state_index & low_mask) +
                        ((state_index & mid_mask) << 1) +
                        ((state_index & high_mask) << 2);

        // gather index
        ITYPE basis_1 = basis_0 + target_mask1;
        ITYPE basis_2 = basis_0 + target_mask2;
        ITYPE basis_3 = basis_1 + target_mask2;

        // fetch values
        CTYPE cval_0 = state[basis_0];
        CTYPE cval_1 = state[basis_1];
        CTYPE cval_2 = state[basis_2];
        CTYPE cval_3 = state[basis_3];

        // set values
        state[basis_0] = matrix[0] * cval_0 + matrix[1] * cval_1 +
                         matrix[2] * cval_2 + matrix[3] * cval_3;
        state[basis_1] = matrix[4] * cval_0 + matrix[5] * cval_1 +
                         matrix[6] * cval_2 + matrix[7] * cval_3;
        state[basis_2] = matrix[8] * cval_0 + matrix[9] * cval_1 +
                         matrix[10] * cval_2 + matrix[11] * cval_3;
        state[basis_3] = matrix[12] * cval_0 + matrix[13] * cval_1 +
                         matrix[14] * cval_2 + matrix[15] * cval_3;
    }
}

#if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
static inline void MatrixVectorProduct4x4(SV_PRED pg, SV_FTYPE input0,
    SV_FTYPE input1, SV_FTYPE input2, SV_FTYPE input3, SV_FTYPE mat0,
    SV_FTYPE mat1, SV_FTYPE mat2, SV_FTYPE mat3, SV_FTYPE* output0,
    SV_FTYPE* output1, SV_FTYPE* output2, SV_FTYPE* output3);

static inline void MatrixVectorProduct4x4(SV_PRED pg, SV_FTYPE input0,
    SV_FTYPE input1, SV_FTYPE input2, SV_FTYPE input3, SV_FTYPE mat0,
    SV_FTYPE mat1, SV_FTYPE mat2, SV_FTYPE mat3, SV_FTYPE* output0,
    SV_FTYPE* output1, SV_FTYPE* output2, SV_FTYPE* output3) {
    // perform matrix-vector product
    *output0 = svmul_z(pg, svdup_lane(mat0, 0), input0);
    *output0 = svcmla_z(pg, *output0, svdupq_lane(mat0, 0), input0, 90);
    *output0 = svmla_z(pg, *output0, svdup_lane(mat0, 2), input1);
    *output0 = svcmla_z(pg, *output0, svdupq_lane(mat0, 1), input1, 90);
    *output0 = svmla_z(pg, *output0, svdup_lane(mat0, 4), input2);
    *output0 = svcmla_z(pg, *output0, svdupq_lane(mat0, 2), input2, 90);
    *output0 = svmla_z(pg, *output0, svdup_lane(mat0, 6), input3);
    *output0 = svcmla_z(pg, *output0, svdupq_lane(mat0, 3), input3, 90);

    *output1 = svmul_z(pg, svdup_lane(mat1, 0), input0);
    *output1 = svcmla_z(pg, *output1, svdupq_lane(mat1, 0), input0, 90);
    *output1 = svmla_z(pg, *output1, svdup_lane(mat1, 2), input1);
    *output1 = svcmla_z(pg, *output1, svdupq_lane(mat1, 1), input1, 90);
    *output1 = svmla_z(pg, *output1, svdup_lane(mat1, 4), input2);
    *output1 = svcmla_z(pg, *output1, svdupq_lane(mat1, 2), input2, 90);
    *output1 = svmla_z(pg, *output1, svdup_lane(mat1, 6), input3);
    *output1 = svcmla_z(pg, *output1, svdupq_lane(mat1, 3), input3, 90);

    *output2 = svmul_z(pg, svdup_lane(mat2, 0), input0);
    *output2 = svcmla_z(pg, *output2, svdupq_lane(mat2, 0), input0, 90);
    *output2 = svmla_z(pg, *output2, svdup_lane(mat2, 2), input1);
    *output2 = svcmla_z(pg, *output2, svdupq_lane(mat2, 1), input1, 90);
    *output2 = svmla_z(pg, *output2, svdup_lane(mat2, 4), input2);
    *output2 = svcmla_z(pg, *output2, svdupq_lane(mat2, 2), input2, 90);
    *output2 = svmla_z(pg, *output2, svdup_lane(mat2, 6), input3);
    *output2 = svcmla_z(pg, *output2, svdupq_lane(mat2, 3), input3, 90);

    *output3 = svmul_z(pg, svdup_lane(mat3, 0), input0);
    *output3 = svcmla_z(pg, *output3, svdupq_lane(mat3, 0), input0, 90);
    *output3 = svmla_z(pg, *output3, svdup_lane(mat3, 2), input1);
    *output3 = svcmla_z(pg, *output3, svdupq_lane(mat3, 1), input1, 90);
    *output3 = svmla_z(pg, *output3, svdup_lane(mat3, 4), input2);
    *output3 = svcmla_z(pg, *output3, svdupq_lane(mat3, 2), input2, 90);
    *output3 = svmla_z(pg, *output3, svdup_lane(mat3, 6), input3);
    *output3 = svcmla_z(pg, *output3, svdupq_lane(mat3, 3), input3, 90);
}

void double_qubit_dense_matrix_gate_sve_high(UINT target_qubit_index1,
    UINT target_qubit_index2, const CTYPE matrix[16], CTYPE* state, ITYPE dim) {
    const UINT min_qubit_index =
        get_min_ui(target_qubit_index1, target_qubit_index2);
    const UINT max_qubit_index =
        get_max_ui(target_qubit_index1, target_qubit_index2);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    const ITYPE target_mask1 = 1ULL << target_qubit_index1;
    const ITYPE target_mask2 = 1ULL << target_qubit_index2;

    // loop variables
    const ITYPE loop_dim = dim / 4;
    ITYPE state_index;
    ITYPE vec_len = getVecLength();

    SV_PRED pg = Svptrue();
    SV_FTYPE mat0, mat1, mat2, mat3;
    SV_FTYPE input0, input1, input2, input3;
    SV_FTYPE output0, output1, output2, output3;

    mat0 = svld1(pg, (ETYPE*)&matrix[0]);
    mat1 = svld1(pg, (ETYPE*)&matrix[4]);
    mat2 = svld1(pg, (ETYPE*)&matrix[8]);
    mat3 = svld1(pg, (ETYPE*)&matrix[12]);
#ifdef _OPENMP
#pragma omp parallel for private(input0, input1, input2, input3, output0, \
    output1, output2, output3) shared(pg, mat0, mat1, mat2, mat3)
#endif
    for (state_index = 0; state_index < loop_dim;
         state_index += (vec_len >> 1)) {
        // create index
        ITYPE basis_0 = (state_index & low_mask) +
                        ((state_index & mid_mask) << 1) +
                        ((state_index & high_mask) << 2);

        // gather index
        ITYPE basis_1 = basis_0 + target_mask1;
        ITYPE basis_2 = basis_0 + target_mask2;
        ITYPE basis_3 = basis_1 + target_mask2;

        // fetch values
        input0 = svld1(pg, (ETYPE*)&state[basis_0]);
        input1 = svld1(pg, (ETYPE*)&state[basis_1]);
        input2 = svld1(pg, (ETYPE*)&state[basis_2]);
        input3 = svld1(pg, (ETYPE*)&state[basis_3]);

        MatrixVectorProduct4x4(pg, input0, input1, input2, input3, mat0, mat1,
            mat2, mat3, &output0, &output1, &output2, &output3);

        // set values
        svst1(pg, (ETYPE*)&state[basis_0], output0);
        svst1(pg, (ETYPE*)&state[basis_1], output1);
        svst1(pg, (ETYPE*)&state[basis_2], output2);
        svst1(pg, (ETYPE*)&state[basis_3], output3);

        if ((4 <= min_qubit_index && min_qubit_index <= 9) ||
            (4 <= max_qubit_index && max_qubit_index <= 9)) {
            // L1 prefetch
            __builtin_prefetch(&state[basis_0 + target_mask1 * 4], 1, 3);
            __builtin_prefetch(&state[basis_1 + target_mask1 * 4], 1, 3);
            // L2 prefetch
            __builtin_prefetch(&state[basis_0 + target_mask1 * 8], 1, 2);
            __builtin_prefetch(&state[basis_1 + target_mask1 * 8], 1, 2);
            // L1 prefetch
            __builtin_prefetch(&state[basis_2 + target_mask2 * 4], 1, 3);
            __builtin_prefetch(&state[basis_3 + target_mask2 * 4], 1, 3);
            // L2 prefetch
            __builtin_prefetch(&state[basis_2 + target_mask2 * 8], 1, 2);
            __builtin_prefetch(&state[basis_3 + target_mask2 * 8], 1, 2);
        }
    }
}

void double_qubit_dense_matrix_gate_sve_middle(UINT target_qubit_index1,
    UINT target_qubit_index2, const CTYPE matrix[16], CTYPE* state, ITYPE dim) {
    const UINT min_qubit_index =
        get_min_ui(target_qubit_index1, target_qubit_index2);
    const UINT max_qubit_index =
        get_max_ui(target_qubit_index1, target_qubit_index2);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    const ITYPE target_mask1 = 1ULL << target_qubit_index1;
    const ITYPE target_mask2 = 1ULL << target_qubit_index2;

    // loop variables
    const ITYPE loop_dim = dim / 4;
    ITYPE state_index;
    ITYPE vec_len =
        getVecLength();  // the number of double elements in a vector
    ITYPE numComplexInVec = vec_len >> 1;

    SV_PRED pg = Svptrue();
    SV_PRED vec_select;
    SV_FTYPE mat0, mat1, mat2, mat3;
    SV_FTYPE input0, input1, input2, input3;
    SV_FTYPE cval0, cval1, cval2, cval3;
    SV_FTYPE result0, result1, result2, result3;
    SV_FTYPE output0, output1, output2, output3;

    mat0 = svld1(pg, (ETYPE*)&matrix[0]);
    mat1 = svld1(pg, (ETYPE*)&matrix[4]);
    mat2 = svld1(pg, (ETYPE*)&matrix[8]);
    mat3 = svld1(pg, (ETYPE*)&matrix[12]);

    if (target_qubit_index2 > target_qubit_index1) {
        // creat element index for shuffling in a vector
        vec_select = svcmpeq(pg,
            svand_z(pg, SvindexI(0, 1), SvdupI(target_mask1 << 1)), SvdupI(0));

        if (target_qubit_index1 == 0) {
#ifdef _OPENMP
#pragma omp parallel for private(input0, input1, input2, input3, output0,    \
    cval0, cval1, cval2, cval3, output1, output2, output3, result0, result1, \
    result2, result3) shared(pg, mat0, mat1, mat2, mat3)
#endif
            for (state_index = 0; state_index < loop_dim;
                 state_index += numComplexInVec) {
                // create index
                ITYPE basis_0 = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2);
                ITYPE basis_1 = state_index + (numComplexInVec >> 1);
                basis_1 = (basis_1 & low_mask) + ((basis_1 & mid_mask) << 1) +
                          ((basis_1 & high_mask) << 2);
                ITYPE basis_2 = basis_0 + target_mask2;
                ITYPE basis_3 = basis_1 + target_mask2;

                // fetch values
                input0 = svld1(pg, (ETYPE*)&state[basis_0]);
                input1 = svld1(pg, (ETYPE*)&state[basis_1]);
                input2 = svld1(pg, (ETYPE*)&state[basis_2]);
                input3 = svld1(pg, (ETYPE*)&state[basis_3]);

                // shuffle
                cval0 = svsel(vec_select, input0, svext(input1, input1, 6));
                cval1 = svsel(vec_select, svext(input0, input0, 2), input1);
                cval2 = svsel(vec_select, input2, svext(input3, input3, 6));
                cval3 = svsel(vec_select, svext(input2, input2, 2), input3);

                // perform matrix-vector product
                MatrixVectorProduct4x4(pg, cval0, cval1, cval2, cval3, mat0,
                    mat1, mat2, mat3, &result0, &result1, &result2, &result3);

                // reshuffle
                output0 =
                    svsel(vec_select, result0, svext(result1, result1, 6));
                output1 =
                    svsel(vec_select, svext(result0, result0, 2), result1);
                output2 =
                    svsel(vec_select, result2, svext(result3, result3, 6));
                output3 =
                    svsel(vec_select, svext(result2, result2, 2), result3);

                // set values
                svst1(pg, (ETYPE*)&state[basis_0], output0);
                svst1(pg, (ETYPE*)&state[basis_1], output1);
                svst1(pg, (ETYPE*)&state[basis_2], output2);
                svst1(pg, (ETYPE*)&state[basis_3], output3);
            }
        } else {  // target_qubit_index1 == 1
#ifdef _OPENMP
#pragma omp parallel for private(input0, input1, input2, input3, output0,    \
    cval0, cval1, cval2, cval3, output1, output2, output3, result0, result1, \
    result2, result3) shared(pg, mat0, mat1, mat2, mat3)
#endif
            for (state_index = 0; state_index < loop_dim;
                 state_index += numComplexInVec) {
                // create index
                ITYPE basis_0 = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2);
                ITYPE basis_1 = state_index + (numComplexInVec >> 1);
                basis_1 = (basis_1 & low_mask) + ((basis_1 & mid_mask) << 1) +
                          ((basis_1 & high_mask) << 2);
                ITYPE basis_2 = basis_0 + target_mask2;
                ITYPE basis_3 = basis_1 + target_mask2;

                // fetch values
                input0 = svld1(pg, (ETYPE*)&state[basis_0]);
                input1 = svld1(pg, (ETYPE*)&state[basis_1]);
                input2 = svld1(pg, (ETYPE*)&state[basis_2]);
                input3 = svld1(pg, (ETYPE*)&state[basis_3]);

                // shuffle
                cval0 = svsel(vec_select, input0, svext(input1, input1, 4));
                cval1 = svsel(vec_select, svext(input0, input0, 4), input1);
                cval2 = svsel(vec_select, input2, svext(input3, input3, 4));
                cval3 = svsel(vec_select, svext(input2, input2, 4), input3);

                // perform matrix-vector product
                MatrixVectorProduct4x4(pg, cval0, cval1, cval2, cval3, mat0,
                    mat1, mat2, mat3, &result0, &result1, &result2, &result3);

                // reshuffle
                output0 =
                    svsel(vec_select, result0, svext(result1, result1, 4));
                output1 =
                    svsel(vec_select, svext(result0, result0, 4), result1);
                output2 =
                    svsel(vec_select, result2, svext(result3, result3, 4));
                output3 =
                    svsel(vec_select, svext(result2, result2, 4), result3);

                // set values
                svst1(pg, (ETYPE*)&state[basis_0], output0);
                svst1(pg, (ETYPE*)&state[basis_1], output1);
                svst1(pg, (ETYPE*)&state[basis_2], output2);
                svst1(pg, (ETYPE*)&state[basis_3], output3);
            }
        }
    } else {  // target_qubit_index1 > target_qubit_index2

        // create element index for shuffling in a vector
        vec_select = svcmpeq(pg,
            svand_z(pg, SvindexI(0, 1), SvdupI(target_mask2 << 1)), SvdupI(0));

        if (target_qubit_index2 == 0) {  // target_qubit_index2 == 0
#ifdef _OPENMP
#pragma omp parallel for private(input0, input1, input2, input3, output0,    \
    cval0, cval1, cval2, cval3, output1, output2, output3, result0, result1, \
    result2, result3) shared(pg, mat0, mat1, mat2, mat3)
#endif
            for (state_index = 0; state_index < loop_dim;
                 state_index += numComplexInVec) {
                // create index
                ITYPE basis_0 = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2);

                ITYPE basis_1 = state_index + (numComplexInVec >> 1);
                basis_1 = (basis_1 & low_mask) + ((basis_1 & mid_mask) << 1) +
                          ((basis_1 & high_mask) << 2);
                ITYPE basis_2 = basis_0 + target_mask1;
                ITYPE basis_3 = basis_1 + target_mask1;

                // fetch values
                input0 = svld1(pg, (ETYPE*)&state[basis_0]);
                input1 = svld1(pg, (ETYPE*)&state[basis_1]);
                input2 = svld1(pg, (ETYPE*)&state[basis_2]);
                input3 = svld1(pg, (ETYPE*)&state[basis_3]);

                // shuffle
                cval0 = svsel(vec_select, input0, svext(input1, input1, 6));
                cval2 = svsel(vec_select, svext(input0, input0, 2), input1);
                cval1 = svsel(vec_select, input2, svext(input3, input3, 6));
                cval3 = svsel(vec_select, svext(input2, input2, 2), input3);

                // perform matrix-vector product
                MatrixVectorProduct4x4(pg, cval0, cval1, cval2, cval3, mat0,
                    mat1, mat2, mat3, &result0, &result1, &result2, &result3);

                // reshuffle
                output0 =
                    svsel(vec_select, result0, svext(result2, result2, 6));
                output1 =
                    svsel(vec_select, svext(result0, result0, 2), result2);
                output2 =
                    svsel(vec_select, result1, svext(result3, result3, 6));
                output3 =
                    svsel(vec_select, svext(result1, result1, 2), result3);

                // set values
                svst1(pg, (ETYPE*)&state[basis_0], output0);
                svst1(pg, (ETYPE*)&state[basis_1], output1);
                svst1(pg, (ETYPE*)&state[basis_2], output2);
                svst1(pg, (ETYPE*)&state[basis_3], output3);
            }
        } else {  // target_qubit_index2 == 1
#ifdef _OPENMP
#pragma omp parallel for private(input0, input1, input2, input3, output0,    \
    cval0, cval1, cval2, cval3, output1, output2, output3, result0, result1, \
    result2, result3) shared(pg, mat0, mat1, mat2, mat3)
#endif
            for (state_index = 0; state_index < loop_dim;
                 state_index += numComplexInVec) {
                // create index
                ITYPE basis_0 = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2);

                ITYPE basis_1 = state_index + (numComplexInVec >> 1);
                basis_1 = (basis_1 & low_mask) + ((basis_1 & mid_mask) << 1) +
                          ((basis_1 & high_mask) << 2);
                ITYPE basis_2 = basis_0 + target_mask1;
                ITYPE basis_3 = basis_1 + target_mask1;

                // fetch values
                input0 = svld1(pg, (ETYPE*)&state[basis_0]);
                input1 = svld1(pg, (ETYPE*)&state[basis_1]);
                input2 = svld1(pg, (ETYPE*)&state[basis_2]);
                input3 = svld1(pg, (ETYPE*)&state[basis_3]);

                // shuffle
                cval0 = svsel(vec_select, input0, svext(input1, input1, 4));
                cval2 = svsel(vec_select, svext(input0, input0, 4), input1);
                cval1 = svsel(vec_select, input2, svext(input3, input3, 4));
                cval3 = svsel(vec_select, svext(input2, input2, 4), input3);

                // perform matrix-vector product
                MatrixVectorProduct4x4(pg, cval0, cval1, cval2, cval3, mat0,
                    mat1, mat2, mat3, &result0, &result1, &result2, &result3);

                // reshuffle
                output0 =
                    svsel(vec_select, result0, svext(result2, result2, 4));
                output1 =
                    svsel(vec_select, svext(result0, result0, 4), result2);
                output2 =
                    svsel(vec_select, result1, svext(result3, result3, 4));
                output3 =
                    svsel(vec_select, svext(result1, result1, 4), result3);

                // set values
                svst1(pg, (ETYPE*)&state[basis_0], output0);
                svst1(pg, (ETYPE*)&state[basis_1], output1);
                svst1(pg, (ETYPE*)&state[basis_2], output2);
                svst1(pg, (ETYPE*)&state[basis_3], output3);
            }
        }
    }
}

void double_qubit_dense_matrix_gate_sve_low(UINT target_qubit_index1,
    UINT target_qubit_index2, const CTYPE matrix[16], CTYPE* state, ITYPE dim) {
    const UINT min_qubit_index =
        get_min_ui(target_qubit_index1, target_qubit_index2);
    const UINT max_qubit_index =
        get_max_ui(target_qubit_index1, target_qubit_index2);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    // loop variables
    const ITYPE loop_dim = dim / 4;
    ITYPE state_index;

    SV_PRED pg = Svptrue();
    SV_PRED vec_select;
    SV_ITYPE vec_shuffle_index;

    // make the following predicate register: (1,1,0,0,1,1,0,0)
    vec_select = svcmpeq(pg, svand_z(pg, SvindexI(0, 1), SvdupI(2)), SvdupI(0));

    // make the following vector: (0, 1, 4, 5, 2, 3, 6, 7)
    if (target_qubit_index1 > target_qubit_index2) {
        vec_shuffle_index = SvindexI(0, 1);
        vec_shuffle_index = svlsr_z(pg, vec_shuffle_index, 1);
        vec_shuffle_index = svorr_z(pg,
            svlsl_z(pg, svand_z(pg, vec_shuffle_index, SvdupI(1)), 1),
            svlsr_z(pg, vec_shuffle_index, 1));
        vec_shuffle_index = svorr_z(pg, svlsl_z(pg, vec_shuffle_index, 1),
            svand_z(pg, SvindexI(0, 1), 1));
    }

    SV_FTYPE mat0, mat1, mat2, mat3;
    SV_FTYPE mat0r, mat1r, mat2r, mat3r;
    SV_FTYPE input;
    SV_FTYPE output;
    SV_FTYPE vec_tmp1, vec_tmp2;

    mat0 = svld1(pg, (ETYPE*)&matrix[0]);
    mat1 = svld1(pg, (ETYPE*)&matrix[4]);
    mat2 = svld1(pg, (ETYPE*)&matrix[8]);
    mat3 = svld1(pg, (ETYPE*)&matrix[12]);

    mat0r = svtrn1(mat0, mat0);
    mat1r = svtrn1(mat1, mat1);
    mat2r = svtrn1(mat2, mat2);
    mat3r = svtrn1(mat3, mat3);
#ifdef _OPENMP
#pragma omp parallel for private(input, output, vec_tmp1, vec_tmp2) \
    shared(pg, mat0, mat1, mat2, mat3, mat0r, mat1r, mat2r, mat3r)
#endif
    for (state_index = 0; state_index < loop_dim; state_index++) {
        // create index
        ITYPE basis_0 = (state_index & low_mask) +
                        ((state_index & mid_mask) << 1) +
                        ((state_index & high_mask) << 2);

        // fetch values
        input = svld1(pg, (ETYPE*)&state[basis_0]);
        if (target_qubit_index1 > target_qubit_index2)
            input = svtbl(input, vec_shuffle_index);

        // perform matrix-vector product
        output = svmul_z(pg, mat0r, input);
        output = svcmla_z(pg, output, mat0, input, 90);
        output = svadd_z(pg, output, svext(output, output, 2));

        vec_tmp1 = svmul_z(pg, mat1r, input);
        vec_tmp1 = svcmla_z(pg, vec_tmp1, mat1, input, 90);
        vec_tmp1 = svadd_z(pg, vec_tmp1, svext(vec_tmp1, vec_tmp1, 2));

        output = svsel(vec_select, output, vec_tmp1);
        output = svadd_z(pg, svext(output, output, 4), output);

        vec_tmp1 = svmul_z(pg, mat2r, input);
        vec_tmp1 = svcmla_z(pg, vec_tmp1, mat2, input, 90);
        vec_tmp1 = svadd_z(pg, vec_tmp1, svext(vec_tmp1, vec_tmp1, 2));

        vec_tmp2 = svmul_z(pg, mat3r, input);
        vec_tmp2 = svcmla_z(pg, vec_tmp2, mat3, input, 90);
        vec_tmp2 = svadd_z(pg, vec_tmp2, svext(vec_tmp2, vec_tmp2, 2));

        vec_tmp1 = svsel(vec_select, vec_tmp1, vec_tmp2);
        vec_tmp1 = svadd_z(pg, svext(vec_tmp1, vec_tmp1, 4), vec_tmp1);

        output = svext(output, vec_tmp1, 4);

        if (target_qubit_index1 > target_qubit_index2)
            output = svtbl(output, vec_shuffle_index);

        svst1(pg, (ETYPE*)&state[basis_0], output);
    }
}

void double_qubit_dense_matrix_gate_sve(UINT target_qubit_index1,
    UINT target_qubit_index2, const CTYPE mat[16], CTYPE* vec, ITYPE dim) {
    ITYPE vec_len = getVecLength();
    ITYPE numComplexInVec = vec_len >> 1;

    assert(target_qubit_index1 != target_qubit_index2);
    if ((dim >= numComplexInVec) && (numComplexInVec == 4) &&
        (target_qubit_index1 < 2) && (target_qubit_index2 < 2)) {
        assert(sizeof(ETYPE) == sizeof(double));
        double_qubit_dense_matrix_gate_sve_low(
            target_qubit_index1, target_qubit_index2, mat, vec, dim);
    } else if ((dim >= (numComplexInVec << 1)) && (numComplexInVec == 4) &&
               ((target_qubit_index1 < 2) || (target_qubit_index2 < 2))) {
        assert(sizeof(ETYPE) == sizeof(double));
        double_qubit_dense_matrix_gate_sve_middle(
            target_qubit_index1, target_qubit_index2, mat, vec, dim);

    } else if ((numComplexInVec >= 4) && (target_qubit_index1 >= 2) &&
               (target_qubit_index2 >= 2)) {
        assert(sizeof(ETYPE) == sizeof(double));
        double_qubit_dense_matrix_gate_sve_high(
            target_qubit_index1, target_qubit_index2, mat, vec, dim);
    } else {
        double_qubit_dense_matrix_gate_nosimd(
            target_qubit_index1, target_qubit_index2, mat, vec, dim);
    }
}

#endif  // #if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)

#ifdef _USE_SIMD
void double_qubit_dense_matrix_gate_simd_high(UINT target_qubit_index1,
    UINT target_qubit_index2, const CTYPE mat[16], CTYPE* vec, ITYPE dim) {
    assert(target_qubit_index1 >= 2);
    assert(target_qubit_index2 >= 2);
    const UINT min_qubit_index =
        get_min_ui(target_qubit_index1, target_qubit_index2);
    const UINT max_qubit_index =
        get_max_ui(target_qubit_index1, target_qubit_index2);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    const ITYPE target_mask1_shift = 1ULL << (target_qubit_index1 + 1);
    const ITYPE target_mask2_shift = 1ULL << (target_qubit_index2 + 1);

    // loop variables
    const ITYPE loop_dim = dim / 4;
    ITYPE state_index;

    double* ptr_vec = (double*)vec;
    const double* ptr_mat = (const double*)mat;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (state_index = 0; state_index < loop_dim; state_index += 4) {
        __m256d res_real_sum, res_imag_sum;
        __m256d vec_before, vec_after;
        __m256d vec_real00, vec_imag00;
        __m256d vec_real01, vec_imag01;
        __m256d vec_real10, vec_imag10;
        __m256d vec_real11, vec_imag11;
        __m256d dup_mr, dup_mi;

        // create index
        ITYPE basis00 = (state_index & low_mask) +
                        ((state_index & mid_mask) << 1) +
                        ((state_index & high_mask) << 2);
        // shited due to index from complex -> double
        basis00 = basis00 << 1;
        ITYPE basis01 = basis00 + target_mask1_shift;
        ITYPE basis10 = basis00 + target_mask2_shift;
        ITYPE basis11 = basis01 + target_mask2_shift;

        //// Pick 4 complex values from basis00
        vec_before = _mm256_loadu_pd(ptr_vec + basis00);     // (i1 r1 i0 r0)
        vec_after = _mm256_loadu_pd(ptr_vec + basis00 + 4);  // (i3 r3 i2 r2)
        //// Split real values and imag values via shuffle
        vec_real00 = _mm256_shuffle_pd(vec_before, vec_after,
            0);  // (i1 r1 i0 r0) (i3 r3 i2 r2) -> (r3 r1 r2 r0) 0000 = 0
        vec_imag00 = _mm256_shuffle_pd(vec_before, vec_after,
            15);  // (i1 r1 i0 r0) (i3 r3 i2 r2) -> (i3 i1 i2 i0) 1111 = 15
        //// Pick matrix elem with 4 dup
        dup_mr = _mm256_set1_pd(ptr_mat[0]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[1]);  // (mi0 mi0 mi0 mi0)
        //// Compute real and imag part
        res_real_sum = _mm256_mul_pd(vec_real00, dup_mr);
        res_real_sum =
            _mm256_fnmadd_pd(vec_imag00, dup_mi, res_real_sum);  // -a*b+c
        res_imag_sum = _mm256_mul_pd(vec_real00, dup_mi);
        res_imag_sum =
            _mm256_fmadd_pd(vec_imag00, dup_mr, res_imag_sum);  // a*b+c

        //// Pick 4 complex values from basis01
        vec_before = _mm256_loadu_pd(ptr_vec + basis01);     // (i1 r1 i0 r0)
        vec_after = _mm256_loadu_pd(ptr_vec + basis01 + 4);  // (i3 r3 i2 r2)
        //// Split real values and imag values via shuffle
        vec_real01 = _mm256_shuffle_pd(vec_before, vec_after,
            0);  // (i1 r1 i0 r0) (i3 r3 i2 r2) -> (r3 r1 r2 r0) 0000 = 0
        vec_imag01 = _mm256_shuffle_pd(vec_before, vec_after,
            15);  // (i1 r1 i0 r0) (i3 r3 i2 r2) -> (i3 i1 i2 i0) 1111 = 15
        //// Pick matrix elem with 4 dup
        dup_mr = _mm256_set1_pd(ptr_mat[2]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[3]);  // (mi0 mi0 mi0 mi0)
        //// Compute real and imag part
        res_real_sum =
            _mm256_fmadd_pd(vec_real01, dup_mr, res_real_sum);  // a*b+c
        res_real_sum =
            _mm256_fnmadd_pd(vec_imag01, dup_mi, res_real_sum);  //-a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_real01, dup_mi, res_imag_sum);  // a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_imag01, dup_mr, res_imag_sum);  // a*b+c

        //// Pick 4 complex values from basis10
        vec_before = _mm256_loadu_pd(ptr_vec + basis10);     // (i1 r1 i0 r0)
        vec_after = _mm256_loadu_pd(ptr_vec + basis10 + 4);  // (i3 r3 i2 r2)
        //// Split real values and imag values via shuffle
        vec_real10 = _mm256_shuffle_pd(vec_before, vec_after,
            0);  // (i1 r1 i0 r0) (i3 r3 i2 r2) -> (r3 r1 r2 r0) 0000 = 0
        vec_imag10 = _mm256_shuffle_pd(vec_before, vec_after,
            15);  // (i1 r1 i0 r0) (i3 r3 i2 r2) -> (i3 i1 i2 i0) 1111 = 15
        //// Pick matrix elem with 4 dup
        dup_mr = _mm256_set1_pd(ptr_mat[4]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[5]);  // (mi0 mi0 mi0 mi0)
        //// Compute real and imag part
        res_real_sum =
            _mm256_fmadd_pd(vec_real10, dup_mr, res_real_sum);  // a*b+c
        res_real_sum =
            _mm256_fnmadd_pd(vec_imag10, dup_mi, res_real_sum);  //-a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_real10, dup_mi, res_imag_sum);  // a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_imag10, dup_mr, res_imag_sum);  // a*b+c

        //// Pick 4 complex values from basis11
        vec_before = _mm256_loadu_pd(ptr_vec + basis11);     // (i1 r1 i0 r0)
        vec_after = _mm256_loadu_pd(ptr_vec + basis11 + 4);  // (i3 r3 i2 r2)
        //// Split real values and imag values via shuffle
        vec_real11 = _mm256_shuffle_pd(vec_before, vec_after,
            0);  // (i1 r1 i0 r0) (i3 r3 i2 r2) -> (r3 r1 r2 r0) 0000 = 0
        vec_imag11 = _mm256_shuffle_pd(vec_before, vec_after,
            15);  // (i1 r1 i0 r0) (i3 r3 i2 r2) -> (i3 i1 i2 i0) 1111 = 15
        //// Pick matrix elem with 4 dup
        dup_mr = _mm256_set1_pd(ptr_mat[6]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[7]);  // (mi0 mi0 mi0 mi0)
        //// Compute real and imag part
        res_real_sum =
            _mm256_fmadd_pd(vec_real11, dup_mr, res_real_sum);  // a*b+c
        res_real_sum =
            _mm256_fnmadd_pd(vec_imag11, dup_mi, res_real_sum);  //-a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_real11, dup_mi, res_imag_sum);  // a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_imag11, dup_mr, res_imag_sum);  // a*b+c

        //// Store
        vec_before = _mm256_shuffle_pd(res_real_sum, res_imag_sum,
            0);  // (r3 r1 r2 r0) (i3 i1 i2 i0) -> (r1 i1 i0 r0) 0000 = 0
        vec_after = _mm256_shuffle_pd(res_real_sum, res_imag_sum,
            15);  // (r3 r1 r2 r0) (i3 i1 i2 i0) -> (r3 i3 i2 r2) 1111 = 15
        _mm256_storeu_pd(ptr_vec + basis00, vec_before);
        _mm256_storeu_pd(ptr_vec + basis00 + 4, vec_after);

        // vector is already fetched, fetch successive matrix elements and
        // perform dot(vec,vec) for other basis
        //// basis01
        //// Pick matrix elem with 4 dup
        dup_mr = _mm256_set1_pd(ptr_mat[8]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[9]);  // (mi0 mi0 mi0 mi0)
        //// Compute real and imag part
        res_real_sum = _mm256_mul_pd(vec_real00, dup_mr);
        res_real_sum =
            _mm256_fnmadd_pd(vec_imag00, dup_mi, res_real_sum);  // -a*b+c
        res_imag_sum = _mm256_mul_pd(vec_real00, dup_mi);
        res_imag_sum =
            _mm256_fmadd_pd(vec_imag00, dup_mr, res_imag_sum);  // a*b+c
        //// Pick matrix elem with 4 dup
        dup_mr = _mm256_set1_pd(ptr_mat[10]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[11]);  // (mi0 mi0 mi0 mi0)
        //// Compute real and imag part
        res_real_sum =
            _mm256_fmadd_pd(vec_real01, dup_mr, res_real_sum);  // a*b+c
        res_real_sum =
            _mm256_fnmadd_pd(vec_imag01, dup_mi, res_real_sum);  //-a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_real01, dup_mi, res_imag_sum);  // a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_imag01, dup_mr, res_imag_sum);  // a*b+c
        //// Pick matrix elem with 4 dup
        dup_mr = _mm256_set1_pd(ptr_mat[12]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[13]);  // (mi0 mi0 mi0 mi0)
        //// Compute real and imag part
        res_real_sum =
            _mm256_fmadd_pd(vec_real10, dup_mr, res_real_sum);  // a*b+c
        res_real_sum =
            _mm256_fnmadd_pd(vec_imag10, dup_mi, res_real_sum);  //-a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_real10, dup_mi, res_imag_sum);  // a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_imag10, dup_mr, res_imag_sum);  // a*b+c
        //// Pick matrix elem with 4 dup
        dup_mr = _mm256_set1_pd(ptr_mat[14]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[15]);  // (mi0 mi0 mi0 mi0)
        //// Compute real and imag part
        res_real_sum =
            _mm256_fmadd_pd(vec_real11, dup_mr, res_real_sum);  // a*b+c
        res_real_sum =
            _mm256_fnmadd_pd(vec_imag11, dup_mi, res_real_sum);  //-a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_real11, dup_mi, res_imag_sum);  // a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_imag11, dup_mr, res_imag_sum);  // a*b+c
        //// Store
        vec_before = _mm256_shuffle_pd(res_real_sum, res_imag_sum,
            0);  // (r3 r1 r2 r0) (i3 i1 i2 i0) -> (r1 i1 i0 r0) 0000 = 0
        vec_after = _mm256_shuffle_pd(res_real_sum, res_imag_sum,
            15);  // (r3 r1 r2 r0) (i3 i1 i2 i0) -> (r3 i3 i2 r2) 1111 = 15
        _mm256_storeu_pd(ptr_vec + basis01, vec_before);
        _mm256_storeu_pd(ptr_vec + basis01 + 4, vec_after);

        //// basis10
        //// Pick matrix elem with 4 dup
        dup_mr = _mm256_set1_pd(ptr_mat[16]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[17]);  // (mi0 mi0 mi0 mi0)
        //// Compute real and imag part
        res_real_sum = _mm256_mul_pd(vec_real00, dup_mr);
        res_real_sum =
            _mm256_fnmadd_pd(vec_imag00, dup_mi, res_real_sum);  // -a*b+c
        res_imag_sum = _mm256_mul_pd(vec_real00, dup_mi);
        res_imag_sum =
            _mm256_fmadd_pd(vec_imag00, dup_mr, res_imag_sum);  // a*b+c
        //// Pick matrix elem with 4 dup
        dup_mr = _mm256_set1_pd(ptr_mat[18]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[19]);  // (mi0 mi0 mi0 mi0)
        //// Compute real and imag part
        res_real_sum =
            _mm256_fmadd_pd(vec_real01, dup_mr, res_real_sum);  // a*b+c
        res_real_sum =
            _mm256_fnmadd_pd(vec_imag01, dup_mi, res_real_sum);  //-a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_real01, dup_mi, res_imag_sum);  // a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_imag01, dup_mr, res_imag_sum);  // a*b+c
        //// Pick matrix elem with 4 dup
        dup_mr = _mm256_set1_pd(ptr_mat[20]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[21]);  // (mi0 mi0 mi0 mi0)
        //// Compute real and imag part
        res_real_sum =
            _mm256_fmadd_pd(vec_real10, dup_mr, res_real_sum);  // a*b+c
        res_real_sum =
            _mm256_fnmadd_pd(vec_imag10, dup_mi, res_real_sum);  //-a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_real10, dup_mi, res_imag_sum);  // a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_imag10, dup_mr, res_imag_sum);  // a*b+c
        //// Pick matrix elem with 4 dup
        dup_mr = _mm256_set1_pd(ptr_mat[22]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[23]);  // (mi0 mi0 mi0 mi0)
        //// Compute real and imag part
        res_real_sum =
            _mm256_fmadd_pd(vec_real11, dup_mr, res_real_sum);  // a*b+c
        res_real_sum =
            _mm256_fnmadd_pd(vec_imag11, dup_mi, res_real_sum);  //-a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_real11, dup_mi, res_imag_sum);  // a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_imag11, dup_mr, res_imag_sum);  // a*b+c
        //// Store
        vec_before = _mm256_shuffle_pd(res_real_sum, res_imag_sum,
            0);  // (r3 r1 r2 r0) (i3 i1 i2 i0) -> (r1 i1 i0 r0) 0000 = 0
        vec_after = _mm256_shuffle_pd(res_real_sum, res_imag_sum,
            15);  // (r3 r1 r2 r0) (i3 i1 i2 i0) -> (r3 i3 i2 r2) 1111 = 15
        _mm256_storeu_pd(ptr_vec + basis10, vec_before);
        _mm256_storeu_pd(ptr_vec + basis10 + 4, vec_after);

        //// basis11
        //// Pick matrix elem with 4 dup
        dup_mr = _mm256_set1_pd(ptr_mat[24]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[25]);  // (mi0 mi0 mi0 mi0)
        //// Compute real and imag part
        res_real_sum = _mm256_mul_pd(vec_real00, dup_mr);
        res_real_sum =
            _mm256_fnmadd_pd(vec_imag00, dup_mi, res_real_sum);  // -a*b+c
        res_imag_sum = _mm256_mul_pd(vec_real00, dup_mi);
        res_imag_sum =
            _mm256_fmadd_pd(vec_imag00, dup_mr, res_imag_sum);  // a*b+c
        //// Pick matrix elem with 4 dup
        dup_mr = _mm256_set1_pd(ptr_mat[26]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[27]);  // (mi0 mi0 mi0 mi0)
        //// Compute real and imag part
        res_real_sum =
            _mm256_fmadd_pd(vec_real01, dup_mr, res_real_sum);  // a*b+c
        res_real_sum =
            _mm256_fnmadd_pd(vec_imag01, dup_mi, res_real_sum);  //-a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_real01, dup_mi, res_imag_sum);  // a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_imag01, dup_mr, res_imag_sum);  // a*b+c
        //// Pick matrix elem with 4 dup
        dup_mr = _mm256_set1_pd(ptr_mat[28]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[29]);  // (mi0 mi0 mi0 mi0)
        //// Compute real and imag part
        res_real_sum =
            _mm256_fmadd_pd(vec_real10, dup_mr, res_real_sum);  // a*b+c
        res_real_sum =
            _mm256_fnmadd_pd(vec_imag10, dup_mi, res_real_sum);  //-a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_real10, dup_mi, res_imag_sum);  // a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_imag10, dup_mr, res_imag_sum);  // a*b+c
        //// Pick matrix elem with 4 dup
        dup_mr = _mm256_set1_pd(ptr_mat[30]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[31]);  // (mi0 mi0 mi0 mi0)
        //// Compute real and imag part
        res_real_sum =
            _mm256_fmadd_pd(vec_real11, dup_mr, res_real_sum);  // a*b+c
        res_real_sum =
            _mm256_fnmadd_pd(vec_imag11, dup_mi, res_real_sum);  //-a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_real11, dup_mi, res_imag_sum);  // a*b+c
        res_imag_sum =
            _mm256_fmadd_pd(vec_imag11, dup_mr, res_imag_sum);  // a*b+c
        //// Store
        vec_before = _mm256_shuffle_pd(res_real_sum, res_imag_sum,
            0);  // (r3 r1 r2 r0) (i3 i1 i2 i0) -> (r1 i1 i0 r0) 0000 = 0
        vec_after = _mm256_shuffle_pd(res_real_sum, res_imag_sum,
            15);  // (r3 r1 r2 r0) (i3 i1 i2 i0) -> (r3 i3 i2 r2) 1111 = 15
        _mm256_storeu_pd(ptr_vec + basis11, vec_before);
        _mm256_storeu_pd(ptr_vec + basis11 + 4, vec_after);
    }
}

void double_qubit_dense_matrix_gate_simd_low(UINT target_qubit_index1,
    UINT target_qubit_index2, const CTYPE mat[16], CTYPE* vec, ITYPE dim) {
    assert(target_qubit_index1 < 2);
    assert(target_qubit_index2 < 2);
    assert(dim >= 8);

    // loop variables
    const ITYPE loop_dim = dim * 2;
    ITYPE state_index;

    double* ptr_vec = (double*)vec;
    const double* ptr_mat = (const double*)mat;
    if (target_qubit_index1 < target_qubit_index2) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 16) {
            __m256d vec1, vec2, vec3, vec4;
            __m256d u1, u2, u3, u4, u1f, u2f, u3f, u4f;
            __m256d mr, mi;

            vec1 = _mm256_loadu_pd(ptr_vec + state_index);  // c1 c0
            vec1 = _mm256_permute4x64_pd(
                vec1, 78);  // (c1 c0) -> (c0 c1) : 1032 = 1*2 + 4*3 + 16*0 +
                            // 64*1 = 64+12+2=78

            vec2 = _mm256_loadu_pd(ptr_vec + state_index + 4);  // c3 c2
            vec2 = _mm256_permute4x64_pd(vec2,
                78);  // (c3 c2) -> (c2 c3) : 1032 = 1*2+4*3+16*0+32*1 = 46

            vec3 = _mm256_loadu_pd(ptr_vec + state_index + 8);  // c5 c4
            u1 = _mm256_blend_pd(
                vec1, vec3, 3);  // (c0 c1) (c5 c4) -> (c0 c4) : 0011 = 3
            u2 = _mm256_blend_pd(
                vec1, vec3, 12);  // (c0 c1) (c5 c4) -> (c5 c1) : 1100 = 12
            u2 = _mm256_permute4x64_pd(
                u2, 78);  // (c5 c1) -> (c1 c5) : 1032 = 1*2+4*3+16*0+64*1 =
                          // 64+12+2=78

            vec4 = _mm256_loadu_pd(ptr_vec + state_index + 12);  // c7 c6
            u3 = _mm256_blend_pd(
                vec2, vec4, 3);  // (c2 c3) (c7 c6) -> (c2 c6) : 0011 = 3
            u4 = _mm256_blend_pd(
                vec2, vec4, 12);  // (c2 c3) (c7 c6) -> (c7 c3) : 1100 = 12
            u4 = _mm256_permute4x64_pd(
                u4, 78);  // (c7 c3) -> (c3 c7) : 1032 = 1*2+4*3+16*0+32*1 = 46

            u1f = _mm256_permute4x64_pd(
                u1, 177);  // 2301 = 64*2+16*3+1 = 128+48+1 = 177
            u2f = _mm256_permute4x64_pd(u2, 177);
            u3f = _mm256_permute4x64_pd(u3, 177);
            u4f = _mm256_permute4x64_pd(u4, 177);

            // u1  = (c0i c0r c4i c4r)
            // u2  = (c1i c1r c5i c5r)
            // u3  = (c2i c2r c6i c6r)
            // u4  = (c3i c3r c7i c7r)
            // u1f = (c0r c0i c4r c4i)
            // u2f = (c1r c1i c5r c5i)
            // u3f = (c2r c2i c6r c6i)
            // u4f = (c3r c3i c7r c7i)

            __m256d res_u1, res_u2, res_u3, res_u4, tmp_inv;
            tmp_inv = _mm256_set_pd(1, -1, 1, -1);

            mr = _mm256_set1_pd(ptr_mat[0]);
            res_u1 = _mm256_mul_pd(mr, u1);  // c0i*m0r, -c0r*m0r
            mi = _mm256_set1_pd(ptr_mat[1]);
            res_u1 = _mm256_fmaddsub_pd(
                mi, u1f, res_u1);  // m0i*c0r + c0i*m0r, m0i*c0i - c0r*m0r
            mr = _mm256_set1_pd(ptr_mat[2]);
            res_u1 = _mm256_fmaddsub_pd(
                mr, u2, res_u1);  // m1r*c1i + m0i*c0r + c0i*m0r, m1r*c1r -
                                  // m0i*c0i + c0r*m0r
            mi = _mm256_set1_pd(ptr_mat[3]);
            res_u1 = _mm256_fmaddsub_pd(
                mi, u2f, res_u1);  // m1i*c1r + m1r*c1i + m0i*c0r + c0i*m0r,
                                   // m1i*c1i - m1r*c1r + m0i*c0i - c0r*m0r
            mr = _mm256_set1_pd(ptr_mat[4]);
            res_u1 = _mm256_fmaddsub_pd(mr, u3, res_u1);
            mi = _mm256_set1_pd(ptr_mat[5]);
            res_u1 = _mm256_fmaddsub_pd(mi, u3f, res_u1);
            mr = _mm256_set1_pd(ptr_mat[6]);
            res_u1 = _mm256_fmaddsub_pd(mr, u4, res_u1);
            mi = _mm256_set1_pd(ptr_mat[7]);
            res_u1 = _mm256_fmaddsub_pd(mi, u4f, res_u1);
            res_u1 = _mm256_mul_pd(res_u1, tmp_inv);

            mr = _mm256_set1_pd(ptr_mat[8]);
            res_u2 = _mm256_mul_pd(mr, u1);  // c0i*m0r, -c0r*m0r
            mi = _mm256_set1_pd(ptr_mat[9]);
            res_u2 = _mm256_fmaddsub_pd(
                mi, u1f, res_u2);  // m0i*c0r + c0i*m0r, m0i*c0i - c0r*m0r
            mr = _mm256_set1_pd(ptr_mat[10]);
            res_u2 = _mm256_fmaddsub_pd(
                mr, u2, res_u2);  // m1r*c1i + m0i*c0r + c0i*m0r, m1r*c1r -
                                  // m0i*c0i + c0r*m0r
            mi = _mm256_set1_pd(ptr_mat[11]);
            res_u2 = _mm256_fmaddsub_pd(
                mi, u2f, res_u2);  // m1i*c1r + m1r*c1i + m0i*c0r + c0i*m0r,
                                   // m1i*c1i - m1r*c1r + m0i*c0i - c0r*m0r
            mr = _mm256_set1_pd(ptr_mat[12]);
            res_u2 = _mm256_fmaddsub_pd(mr, u3, res_u2);
            mi = _mm256_set1_pd(ptr_mat[13]);
            res_u2 = _mm256_fmaddsub_pd(mi, u3f, res_u2);
            mr = _mm256_set1_pd(ptr_mat[14]);
            res_u2 = _mm256_fmaddsub_pd(mr, u4, res_u2);
            mi = _mm256_set1_pd(ptr_mat[15]);
            res_u2 = _mm256_fmaddsub_pd(mi, u4f, res_u2);
            res_u2 = _mm256_mul_pd(res_u2, tmp_inv);

            res_u2 = _mm256_permute4x64_pd(res_u2, 78);  // flip
            vec1 = _mm256_blend_pd(res_u1, res_u2, 3);   // blend
            vec2 = _mm256_blend_pd(res_u1, res_u2, 12);  // blend
            vec1 = _mm256_permute4x64_pd(vec1, 78);      // flip
            _mm256_storeu_pd(ptr_vec + state_index, vec1);
            _mm256_storeu_pd(ptr_vec + state_index + 8, vec2);

            mr = _mm256_set1_pd(ptr_mat[16]);
            res_u3 = _mm256_mul_pd(mr, u1);  // c0i*m0r, -c0r*m0r
            mi = _mm256_set1_pd(ptr_mat[17]);
            res_u3 = _mm256_fmaddsub_pd(
                mi, u1f, res_u3);  // m0i*c0r + c0i*m0r, m0i*c0i - c0r*m0r
            mr = _mm256_set1_pd(ptr_mat[18]);
            res_u3 = _mm256_fmaddsub_pd(
                mr, u2, res_u3);  // m1r*c1i + m0i*c0r + c0i*m0r, m1r*c1r -
                                  // m0i*c0i + c0r*m0r
            mi = _mm256_set1_pd(ptr_mat[19]);
            res_u3 = _mm256_fmaddsub_pd(
                mi, u2f, res_u3);  // m1i*c1r + m1r*c1i + m0i*c0r + c0i*m0r,
                                   // m1i*c1i - m1r*c1r + m0i*c0i - c0r*m0r
            mr = _mm256_set1_pd(ptr_mat[20]);
            res_u3 = _mm256_fmaddsub_pd(mr, u3, res_u3);
            mi = _mm256_set1_pd(ptr_mat[21]);
            res_u3 = _mm256_fmaddsub_pd(mi, u3f, res_u3);
            mr = _mm256_set1_pd(ptr_mat[22]);
            res_u3 = _mm256_fmaddsub_pd(mr, u4, res_u3);
            mi = _mm256_set1_pd(ptr_mat[23]);
            res_u3 = _mm256_fmaddsub_pd(mi, u4f, res_u3);
            res_u3 = _mm256_mul_pd(res_u3, tmp_inv);

            mr = _mm256_set1_pd(ptr_mat[24]);
            res_u4 = _mm256_mul_pd(mr, u1);  // c0i*m0r, -c0r*m0r
            mi = _mm256_set1_pd(ptr_mat[25]);
            res_u4 = _mm256_fmaddsub_pd(
                mi, u1f, res_u4);  // m0i*c0r + c0i*m0r, m0i*c0i - c0r*m0r
            mr = _mm256_set1_pd(ptr_mat[26]);
            res_u4 = _mm256_fmaddsub_pd(
                mr, u2, res_u4);  // m1r*c1i + m0i*c0r + c0i*m0r, m1r*c1r -
                                  // m0i*c0i + c0r*m0r
            mi = _mm256_set1_pd(ptr_mat[27]);
            res_u4 = _mm256_fmaddsub_pd(
                mi, u2f, res_u4);  // m1i*c1r + m1r*c1i + m0i*c0r + c0i*m0r,
                                   // m1i*c1i - m1r*c1r + m0i*c0i - c0r*m0r
            mr = _mm256_set1_pd(ptr_mat[28]);
            res_u4 = _mm256_fmaddsub_pd(mr, u3, res_u4);
            mi = _mm256_set1_pd(ptr_mat[29]);
            res_u4 = _mm256_fmaddsub_pd(mi, u3f, res_u4);
            mr = _mm256_set1_pd(ptr_mat[30]);
            res_u4 = _mm256_fmaddsub_pd(mr, u4, res_u4);
            mi = _mm256_set1_pd(ptr_mat[31]);
            res_u4 = _mm256_fmaddsub_pd(mi, u4f, res_u4);
            res_u4 = _mm256_mul_pd(res_u4, tmp_inv);

            res_u4 = _mm256_permute4x64_pd(res_u4, 78);  // flip
            vec3 = _mm256_blend_pd(res_u3, res_u4, 3);   // blend
            vec4 = _mm256_blend_pd(res_u3, res_u4, 12);  // blend
            vec3 = _mm256_permute4x64_pd(vec3, 78);      // flip
            _mm256_storeu_pd(ptr_vec + state_index + 4, vec3);
            _mm256_storeu_pd(ptr_vec + state_index + 12, vec4);
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 16) {
            __m256d vec1, vec2, vec3, vec4;
            __m256d u1, u2, u3, u4, u1f, u2f, u3f, u4f;
            __m256d mr, mi;

            vec1 = _mm256_loadu_pd(ptr_vec + state_index);  // c1 c0
            vec1 = _mm256_permute4x64_pd(
                vec1, 78);  // (c1 c0) -> (c0 c1) : 1032 = 1*2 + 4*3 + 16*0 +
                            // 64*1 = 64+12+2=78

            vec2 = _mm256_loadu_pd(ptr_vec + state_index + 4);  // c3 c2
            vec2 = _mm256_permute4x64_pd(vec2,
                78);  // (c3 c2) -> (c2 c3) : 1032 = 1*2+4*3+16*0+32*1 = 46

            vec3 = _mm256_loadu_pd(ptr_vec + state_index + 8);  // c5 c4
            u1 = _mm256_blend_pd(
                vec1, vec3, 3);  // (c0 c1) (c5 c4) -> (c0 c4) : 0011 = 3
            u2 = _mm256_blend_pd(
                vec1, vec3, 12);  // (c0 c1) (c5 c4) -> (c5 c1) : 1100 = 12
            u2 = _mm256_permute4x64_pd(
                u2, 78);  // (c5 c1) -> (c1 c5) : 1032 = 1*2+4*3+16*0+64*1 =
                          // 64+12+2=78

            vec4 = _mm256_loadu_pd(ptr_vec + state_index + 12);  // c7 c6
            u3 = _mm256_blend_pd(
                vec2, vec4, 3);  // (c2 c3) (c7 c6) -> (c2 c6) : 0011 = 3
            u4 = _mm256_blend_pd(
                vec2, vec4, 12);  // (c2 c3) (c7 c6) -> (c7 c3) : 1100 = 12
            u4 = _mm256_permute4x64_pd(
                u4, 78);  // (c7 c3) -> (c3 c7) : 1032 = 1*2+4*3+16*0+32*1 = 46

            u1f = _mm256_permute4x64_pd(
                u1, 177);  // 2301 = 64*2+16*3+1 = 128+48+1 = 177
            u2f = _mm256_permute4x64_pd(u2, 177);
            u3f = _mm256_permute4x64_pd(u3, 177);
            u4f = _mm256_permute4x64_pd(u4, 177);

            // u1  = (c0i c0r c4i c4r)
            // u2  = (c1i c1r c5i c5r)
            // u3  = (c2i c2r c6i c6r)
            // u4  = (c3i c3r c7i c7r)
            // u1f = (c0r c0i c4r c4i)
            // u2f = (c1r c1i c5r c5i)
            // u3f = (c2r c2i c6r c6i)
            // u4f = (c3r c3i c7r c7i)

            __m256d res_u1, res_u2, res_u3, res_u4, tmp_inv;
            tmp_inv = _mm256_set_pd(1, -1, 1, -1);

            mr = _mm256_set1_pd(ptr_mat[0]);
            res_u1 = _mm256_mul_pd(mr, u1);  // c0i*m0r, -c0r*m0r
            mi = _mm256_set1_pd(ptr_mat[1]);
            res_u1 = _mm256_fmaddsub_pd(
                mi, u1f, res_u1);  // m0i*c0r + c0i*m0r, m0i*c0i - c0r*m0r
            mr = _mm256_set1_pd(ptr_mat[2]);
            res_u1 = _mm256_fmaddsub_pd(
                mr, u3, res_u1);  // m1r*c1i + m0i*c0r + c0i*m0r, m1r*c1r -
                                  // m0i*c0i + c0r*m0r
            mi = _mm256_set1_pd(ptr_mat[3]);
            res_u1 = _mm256_fmaddsub_pd(
                mi, u3f, res_u1);  // m1i*c1r + m1r*c1i + m0i*c0r + c0i*m0r,
                                   // m1i*c1i - m1r*c1r + m0i*c0i - c0r*m0r
            mr = _mm256_set1_pd(ptr_mat[4]);
            res_u1 = _mm256_fmaddsub_pd(mr, u2, res_u1);
            mi = _mm256_set1_pd(ptr_mat[5]);
            res_u1 = _mm256_fmaddsub_pd(mi, u2f, res_u1);
            mr = _mm256_set1_pd(ptr_mat[6]);
            res_u1 = _mm256_fmaddsub_pd(mr, u4, res_u1);
            mi = _mm256_set1_pd(ptr_mat[7]);
            res_u1 = _mm256_fmaddsub_pd(mi, u4f, res_u1);
            res_u1 = _mm256_mul_pd(res_u1, tmp_inv);

            mr = _mm256_set1_pd(ptr_mat[16]);
            res_u3 = _mm256_mul_pd(mr, u1);  // c0i*m0r, -c0r*m0r
            mi = _mm256_set1_pd(ptr_mat[17]);
            res_u3 = _mm256_fmaddsub_pd(
                mi, u1f, res_u3);  // m0i*c0r + c0i*m0r, m0i*c0i - c0r*m0r
            mr = _mm256_set1_pd(ptr_mat[18]);
            res_u3 = _mm256_fmaddsub_pd(
                mr, u3, res_u3);  // m1r*c1i + m0i*c0r + c0i*m0r, m1r*c1r -
                                  // m0i*c0i + c0r*m0r
            mi = _mm256_set1_pd(ptr_mat[19]);
            res_u3 = _mm256_fmaddsub_pd(
                mi, u3f, res_u3);  // m1i*c1r + m1r*c1i + m0i*c0r + c0i*m0r,
                                   // m1i*c1i - m1r*c1r + m0i*c0i - c0r*m0r
            mr = _mm256_set1_pd(ptr_mat[20]);
            res_u3 = _mm256_fmaddsub_pd(mr, u2, res_u3);
            mi = _mm256_set1_pd(ptr_mat[21]);
            res_u3 = _mm256_fmaddsub_pd(mi, u2f, res_u3);
            mr = _mm256_set1_pd(ptr_mat[22]);
            res_u3 = _mm256_fmaddsub_pd(mr, u4, res_u3);
            mi = _mm256_set1_pd(ptr_mat[23]);
            res_u3 = _mm256_fmaddsub_pd(mi, u4f, res_u3);
            res_u3 = _mm256_mul_pd(res_u3, tmp_inv);

            res_u3 = _mm256_permute4x64_pd(res_u3, 78);  // flip
            vec1 = _mm256_blend_pd(res_u1, res_u3, 3);   // blend
            vec3 = _mm256_blend_pd(res_u1, res_u3, 12);  // blend
            vec1 = _mm256_permute4x64_pd(vec1, 78);      // flip
            _mm256_storeu_pd(ptr_vec + state_index, vec1);
            _mm256_storeu_pd(ptr_vec + state_index + 8, vec3);

            mr = _mm256_set1_pd(ptr_mat[8]);
            res_u2 = _mm256_mul_pd(mr, u1);  // c0i*m0r, -c0r*m0r
            mi = _mm256_set1_pd(ptr_mat[9]);
            res_u2 = _mm256_fmaddsub_pd(
                mi, u1f, res_u2);  // m0i*c0r + c0i*m0r, m0i*c0i - c0r*m0r
            mr = _mm256_set1_pd(ptr_mat[10]);
            res_u2 = _mm256_fmaddsub_pd(
                mr, u3, res_u2);  // m1r*c1i + m0i*c0r + c0i*m0r, m1r*c1r -
                                  // m0i*c0i + c0r*m0r
            mi = _mm256_set1_pd(ptr_mat[11]);
            res_u2 = _mm256_fmaddsub_pd(
                mi, u3f, res_u2);  // m1i*c1r + m1r*c1i + m0i*c0r + c0i*m0r,
                                   // m1i*c1i - m1r*c1r + m0i*c0i - c0r*m0r
            mr = _mm256_set1_pd(ptr_mat[12]);
            res_u2 = _mm256_fmaddsub_pd(mr, u2, res_u2);
            mi = _mm256_set1_pd(ptr_mat[13]);
            res_u2 = _mm256_fmaddsub_pd(mi, u2f, res_u2);
            mr = _mm256_set1_pd(ptr_mat[14]);
            res_u2 = _mm256_fmaddsub_pd(mr, u4, res_u2);
            mi = _mm256_set1_pd(ptr_mat[15]);
            res_u2 = _mm256_fmaddsub_pd(mi, u4f, res_u2);
            res_u2 = _mm256_mul_pd(res_u2, tmp_inv);

            mr = _mm256_set1_pd(ptr_mat[24]);
            res_u4 = _mm256_mul_pd(mr, u1);  // c0i*m0r, -c0r*m0r
            mi = _mm256_set1_pd(ptr_mat[25]);
            res_u4 = _mm256_fmaddsub_pd(
                mi, u1f, res_u4);  // m0i*c0r + c0i*m0r, m0i*c0i - c0r*m0r
            mr = _mm256_set1_pd(ptr_mat[26]);
            res_u4 = _mm256_fmaddsub_pd(
                mr, u3, res_u4);  // m1r*c1i + m0i*c0r + c0i*m0r, m1r*c1r -
                                  // m0i*c0i + c0r*m0r
            mi = _mm256_set1_pd(ptr_mat[27]);
            res_u4 = _mm256_fmaddsub_pd(
                mi, u3f, res_u4);  // m1i*c1r + m1r*c1i + m0i*c0r + c0i*m0r,
                                   // m1i*c1i - m1r*c1r + m0i*c0i - c0r*m0r
            mr = _mm256_set1_pd(ptr_mat[28]);
            res_u4 = _mm256_fmaddsub_pd(mr, u2, res_u4);
            mi = _mm256_set1_pd(ptr_mat[29]);
            res_u4 = _mm256_fmaddsub_pd(mi, u2f, res_u4);
            mr = _mm256_set1_pd(ptr_mat[30]);
            res_u4 = _mm256_fmaddsub_pd(mr, u4, res_u4);
            mi = _mm256_set1_pd(ptr_mat[31]);
            res_u4 = _mm256_fmaddsub_pd(mi, u4f, res_u4);
            res_u4 = _mm256_mul_pd(res_u4, tmp_inv);

            res_u4 = _mm256_permute4x64_pd(res_u4, 78);  // flip
            vec2 = _mm256_blend_pd(res_u2, res_u4, 3);   // blend
            vec4 = _mm256_blend_pd(res_u2, res_u4, 12);  // blend
            vec2 = _mm256_permute4x64_pd(vec2, 78);      // flip
            _mm256_storeu_pd(ptr_vec + state_index + 4, vec2);
            _mm256_storeu_pd(ptr_vec + state_index + 12, vec4);
        }
    }
}

__inline void _element_swap(CTYPE* vec, UINT i1, UINT i2) {
    CTYPE temp = vec[i1];
    vec[i1] = vec[i2];
    vec[i2] = temp;
}

void double_qubit_dense_matrix_gate_simd_middle(UINT target_qubit_index1,
    UINT target_qubit_index2, const CTYPE _mat[16], CTYPE* vec, ITYPE dim) {
    CTYPE mat[16];
    memcpy(mat, _mat, sizeof(CTYPE) * 16);
    if (target_qubit_index2 < target_qubit_index1) {
        UINT temp = target_qubit_index1;
        target_qubit_index1 = target_qubit_index2;
        target_qubit_index2 = temp;
        _element_swap(mat, 1, 2);
        _element_swap(mat, 4, 8);
        _element_swap(mat, 7, 11);
        _element_swap(mat, 13, 14);
        _element_swap(mat, 5, 10);
        _element_swap(mat, 6, 9);
    }
    assert(target_qubit_index1 < 2);
    assert(target_qubit_index2 >= 2);

    const UINT min_qubit_index =
        get_min_ui(target_qubit_index1, target_qubit_index2);
    const UINT max_qubit_index =
        get_max_ui(target_qubit_index1, target_qubit_index2);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    const ITYPE target_mask1_shift = 1ULL << (target_qubit_index1 + 1);
    const ITYPE target_mask2_shift = 1ULL << (target_qubit_index2 + 1);

    // loop variables
    const ITYPE loop_dim = dim / 4;
    ITYPE state_index;

    double* ptr_vec = (double*)vec;
    const double* ptr_mat = (const double*)mat;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (state_index = 0; state_index < loop_dim; state_index += 2) {
        // create index
        ITYPE basis00 = (state_index & low_mask) +
                        ((state_index & mid_mask) << 1) +
                        ((state_index & high_mask) << 2);
        // shited due to index from complex -> double
        basis00 = basis00 << 1;
        // ITYPE basis01 = basis00 + target_mask1_shift;
        ITYPE basis10 = basis00 + target_mask2_shift;
        // ITYPE basis11 = basis01 + target_mask2_shift;

        //// Pick 4 complex values from basis00
        __m256d vec_bef0, vec_aft0, vec_bef1, vec_aft1;
        vec_bef0 = _mm256_loadu_pd(ptr_vec + basis00);      // (i1 r1 i0 r0)
        vec_aft0 = _mm256_loadu_pd(ptr_vec + basis00 + 4);  // (i3 r3 i2 r2)
        vec_bef1 = _mm256_loadu_pd(ptr_vec + basis10);
        vec_aft1 = _mm256_loadu_pd(ptr_vec + basis10 + 4);

        __m256d vec_u0, vec_u1, vec_u2, vec_u3;
        __m256d vec_u0f, vec_u1f, vec_u2f, vec_u3f;
        __m256d vec_inv;
        vec_inv = _mm256_set_pd(1, -1, 1, -1);
        if (target_qubit_index1 == 0) {
            vec_aft0 = _mm256_permute4x64_pd(
                vec_aft0, 78);  // (3 2 1 0) -> (1 0 3 2) 1*2 + 4*3 + 16*0 +
                                // 64*1 = 64+12+2 = 78
            vec_aft1 = _mm256_permute4x64_pd(vec_aft1, 78);
            vec_u0 = _mm256_blend_pd(vec_bef0, vec_aft0,
                12);  // (a a b b) = 1*0 + 2*0 + 4*1 + 8*1 = 12
            vec_u1 = _mm256_blend_pd(vec_bef0, vec_aft0,
                3);  // (a a b b) = 1*0 + 2*0 + 4*1 + 8*1 = 12
            vec_u2 = _mm256_blend_pd(vec_bef1, vec_aft1,
                12);  // (a a b b) = 1*0 + 2*0 + 4*1 + 8*1 = 12
            vec_u3 = _mm256_blend_pd(vec_bef1, vec_aft1,
                3);  // (a a b b) = 1*0 + 2*0 + 4*1 + 8*1 = 12
            vec_u1 = _mm256_permute4x64_pd(
                vec_u1, 78);  // (3 2 1 0) -> (1 0 3 2) 1*2 + 4*3 + 16*0 + 64*1
                              // = 64+12+2 = 78
            vec_u3 = _mm256_permute4x64_pd(vec_u3, 78);
        } else {
            vec_u0 = vec_bef0;
            vec_u1 = vec_aft0;
            vec_u2 = vec_bef1;
            vec_u3 = vec_aft1;
        }
        vec_u0f = _mm256_permute_pd(vec_u0, 5);  // 1*1 + 2*0 + 4*1 + 8*0
        vec_u1f = _mm256_permute_pd(vec_u1, 5);  // 1*1 + 2*0 + 4*1 + 8*0
        vec_u2f = _mm256_permute_pd(vec_u2, 5);  // 1*1 + 2*0 + 4*1 + 8*0
        vec_u3f = _mm256_permute_pd(vec_u3, 5);  // 1*1 + 2*0 + 4*1 + 8*0
        vec_u0f = _mm256_mul_pd(vec_u0f, vec_inv);
        vec_u1f = _mm256_mul_pd(vec_u1f, vec_inv);
        vec_u2f = _mm256_mul_pd(vec_u2f, vec_inv);
        vec_u3f = _mm256_mul_pd(vec_u3f, vec_inv);

        __m256d dup_mr, dup_mi;

        __m256d res_sum0, res_sum1, res_sum2, res_sum3;
        dup_mr = _mm256_set1_pd(ptr_mat[0]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[1]);  // (mi0 mi0 mi0 mi0)
        res_sum0 = _mm256_mul_pd(vec_u0, dup_mr);
        res_sum0 = _mm256_fmadd_pd(vec_u0f, dup_mi, res_sum0);
        dup_mr = _mm256_set1_pd(ptr_mat[2]);  // (mr1 mr1 mr1 mr1)
        dup_mi = _mm256_set1_pd(ptr_mat[3]);  // (mi1 mi1 mi1 mi1)
        res_sum0 = _mm256_fmadd_pd(vec_u1, dup_mr, res_sum0);
        res_sum0 = _mm256_fmadd_pd(vec_u1f, dup_mi, res_sum0);
        dup_mr = _mm256_set1_pd(ptr_mat[4]);  // (mr1 mr1 mr1 mr1)
        dup_mi = _mm256_set1_pd(ptr_mat[5]);  // (mi1 mi1 mi1 mi1)
        res_sum0 = _mm256_fmadd_pd(vec_u2, dup_mr, res_sum0);
        res_sum0 = _mm256_fmadd_pd(vec_u2f, dup_mi, res_sum0);
        dup_mr = _mm256_set1_pd(ptr_mat[6]);  // (mr1 mr1 mr1 mr1)
        dup_mi = _mm256_set1_pd(ptr_mat[7]);  // (mi1 mi1 mi1 mi1)
        res_sum0 = _mm256_fmadd_pd(vec_u3, dup_mr, res_sum0);
        res_sum0 = _mm256_fmadd_pd(vec_u3f, dup_mi, res_sum0);

        dup_mr = _mm256_set1_pd(ptr_mat[8]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[9]);  // (mi0 mi0 mi0 mi0)
        res_sum1 = _mm256_mul_pd(vec_u0, dup_mr);
        res_sum1 = _mm256_fmadd_pd(vec_u0f, dup_mi, res_sum1);
        dup_mr = _mm256_set1_pd(ptr_mat[10]);  // (mr1 mr1 mr1 mr1)
        dup_mi = _mm256_set1_pd(ptr_mat[11]);  // (mi1 mi1 mi1 mi1)
        res_sum1 = _mm256_fmadd_pd(vec_u1, dup_mr, res_sum1);
        res_sum1 = _mm256_fmadd_pd(vec_u1f, dup_mi, res_sum1);
        dup_mr = _mm256_set1_pd(ptr_mat[12]);  // (mr1 mr1 mr1 mr1)
        dup_mi = _mm256_set1_pd(ptr_mat[13]);  // (mi1 mi1 mi1 mi1)
        res_sum1 = _mm256_fmadd_pd(vec_u2, dup_mr, res_sum1);
        res_sum1 = _mm256_fmadd_pd(vec_u2f, dup_mi, res_sum1);
        dup_mr = _mm256_set1_pd(ptr_mat[14]);  // (mr1 mr1 mr1 mr1)
        dup_mi = _mm256_set1_pd(ptr_mat[15]);  // (mi1 mi1 mi1 mi1)
        res_sum1 = _mm256_fmadd_pd(vec_u3, dup_mr, res_sum1);
        res_sum1 = _mm256_fmadd_pd(vec_u3f, dup_mi, res_sum1);

        dup_mr = _mm256_set1_pd(ptr_mat[16]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[17]);  // (mi0 mi0 mi0 mi0)
        res_sum2 = _mm256_mul_pd(vec_u0, dup_mr);
        res_sum2 = _mm256_fmadd_pd(vec_u0f, dup_mi, res_sum2);
        dup_mr = _mm256_set1_pd(ptr_mat[18]);  // (mr1 mr1 mr1 mr1)
        dup_mi = _mm256_set1_pd(ptr_mat[19]);  // (mi1 mi1 mi1 mi1)
        res_sum2 = _mm256_fmadd_pd(vec_u1, dup_mr, res_sum2);
        res_sum2 = _mm256_fmadd_pd(vec_u1f, dup_mi, res_sum2);
        dup_mr = _mm256_set1_pd(ptr_mat[20]);  // (mr1 mr1 mr1 mr1)
        dup_mi = _mm256_set1_pd(ptr_mat[21]);  // (mi1 mi1 mi1 mi1)
        res_sum2 = _mm256_fmadd_pd(vec_u2, dup_mr, res_sum2);
        res_sum2 = _mm256_fmadd_pd(vec_u2f, dup_mi, res_sum2);
        dup_mr = _mm256_set1_pd(ptr_mat[22]);  // (mr1 mr1 mr1 mr1)
        dup_mi = _mm256_set1_pd(ptr_mat[23]);  // (mi1 mi1 mi1 mi1)
        res_sum2 = _mm256_fmadd_pd(vec_u3, dup_mr, res_sum2);
        res_sum2 = _mm256_fmadd_pd(vec_u3f, dup_mi, res_sum2);

        dup_mr = _mm256_set1_pd(ptr_mat[24]);  // (mr0 mr0 mr0 mr0)
        dup_mi = _mm256_set1_pd(ptr_mat[25]);  // (mi0 mi0 mi0 mi0)
        res_sum3 = _mm256_mul_pd(vec_u0, dup_mr);
        res_sum3 = _mm256_fmadd_pd(vec_u0f, dup_mi, res_sum3);
        dup_mr = _mm256_set1_pd(ptr_mat[26]);  // (mr1 mr1 mr1 mr1)
        dup_mi = _mm256_set1_pd(ptr_mat[27]);  // (mi1 mi1 mi1 mi1)
        res_sum3 = _mm256_fmadd_pd(vec_u1, dup_mr, res_sum3);
        res_sum3 = _mm256_fmadd_pd(vec_u1f, dup_mi, res_sum3);
        dup_mr = _mm256_set1_pd(ptr_mat[28]);  // (mr1 mr1 mr1 mr1)
        dup_mi = _mm256_set1_pd(ptr_mat[29]);  // (mi1 mi1 mi1 mi1)
        res_sum3 = _mm256_fmadd_pd(vec_u2, dup_mr, res_sum3);
        res_sum3 = _mm256_fmadd_pd(vec_u2f, dup_mi, res_sum3);
        dup_mr = _mm256_set1_pd(ptr_mat[30]);  // (mr1 mr1 mr1 mr1)
        dup_mi = _mm256_set1_pd(ptr_mat[31]);  // (mi1 mi1 mi1 mi1)
        res_sum3 = _mm256_fmadd_pd(vec_u3, dup_mr, res_sum3);
        res_sum3 = _mm256_fmadd_pd(vec_u3f, dup_mi, res_sum3);

        if (target_qubit_index1 == 0) {
            res_sum1 = _mm256_permute4x64_pd(
                res_sum1, 78);  // (3 2 1 0) -> (1 0 3 2) 1*2 + 4*3 + 16*0 +
                                // 64*1 = 64+12+2 = 78
            res_sum3 = _mm256_permute4x64_pd(res_sum3, 78);
            vec_bef0 = _mm256_blend_pd(res_sum0, res_sum1,
                12);  // (a a b b) = 1*0 + 2*0 + 4*1 + 8*1 = 12
            vec_aft0 = _mm256_blend_pd(res_sum0, res_sum1,
                3);  // (a a b b) = 1*0 + 2*0 + 4*1 + 8*1 = 12
            vec_bef1 = _mm256_blend_pd(res_sum2, res_sum3,
                12);  // (a a b b) = 1*0 + 2*0 + 4*1 + 8*1 = 12
            vec_aft1 = _mm256_blend_pd(res_sum2, res_sum3,
                3);  // (a a b b) = 1*0 + 2*0 + 4*1 + 8*1 = 12
            vec_aft0 = _mm256_permute4x64_pd(
                vec_aft0, 78);  // (3 2 1 0) -> (1 0 3 2) 1*2 + 4*3 + 16*0 +
                                // 64*1 = 64+12+2 = 78
            vec_aft1 = _mm256_permute4x64_pd(vec_aft1, 78);
        } else {
            vec_bef0 = res_sum0;
            vec_aft0 = res_sum1;
            vec_bef1 = res_sum2;
            vec_aft1 = res_sum3;
        }
        //// Store
        _mm256_storeu_pd(ptr_vec + basis00, vec_bef0);      // (i1 r1 i0 r0)
        _mm256_storeu_pd(ptr_vec + basis00 + 4, vec_aft0);  // (i3 r3 i2 r2)
        _mm256_storeu_pd(ptr_vec + basis10, vec_bef1);
        _mm256_storeu_pd(ptr_vec + basis10 + 4, vec_aft1);
    }
}

void double_qubit_dense_matrix_gate_simd(UINT target_qubit_index1,
    UINT target_qubit_index2, const CTYPE mat[16], CTYPE* vec, ITYPE dim) {
    assert(target_qubit_index1 != target_qubit_index2);
    if (dim == 4) {
        // avx2 code cannot use for 2-qubit state
        double_qubit_dense_matrix_gate_nosimd(
            target_qubit_index1, target_qubit_index2, mat, vec, dim);
    } else if (target_qubit_index1 >= 2 && target_qubit_index2 >= 2) {
        double_qubit_dense_matrix_gate_simd_high(
            target_qubit_index1, target_qubit_index2, mat, vec, dim);
    } else if (target_qubit_index1 >= 2 || target_qubit_index2 >= 2) {
        double_qubit_dense_matrix_gate_simd_middle(
            target_qubit_index1, target_qubit_index2, mat, vec, dim);
    } else {
        double_qubit_dense_matrix_gate_simd_low(
            target_qubit_index1, target_qubit_index2, mat, vec, dim);
    }
}
#endif
