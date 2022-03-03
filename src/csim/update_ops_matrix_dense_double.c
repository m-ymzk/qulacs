
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
static inline void MatrixVectorProduct4x4(SV_PRED pg, SV_FTYPE input0ir,
    SV_FTYPE input1ir, SV_FTYPE input2ir, SV_FTYPE input3ir, SV_FTYPE input0ri,
    SV_FTYPE input1ri, SV_FTYPE input2ri, SV_FTYPE input3ri, SV_FTYPE mat01rr,
    SV_FTYPE mat23rr, SV_FTYPE mat0ii, SV_FTYPE mat1ii, SV_FTYPE mat2ii,
    SV_FTYPE mat3ii, SV_FTYPE* output0, SV_FTYPE* output1, SV_FTYPE* output2,
    SV_FTYPE* output3);

static inline void PrepareMatrixElements(SV_PRED pg, const CTYPE matrix[16],
    SV_FTYPE* mat01rr, SV_FTYPE* mat23rr, SV_FTYPE* mat0ii, SV_FTYPE* mat1ii,
    SV_FTYPE* mat2ii, SV_FTYPE* mat3ii);

static inline void MatrixVectorProduct4x4MT1(SV_PRED pg, SV_FTYPE input0ir,
    SV_FTYPE input1ir, SV_FTYPE input2ir, SV_FTYPE input3ir, SV_FTYPE input0ri,
    SV_FTYPE input1ri, SV_FTYPE input2ri, SV_FTYPE input3ri, SV_FTYPE mat01rr,
    SV_FTYPE mat23rr, SV_FTYPE mat01ii, SV_FTYPE mat23ii, SV_FTYPE* output0,
    SV_FTYPE* output1, SV_FTYPE* output2, SV_FTYPE* output3);

static inline void PrepareMatrixElementsMT1(SV_PRED pg, const CTYPE matrix[16],
    SV_FTYPE* mat01rr, SV_FTYPE* mat23rr, SV_FTYPE* mat01ii, SV_FTYPE* mat23ii);

// clang-format off
/*
 * This function performs multiplication of a 4x4 matrix and four vectors
 *
 *            4x4 matrix                              four vectors
 * [ x_00 + iy_00 ... x_03+iy_03]   [ a_0+ib_0 ][ c_0+id_0 ][ e_0+if_0 ][ g_0+ih_0 ]
 * [              ...           ] * [    ...   ][    ...   ][    ...   ][    ...   ]
 * [ x_30 + iy_30 ... x_33+iy_33]   [ a_3+ib_3 ][ c_3+id_3 ][ e_3+if_3 ][ g_3+ih_3 ]
 *
 * params
 * - pg: All 1's predecate register 
 * - input0ir: An SVE register has the first component of the four vectors
 *   - e.g.) 512-bit SVE & FP64: [ a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0]
 * - input1ir: An SVE register has the second component of the four vectors
 *   - e.g.) 512-bit SVE & FP64: [ a_1, b_1, c_1, d_1, e_1, f_1, g_1, h_1]
 * - input2ir: An SVE register has the third component of the four vectors
 *   - e.g.) 512-bit SVE & FP64: [ a_2, b_2, c_2, d_2, e_2, f_2, g_2, h_2]
 * - input3ir: An SVE register has the fourth component of the four vectors
 *   - e.g.) 512-bit SVE & FP64: [ a_3, b_3, c_3, d_3, e_3, f_3, g_3, h_3]
 * - input0ri: An SVE register in which the order of real and imag. numbers is reversed
 *   - e.g.) 512-bit SVE & FP64: [ b_0, a_0, d_0, c_0, f_0, e_0, h_0, g_0]
 * - input1ri: An SVE register in which the order of real and imag. numbers is reversed
 *   - e.g.) 512-bit SVE & FP64: [ b_1, a_1, d_1, c_1, f_1, e_1, h_1, g_1]
 * - input2ri: An SVE register in which the order of real and imag. numbers is reversed
 *   - e.g.) 512-bit SVE & FP64: [ b_2, a_2, d_2, c_2, f_2, e_2, h_2, g_2]
 * - input3ri: An SVE register in which the order of real and imag. numbers is reversed
 *   - e.g.) 512-bit SVE & FP64: [ b_3, a_3, d_3, c_3, f_3, e_3, h_3, g_3]
 * - mat01rr: An SVE register has the real part of the first and second rows of the matrix
 *   - e.g.) 512-bit SVE & FP64: [ x_00, x_01, x_02, x_03, x_10, x_11, x_12, x_13]
 * - mat23rr: An SVE register has the real part of the third and fourth rows of the matrix
 *   - e.g.) 512-bit SVE & FP64: [ x_20, x_21, x_22, x_23, x_20, x_21, x_22, x_23]
 * - mat0ii: An SVE register has the imag. part of the first row of the matrix
 *   - Even elements are sign-reversed
 *   - e.g.) 512-bit SVE & FP64: [ y_00, -y_00, y_01, -y_01, y_02, -y_02, y_03, -y_03]
 * - mat1ii: An SVE register has the imag. part of the second row of the matrix
 *   - e.g.) 512-bit SVE & FP64: [ y_10, -y_10, y_11, -y_11, y_12, -y_12, y_13, -y_13]
 * - mat2ii: An SVE register has the imag. part of the third row of the matrix
 *   - e.g.) 512-bit SVE & FP64: [ y_20, -y_20, y_21, -y_21, y_22, -y_22, y_23, -y_23]
 * - mat3ii: An SVE register has the imag. part of the fourth row of the matrix
 *   - e.g.) 512-bit SVE & FP64: [ y_30, -y_30, y_31, -y_31, y_32, -y_32, y_33, -y_33]
 * - output*: SVE register store results of matrix-vector products
 *
 */
// clang-format on
static inline void MatrixVectorProduct4x4(SV_PRED pg, SV_FTYPE input0ir,
    SV_FTYPE input1ir, SV_FTYPE input2ir, SV_FTYPE input3ir, SV_FTYPE input0ri,
    SV_FTYPE input1ri, SV_FTYPE input2ri, SV_FTYPE input3ri, SV_FTYPE mat01rr,
    SV_FTYPE mat23rr, SV_FTYPE mat0ii, SV_FTYPE mat1ii, SV_FTYPE mat2ii,
    SV_FTYPE mat3ii, SV_FTYPE* output0, SV_FTYPE* output1, SV_FTYPE* output2,
    SV_FTYPE* output3) {
    // perform matrix-vector product
    *output0 = svmul_z(pg, svdup_lane(mat01rr, 0), input0ir);
    *output0 = svmla_z(pg, *output0, svdupq_lane(mat0ii, 0), input0ri);
    *output0 = svmla_z(pg, *output0, svdup_lane(mat01rr, 2), input1ir);
    *output0 = svmla_z(pg, *output0, svdupq_lane(mat0ii, 1), input1ri);
    *output0 = svmla_z(pg, *output0, svdup_lane(mat01rr, 4), input2ir);
    *output0 = svmla_z(pg, *output0, svdupq_lane(mat0ii, 2), input2ri);
    *output0 = svmla_z(pg, *output0, svdup_lane(mat01rr, 6), input3ir);
    *output0 = svmla_z(pg, *output0, svdupq_lane(mat0ii, 3), input3ri);

    *output1 = svmul_z(pg, svdup_lane(mat01rr, 1), input0ir);
    *output1 = svmla_z(pg, *output1, svdupq_lane(mat1ii, 0), input0ri);
    *output1 = svmla_z(pg, *output1, svdup_lane(mat01rr, 3), input1ir);
    *output1 = svmla_z(pg, *output1, svdupq_lane(mat1ii, 1), input1ri);
    *output1 = svmla_z(pg, *output1, svdup_lane(mat01rr, 5), input2ir);
    *output1 = svmla_z(pg, *output1, svdupq_lane(mat1ii, 2), input2ri);
    *output1 = svmla_z(pg, *output1, svdup_lane(mat01rr, 7), input3ir);
    *output1 = svmla_z(pg, *output1, svdupq_lane(mat1ii, 3), input3ri);

    *output2 = svmul_z(pg, svdup_lane(mat23rr, 0), input0ir);
    *output2 = svmla_z(pg, *output2, svdupq_lane(mat2ii, 0), input0ri);
    *output2 = svmla_z(pg, *output2, svdup_lane(mat23rr, 2), input1ir);
    *output2 = svmla_z(pg, *output2, svdupq_lane(mat2ii, 1), input1ri);
    *output2 = svmla_z(pg, *output2, svdup_lane(mat23rr, 4), input2ir);
    *output2 = svmla_z(pg, *output2, svdupq_lane(mat2ii, 2), input2ri);
    *output2 = svmla_z(pg, *output2, svdup_lane(mat23rr, 6), input3ir);
    *output2 = svmla_z(pg, *output2, svdupq_lane(mat2ii, 3), input3ri);

    *output3 = svmul_z(pg, svdup_lane(mat23rr, 1), input0ir);
    *output3 = svmla_z(pg, *output3, svdupq_lane(mat3ii, 0), input0ri);
    *output3 = svmla_z(pg, *output3, svdup_lane(mat23rr, 3), input1ir);
    *output3 = svmla_z(pg, *output3, svdupq_lane(mat3ii, 1), input1ri);
    *output3 = svmla_z(pg, *output3, svdup_lane(mat23rr, 5), input2ir);
    *output3 = svmla_z(pg, *output3, svdupq_lane(mat3ii, 2), input2ri);
    *output3 = svmla_z(pg, *output3, svdup_lane(mat23rr, 7), input3ir);
    *output3 = svmla_z(pg, *output3, svdupq_lane(mat3ii, 3), input3ri);
}

// clang-format off
/*
 * This function prepare SVE registers mat01rr, mat23rr, mat0ii, mat1ii,
 * mat2ii, and mat3ii for the MatrixVectorProduct4x4 function.
 *
 */
// clang-format on
static inline void PrepareMatrixElements(SV_PRED pg, const CTYPE matrix[16],
    SV_FTYPE* mat01rr, SV_FTYPE* mat23rr, SV_FTYPE* mat0ii, SV_FTYPE* mat1ii,
    SV_FTYPE* mat2ii, SV_FTYPE* mat3ii) {
    SV_PRED pred_01;
    SV_ITYPE vec_tbl;

    // load each row
    *mat0ii = svld1(pg, (ETYPE*)&matrix[0]);
    *mat1ii = svld1(pg, (ETYPE*)&matrix[4]);
    *mat2ii = svld1(pg, (ETYPE*)&matrix[8]);
    *mat3ii = svld1(pg, (ETYPE*)&matrix[12]);

    // gather the real parts of every two rows
    *mat01rr = svtrn1(*mat0ii, *mat1ii);
    *mat23rr = svtrn1(*mat2ii, *mat3ii);

    // gather the imag. parts in each row
    vec_tbl = svorr_z(pg, SvindexI(0, 1), SvdupI(1));
    *mat0ii = svtbl(*mat0ii, vec_tbl);
    *mat1ii = svtbl(*mat1ii, vec_tbl);
    *mat2ii = svtbl(*mat2ii, vec_tbl);
    *mat3ii = svtbl(*mat3ii, vec_tbl);

    // even elements are sign-reversed
    pred_01 = svcmpeq(pg, svand_z(pg, SvindexI(0, 1), SvdupI(1)), SvdupI(0));
    *mat0ii = svneg_m(*mat0ii, pred_01, *mat0ii);
    *mat1ii = svneg_m(*mat1ii, pred_01, *mat1ii);
    *mat2ii = svneg_m(*mat2ii, pred_01, *mat2ii);
    *mat3ii = svneg_m(*mat3ii, pred_01, *mat3ii);
}

// clang-format off
/*
 * This function performs multiplication of a 4x4 matrix and four vectors
 *  - called when either of two targets is one and the other is greater than one
 *
 *            4x4 matrix                              four vectors
 * [ x_00 + iy_00 ... x_03+iy_03]   [ a_0+ib_0 ][ c_0+id_0 ][ e_0+if_0 ][ g_0+ih_0 ]
 * [              ...           ] * [    ...   ][    ...   ][    ...   ][    ...   ]
 * [ x_30 + iy_30 ... x_33+iy_33]   [ a_3+ib_3 ][ c_3+id_3 ][ e_3+if_3 ][ g_3+ih_3 ]
 *
 */
// clang-format off
static inline void MatrixVectorProduct4x4MT1(SV_PRED pg, SV_FTYPE input0ir,
    SV_FTYPE input1ir, SV_FTYPE input2ir, SV_FTYPE input3ir, SV_FTYPE input0ri,
    SV_FTYPE input1ri, SV_FTYPE input2ri, SV_FTYPE input3ri, SV_FTYPE mat01rr,
    SV_FTYPE mat23rr, SV_FTYPE mat01ii, SV_FTYPE mat23ii, SV_FTYPE* output0,
    SV_FTYPE* output1, SV_FTYPE* output2, SV_FTYPE* output3) {
    // perform matrix-vector product
    *output0 = svmul_z(pg, svdup_lane(mat01rr, 0), input0ir);
    *output0 = svmla_z(pg, *output0, svdup_lane(mat01ii, 0), input0ri);
    *output0 = svmla_z(pg, *output0, svdup_lane(mat01rr, 2), input1ir);
    *output0 = svmla_z(pg, *output0, svdup_lane(mat01ii, 2), input1ri);
    *output0 = svmla_z(pg, *output0, svdup_lane(mat01rr, 4), input2ir);
    *output0 = svmla_z(pg, *output0, svdup_lane(mat01ii, 4), input2ri);
    *output0 = svmla_z(pg, *output0, svdup_lane(mat01rr, 6), input3ir);
    *output0 = svmla_z(pg, *output0, svdup_lane(mat01ii, 6), input3ri);

    *output1 = svmul_z(pg, svdup_lane(mat01rr, 1), input0ir);
    *output1 = svmla_z(pg, *output1, svdup_lane(mat01ii, 1), input0ri);
    *output1 = svmla_z(pg, *output1, svdup_lane(mat01rr, 3), input1ir);
    *output1 = svmla_z(pg, *output1, svdup_lane(mat01ii, 3), input1ri);
    *output1 = svmla_z(pg, *output1, svdup_lane(mat01rr, 5), input2ir);
    *output1 = svmla_z(pg, *output1, svdup_lane(mat01ii, 5), input2ri);
    *output1 = svmla_z(pg, *output1, svdup_lane(mat01rr, 7), input3ir);
    *output1 = svmla_z(pg, *output1, svdup_lane(mat01ii, 7), input3ri);

    *output2 = svmul_z(pg, svdup_lane(mat23rr, 0), input0ir);
    *output2 = svmla_z(pg, *output2, svdup_lane(mat23ii, 0), input0ri);
    *output2 = svmla_z(pg, *output2, svdup_lane(mat23rr, 2), input1ir);
    *output2 = svmla_z(pg, *output2, svdup_lane(mat23ii, 2), input1ri);
    *output2 = svmla_z(pg, *output2, svdup_lane(mat23rr, 4), input2ir);
    *output2 = svmla_z(pg, *output2, svdup_lane(mat23ii, 4), input2ri);
    *output2 = svmla_z(pg, *output2, svdup_lane(mat23rr, 6), input3ir);
    *output2 = svmla_z(pg, *output2, svdup_lane(mat23ii, 6), input3ri);

    *output3 = svmul_z(pg, svdup_lane(mat23rr, 1), input0ir);
    *output3 = svmla_z(pg, *output3, svdup_lane(mat23ii, 1), input0ri);
    *output3 = svmla_z(pg, *output3, svdup_lane(mat23rr, 3), input1ir);
    *output3 = svmla_z(pg, *output3, svdup_lane(mat23ii, 3), input1ri);
    *output3 = svmla_z(pg, *output3, svdup_lane(mat23rr, 5), input2ir);
    *output3 = svmla_z(pg, *output3, svdup_lane(mat23ii, 5), input2ri);
    *output3 = svmla_z(pg, *output3, svdup_lane(mat23rr, 7), input3ir);
    *output3 = svmla_z(pg, *output3, svdup_lane(mat23ii, 7), input3ri);
}

static inline void PrepareMatrixElementsMT1(SV_PRED pg, const CTYPE matrix[16],
     SV_FTYPE* mat01rr, SV_FTYPE* mat23rr, SV_FTYPE* mat01ii, SV_FTYPE* mat23ii){

    // load each row of the matrix
    SV_FTYPE input0 = svld1(pg, (ETYPE*)&matrix[0]);
    SV_FTYPE input1 = svld1(pg, (ETYPE*)&matrix[4]);
    SV_FTYPE input2 = svld1(pg, (ETYPE*)&matrix[8]);
    SV_FTYPE input3 = svld1(pg, (ETYPE*)&matrix[12]);

    // gather the real parts of every two rows
    *mat01rr = svtrn1(input0, input1);
    *mat23rr = svtrn1(input2, input3);

    // gather the imag. parts of every two rows
    *mat01ii = svtrn2(input0, input1);
    *mat23ii = svtrn2(input2, input3);
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
    ITYPE prefetch_taget1, prefetch_taget2;

    SV_PRED pg = Svptrue();
    SV_ITYPE vec_tbl;

    SV_FTYPE mat01rr, mat23rr;
    SV_FTYPE mat0ii, mat1ii, mat2ii, mat3ii;

    // load each row of the matrix
    PrepareMatrixElements(
        pg, matrix, &mat01rr, &mat23rr, &mat0ii, &mat1ii, &mat2ii, &mat3ii);

    // create a table for swap
    vec_tbl = sveor_z(pg, SvindexI(0, 1), SvdupI(1));

    prefetch_taget1 =
        ((5 <= target_qubit_index1) && (target_qubit_index1 <= 9));
    prefetch_taget2 =
        ((5 <= target_qubit_index2) && (target_qubit_index2 <= 9));

    if (prefetch_taget1 && prefetch_taget2) {
#ifdef _OPENMP
#pragma omp parallel for \
    shared(pg, vec_tbl, mat01rr, mat23rr, mat0ii, mat1ii, mat2ii, mat3ii)
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
            SV_FTYPE input0ir = svld1(pg, (ETYPE*)&state[basis_0]);
            SV_FTYPE input1ir = svld1(pg, (ETYPE*)&state[basis_1]);
            SV_FTYPE input2ir = svld1(pg, (ETYPE*)&state[basis_2]);
            SV_FTYPE input3ir = svld1(pg, (ETYPE*)&state[basis_3]);

            // swap the real and imag. parts
            SV_FTYPE input0ri = svtbl(input0ir, vec_tbl);
            SV_FTYPE input1ri = svtbl(input1ir, vec_tbl);
            SV_FTYPE input2ri = svtbl(input2ir, vec_tbl);
            SV_FTYPE input3ri = svtbl(input3ir, vec_tbl);

            SV_FTYPE output0, output1, output2, output3;
            MatrixVectorProduct4x4(pg, input0ir, input1ir, input2ir, input3ir,
                input0ri, input1ri, input2ri, input3ri, mat01rr, mat23rr,
                mat0ii, mat1ii, mat2ii, mat3ii, &output0, &output1, &output2,
                &output3);

            // set values
            svst1(pg, (ETYPE*)&state[basis_0], output0);
            svst1(pg, (ETYPE*)&state[basis_1], output1);
            svst1(pg, (ETYPE*)&state[basis_2], output2);
            svst1(pg, (ETYPE*)&state[basis_3], output3);

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

    } else if (prefetch_taget1) {
#ifdef _OPENMP
#pragma omp parallel for \
    shared(pg, vec_tbl, mat01rr, mat23rr, mat0ii, mat1ii, mat2ii, mat3ii)
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
            SV_FTYPE input0ir = svld1(pg, (ETYPE*)&state[basis_0]);
            SV_FTYPE input1ir = svld1(pg, (ETYPE*)&state[basis_1]);
            SV_FTYPE input2ir = svld1(pg, (ETYPE*)&state[basis_2]);
            SV_FTYPE input3ir = svld1(pg, (ETYPE*)&state[basis_3]);

            // swap the real and imag. parts
            SV_FTYPE input0ri = svtbl(input0ir, vec_tbl);
            SV_FTYPE input1ri = svtbl(input1ir, vec_tbl);
            SV_FTYPE input2ri = svtbl(input2ir, vec_tbl);
            SV_FTYPE input3ri = svtbl(input3ir, vec_tbl);

            SV_FTYPE output0, output1, output2, output3;
            MatrixVectorProduct4x4(pg, input0ir, input1ir, input2ir, input3ir,
                input0ri, input1ri, input2ri, input3ri, mat01rr, mat23rr,
                mat0ii, mat1ii, mat2ii, mat3ii, &output0, &output1, &output2,
                &output3);

            // set values
            svst1(pg, (ETYPE*)&state[basis_0], output0);
            svst1(pg, (ETYPE*)&state[basis_1], output1);
            svst1(pg, (ETYPE*)&state[basis_2], output2);
            svst1(pg, (ETYPE*)&state[basis_3], output3);

            // L1 prefetch
            __builtin_prefetch(&state[basis_0 + target_mask1 * 4], 1, 3);
            __builtin_prefetch(&state[basis_1 + target_mask1 * 4], 1, 3);
            __builtin_prefetch(&state[basis_2 + target_mask1 * 4], 1, 3);
            __builtin_prefetch(&state[basis_3 + target_mask1 * 4], 1, 3);
            // L2 prefetch
            __builtin_prefetch(&state[basis_0 + target_mask1 * 8], 1, 2);
            __builtin_prefetch(&state[basis_1 + target_mask1 * 8], 1, 2);
            __builtin_prefetch(&state[basis_2 + target_mask1 * 8], 1, 2);
            __builtin_prefetch(&state[basis_3 + target_mask1 * 8], 1, 2);
        }
    } else if (prefetch_taget2) {
#ifdef _OPENMP
#pragma omp parallel for \
    shared(pg, vec_tbl, mat01rr, mat23rr, mat0ii, mat1ii, mat2ii, mat3ii)
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
            SV_FTYPE input0ir = svld1(pg, (ETYPE*)&state[basis_0]);
            SV_FTYPE input1ir = svld1(pg, (ETYPE*)&state[basis_1]);
            SV_FTYPE input2ir = svld1(pg, (ETYPE*)&state[basis_2]);
            SV_FTYPE input3ir = svld1(pg, (ETYPE*)&state[basis_3]);

            // swap the real and imag. parts
            SV_FTYPE input0ri = svtbl(input0ir, vec_tbl);
            SV_FTYPE input1ri = svtbl(input1ir, vec_tbl);
            SV_FTYPE input2ri = svtbl(input2ir, vec_tbl);
            SV_FTYPE input3ri = svtbl(input3ir, vec_tbl);

            SV_FTYPE output0, output1, output2, output3;
            MatrixVectorProduct4x4(pg, input0ir, input1ir, input2ir, input3ir,
                input0ri, input1ri, input2ri, input3ri, mat01rr, mat23rr,
                mat0ii, mat1ii, mat2ii, mat3ii, &output0, &output1, &output2,
                &output3);

            // set values
            svst1(pg, (ETYPE*)&state[basis_0], output0);
            svst1(pg, (ETYPE*)&state[basis_1], output1);
            svst1(pg, (ETYPE*)&state[basis_2], output2);
            svst1(pg, (ETYPE*)&state[basis_3], output3);

            // L1 prefetch
            __builtin_prefetch(&state[basis_0 + target_mask2 * 4], 1, 3);
            __builtin_prefetch(&state[basis_1 + target_mask2 * 4], 1, 3);
            __builtin_prefetch(&state[basis_2 + target_mask2 * 4], 1, 3);
            __builtin_prefetch(&state[basis_3 + target_mask2 * 4], 1, 3);
            // L2 prefetch
            __builtin_prefetch(&state[basis_0 + target_mask2 * 8], 1, 2);
            __builtin_prefetch(&state[basis_1 + target_mask2 * 8], 1, 2);
            __builtin_prefetch(&state[basis_2 + target_mask2 * 8], 1, 2);
            __builtin_prefetch(&state[basis_3 + target_mask2 * 8], 1, 2);
        }

    } else {
#ifdef _OPENMP
#pragma omp parallel for \
    shared(pg, vec_tbl, mat01rr, mat23rr, mat0ii, mat1ii, mat2ii, mat3ii)
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
            SV_FTYPE input0ir = svld1(pg, (ETYPE*)&state[basis_0]);
            SV_FTYPE input1ir = svld1(pg, (ETYPE*)&state[basis_1]);
            SV_FTYPE input2ir = svld1(pg, (ETYPE*)&state[basis_2]);
            SV_FTYPE input3ir = svld1(pg, (ETYPE*)&state[basis_3]);

            // swap the real and imag. parts
            SV_FTYPE input0ri = svtbl(input0ir, vec_tbl);
            SV_FTYPE input1ri = svtbl(input1ir, vec_tbl);
            SV_FTYPE input2ri = svtbl(input2ir, vec_tbl);
            SV_FTYPE input3ri = svtbl(input3ir, vec_tbl);

            SV_FTYPE output0, output1, output2, output3;
            MatrixVectorProduct4x4(pg, input0ir, input1ir, input2ir, input3ir,
                input0ri, input1ri, input2ri, input3ri, mat01rr, mat23rr,
                mat0ii, mat1ii, mat2ii, mat3ii, &output0, &output1, &output2,
                &output3);

            // set values
            svst1(pg, (ETYPE*)&state[basis_0], output0);
            svst1(pg, (ETYPE*)&state[basis_1], output1);
            svst1(pg, (ETYPE*)&state[basis_2], output2);
            svst1(pg, (ETYPE*)&state[basis_3], output3);
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
    if (min_qubit_index == 1) {
        SV_PRED pg_neg;
        SV_ITYPE vec_tbl;

        SV_FTYPE mat01rr, mat23rr, mat01ii, mat23ii;

        // load elements of the matrix
        PrepareMatrixElementsMT1(pg, matrix, &mat01rr, &mat23rr, &mat01ii, &mat23ii);

        // create a table for swapping the real and imag. parts
        vec_tbl = sveor_z(pg, SvindexI(0, 1), SvdupI(2));

        // creat a predicate register for sign reversal
        pg_neg = svcmpeq(pg, svand_z(pg, SvindexI(0, 1), SvdupI(2)), SvdupI(0));

        if (target_qubit_index1 == 1) {
            ITYPE prefetch_flag =
                (5 <= target_qubit_index2) && (target_qubit_index2 <= 8);

            if (prefetch_flag) {
#ifdef _OPENMP
#pragma omp parallel for \
    shared(pg, pg_neg, vec_tbl, mat01ii, mat23ii, mat01rr, mat23rr)
#endif
                for (state_index = 0; state_index < loop_dim;
                     state_index += numComplexInVec) {
                    // create index
                    ITYPE basis_0 = (state_index & low_mask) +
                                    ((state_index & mid_mask) << 1) +
                                    ((state_index & high_mask) << 2);
                    ITYPE basis_1 = state_index + (numComplexInVec >> 1);
                    basis_1 = (basis_1 & low_mask) +
                              ((basis_1 & mid_mask) << 1) +
                              ((basis_1 & high_mask) << 2);
                    ITYPE basis_2 = basis_0 + target_mask2;
                    ITYPE basis_3 = basis_1 + target_mask2;

                    // fetch values
                    SV_FTYPE input0 = svld1(pg, (ETYPE*)&state[basis_0]);
                    SV_FTYPE input1 = svld1(pg, (ETYPE*)&state[basis_1]);
                    SV_FTYPE input2 = svld1(pg, (ETYPE*)&state[basis_2]);
                    SV_FTYPE input3 = svld1(pg, (ETYPE*)&state[basis_3]);

                    // shuffle
                    SV_FTYPE cval0ir = svzip1(input0, input1);
                    SV_FTYPE cval1ir = svzip2(input0, input1);
                    SV_FTYPE cval2ir = svzip1(input2, input3);
                    SV_FTYPE cval3ir = svzip2(input2, input3);

                    // swap the real and imag. parts
                    SV_FTYPE cval0ri = svtbl(cval0ir, vec_tbl);
                    SV_FTYPE cval1ri = svtbl(cval1ir, vec_tbl);
                    SV_FTYPE cval2ri = svtbl(cval2ir, vec_tbl);
                    SV_FTYPE cval3ri = svtbl(cval3ir, vec_tbl);

                    // reverse the sign every two elements
                    cval0ri = svneg_m(cval0ri, pg_neg, cval0ri);
                    cval1ri = svneg_m(cval1ri, pg_neg, cval1ri);
                    cval2ri = svneg_m(cval2ri, pg_neg, cval2ri);
                    cval3ri = svneg_m(cval3ri, pg_neg, cval3ri);

                    // perform matrix-vector product
                    SV_FTYPE result0, result1, result2, result3;
                    SV_FTYPE output0, output1, output2, output3;
                    MatrixVectorProduct4x4MT1(pg, cval0ir, cval1ir, cval2ir,
                        cval3ir, cval0ri, cval1ri, cval2ri, cval3ri, mat01rr,
                        mat23rr, mat01ii, mat23ii, &result0, &result1, &result2,
                        &result3);

                    // reshuffle
                    output0 = svuzp1(result0, result1);
                    output1 = svuzp2(result0, result1);
                    output2 = svuzp1(result2, result3);
                    output3 = svuzp2(result2, result3);

                    // set values
                    svst1(pg, (ETYPE*)&state[basis_0], output0);
                    svst1(pg, (ETYPE*)&state[basis_1], output1);
                    svst1(pg, (ETYPE*)&state[basis_2], output2);
                    svst1(pg, (ETYPE*)&state[basis_3], output3);

                    // L1 prefetch
                    __builtin_prefetch(
                        &state[basis_0 + target_mask2 * 4], 1, 3);
                    __builtin_prefetch(
                        &state[basis_2 + target_mask2 * 4], 1, 3);
                    // L2 prefetch
                    __builtin_prefetch(
                        &state[basis_0 + target_mask2 * 8], 1, 2);
                    __builtin_prefetch(
                        &state[basis_2 + target_mask2 * 8], 1, 2);
                }

            } else {
#ifdef _OPENMP
#pragma omp parallel for  \
    shared(pg, pg_neg, vec_tbl, mat01ii, mat23ii, mat01rr, mat23rr)
#endif
                for (state_index = 0; state_index < loop_dim;
                     state_index += numComplexInVec) {
                    // create index
                    ITYPE basis_0 = (state_index & low_mask) +
                                    ((state_index & mid_mask) << 1) +
                                    ((state_index & high_mask) << 2);
                    ITYPE basis_1 = state_index + (numComplexInVec >> 1);
                    basis_1 = (basis_1 & low_mask) +
                              ((basis_1 & mid_mask) << 1) +
                              ((basis_1 & high_mask) << 2);
                    ITYPE basis_2 = basis_0 + target_mask2;
                    ITYPE basis_3 = basis_1 + target_mask2;

                    // fetch values
                    SV_FTYPE input0 = svld1(pg, (ETYPE*)&state[basis_0]);
                    SV_FTYPE input1 = svld1(pg, (ETYPE*)&state[basis_1]);
                    SV_FTYPE input2 = svld1(pg, (ETYPE*)&state[basis_2]);
                    SV_FTYPE input3 = svld1(pg, (ETYPE*)&state[basis_3]);

                    // shuffle
                    SV_FTYPE cval0ir = svzip1(input0, input1);
                    SV_FTYPE cval1ir = svzip2(input0, input1);
                    SV_FTYPE cval2ir = svzip1(input2, input3);
                    SV_FTYPE cval3ir = svzip2(input2, input3);

                    // swap the real and imag. parts
                    SV_FTYPE cval0ri = svtbl(cval0ir, vec_tbl);
                    SV_FTYPE cval1ri = svtbl(cval1ir, vec_tbl);
                    SV_FTYPE cval2ri = svtbl(cval2ir, vec_tbl);
                    SV_FTYPE cval3ri = svtbl(cval3ir, vec_tbl);

                    // reverse the sign every two elements
                    cval0ri = svneg_m(cval0ri, pg_neg, cval0ri);
                    cval1ri = svneg_m(cval1ri, pg_neg, cval1ri);
                    cval2ri = svneg_m(cval2ri, pg_neg, cval2ri);
                    cval3ri = svneg_m(cval3ri, pg_neg, cval3ri);

                    // perform matrix-vector product
                    SV_FTYPE result0, result1, result2, result3;
                    SV_FTYPE output0, output1, output2, output3;
                    MatrixVectorProduct4x4MT1(pg, cval0ir, cval1ir, cval2ir,
                        cval3ir, cval0ri, cval1ri, cval2ri, cval3ri, mat01rr,
                        mat23rr, mat01ii, mat23ii, &result0, &result1, &result2,
                        &result3);

                    // reshuffle
                    output0 = svuzp1(result0, result1);
                    output1 = svuzp2(result0, result1);
                    output2 = svuzp1(result2, result3);
                    output3 = svuzp2(result2, result3);

                    // set values
                    svst1(pg, (ETYPE*)&state[basis_0], output0);
                    svst1(pg, (ETYPE*)&state[basis_1], output1);
                    svst1(pg, (ETYPE*)&state[basis_2], output2);
                    svst1(pg, (ETYPE*)&state[basis_3], output3);
                }
            }
        } else {  // target_qubit_index2 == 1
            ITYPE prefetch_flag =
                (5 <= target_qubit_index1) && (target_qubit_index1 <= 8);

            if (prefetch_flag) {
#ifdef _OPENMP
#pragma omp parallel for  \
    shared(pg, pg_neg, vec_tbl, mat01ii, mat23ii, mat01rr, mat23rr)
#endif
                for (state_index = 0; state_index < loop_dim;
                     state_index += numComplexInVec) {
                    // create index
                    ITYPE basis_0 = (state_index & low_mask) +
                                    ((state_index & mid_mask) << 1) +
                                    ((state_index & high_mask) << 2);
                    ITYPE basis_1 = state_index + (numComplexInVec >> 1);
                    basis_1 = (basis_1 & low_mask) +
                              ((basis_1 & mid_mask) << 1) +
                              ((basis_1 & high_mask) << 2);
                    ITYPE basis_2 = basis_0 + target_mask1;
                    ITYPE basis_3 = basis_1 + target_mask1;

                    // fetch values
                    SV_FTYPE input0 = svld1(pg, (ETYPE*)&state[basis_0]);
                    SV_FTYPE input1 = svld1(pg, (ETYPE*)&state[basis_1]);
                    SV_FTYPE input2 = svld1(pg, (ETYPE*)&state[basis_2]);
                    SV_FTYPE input3 = svld1(pg, (ETYPE*)&state[basis_3]);

                    // shuffle
                    SV_FTYPE cval0ir = svzip1(input0, input1);
                    SV_FTYPE cval2ir = svzip2(input0, input1);
                    SV_FTYPE cval1ir = svzip1(input2, input3);
                    SV_FTYPE cval3ir = svzip2(input2, input3);

                    // swap the real and imag. parts
                    SV_FTYPE cval0ri = svtbl(cval0ir, vec_tbl);
                    SV_FTYPE cval1ri = svtbl(cval1ir, vec_tbl);
                    SV_FTYPE cval2ri = svtbl(cval2ir, vec_tbl);
                    SV_FTYPE cval3ri = svtbl(cval3ir, vec_tbl);

                    // reverse the sign every two elements
                    cval0ri = svneg_m(cval0ri, pg_neg, cval0ri);
                    cval1ri = svneg_m(cval1ri, pg_neg, cval1ri);
                    cval2ri = svneg_m(cval2ri, pg_neg, cval2ri);
                    cval3ri = svneg_m(cval3ri, pg_neg, cval3ri);

                    // perform matrix-vector product
                    SV_FTYPE result0, result1, result2, result3;
                    SV_FTYPE output0, output1, output2, output3;
                    MatrixVectorProduct4x4MT1(pg, cval0ir, cval1ir, cval2ir,
                        cval3ir, cval0ri, cval1ri, cval2ri, cval3ri, mat01rr,
                        mat23rr, mat01ii, mat23ii, &result0, &result1, &result2,
                        &result3);

                    // reshuffle
                    output0 = svuzp1(result0, result2);
                    output1 = svuzp2(result0, result2);
                    output2 = svuzp1(result1, result3);
                    output3 = svuzp2(result1, result3);

                    // set values
                    svst1(pg, (ETYPE*)&state[basis_0], output0);
                    svst1(pg, (ETYPE*)&state[basis_1], output1);
                    svst1(pg, (ETYPE*)&state[basis_2], output2);
                    svst1(pg, (ETYPE*)&state[basis_3], output3);

                    // L1 prefetch
                    __builtin_prefetch(
                        &state[basis_0 + target_mask1 * 4], 1, 3);
                    __builtin_prefetch(
                        &state[basis_2 + target_mask1 * 4], 1, 3);
                    // L2 prefetch
                    __builtin_prefetch(
                        &state[basis_0 + target_mask1 * 8], 1, 2);
                    __builtin_prefetch(
                        &state[basis_2 + target_mask1 * 8], 1, 2);
                }

            } else {
#ifdef _OPENMP
#pragma omp parallel for \
    shared(pg, pg_neg, vec_tbl, mat01ii, mat23ii, mat01rr, mat23rr)
#endif
                for (state_index = 0; state_index < loop_dim;
                     state_index += numComplexInVec) {
                    // create index
                    ITYPE basis_0 = (state_index & low_mask) +
                                    ((state_index & mid_mask) << 1) +
                                    ((state_index & high_mask) << 2);
                    ITYPE basis_1 = state_index + (numComplexInVec >> 1);
                    basis_1 = (basis_1 & low_mask) +
                              ((basis_1 & mid_mask) << 1) +
                              ((basis_1 & high_mask) << 2);
                    ITYPE basis_2 = basis_0 + target_mask1;
                    ITYPE basis_3 = basis_1 + target_mask1;

                    // fetch values
                    SV_FTYPE input0 = svld1(pg, (ETYPE*)&state[basis_0]);
                    SV_FTYPE input1 = svld1(pg, (ETYPE*)&state[basis_1]);
                    SV_FTYPE input2 = svld1(pg, (ETYPE*)&state[basis_2]);
                    SV_FTYPE input3 = svld1(pg, (ETYPE*)&state[basis_3]);

                    // shuffle
                    SV_FTYPE cval0ir = svzip1(input0, input1);
                    SV_FTYPE cval2ir = svzip2(input0, input1);
                    SV_FTYPE cval1ir = svzip1(input2, input3);
                    SV_FTYPE cval3ir = svzip2(input2, input3);

                    // swap the real and imag. parts
                    SV_FTYPE cval0ri = svtbl(cval0ir, vec_tbl);
                    SV_FTYPE cval1ri = svtbl(cval1ir, vec_tbl);
                    SV_FTYPE cval2ri = svtbl(cval2ir, vec_tbl);
                    SV_FTYPE cval3ri = svtbl(cval3ir, vec_tbl);

                    // reverse the sign every two elements
                    cval0ri = svneg_m(cval0ri, pg_neg, cval0ri);
                    cval1ri = svneg_m(cval1ri, pg_neg, cval1ri);
                    cval2ri = svneg_m(cval2ri, pg_neg, cval2ri);
                    cval3ri = svneg_m(cval3ri, pg_neg, cval3ri);

                    // perform matrix-vector product
                    SV_FTYPE result0, result1, result2, result3;
                    SV_FTYPE output0, output1, output2, output3;
                    MatrixVectorProduct4x4MT1(pg, cval0ir, cval1ir, cval2ir,
                        cval3ir, cval0ri, cval1ri, cval2ri, cval3ri, mat01rr,
                        mat23rr, mat01ii, mat23ii, &result0, &result1, &result2,
                        &result3);

                    // reshuffle
                    output0 = svuzp1(result0, result2);
                    output1 = svuzp2(result0, result2);
                    output2 = svuzp1(result1, result3);
                    output3 = svuzp2(result1, result3);

                    // set values
                    svst1(pg, (ETYPE*)&state[basis_0], output0);
                    svst1(pg, (ETYPE*)&state[basis_1], output1);
                    svst1(pg, (ETYPE*)&state[basis_2], output2);
                    svst1(pg, (ETYPE*)&state[basis_3], output3);
                }
            }
        }
    } else {  // min_qubit_index == 0

        SV_PRED pg_select;
        SV_ITYPE vec_tbl;

        SV_FTYPE mat0ii, mat1ii, mat2ii, mat3ii;
        SV_FTYPE mat01rr, mat23rr;

        // load each row of the matrix
        PrepareMatrixElements(
            pg, matrix, &mat01rr, &mat23rr, &mat0ii, &mat1ii, &mat2ii, &mat3ii);

        // create a table for swap
        vec_tbl = sveor_z(pg, SvindexI(0, 1), SvdupI(1));

        if (target_qubit_index2 >
            target_qubit_index1) {  // target_qubit_index1 == 0
            ITYPE prefetch_flag =
                (5 <= target_qubit_index2) && (target_qubit_index2 <= 8);

            // creat element index for shuffling in a vector
            pg_select = svcmpeq(pg,
                svand_z(pg, SvindexI(0, 1), SvdupI(target_mask1 << 1)),
                SvdupI(0));

            if (prefetch_flag) {
#ifdef _OPENMP
#pragma omp parallel for shared(pg, \
    pg_select, vec_tbl, mat0ii, mat1ii, mat2ii, mat3ii, mat01rr, mat23rr)
#endif
                for (state_index = 0; state_index < loop_dim;
                     state_index += numComplexInVec) {
                    // create index
                    ITYPE basis_0 = (state_index & low_mask) +
                                    ((state_index & mid_mask) << 1) +
                                    ((state_index & high_mask) << 2);
                    ITYPE basis_1 = state_index + (numComplexInVec >> 1);
                    basis_1 = (basis_1 & low_mask) +
                              ((basis_1 & mid_mask) << 1) +
                              ((basis_1 & high_mask) << 2);
                    ITYPE basis_2 = basis_0 + target_mask2;
                    ITYPE basis_3 = basis_1 + target_mask2;

                    // fetch values
                    SV_FTYPE input0 = svld1(pg, (ETYPE*)&state[basis_0]);
                    SV_FTYPE input1 = svld1(pg, (ETYPE*)&state[basis_1]);
                    SV_FTYPE input2 = svld1(pg, (ETYPE*)&state[basis_2]);
                    SV_FTYPE input3 = svld1(pg, (ETYPE*)&state[basis_3]);

                    // shuffle
                    SV_FTYPE cval0ir =
                         svsel(pg_select, input0, svext(input1, input1, 6));
                    SV_FTYPE cval1ir =
                         svsel(pg_select, svext(input0, input0, 2), input1);
                    SV_FTYPE cval2ir =
                         svsel(pg_select, input2, svext(input3, input3, 6));
                    SV_FTYPE cval3ir =
                        svsel(pg_select, svext(input2, input2, 2), input3);
                    // swap
                    SV_FTYPE cval0ri = svtbl(cval0ir, vec_tbl);
                    SV_FTYPE cval1ri = svtbl(cval1ir, vec_tbl);
                    SV_FTYPE cval2ri = svtbl(cval2ir, vec_tbl);
                    SV_FTYPE cval3ri = svtbl(cval3ir, vec_tbl);

                    // perform matrix-vector product
                    SV_FTYPE result0, result1, result2, result3;
                    SV_FTYPE output0, output1, output2, output3;
                    MatrixVectorProduct4x4(pg, cval0ir, cval1ir, cval2ir,
                        cval3ir, cval0ri, cval1ri, cval2ri, cval3ri, mat01rr,
                        mat23rr, mat0ii, mat1ii, mat2ii, mat3ii, &result0,
                        &result1, &result2, &result3);

                    // reshuffle
                    output0 =
                        svsel(pg_select, result0, svext(result1, result1, 6));
                    output1 =
                        svsel(pg_select, svext(result0, result0, 2), result1);
                    output2 =
                        svsel(pg_select, result2, svext(result3, result3, 6));
                    output3 =
                        svsel(pg_select, svext(result2, result2, 2), result3);

                    // set values
                    svst1(pg, (ETYPE*)&state[basis_0], output0);
                    svst1(pg, (ETYPE*)&state[basis_1], output1);
                    svst1(pg, (ETYPE*)&state[basis_2], output2);
                    svst1(pg, (ETYPE*)&state[basis_3], output3);

                    // L1 prefetch
                    __builtin_prefetch(
                        &state[basis_0 + target_mask2 * 4], 1, 3);
                    __builtin_prefetch(
                        &state[basis_2 + target_mask2 * 4], 1, 3);
                    // L2 prefetch
                    __builtin_prefetch(
                        &state[basis_0 + target_mask2 * 8], 1, 2);
                    __builtin_prefetch(
                        &state[basis_2 + target_mask2 * 8], 1, 2);
                }
            } else {
#ifdef _OPENMP
#pragma omp parallel for shared(pg, \
    pg_select, vec_tbl, mat0ii, mat1ii, mat2ii, mat3ii, mat01rr, mat23rr)
#endif
                for (state_index = 0; state_index < loop_dim;
                     state_index += numComplexInVec) {
                    // create index
                    ITYPE basis_0 = (state_index & low_mask) +
                                    ((state_index & mid_mask) << 1) +
                                    ((state_index & high_mask) << 2);
                    ITYPE basis_1 = state_index + (numComplexInVec >> 1);
                    basis_1 = (basis_1 & low_mask) +
                              ((basis_1 & mid_mask) << 1) +
                              ((basis_1 & high_mask) << 2);
                    ITYPE basis_2 = basis_0 + target_mask2;
                    ITYPE basis_3 = basis_1 + target_mask2;

                    // fetch values
                    SV_FTYPE input0 = svld1(pg, (ETYPE*)&state[basis_0]);
                    SV_FTYPE input1 = svld1(pg, (ETYPE*)&state[basis_1]);
                    SV_FTYPE input2 = svld1(pg, (ETYPE*)&state[basis_2]);
                    SV_FTYPE input3 = svld1(pg, (ETYPE*)&state[basis_3]);

                    // shuffle
                    SV_FTYPE cval0ir =
                         svsel(pg_select, input0, svext(input1, input1, 6));
                    SV_FTYPE cval1ir =
                         svsel(pg_select, svext(input0, input0, 2), input1);
                    SV_FTYPE cval2ir =
                         svsel(pg_select, input2, svext(input3, input3, 6));
                    SV_FTYPE cval3ir =
                        svsel(pg_select, svext(input2, input2, 2), input3);
                    // swap
                    SV_FTYPE cval0ri = svtbl(cval0ir, vec_tbl);
                    SV_FTYPE cval1ri = svtbl(cval1ir, vec_tbl);
                    SV_FTYPE cval2ri = svtbl(cval2ir, vec_tbl);
                    SV_FTYPE cval3ri = svtbl(cval3ir, vec_tbl);

                    // perform matrix-vector product
                    SV_FTYPE result0, result1, result2, result3;
                    SV_FTYPE output0, output1, output2, output3;
                    MatrixVectorProduct4x4(pg, cval0ir, cval1ir, cval2ir,
                        cval3ir, cval0ri, cval1ri, cval2ri, cval3ri, mat01rr,
                        mat23rr, mat0ii, mat1ii, mat2ii, mat3ii, &result0,
                        &result1, &result2, &result3);

                    // reshuffle
                    output0 =
                        svsel(pg_select, result0, svext(result1, result1, 6));
                    output1 =
                        svsel(pg_select, svext(result0, result0, 2), result1);
                    output2 =
                        svsel(pg_select, result2, svext(result3, result3, 6));
                    output3 =
                        svsel(pg_select, svext(result2, result2, 2), result3);

                    // set values
                    svst1(pg, (ETYPE*)&state[basis_0], output0);
                    svst1(pg, (ETYPE*)&state[basis_1], output1);
                    svst1(pg, (ETYPE*)&state[basis_2], output2);
                    svst1(pg, (ETYPE*)&state[basis_3], output3);
                }
            }
        } else {  // target_qubit_index2 == 0

            ITYPE prefetch_flag =
                (5 <= target_qubit_index1) && (target_qubit_index1 <= 8);

            // create element index for shuffling in a vector
            pg_select = svcmpeq(pg,
                svand_z(pg, SvindexI(0, 1), SvdupI(target_mask2 << 1)),
                SvdupI(0));

            if (prefetch_flag) {
#ifdef _OPENMP
#pragma omp parallel for shared(pg, \
    pg_select, vec_tbl, mat0ii, mat1ii, mat2ii, mat3ii, mat01rr, mat23rr)
#endif
                for (state_index = 0; state_index < loop_dim;
                     state_index += numComplexInVec) {
                    // create index
                    ITYPE basis_0 = (state_index & low_mask) +
                                    ((state_index & mid_mask) << 1) +
                                    ((state_index & high_mask) << 2);

                    ITYPE basis_1 = state_index + (numComplexInVec >> 1);
                    basis_1 = (basis_1 & low_mask) +
                              ((basis_1 & mid_mask) << 1) +
                              ((basis_1 & high_mask) << 2);
                    ITYPE basis_2 = basis_0 + target_mask1;
                    ITYPE basis_3 = basis_1 + target_mask1;

                    // fetch values
                    SV_FTYPE input0 = svld1(pg, (ETYPE*)&state[basis_0]);
                    SV_FTYPE input1 = svld1(pg, (ETYPE*)&state[basis_1]);
                    SV_FTYPE input2 = svld1(pg, (ETYPE*)&state[basis_2]);
                    SV_FTYPE input3 = svld1(pg, (ETYPE*)&state[basis_3]);

                    // shuffle
                    SV_FTYPE cval0ir =
                         svsel(pg_select, input0, svext(input1, input1, 6));
                    SV_FTYPE cval2ir =
                         svsel(pg_select, svext(input0, input0, 2), input1);
                    SV_FTYPE cval1ir =
                         svsel(pg_select, input2, svext(input3, input3, 6));
                    SV_FTYPE cval3ir =
                        svsel(pg_select, svext(input2, input2, 2), input3);
                    // swap
                    SV_FTYPE cval0ri = svtbl(cval0ir, vec_tbl);
                    SV_FTYPE cval1ri = svtbl(cval1ir, vec_tbl);
                    SV_FTYPE cval2ri = svtbl(cval2ir, vec_tbl);
                    SV_FTYPE cval3ri = svtbl(cval3ir, vec_tbl);

                    // perform matrix-vector product
                    SV_FTYPE result0, result1, result2, result3;
                    SV_FTYPE output0, output1, output2, output3;
                    MatrixVectorProduct4x4(pg, cval0ir, cval1ir, cval2ir,
                        cval3ir, cval0ri, cval1ri, cval2ri, cval3ri, mat01rr,
                        mat23rr, mat0ii, mat1ii, mat2ii, mat3ii, &result0,
                        &result1, &result2, &result3);

                    // reshuffle
                    output0 =
                        svsel(pg_select, result0, svext(result2, result2, 6));
                    output1 =
                        svsel(pg_select, svext(result0, result0, 2), result2);
                    output2 =
                        svsel(pg_select, result1, svext(result3, result3, 6));
                    output3 =
                        svsel(pg_select, svext(result1, result1, 2), result3);

                    // set values
                    svst1(pg, (ETYPE*)&state[basis_0], output0);
                    svst1(pg, (ETYPE*)&state[basis_1], output1);
                    svst1(pg, (ETYPE*)&state[basis_2], output2);
                    svst1(pg, (ETYPE*)&state[basis_3], output3);

                    // L1 prefetch
                    __builtin_prefetch(
                        &state[basis_0 + target_mask1 * 4], 1, 3);
                    __builtin_prefetch(
                        &state[basis_2 + target_mask1 * 4], 1, 3);
                    // L2 prefetch
                    __builtin_prefetch(
                        &state[basis_0 + target_mask1 * 8], 1, 2);
                    __builtin_prefetch(
                        &state[basis_2 + target_mask1 * 8], 1, 2);
                }
            } else {
#ifdef _OPENMP
#pragma omp parallel for shared(pg, \
    pg_select, vec_tbl, mat0ii, mat1ii, mat2ii, mat3ii, mat01rr, mat23rr)
#endif
                for (state_index = 0; state_index < loop_dim;
                     state_index += numComplexInVec) {
                    // create index
                    ITYPE basis_0 = (state_index & low_mask) +
                                    ((state_index & mid_mask) << 1) +
                                    ((state_index & high_mask) << 2);

                    ITYPE basis_1 = state_index + (numComplexInVec >> 1);
                    basis_1 = (basis_1 & low_mask) +
                              ((basis_1 & mid_mask) << 1) +
                              ((basis_1 & high_mask) << 2);
                    ITYPE basis_2 = basis_0 + target_mask1;
                    ITYPE basis_3 = basis_1 + target_mask1;

                    // fetch values
                    SV_FTYPE input0 = svld1(pg, (ETYPE*)&state[basis_0]);
                    SV_FTYPE input1 = svld1(pg, (ETYPE*)&state[basis_1]);
                    SV_FTYPE input2 = svld1(pg, (ETYPE*)&state[basis_2]);
                    SV_FTYPE input3 = svld1(pg, (ETYPE*)&state[basis_3]);

                    // shuffle
                    SV_FTYPE cval0ir =
                        svsel(pg_select, input0, svext(input1, input1, 6));
                    SV_FTYPE cval2ir =
                        svsel(pg_select, svext(input0, input0, 2), input1);
                    SV_FTYPE cval1ir =
                        svsel(pg_select, input2, svext(input3, input3, 6));
                    SV_FTYPE cval3ir =
                        svsel(pg_select, svext(input2, input2, 2), input3);
                    // swap
                    SV_FTYPE cval0ri = svtbl(cval0ir, vec_tbl);
                    SV_FTYPE cval1ri = svtbl(cval1ir, vec_tbl);
                    SV_FTYPE cval2ri = svtbl(cval2ir, vec_tbl);
                    SV_FTYPE cval3ri = svtbl(cval3ir, vec_tbl);

                    // perform matrix-vector product
                    SV_FTYPE result0, result1, result2, result3;
                    SV_FTYPE output0, output1, output2, output3;
                    MatrixVectorProduct4x4(pg, cval0ir, cval1ir, cval2ir,
                        cval3ir, cval0ri, cval1ri, cval2ri, cval3ri, mat01rr,
                        mat23rr, mat0ii, mat1ii, mat2ii, mat3ii, &result0,
                        &result1, &result2, &result3);

                    // reshuffle
                    output0 =
                        svsel(pg_select, result0, svext(result2, result2, 6));
                    output1 =
                        svsel(pg_select, svext(result0, result0, 2), result2);
                    output2 =
                        svsel(pg_select, result1, svext(result3, result3, 6));
                    output3 =
                        svsel(pg_select, svext(result1, result1, 2), result3);

                    // set values
                    svst1(pg, (ETYPE*)&state[basis_0], output0);
                    svst1(pg, (ETYPE*)&state[basis_1], output1);
                    svst1(pg, (ETYPE*)&state[basis_2], output2);
                    svst1(pg, (ETYPE*)&state[basis_3], output3);
                }
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
    // create tables to extract real or imag. parts
    SV_ITYPE vec_tbl_real = svand_z(pg, SvindexI(0, 1), SvdupI(62));
    SV_ITYPE vec_tbl_imag = svorr_z(pg, SvindexI(0, 1), SvdupI(1));
    // create a predicate register for sign-reversal
    SV_PRED pg_neg = svcmpeq(pg, svand_z(pg, SvindexI(0, 1), SvdupI(1)), SvdupI(0));
    // create a table for swapping the real and imag. parts
    SV_ITYPE vec_tbl = sveor_z(pg, SvindexI(0, 1), SvdupI(1));

    // make the following vector: (0, 1, 4, 5, 2, 3, 6, 7)
    SV_ITYPE vec_shuffle_index;
    if (target_qubit_index1 > target_qubit_index2) {
        vec_shuffle_index = SvindexI(0, 1);
        vec_shuffle_index = svlsr_z(pg, vec_shuffle_index, 1);
        vec_shuffle_index = svorr_z(pg,
            svlsl_z(pg, svand_z(pg, vec_shuffle_index, SvdupI(1)), 1),
            svlsr_z(pg, vec_shuffle_index, 1));
        vec_shuffle_index = svorr_z(pg, svlsl_z(pg, vec_shuffle_index, 1),
            svand_z(pg, SvindexI(0, 1), 1));
    }

    SV_FTYPE mat0ir, mat1ir, mat2ir, mat3ir;
    SV_FTYPE mat0ri, mat1ri, mat2ri, mat3ri;

    mat0ir = svld1(pg, (ETYPE*)&matrix[0]);
    mat1ir = svld1(pg, (ETYPE*)&matrix[4]);
    mat2ir = svld1(pg, (ETYPE*)&matrix[8]);
    mat3ir = svld1(pg, (ETYPE*)&matrix[12]);

    // transpose 4x4 ComplexMatrix
    mat0ri = svuzp1(mat0ir, mat1ir);
    mat1ri = svuzp2(mat0ir, mat1ir);
    mat2ri = svuzp1(mat2ir, mat3ir);
    mat3ri = svuzp2(mat2ir, mat3ir);

    mat0ir = svtrn1(mat0ri, mat1ri);
    mat1ir = svtrn2(mat0ri, mat1ri);
    mat2ir = svtrn1(mat2ri, mat3ri);
    mat3ir = svtrn2(mat2ri, mat3ri);
 
    mat0ri = svuzp1(mat0ir, mat2ir);
    mat2ri = svuzp2(mat0ir, mat2ir);
    mat1ri = svuzp1(mat1ir, mat3ir);
    mat3ri = svuzp2(mat1ir, mat3ir);

    mat0ir = svtrn1(mat0ri, mat2ri);
    mat2ir = svtrn2(mat0ri, mat2ri);
    mat1ir = svtrn1(mat1ri, mat3ri);
    mat3ir = svtrn2(mat1ri, mat3ri);

    //
    mat0ri = svtbl(mat0ir, vec_tbl);
    mat1ri = svtbl(mat1ir, vec_tbl);
    mat2ri = svtbl(mat2ir, vec_tbl);
    mat3ri = svtbl(mat3ir, vec_tbl);
 
    mat0ri = svneg_m(mat0ri, pg_neg, mat0ri);
    mat1ri = svneg_m(mat1ri, pg_neg, mat1ri);
    mat2ri = svneg_m(mat2ri, pg_neg, mat2ri);
    mat3ri = svneg_m(mat3ri, pg_neg, mat3ri);
   
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (state_index = 0; state_index < loop_dim; state_index++) {
        // create index
        ITYPE basis_0 = (state_index & low_mask) +
                        ((state_index & mid_mask) << 1) +
                        ((state_index & high_mask) << 2);

        // fetch values
        SV_FTYPE input = svld1(pg, (ETYPE*)&state[basis_0]);
        if (target_qubit_index1 > target_qubit_index2)
            input = svtbl(input, vec_shuffle_index);

        SV_FTYPE inputrr = svtbl(input, vec_tbl_real);
        SV_FTYPE inputii = svtbl(input, vec_tbl_imag);

        // perform matrix-vector product
        SV_FTYPE output;
        output = svmul_z(pg, svdupq_lane(inputrr, 0), mat0ir);
        output = svmla_z(pg, output, svdupq_lane(inputii, 0), mat0ri);
        output = svmla_z(pg, output, svdupq_lane(inputrr, 1), mat1ir);
        output = svmla_z(pg, output, svdupq_lane(inputii, 1), mat1ri);
        output = svmla_z(pg, output, svdupq_lane(inputrr, 2), mat2ir);
        output = svmla_z(pg, output, svdupq_lane(inputii, 2), mat2ri);
        output = svmla_z(pg, output, svdupq_lane(inputrr, 3), mat3ir);
        output = svmla_z(pg, output, svdupq_lane(inputii, 3), mat3ri);

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
