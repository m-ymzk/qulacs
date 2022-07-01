
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void single_qubit_control_single_qubit_dense_matrix_gate(
    UINT control_qubit_index, UINT control_value, UINT target_qubit_index,
    const CTYPE matrix[4], CTYPE* state, ITYPE dim) {
#if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)

// ARMV8.2-A + SVE
#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim / 4, 13);  // =loop_dim
#endif

    single_qubit_control_single_qubit_dense_matrix_gate_sve(control_qubit_index,
        control_value, target_qubit_index, matrix, state, dim);

#ifdef _OPENMP
    omputil->reset_qulacs_num_threads();
#endif  // ifdef _OPENMP

#else  // if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)

#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim / 4, 13);  // =loop_dim
#endif

#ifdef _USE_SIMD
    single_qubit_control_single_qubit_dense_matrix_gate_parallel_simd(
        control_qubit_index, control_value, target_qubit_index, matrix, state,
        dim);
#else
    single_qubit_control_single_qubit_dense_matrix_gate_parallel_unroll(
        control_qubit_index, control_value, target_qubit_index, matrix, state,
        dim);
#endif

#ifdef _OPENMP
    omputil->reset_qulacs_num_threads();
#endif

#endif  // if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
}

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
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
#ifdef _OPENMP
#pragma omp parallel for
#endif
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

#ifdef _USE_SIMD
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
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
#ifdef _OPENMP
#pragma omp parallel for
#endif
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

#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim, 13);
#endif

    if (control_qubit_index < inner_qc) {
        if (target_qubit_index < inner_qc) {  // control, target: inner, inner

            single_qubit_control_single_qubit_dense_matrix_gate(
                control_qubit_index, control_value, target_qubit_index, matrix,
                state, dim);

        } else {  // control, target: inner, outer

            for (ITYPE iter = 0; iter < num_work; ++iter) {
                m->m_DC_sendrecv(si, t, dim_work, pair_rank);

                UINT index_offset = iter * dim_work;
                single_qubit_control_single_qubit_dense_matrix_gate_mpi_OI(
                    control_qubit_index, control_value, t, matrix, si, dim_work,
                    rank & pair_rank_bit, index_offset);

                si += dim_work;
            }
        }
    } else {
        if (target_qubit_index < inner_qc) {  // control, target: outer, inner
            if (((rank & control_rank_bit) && (control_value == 1)) ||
                (!(rank & control_rank_bit) && (control_value == 0)))
                single_qubit_dense_matrix_gate(
                    target_qubit_index, matrix, state, dim);
        } else {  // control, target: outer, outer
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
        }
    }

#ifdef _OPENMP
    omputil->reset_qulacs_num_threads();
#endif
}

void single_qubit_control_single_qubit_dense_matrix_gate_mpi_OI(
    UINT control_qubit_index, UINT control_value, CTYPE* t,
    const CTYPE matrix[4], CTYPE* state, ITYPE dim, int flag,
    UINT index_offset) {
    UINT control_qubit_mask = 1ULL << control_qubit_index;

#ifdef _OPENMP
#pragma omp parallel for
#endif
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

#if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
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
#endif

void single_qubit_control_single_qubit_dense_matrix_gate_mpi_OO(
    CTYPE* t, const CTYPE matrix[4], CTYPE* state, ITYPE dim, int flag) {
#if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
    ITYPE vec_len =
        getVecLength();  // length of SVE registers (# of 64-bit elements)
    if (dim >= vec_len) {
        SV_PRED pg = Svptrue();  // this predicate register is all 1.

        CTYPE *s0, *s1;
        s0 = (flag) ? t : state;
        s1 = (flag) ? state : t;

        // SVE registers for matrix[4]
        SV_FTYPE mat0r, mat0i, mat1r, mat1i;

        // load matrix elements
        mat0r = SvdupF(creal(matrix[0 + (flag != 0) * 2]));
        mat0i = SvdupF(cimag(matrix[0 + (flag != 0) * 2]));
        mat1r = SvdupF(creal(matrix[1 + (flag != 0) * 2]));
        mat1i = SvdupF(cimag(matrix[1 + (flag != 0) * 2]));

#ifdef _OPENMP
#pragma omp parallel for shared(pg, mat0r, mat0i, mat1r, mat1i)
#endif
        for (ITYPE state_index = 0; state_index < dim; state_index += vec_len) {
            // fetch values
            SV_FTYPE input0 = svld1(pg, (ETYPE*)&s0[state_index]);
            SV_FTYPE input1 = svld1(pg, (ETYPE*)&s1[state_index]);
            SV_FTYPE input2 =
                svld1(pg, (ETYPE*)&s0[state_index + (vec_len >> 1)]);
            SV_FTYPE input3 =
                svld1(pg, (ETYPE*)&s1[state_index + (vec_len >> 1)]);

            // select odd or even elements from two vectors
            SV_FTYPE cval02r = svuzp1(input0, input2);
            SV_FTYPE cval02i = svuzp2(input0, input2);
            SV_FTYPE cval13r = svuzp1(input1, input3);
            SV_FTYPE cval13i = svuzp2(input1, input3);

            // perform matrix-vector product
            SV_FTYPE result01r, result01i;

            MatrixVectorProduct2x2Half(pg, mat0r, mat0i, mat1r, mat1i, cval02r,
                cval02i, cval13r, cval13i, &result01r, &result01i);

            // interleave elements from low halves of two vectors
            SV_FTYPE output0 = svzip1(result01r, result01i);
            SV_FTYPE output1 = svzip2(result01r, result01i);

            // set values
            svst1(pg, (ETYPE*)&state[state_index], output0);
            svst1(pg, (ETYPE*)&state[state_index + (vec_len >> 1)], output1);
        }
    } else
#endif  // #if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
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

#if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)

static inline void MatrixVectorProduct2x2(SV_PRED pg, SV_FTYPE in00r,
    SV_FTYPE in00i, SV_FTYPE in11r, SV_FTYPE in11i, SV_FTYPE mat02r,
    SV_FTYPE mat02i, SV_FTYPE mat13r, SV_FTYPE mat13i, SV_FTYPE* out01r,
    SV_FTYPE* out01i);

static inline void MatrixVectorProduct2x2(SV_PRED pg, SV_FTYPE in00r,
    SV_FTYPE in00i, SV_FTYPE in11r, SV_FTYPE in11i, SV_FTYPE mat02r,
    SV_FTYPE mat02i, SV_FTYPE mat13r, SV_FTYPE mat13i, SV_FTYPE* out01r,
    SV_FTYPE* out01i) {
    *out01r = svmul_x(pg, in00r, mat02r);
    *out01i = svmul_x(pg, in00i, mat02r);

    *out01r = svmsb_x(pg, in00i, mat02i, *out01r);
    *out01i = svmad_x(pg, in00r, mat02i, *out01i);

    *out01r = svmad_x(pg, in11r, mat13r, *out01r);
    *out01i = svmad_x(pg, in11r, mat13i, *out01i);

    *out01r = svmsb_x(pg, in11i, mat13i, *out01r);
    *out01i = svmad_x(pg, in11i, mat13r, *out01i);
}

void single_qubit_control_single_qubit_dense_matrix_gate_sve(
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

        ITYPE vec_step = (vec_len >> 1);
        if (min_qubit_mask >= vec_step) {
            // If the above condition is met, the continuous elements loaded
            // into SVE variables can be applied without reordering.
            if ((4 <= control_qubit_index && control_qubit_index <= 11) ||
                (4 <= target_qubit_index && target_qubit_index <= 11)) {
#pragma omp parallel for
                for (state_index = 0; state_index < loop_dim;
                     state_index += (vec_step << 1)) {
                    // Calculate indices
                    ITYPE basis_0 = (state_index & low_mask) +
                                    ((state_index & mid_mask) << 1) +
                                    ((state_index & high_mask) << 2) +
                                    control_mask * control_value;
                    ITYPE basis_1 = basis_0 + target_mask;
                    ITYPE basis_2 =
                        ((state_index + vec_step) & low_mask) +
                        (((state_index + vec_step) & mid_mask) << 1) +
                        (((state_index + vec_step) & high_mask) << 2) +
                        control_mask * control_value;
                    ITYPE basis_3 = basis_2 + target_mask;

                    // Load values
                    SV_FTYPE input0 = svld1(pg, (ETYPE*)&state[basis_0]);
                    SV_FTYPE input1 = svld1(pg, (ETYPE*)&state[basis_1]);
                    SV_FTYPE input2 = svld1(pg, (ETYPE*)&state[basis_2]);
                    SV_FTYPE input3 = svld1(pg, (ETYPE*)&state[basis_3]);

                    // Select odd or even elements from two vectors
                    SV_FTYPE cval00r = svuzp1(input0, input0);
                    SV_FTYPE cval00i = svuzp2(input0, input0);
                    SV_FTYPE cval11r = svuzp1(input1, input1);
                    SV_FTYPE cval11i = svuzp2(input1, input1);

                    SV_FTYPE cval22r = svuzp1(input2, input2);
                    SV_FTYPE cval22i = svuzp2(input2, input2);
                    SV_FTYPE cval33r = svuzp1(input3, input3);
                    SV_FTYPE cval33i = svuzp2(input3, input3);

                    // Perform matrix-vector products
                    SV_FTYPE result01r, result01i;
                    MatrixVectorProduct2x2(pg, cval00r, cval00i, cval11r,
                        cval11i, mat02r, mat02i, mat13r, mat13i, &result01r,
                        &result01i);
                    SV_FTYPE result02r, result02i;
                    MatrixVectorProduct2x2(pg, cval22r, cval22i, cval33r,
                        cval33i, mat02r, mat02i, mat13r, mat13i, &result02r,
                        &result02i);
                    // Interleave elements from low or high halves of two
                    // vectors
                    SV_FTYPE output0 = svzip1(result01r, result01i);
                    SV_FTYPE output1 = svzip2(result01r, result01i);
                    SV_FTYPE output2 = svzip1(result02r, result02i);
                    SV_FTYPE output3 = svzip2(result02r, result02i);

                    // L1 prefetch
#undef _PRF_L1_ITR
#define _PRF_L1_ITR 2
                    ITYPE basis_index_l1pf0 =
                        ((state_index + (vec_step << 1) * _PRF_L1_ITR) &
                            low_mask) +
                        (((state_index + (vec_step << 1) * _PRF_L1_ITR) &
                             mid_mask)
                            << 1) +
                        (((state_index + (vec_step << 1) * _PRF_L1_ITR) &
                             high_mask)
                            << 2) +
                        control_mask * control_value;
                    ITYPE basis_index_l1pf1 = basis_index_l1pf0 + target_mask;

                    __builtin_prefetch(&state[basis_index_l1pf0], 1, 3);
                    __builtin_prefetch(&state[basis_index_l1pf1], 1, 3);

                    // L2 prefetch
#undef _PRF_L2_ITR
#define _PRF_L2_ITR 16
                    ITYPE basis_index_l2pf0 =
                        ((state_index + (vec_step << 1) * _PRF_L2_ITR) &
                            low_mask) +
                        (((state_index + (vec_step << 1) * _PRF_L2_ITR) &
                             mid_mask)
                            << 1) +
                        (((state_index + (vec_step << 1) * _PRF_L2_ITR) &
                             high_mask)
                            << 2) +
                        control_mask * control_value;
                    ITYPE basis_index_l2pf1 = basis_index_l2pf0 + target_mask;

                    __builtin_prefetch(&state[basis_index_l2pf0], 1, 2);
                    __builtin_prefetch(&state[basis_index_l2pf1], 1, 2);

                    // Store values
                    svst1(pg, (ETYPE*)&state[basis_0], output0);
                    svst1(pg, (ETYPE*)&state[basis_1], output1);
                    svst1(pg, (ETYPE*)&state[basis_2], output2);
                    svst1(pg, (ETYPE*)&state[basis_3], output3);
                }
            } else {
#pragma omp parallel for
                for (state_index = 0; state_index < loop_dim;
                     state_index += vec_step) {
                    // Calculate indices
                    ITYPE basis_0 = (state_index & low_mask) +
                                    ((state_index & mid_mask) << 1) +
                                    ((state_index & high_mask) << 2) +
                                    control_mask * control_value;
                    ITYPE basis_1 = basis_0 + target_mask;

                    // Load values
                    SV_FTYPE input0 = svld1(pg, (ETYPE*)&state[basis_0]);
                    SV_FTYPE input1 = svld1(pg, (ETYPE*)&state[basis_1]);

                    // Select odd or even elements from two vectors
                    SV_FTYPE cval00r = svuzp1(input0, input0);
                    SV_FTYPE cval00i = svuzp2(input0, input0);
                    SV_FTYPE cval11r = svuzp1(input1, input1);
                    SV_FTYPE cval11i = svuzp2(input1, input1);

                    // Perform matrix-vector products
                    SV_FTYPE result01r, result01i;
                    MatrixVectorProduct2x2(pg, cval00r, cval00i, cval11r,
                        cval11i, mat02r, mat02i, mat13r, mat13i, &result01r,
                        &result01i);
                    // Interleave elements from low or high halves of two
                    // vectors
                    SV_FTYPE output0 = svzip1(result01r, result01i);
                    SV_FTYPE output1 = svzip2(result01r, result01i);

                    // Store values
                    svst1(pg, (ETYPE*)&state[basis_0], output0);
                    svst1(pg, (ETYPE*)&state[basis_1], output1);
                }
            }
        } else {
            if ((loop_dim % (vec_step * 4)) == 0) {
                single_qubit_control_single_qubit_dense_matrix_gate_sve_gather_scatter_unroll4(
                    control_qubit_index, control_value, target_qubit_index,
                    matrix, state, dim);

            } else {
#pragma omp parallel for
                for (state_index = 0; state_index < loop_dim; ++state_index) {
                    ITYPE basis_0 = (state_index & low_mask) +
                                    ((state_index & mid_mask) << 1) +
                                    ((state_index & high_mask) << 2) +
                                    control_mask * control_value;
                    ITYPE basis_1 = basis_0 + target_mask;

                    // fetch values
                    CTYPE cval_0 = state[basis_0];
                    CTYPE cval_1 = state[basis_1];

                    // set values
                    state[basis_0] = matrix[0] * cval_0 + matrix[1] * cval_1;
                    state[basis_1] = matrix[2] * cval_0 + matrix[3] * cval_1;
                }
            }
        }
    } else {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask * control_value;
            ITYPE basis_1 = basis_0 + target_mask;

            // fetch values
            CTYPE cval_0 = state[basis_0];
            CTYPE cval_1 = state[basis_1];

            // set values
            state[basis_0] = matrix[0] * cval_0 + matrix[1] * cval_1;
            state[basis_1] = matrix[2] * cval_0 + matrix[3] * cval_1;
        }
    }
}

void single_qubit_control_single_qubit_dense_matrix_gate_sve_gather_scatter_unroll4(
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

    ITYPE vec_len =
        getVecLength();  // length of SVE registers (# of 64-bit elements)

    SV_PRED pg = Svptrue();  // this predicate register is all 1.
    SV_FTYPE mat02r, mat02i, mat13r, mat13i;

    // Load matrix elements to SVE variables
    // e.g.) mat02_real is [matrix[0].real, matrix[0].real, matrix[2].real,
    // matrix[2].real],
    //       if # of elements in SVE variables is four.
    mat02r = svuzp1(SvdupF(creal(matrix[0])), SvdupF(creal(matrix[2])));
    mat02i = svuzp1(SvdupF(cimag(matrix[0])), SvdupF(cimag(matrix[2])));
    mat13r = svuzp1(SvdupF(creal(matrix[1])), SvdupF(creal(matrix[3])));
    mat13i = svuzp1(SvdupF(cimag(matrix[1])), SvdupF(cimag(matrix[3])));

    SV_ITYPE sv_low_mask = SvdupI(low_mask);
    SV_ITYPE sv_mid_mask = SvdupI(mid_mask);
    SV_ITYPE sv_high_mask = SvdupI(high_mask);
    SV_ITYPE sv_control =
        svmul_z(pg, SvdupI(control_mask), SvdupI(control_value));

    SV_ITYPE vec_index = SvindexI(0, 1);    // {0,1,2,3,4,..,7}
    vec_index = svlsr_z(pg, vec_index, 1);  // {0,0,1,1,2,..,3}

    UINT loop_step = (vec_len >> 1);
    UINT vec_step = loop_step * 4;

    ITYPE state_index;

#pragma omp parallel for
    for (state_index = 0; state_index < loop_dim; state_index += vec_step) {
        SV_ITYPE sv_vec_index1 = SvdupI(state_index);
        SV_ITYPE sv_vec_index2 = SvdupI(state_index + loop_step);
        SV_ITYPE sv_vec_index3 = SvdupI(state_index + loop_step * 2);
        SV_ITYPE sv_vec_index4 = SvdupI(state_index + loop_step * 3);
        sv_vec_index1 = svadd_z(pg, sv_vec_index1, vec_index);
        sv_vec_index2 = svadd_z(pg, sv_vec_index2, vec_index);
        sv_vec_index3 = svadd_z(pg, sv_vec_index3, vec_index);
        sv_vec_index4 = svadd_z(pg, sv_vec_index4, vec_index);

        /* prefetch */
        if ((6 <= target_qubit_index && target_qubit_index <= 9) ||
            (4 <= control_qubit_index && control_qubit_index <= 9)) {
            // L1 prefetch
#undef _PRF_L1_ITR
#define _PRF_L1_ITR 4
            ITYPE basis_index_l1pf0 =
                ((state_index + vec_step * _PRF_L1_ITR) & low_mask) +
                (((state_index + vec_step * _PRF_L1_ITR) & mid_mask) << 1) +
                (((state_index + vec_step * _PRF_L1_ITR) & high_mask) << 2) +
                control_mask * control_value;
            ITYPE basis_index_l1pf1 = basis_index_l1pf0 + target_mask;
            ITYPE basis_index_l1pf2 =
                ((state_index + loop_step + vec_step * _PRF_L1_ITR) &
                    low_mask) +
                (((state_index + loop_step + vec_step * _PRF_L1_ITR) & mid_mask)
                    << 1) +
                (((state_index + loop_step + vec_step * _PRF_L1_ITR) &
                     high_mask)
                    << 2) +
                control_mask * control_value;
            ITYPE basis_index_l1pf3 = basis_index_l1pf2 + target_mask;
            ITYPE basis_index_l1pf4 =
                ((state_index + (loop_step * 2) + vec_step * _PRF_L1_ITR) &
                    low_mask) +
                (((state_index + (loop_step * 2) + vec_step * _PRF_L1_ITR) &
                     mid_mask)
                    << 1) +
                (((state_index + (loop_step * 2) + vec_step * _PRF_L1_ITR) &
                     high_mask)
                    << 2) +
                control_mask * control_value;
            ITYPE basis_index_l1pf5 = basis_index_l1pf4 + target_mask;
            ITYPE basis_index_l1pf6 =
                ((state_index + (loop_step * 3) + vec_step * _PRF_L1_ITR) &
                    low_mask) +
                (((state_index + (loop_step * 3) + vec_step * _PRF_L1_ITR) &
                     mid_mask)
                    << 1) +
                (((state_index + (loop_step * 3) + vec_step * _PRF_L1_ITR) &
                     high_mask)
                    << 2) +
                control_mask * control_value;
            ITYPE basis_index_l1pf7 = basis_index_l1pf6 + target_mask;

            __builtin_prefetch(&state[basis_index_l1pf0], 1, 3);
            __builtin_prefetch(&state[basis_index_l1pf1], 1, 3);
            __builtin_prefetch(&state[basis_index_l1pf2], 1, 3);
            __builtin_prefetch(&state[basis_index_l1pf3], 1, 3);
            __builtin_prefetch(&state[basis_index_l1pf4], 1, 3);
            __builtin_prefetch(&state[basis_index_l1pf5], 1, 3);
            __builtin_prefetch(&state[basis_index_l1pf6], 1, 3);
            __builtin_prefetch(&state[basis_index_l1pf7], 1, 3);
            // L2 prefetch
#undef _PRF_L2_ITR
#define _PRF_L2_ITR 32
            ITYPE basis_index_l2pf0 =
                ((state_index + vec_step * _PRF_L2_ITR) & low_mask) +
                (((state_index + vec_step * _PRF_L2_ITR) & mid_mask) << 1) +
                (((state_index + vec_step * _PRF_L2_ITR) & high_mask) << 2) +
                control_mask * control_value;
            ITYPE basis_index_l2pf1 = basis_index_l2pf0 + target_mask;
            ITYPE basis_index_l2pf2 =
                ((state_index + loop_step + vec_step * _PRF_L2_ITR) &
                    low_mask) +
                (((state_index + loop_step + vec_step * _PRF_L2_ITR) & mid_mask)
                    << 1) +
                (((state_index + loop_step + vec_step * _PRF_L2_ITR) &
                     high_mask)
                    << 2) +
                control_mask * control_value;
            ITYPE basis_index_l2pf3 = basis_index_l2pf2 + target_mask;
            ITYPE basis_index_l2pf4 =
                ((state_index + (loop_step * 2) + vec_step * _PRF_L2_ITR) &
                    low_mask) +
                (((state_index + (loop_step * 2) + vec_step * _PRF_L2_ITR) &
                     mid_mask)
                    << 1) +
                (((state_index + (loop_step * 2) + vec_step * _PRF_L2_ITR) &
                     high_mask)
                    << 2) +
                control_mask * control_value;
            ITYPE basis_index_l2pf5 = basis_index_l2pf4 + target_mask;
            ITYPE basis_index_l2pf6 =
                ((state_index + (loop_step * 3) + vec_step * _PRF_L2_ITR) &
                    low_mask) +
                (((state_index + (loop_step * 3) + vec_step * _PRF_L2_ITR) &
                     mid_mask)
                    << 1) +
                (((state_index + (loop_step * 3) + vec_step * _PRF_L2_ITR) &
                     high_mask)
                    << 2) +
                control_mask * control_value;
            ITYPE basis_index_l2pf7 = basis_index_l2pf6 + target_mask;

            __builtin_prefetch(&state[basis_index_l2pf0], 1, 2);
            __builtin_prefetch(&state[basis_index_l2pf1], 1, 2);
            __builtin_prefetch(&state[basis_index_l2pf2], 1, 2);
            __builtin_prefetch(&state[basis_index_l2pf3], 1, 2);
            __builtin_prefetch(&state[basis_index_l2pf4], 1, 2);
            __builtin_prefetch(&state[basis_index_l2pf5], 1, 2);
            __builtin_prefetch(&state[basis_index_l2pf6], 1, 2);
            __builtin_prefetch(&state[basis_index_l2pf7], 1, 2);
        }

        /* calclate the index */
        // (state_index & low_mask) 1/4
        SV_ITYPE sv_tmp_index1 = svand_z(pg, sv_vec_index1, sv_low_mask);
        // ((state_index & mid_mask) << 1) 1/4
        SV_ITYPE sv_tmp_index2 =
            svlsl_z(pg, svand_z(pg, sv_vec_index1, sv_mid_mask), SvdupI(1));
        // (state_index & high_mask) << 2 1/4
        SV_ITYPE sv_tmp_index3 =
            svlsl_z(pg, svand_z(pg, sv_vec_index1, sv_high_mask), SvdupI(2));
        // (state_index & low_mask) 2/4
        SV_ITYPE sv_tmp_index4 = svand_z(pg, sv_vec_index2, sv_low_mask);
        // ((state_index & mid_mask) << 1) 2/4
        SV_ITYPE sv_tmp_index5 =
            svlsl_z(pg, svand_z(pg, sv_vec_index2, sv_mid_mask), SvdupI(1));
        // (state_index & high_mask) << 2 2/4
        SV_ITYPE sv_tmp_index6 =
            svlsl_z(pg, svand_z(pg, sv_vec_index2, sv_high_mask), SvdupI(2));
        // (state_index & low_mask) 3/4
        SV_ITYPE sv_tmp_index7 = svand_z(pg, sv_vec_index3, sv_low_mask);
        // ((state_index & mid_mask) << 1) 3/4
        SV_ITYPE sv_tmp_index8 =
            svlsl_z(pg, svand_z(pg, sv_vec_index3, sv_mid_mask), SvdupI(1));
        // (state_index & high_mask) << 2 3/4
        SV_ITYPE sv_tmp_index9 =
            svlsl_z(pg, svand_z(pg, sv_vec_index3, sv_high_mask), SvdupI(2));
        // (state_index & low_mask) 4/4
        SV_ITYPE sv_tmp_index10 = svand_z(pg, sv_vec_index4, sv_low_mask);
        // ((state_index & mid_mask) << 1) 4/4
        SV_ITYPE sv_tmp_index11 =
            svlsl_z(pg, svand_z(pg, sv_vec_index4, sv_mid_mask), SvdupI(1));
        // (state_index & high_mask) << 2 4/4
        SV_ITYPE sv_tmp_index12 =
            svlsl_z(pg, svand_z(pg, sv_vec_index4, sv_high_mask), SvdupI(2));

        SV_ITYPE sv_basis_0, sv_basis_1;
        sv_basis_0 = svadd_z(pg, sv_tmp_index1, sv_tmp_index2);
        sv_basis_0 = svadd_z(pg, sv_basis_0, sv_tmp_index3);
        sv_basis_0 = svadd_z(pg, sv_basis_0, sv_control);
        sv_basis_1 = svadd_z(pg, sv_basis_0, SvdupI(target_mask));
        SV_ITYPE sv_basis_2, sv_basis_3;
        sv_basis_2 = svadd_z(pg, sv_tmp_index4, sv_tmp_index5);
        sv_basis_2 = svadd_z(pg, sv_basis_2, sv_tmp_index6);
        sv_basis_2 = svadd_z(pg, sv_basis_2, sv_control);
        sv_basis_3 = svadd_z(pg, sv_basis_2, SvdupI(target_mask));
        SV_ITYPE sv_basis_4, sv_basis_5;
        sv_basis_4 = svadd_z(pg, sv_tmp_index7, sv_tmp_index8);
        sv_basis_4 = svadd_z(pg, sv_basis_4, sv_tmp_index9);
        sv_basis_4 = svadd_z(pg, sv_basis_4, sv_control);
        sv_basis_5 = svadd_z(pg, sv_basis_4, SvdupI(target_mask));
        SV_ITYPE sv_basis_6, sv_basis_7;
        sv_basis_6 = svadd_z(pg, sv_tmp_index10, sv_tmp_index11);
        sv_basis_6 = svadd_z(pg, sv_basis_6, sv_tmp_index12);
        sv_basis_6 = svadd_z(pg, sv_basis_6, sv_control);
        sv_basis_7 = svadd_z(pg, sv_basis_6, SvdupI(target_mask));

        // complex -> double
        SV_ITYPE zero_one = svzip1(SvdupI(0), SvdupI(1));
        sv_basis_0 = svmul_z(pg, sv_basis_0, SvdupI(2));
        sv_basis_1 = svmul_z(pg, sv_basis_1, SvdupI(2));
        sv_basis_0 = svadd_z(pg, sv_basis_0, zero_one);
        sv_basis_1 = svadd_z(pg, sv_basis_1, zero_one);
        sv_basis_2 = svmul_z(pg, sv_basis_2, SvdupI(2));
        sv_basis_3 = svmul_z(pg, sv_basis_3, SvdupI(2));
        sv_basis_2 = svadd_z(pg, sv_basis_2, zero_one);
        sv_basis_3 = svadd_z(pg, sv_basis_3, zero_one);
        sv_basis_4 = svmul_z(pg, sv_basis_4, SvdupI(2));
        sv_basis_5 = svmul_z(pg, sv_basis_5, SvdupI(2));
        sv_basis_4 = svadd_z(pg, sv_basis_4, zero_one);
        sv_basis_5 = svadd_z(pg, sv_basis_5, zero_one);
        sv_basis_6 = svmul_z(pg, sv_basis_6, SvdupI(2));
        sv_basis_7 = svmul_z(pg, sv_basis_7, SvdupI(2));
        sv_basis_6 = svadd_z(pg, sv_basis_6, zero_one);
        sv_basis_7 = svadd_z(pg, sv_basis_7, zero_one);

        // Load values (Gather)
        ETYPE* ptr = (ETYPE*)&state[0];
        const SV_FTYPE input0 = svld1_gather_index(pg, ptr, sv_basis_0);
        const SV_FTYPE input1 = svld1_gather_index(pg, ptr, sv_basis_1);
        const SV_FTYPE input2 = svld1_gather_index(pg, ptr, sv_basis_2);
        const SV_FTYPE input3 = svld1_gather_index(pg, ptr, sv_basis_3);
        const SV_FTYPE input4 = svld1_gather_index(pg, ptr, sv_basis_4);
        const SV_FTYPE input5 = svld1_gather_index(pg, ptr, sv_basis_5);
        const SV_FTYPE input6 = svld1_gather_index(pg, ptr, sv_basis_6);
        const SV_FTYPE input7 = svld1_gather_index(pg, ptr, sv_basis_7);

        // Select odd or even elements from two vectors
        SV_FTYPE cval00r = svuzp1(input0, input0);
        SV_FTYPE cval00i = svuzp2(input0, input0);
        SV_FTYPE cval11r = svuzp1(input1, input1);
        SV_FTYPE cval11i = svuzp2(input1, input1);
        SV_FTYPE cval22r = svuzp1(input2, input2);
        SV_FTYPE cval22i = svuzp2(input2, input2);
        SV_FTYPE cval33r = svuzp1(input3, input3);
        SV_FTYPE cval33i = svuzp2(input3, input3);
        SV_FTYPE cval44r = svuzp1(input4, input4);
        SV_FTYPE cval44i = svuzp2(input4, input4);
        SV_FTYPE cval55r = svuzp1(input5, input5);
        SV_FTYPE cval55i = svuzp2(input5, input5);
        SV_FTYPE cval66r = svuzp1(input6, input6);
        SV_FTYPE cval66i = svuzp2(input6, input6);
        SV_FTYPE cval77r = svuzp1(input7, input7);
        SV_FTYPE cval77i = svuzp2(input7, input7);

        // Perform matrix-vector products
        SV_FTYPE result01r, result01i;
        MatrixVectorProduct2x2(pg, cval00r, cval00i, cval11r, cval11i, mat02r,
            mat02i, mat13r, mat13i, &result01r, &result01i);
        SV_FTYPE result02r, result02i;
        MatrixVectorProduct2x2(pg, cval22r, cval22i, cval33r, cval33i, mat02r,
            mat02i, mat13r, mat13i, &result02r, &result02i);
        SV_FTYPE result03r, result03i;
        MatrixVectorProduct2x2(pg, cval44r, cval44i, cval55r, cval55i, mat02r,
            mat02i, mat13r, mat13i, &result03r, &result03i);
        SV_FTYPE result04r, result04i;
        MatrixVectorProduct2x2(pg, cval66r, cval66i, cval77r, cval77i, mat02r,
            mat02i, mat13r, mat13i, &result04r, &result04i);

        // Interleave elements from low or high halves of two vectors
        SV_FTYPE output0 = svzip1(result01r, result01i);
        SV_FTYPE output1 = svzip2(result01r, result01i);
        SV_FTYPE output2 = svzip1(result02r, result02i);
        SV_FTYPE output3 = svzip2(result02r, result02i);
        SV_FTYPE output4 = svzip1(result03r, result03i);
        SV_FTYPE output5 = svzip2(result03r, result03i);
        SV_FTYPE output6 = svzip1(result04r, result04i);
        SV_FTYPE output7 = svzip2(result04r, result04i);

        // Store values (Scatter)
        svst1_scatter_index(pg, ptr, sv_basis_0, output0);
        svst1_scatter_index(pg, ptr, sv_basis_1, output1);
        svst1_scatter_index(pg, ptr, sv_basis_2, output2);
        svst1_scatter_index(pg, ptr, sv_basis_3, output3);
        svst1_scatter_index(pg, ptr, sv_basis_4, output4);
        svst1_scatter_index(pg, ptr, sv_basis_5, output5);
        svst1_scatter_index(pg, ptr, sv_basis_6, output6);
        svst1_scatter_index(pg, ptr, sv_basis_7, output7);
    }
}
#endif  // if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
