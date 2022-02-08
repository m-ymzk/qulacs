
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

// void single_qubit_diagonal_matrix_gate_old_single(UINT target_qubit_index,
// const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim); void
// single_qubit_diagonal_matrix_gate_old_parallel(UINT target_qubit_index, const
// CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim);

void single_qubit_diagonal_matrix_gate(UINT target_qubit_index,
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
    // single_qubit_diagonal_matrix_gate_old_single(target_qubit_index,
    // diagonal_matrix, state, dim);
    // single_qubit_diagonal_matrix_gate_old_parallel(target_qubit_index,
    // diagonal_matrix, state, dim);
    // single_qubit_diagonal_matrix_gate_single_unroll(target_qubit_index,
    // diagonal_matrix, state, dim);
    // single_qubit_diagonal_matrix_gate_single_simd(target_qubit_index,
    // diagonal_matrix, state, dim);
    // single_qubit_diagonal_matrix_gate_parallel_simd(target_qubit_index,
    // diagonal_matrix, state, dim);

#ifdef _USE_SIMD
#ifdef _OPENMP
    UINT threshold = 12;
    if (dim < (((ITYPE)1) << threshold)) {
        single_qubit_diagonal_matrix_gate_single_simd(
            target_qubit_index, diagonal_matrix, state, dim);
    } else {
        single_qubit_diagonal_matrix_gate_parallel_simd(
            target_qubit_index, diagonal_matrix, state, dim);
    }
#else                                                  // #ifdef _OPENMP
    single_qubit_diagonal_matrix_gate_single_simd(
        target_qubit_index, diagonal_matrix, state, dim);
#endif                                                 // #ifdef _OPENMP
#elif defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)  // #ifdef _USE_SIMD
#ifdef _OPENMP
    UINT threshold = 12;
    if (dim < (((ITYPE)1) << threshold)) {
        single_qubit_diagonal_matrix_gate_single_sve(
            target_qubit_index, diagonal_matrix, state, dim);
    } else {
        single_qubit_diagonal_matrix_gate_parallel_sve(
            target_qubit_index, diagonal_matrix, state, dim);
    }
#else   // #ifdef _OPENMP
    single_qubit_diagonal_matrix_gate_single_sve(
        target_qubit_index, diagonal_matrix, state, dim);
#endif  // #ifdef _OPENMP
#else   // #ifdef _USE_SIMD
#ifdef _OPENMP
    UINT threshold = 12;
    if (dim < (((ITYPE)1) << threshold)) {
        single_qubit_diagonal_matrix_gate_single_unroll(
            target_qubit_index, diagonal_matrix, state, dim);
    } else {
        single_qubit_diagonal_matrix_gate_parallel_unroll(
            target_qubit_index, diagonal_matrix, state, dim);
    }
#else
    single_qubit_diagonal_matrix_gate_single_unroll(
        target_qubit_index, diagonal_matrix, state, dim);
#endif
#endif
}

void single_qubit_diagonal_matrix_gate_single_unroll(UINT target_qubit_index,
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
    // loop variables
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    if (target_qubit_index == 0) {
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            state[state_index] *= diagonal_matrix[0];
            state[state_index + 1] *= diagonal_matrix[1];
        }
    } else {
        ITYPE mask = 1ULL << target_qubit_index;
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            int bitval = ((state_index & mask) != 0);
            state[state_index] *= diagonal_matrix[bitval];
            state[state_index + 1] *= diagonal_matrix[bitval];
        }
    }
}

#ifdef _OPENMP
void single_qubit_diagonal_matrix_gate_parallel_unroll(UINT target_qubit_index,
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
    // loop variables
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    if (target_qubit_index == 0) {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            state[state_index] *= diagonal_matrix[0];
            state[state_index + 1] *= diagonal_matrix[1];
        }
    } else {
        ITYPE mask = 1ULL << target_qubit_index;
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            int bitval = ((state_index & mask) != 0);
            state[state_index] *= diagonal_matrix[bitval];
            state[state_index + 1] *= diagonal_matrix[bitval];
        }
    }
}
#endif

#if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)

void single_qubit_diagonal_matrix_gate_single_sve(UINT target_qubit_index,
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
    // loop variables
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    ITYPE vec_len =
        getVecLength();  // length of SVE registers (# of 64-bit elements)
    ITYPE mask = 1ULL << target_qubit_index;

    if (target_qubit_index >= vec_len) {
        SV_PRED pg = Svptrue();  // this predicate register is all 1.

        SV_FTYPE mat0_real, mat0_imag, mat1_real, mat1_imag;
        SV_FTYPE mat_real, mat_imag;
        SV_FTYPE input0, input1, output0, output1;
        SV_FTYPE cval_real, cval_imag, result_real, result_imag;

        mat0_real = SvdupF(creal(diagonal_matrix[0]));
        mat0_imag = SvdupF(cimag(diagonal_matrix[0]));
        mat1_real = SvdupF(creal(diagonal_matrix[1]));
        mat1_imag = SvdupF(cimag(diagonal_matrix[1]));

        for (state_index = 0; state_index < loop_dim; state_index += vec_len) {
            int bitval = ((state_index & mask) != 0);

            // fetch values
            input0 = svld1(pg, (ETYPE *)&state[state_index]);
            input1 = svld1(pg, (ETYPE *)&state[state_index + (vec_len >> 1)]);

            // select matrix elements
            mat_real = (bitval != 0) ? mat1_real : mat0_real;
            mat_imag = (bitval != 0) ? mat1_imag : mat0_imag;

            // select odd or even elements from two vectors
            cval_real = svuzp1(input0, input1);
            cval_imag = svuzp2(input0, input1);

            // perform complex multiplication
            result_real = svmul_x(pg, cval_real, mat_real);
            result_imag = svmul_x(pg, cval_imag, mat_real);

            result_real = svmsb_x(pg, cval_imag, mat_imag, result_real);
            result_imag = svmad_x(pg, cval_real, mat_imag, result_imag);

            // interleave elements from low halves of two vectors
            output0 = svzip1(result_real, result_imag);
            output1 = svzip2(result_real, result_imag);

            // set values
            svst1(pg, (ETYPE *)&state[state_index], output0);
            svst1(pg, (ETYPE *)&state[state_index + (vec_len >> 1)], output1);
        }
    } else {
        if (loop_dim >= vec_len) {
            SV_PRED pg = Svptrue();  // this predicate register is all 1.
            SV_PRED select_flag;

            SV_FTYPE mat0_real, mat0_imag, mat1_real, mat1_imag;
            SV_FTYPE mat_real, mat_imag;
            SV_FTYPE input0, input1, output0, output1;
            SV_FTYPE cval_real, cval_imag, result_real, result_imag;

            // SVE registers for control factor elements
            SV_ITYPE vec_index_diff, vec_bitval;
            vec_index_diff = SvindexI(0, 1);  // (0, 1, 2, 3, 4,...)

            mat0_real = SvdupF(creal(diagonal_matrix[0]));
            mat0_imag = SvdupF(cimag(diagonal_matrix[0]));
            mat1_real = SvdupF(creal(diagonal_matrix[1]));
            mat1_imag = SvdupF(cimag(diagonal_matrix[1]));
            for (state_index = 0; state_index < loop_dim;
                 state_index += vec_len) {
                // fetch values
                input0 = svld1(pg, (ETYPE *)&state[state_index]);
                input1 =
                    svld1(pg, (ETYPE *)&state[state_index + (vec_len >> 1)]);

                // select matrix elements
                vec_bitval =
                    svadd_z(pg, vec_index_diff, SvdupI(state_index));
                vec_bitval = svand_z(pg, vec_bitval, SvdupI(mask));
                select_flag = svcmpne(pg, vec_bitval, SvdupI(0));

                mat_real = svsel(select_flag, mat1_real, mat0_real);
                mat_imag = svsel(select_flag, mat1_imag, mat0_imag);

                // select odd or even elements from two vectors
                cval_real = svuzp1(input0, input1);
                cval_imag = svuzp2(input0, input1);

                // perform complex multiplication
                result_real = svmul_x(pg, cval_real, mat_real);
                result_imag = svmul_x(pg, cval_imag, mat_real);

                result_real = svmsb_x(pg, cval_imag, mat_imag, result_real);
                result_imag = svmad_x(pg, cval_real, mat_imag, result_imag);

                // interleave elements from low halves of two vectors
                output0 = svzip1(result_real, result_imag);
                output1 = svzip2(result_real, result_imag);

                // set values
                svst1(pg, (ETYPE *)&state[state_index], output0);
                svst1(pg, (ETYPE *)&state[state_index + (vec_len >> 1)],
                    output1);
            }
        } else {
            for (state_index = 0; state_index < loop_dim; state_index++) {
                int bitval = ((state_index & mask) != 0);
                state[state_index] *= diagonal_matrix[bitval];
            }
        }
    }
}

#ifdef _OPENMP
void single_qubit_diagonal_matrix_gate_parallel_sve(UINT target_qubit_index,
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
    // loop variables
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    ITYPE vec_len =
        getVecLength();  // length of SVE registers (# of 64-bit elements)
    ITYPE mask = 1ULL << target_qubit_index;

    if (target_qubit_index >= vec_len) {
        SV_PRED pg = Svptrue();  // this predicate register is all 1.

        SV_FTYPE mat0_real, mat0_imag, mat1_real, mat1_imag;
        SV_FTYPE mat_real, mat_imag;
        SV_FTYPE input0, input1, output0, output1;
        SV_FTYPE cval_real, cval_imag, result_real, result_imag;

        mat0_real = SvdupF(creal(diagonal_matrix[0]));
        mat0_imag = SvdupF(cimag(diagonal_matrix[0]));
        mat1_real = SvdupF(creal(diagonal_matrix[1]));
        mat1_imag = SvdupF(cimag(diagonal_matrix[1]));

#pragma omp parallel for private(mat_real, mat_imag, input0, input1, output0, \
    output1, cval_real, cval_imag, result_real, result_imag)                  \
    shared(mask, pg, mat0_real, mat0_imag, mat1_real, mat1_imag)
        for (state_index = 0; state_index < loop_dim; state_index += vec_len) {
            int bitval = ((state_index & mask) != 0);

            // fetch values
            input0 = svld1(pg, (ETYPE *)&state[state_index]);
            input1 = svld1(pg, (ETYPE *)&state[state_index + (vec_len >> 1)]);

            // select matrix elements
            mat_real = (bitval != 0) ? mat1_real : mat0_real;
            mat_imag = (bitval != 0) ? mat1_imag : mat0_imag;

            // select odd or even elements from two vectors
            cval_real = svuzp1(input0, input1);
            cval_imag = svuzp2(input0, input1);

            // perform complex multiplication
            result_real = svmul_x(pg, cval_real, mat_real);
            result_imag = svmul_x(pg, cval_imag, mat_real);

            result_real = svmsb_x(pg, cval_imag, mat_imag, result_real);
            result_imag = svmad_x(pg, cval_real, mat_imag, result_imag);

            // interleave elements from low halves of two vectors
            output0 = svzip1(result_real, result_imag);
            output1 = svzip2(result_real, result_imag);

            // set values
            svst1(pg, (ETYPE *)&state[state_index], output0);
            svst1(pg, (ETYPE *)&state[state_index + (vec_len >> 1)], output1);
        }
    } else {
        if (loop_dim >= vec_len) {
            SV_PRED pg = Svptrue();  // this predicate register is all 1.
            SV_PRED select_flag;

            SV_FTYPE mat0_real, mat0_imag, mat1_real, mat1_imag;
            SV_FTYPE mat_real, mat_imag;
            SV_FTYPE input0, input1, output0, output1;
            SV_FTYPE cval_real, cval_imag, result_real, result_imag;

            // SVE registers for control factor elements
            SV_ITYPE vec_index_diff, vec_bitval;
            vec_index_diff = SvindexI(0, 1);  // (0, 1, 2, 3, 4,...)

            mat0_real = SvdupF(creal(diagonal_matrix[0]));
            mat0_imag = SvdupF(cimag(diagonal_matrix[0]));
            mat1_real = SvdupF(creal(diagonal_matrix[1]));
            mat1_imag = SvdupF(cimag(diagonal_matrix[1]));

#pragma omp parallel for private(mat_real, mat_imag, input0, input1, output0, \
    output1, cval_real, cval_imag, result_real, result_imag, vec_bitval,      \
    select_flag) shared(mask, pg, vec_index_diff, mat0_real, mat0_imag,       \
    mat1_real, mat1_imag)
            for (state_index = 0; state_index < loop_dim;
                 state_index += vec_len) {
                // fetch values
                input0 = svld1(pg, (ETYPE *)&state[state_index]);
                input1 =
                    svld1(pg, (ETYPE *)&state[state_index + (vec_len >> 1)]);

                // select matrix elements
                vec_bitval =
                    svadd_z(pg, vec_index_diff, SvdupI(state_index));
                vec_bitval = svand_z(pg, vec_bitval, SvdupI(mask));
                select_flag = svcmpne(pg, vec_bitval, SvdupI(0));

                mat_real = svsel(select_flag, mat1_real, mat0_real);
                mat_imag = svsel(select_flag, mat1_imag, mat0_imag);

                // select odd or even elements from two vectors
                cval_real = svuzp1(input0, input1);
                cval_imag = svuzp2(input0, input1);

                // perform complex multiplication
                result_real = svmul_x(pg, cval_real, mat_real);
                result_imag = svmul_x(pg, cval_imag, mat_real);

                result_real = svmsb_x(pg, cval_imag, mat_imag, result_real);
                result_imag = svmad_x(pg, cval_real, mat_imag, result_imag);

                // interleave elements from low halves of two vectors
                output0 = svzip1(result_real, result_imag);
                output1 = svzip2(result_real, result_imag);

                // set values
                svst1(pg, (ETYPE *)&state[state_index], output0);
                svst1(pg, (ETYPE *)&state[state_index + (vec_len >> 1)],
                    output1);
            }
        } else {
#pragma omp parallel for
            for (state_index = 0; state_index < loop_dim; state_index++) {
                int bitval = ((state_index & mask) != 0);
                state[state_index] *= diagonal_matrix[bitval];
            }
        }
    }
}

#endif  // #ifdef _OPENMP
#endif  // #if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)

#ifdef _USE_SIMD
void single_qubit_diagonal_matrix_gate_single_simd(UINT target_qubit_index,
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
    // loop variables
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    if (target_qubit_index == 0) {
        __m256d mv0 =
            _mm256_set_pd(-cimag(diagonal_matrix[1]), creal(diagonal_matrix[1]),
                -cimag(diagonal_matrix[0]), creal(diagonal_matrix[0]));
        __m256d mv1 =
            _mm256_set_pd(creal(diagonal_matrix[1]), cimag(diagonal_matrix[1]),
                creal(diagonal_matrix[0]), cimag(diagonal_matrix[0]));
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            double *ptr = (double *)(state + state_index);
            __m256d data = _mm256_loadu_pd(ptr);
            __m256d data0 = _mm256_mul_pd(data, mv0);
            __m256d data1 = _mm256_mul_pd(data, mv1);
            data = _mm256_hadd_pd(data0, data1);
            _mm256_storeu_pd(ptr, data);
        }
    } else {
        __m256d mv0 =
            _mm256_set_pd(-cimag(diagonal_matrix[0]), creal(diagonal_matrix[0]),
                -cimag(diagonal_matrix[0]), creal(diagonal_matrix[0]));
        __m256d mv1 =
            _mm256_set_pd(creal(diagonal_matrix[0]), cimag(diagonal_matrix[0]),
                creal(diagonal_matrix[0]), cimag(diagonal_matrix[0]));
        __m256d mv2 =
            _mm256_set_pd(-cimag(diagonal_matrix[1]), creal(diagonal_matrix[1]),
                -cimag(diagonal_matrix[1]), creal(diagonal_matrix[1]));
        __m256d mv3 =
            _mm256_set_pd(creal(diagonal_matrix[1]), cimag(diagonal_matrix[1]),
                creal(diagonal_matrix[1]), cimag(diagonal_matrix[1]));
        //__m256i mask = _mm256_set1_epi64x(1LL<<target_qubit_index);
        ITYPE mask = 1LL << target_qubit_index;
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            double *ptr = (double *)(state + state_index);
            ITYPE flag = (state_index & mask);
            __m256d mv4 = flag ? mv2 : mv0;
            __m256d mv5 = flag ? mv3 : mv1;
            __m256d data = _mm256_loadu_pd(ptr);
            __m256d data0 = _mm256_mul_pd(data, mv4);
            __m256d data1 = _mm256_mul_pd(data, mv5);
            data = _mm256_hadd_pd(data0, data1);
            _mm256_storeu_pd(ptr, data);
        }
    }
}

#ifdef _OPENMP
void single_qubit_diagonal_matrix_gate_parallel_simd(UINT target_qubit_index,
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
    // loop variables
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    if (target_qubit_index == 0) {
        __m256d mv0 =
            _mm256_set_pd(-cimag(diagonal_matrix[1]), creal(diagonal_matrix[1]),
                -cimag(diagonal_matrix[0]), creal(diagonal_matrix[0]));
        __m256d mv1 =
            _mm256_set_pd(creal(diagonal_matrix[1]), cimag(diagonal_matrix[1]),
                creal(diagonal_matrix[0]), cimag(diagonal_matrix[0]));
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            double *ptr = (double *)(state + state_index);
            __m256d data = _mm256_loadu_pd(ptr);
            __m256d data0 = _mm256_mul_pd(data, mv0);
            __m256d data1 = _mm256_mul_pd(data, mv1);
            data = _mm256_hadd_pd(data0, data1);
            _mm256_storeu_pd(ptr, data);
        }
    } else {
        __m256d mv0 =
            _mm256_set_pd(-cimag(diagonal_matrix[0]), creal(diagonal_matrix[0]),
                -cimag(diagonal_matrix[0]), creal(diagonal_matrix[0]));
        __m256d mv1 =
            _mm256_set_pd(creal(diagonal_matrix[0]), cimag(diagonal_matrix[0]),
                creal(diagonal_matrix[0]), cimag(diagonal_matrix[0]));
        __m256d mv2 =
            _mm256_set_pd(-cimag(diagonal_matrix[1]), creal(diagonal_matrix[1]),
                -cimag(diagonal_matrix[1]), creal(diagonal_matrix[1]));
        __m256d mv3 =
            _mm256_set_pd(creal(diagonal_matrix[1]), cimag(diagonal_matrix[1]),
                creal(diagonal_matrix[1]), cimag(diagonal_matrix[1]));
        //__m256i mask = _mm256_set1_epi64x(1LL<<target_qubit_index);
        ITYPE mask = 1LL << target_qubit_index;
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            double *ptr = (double *)(state + state_index);
            ITYPE flag = (state_index & mask);
            __m256d mv4 = flag ? mv2 : mv0;
            __m256d mv5 = flag ? mv3 : mv1;
            __m256d data = _mm256_loadu_pd(ptr);
            __m256d data0 = _mm256_mul_pd(data, mv4);
            __m256d data1 = _mm256_mul_pd(data, mv5);
            data = _mm256_hadd_pd(data0, data1);
            _mm256_storeu_pd(ptr, data);
        }
    }
}
#endif
#endif

#ifdef _USE_MPI
void single_qubit_diagonal_matrix_gate_mpi(UINT target_qubit_index,
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim, UINT inner_qc) {
    if (target_qubit_index < inner_qc) {
        single_qubit_diagonal_matrix_gate(
            target_qubit_index, diagonal_matrix, state, dim);
    } else {
        const MPIutil m = get_mpiutil();
        const int rank = m->get_rank();
        const int pair_rank_bit = 1 << (target_qubit_index - inner_qc);

#ifdef _OPENMP
        UINT threshold = 12;
        if (dim < (((ITYPE)1) << threshold)) {
            single_qubit_diagonal_matrix_gate_single_unroll_mpi(
                diagonal_matrix, state, dim, (rank & pair_rank_bit) != 0);
        } else {
            single_qubit_diagonal_matrix_gate_parallel_unroll_mpi(
                diagonal_matrix, state, dim, (rank & pair_rank_bit) != 0);
        }
#else
        single_qubit_diagonal_matrix_gate_single_unroll_mpi(
            diagonal_matrix, state, dim, rank & pair_rank_bit);
#endif
    }
}

// flag: My qubit(target in outer_qubit) value.
void single_qubit_diagonal_matrix_gate_single_unroll_mpi(
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim, int isone) {
    // loop variables
    ITYPE state_index;
#if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
    ITYPE vec_len =
        getVecLength();  // length of SVE registers (# of 64-bit elements)

    if (dim >= vec_len) {
        SV_PRED pg = Svptrue();  // this predicate register is all 1.

        SV_FTYPE mat_real, mat_imag;
        SV_FTYPE input0, input1, output0, output1;
        SV_FTYPE cval_real, cval_imag, result_real, result_imag;

        mat_real = SvdupF(creal(diagonal_matrix[isone]));
        mat_imag = SvdupF(cimag(diagonal_matrix[isone]));
        for (state_index = 0; state_index < dim; state_index += vec_len) {
            // fetch values
            input0 = svld1(pg, (ETYPE *)&state[state_index]);
            input1 = svld1(pg, (ETYPE *)&state[state_index + (vec_len >> 1)]);

            // select odd or even elements from two vectors
            cval_real = svuzp1(input0, input1);
            cval_imag = svuzp2(input0, input1);

            // perform complex multiplication
            result_real = svmul_x(pg, cval_real, mat_real);
            result_imag = svmul_x(pg, cval_imag, mat_real);

            result_real = svmsb_x(pg, cval_imag, mat_imag, result_real);
            result_imag = svmad_x(pg, cval_real, mat_imag, result_imag);

            // interleave elements from low halves of two vectors
            output0 = svzip1(result_real, result_imag);
            output1 = svzip2(result_real, result_imag);

            // set values
            svst1(pg, (ETYPE *)&state[state_index], output0);
            svst1(pg, (ETYPE *)&state[state_index + (vec_len >> 1)], output1);
        }
    } else
#endif  // #if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
    {
        for (state_index = 0; state_index < dim; state_index += 2) {
            state[state_index] *= diagonal_matrix[isone];
            state[state_index + 1] *= diagonal_matrix[isone];
        }
    }
}

#ifdef _OPENMP
void single_qubit_diagonal_matrix_gate_parallel_unroll_mpi(
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim, int isone) {
    // loop variables
    ITYPE state_index;
#if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
    ITYPE vec_len =
        getVecLength();  // length of SVE registers (# of 64-bit elements)

    if (dim >= vec_len) {
        SV_PRED pg = Svptrue();  // this predicate register is all 1.

        SV_FTYPE mat_real, mat_imag;
        SV_FTYPE input0, input1, output0, output1;
        SV_FTYPE cval_real, cval_imag, result_real, result_imag;

        mat_real = SvdupF(creal(diagonal_matrix[isone]));
        mat_imag = SvdupF(cimag(diagonal_matrix[isone]));

#pragma omp parallel for private(input0, input1, output0, output1, cval_real, \
    cval_imag, result_real, result_imag) shared(pg, mat_real, mat_imag)
        for (state_index = 0; state_index < dim; state_index += vec_len) {
            // fetch values
            input0 = svld1(pg, (ETYPE *)&state[state_index]);
            input1 = svld1(pg, (ETYPE *)&state[state_index + (vec_len >> 1)]);

            // select odd or even elements from two vectors
            cval_real = svuzp1(input0, input1);
            cval_imag = svuzp2(input0, input1);

            // perform complex multiplication
            result_real = svmul_x(pg, cval_real, mat_real);
            result_imag = svmul_x(pg, cval_imag, mat_real);

            result_real = svmsb_x(pg, cval_imag, mat_imag, result_real);
            result_imag = svmad_x(pg, cval_real, mat_imag, result_imag);

            // interleave elements from low halves of two vectors
            output0 = svzip1(result_real, result_imag);
            output1 = svzip2(result_real, result_imag);

            // set values
            svst1(pg, (ETYPE *)&state[state_index], output0);
            svst1(pg, (ETYPE *)&state[state_index + (vec_len >> 1)], output1);
        }
    } else
#endif  // #if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
    {
#pragma omp parallel for
        for (state_index = 0; state_index < dim; state_index += 2) {
            state[state_index] *= diagonal_matrix[isone];
            state[state_index + 1] *= diagonal_matrix[isone];
        }
    }
}
#endif
#endif  //#ifdef _USE_MPI

/*

void single_qubit_diagonal_matrix_gate_old_single(UINT target_qubit_index, const
CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
        // loop variables
        const ITYPE loop_dim = dim;
        ITYPE state_index;
        for (state_index = 0; state_index < loop_dim; ++state_index) {
                // determin matrix pos
                UINT bit_val = (state_index >> target_qubit_index) % 2;

                // set value
                state[state_index] *= diagonal_matrix[bit_val];
        }
}

#ifdef _OPENMP
void single_qubit_diagonal_matrix_gate_old_parallel(UINT target_qubit_index,
const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
        // loop variables
        const ITYPE loop_dim = dim;
        ITYPE state_index;
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
                // determin matrix pos
                UINT bit_val = (state_index >> target_qubit_index) % 2;

                // set value
                state[state_index] *= diagonal_matrix[bit_val];
        }
}
#endif
*/
