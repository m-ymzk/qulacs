
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

void single_qubit_diagonal_matrix_gate(UINT target_qubit_index,
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
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
    UINT default_thread_count = omp_get_max_threads();
    if (dim < (((ITYPE)1) << threshold)) omp_set_num_threads(1);
#endif
    single_qubit_diagonal_matrix_gate_sve(
        target_qubit_index, diagonal_matrix, state, dim);

#ifdef _OPENMP
    omp_set_num_threads(default_thread_count);
#endif

#else  // #ifdef _USE_SIMD
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

void single_qubit_diagonal_matrix_gate_sve(UINT target_qubit_index,
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim) {
    // loop variables
    const ITYPE loop_dim = dim;
    ITYPE state_index;
    ITYPE vec_len =
        getVecLength();  // length of SVE registers (# of 64-bit elements)
    ITYPE mask = 1ULL << target_qubit_index;

    if (target_qubit_index >= vec_len) {
        SV_PRED pg = Svptrue();  // this predicate register is all 1.

        SV_FTYPE mat0r, mat0i, mat1r, mat1i;

        mat0r = SvdupF(creal(diagonal_matrix[0]));
        mat0i = SvdupF(cimag(diagonal_matrix[0]));
        mat1r = SvdupF(creal(diagonal_matrix[1]));
        mat1i = SvdupF(cimag(diagonal_matrix[1]));

#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += vec_len) {
            int bitval = ((state_index & mask) != 0);

            // fetch values
            SV_FTYPE input0 = svld1(pg, (ETYPE *)&state[state_index]);
            SV_FTYPE input1 =
                svld1(pg, (ETYPE *)&state[state_index + (vec_len >> 1)]);

            // select matrix elements
            SV_FTYPE matr = (bitval != 0) ? mat1r : mat0r;
            SV_FTYPE mati = (bitval != 0) ? mat1i : mat0i;

            // select odd or even elements from two vectors
            SV_FTYPE cvalr = svuzp1(input0, input1);
            SV_FTYPE cvali = svuzp2(input0, input1);

            // perform complex multiplication
            SV_FTYPE resultr = svmul_x(pg, cvalr, matr);
            SV_FTYPE resulti = svmul_x(pg, cvali, matr);

            resultr = svmsb_x(pg, cvali, mati, resultr);
            resulti = svmad_x(pg, cvalr, mati, resulti);

            // interleave elements from low halves of two vectors
            SV_FTYPE output0 = svzip1(resultr, resulti);
            SV_FTYPE output1 = svzip2(resultr, resulti);

            // set values
            svst1(pg, (ETYPE *)&state[state_index], output0);
            svst1(pg, (ETYPE *)&state[state_index + (vec_len >> 1)], output1);
        }
    } else {
        if (loop_dim >= vec_len) {
            SV_PRED pg = Svptrue();  // this predicate register is all 1.

            SV_FTYPE mat0r, mat0i, mat1r, mat1i;

            // SVE registers for control factor elements
            SV_ITYPE vec_index_diff;
            vec_index_diff = SvindexI(0, 1);  // (0, 1, 2, 3, 4,...)

            mat0r = SvdupF(creal(diagonal_matrix[0]));
            mat0i = SvdupF(cimag(diagonal_matrix[0]));
            mat1r = SvdupF(creal(diagonal_matrix[1]));
            mat1i = SvdupF(cimag(diagonal_matrix[1]));

#pragma omp parallel for
            for (state_index = 0; state_index < loop_dim;
                 state_index += vec_len) {
                // fetch values
                SV_FTYPE input0 = svld1(pg, (ETYPE *)&state[state_index]);
                SV_FTYPE input1 =
                    svld1(pg, (ETYPE *)&state[state_index + (vec_len >> 1)]);

                // select matrix elements
                SV_ITYPE vec_bitval =
                    svadd_z(pg, vec_index_diff, SvdupI(state_index));
                vec_bitval = svand_z(pg, vec_bitval, SvdupI(mask));

                SV_PRED select_flag = svcmpne(pg, vec_bitval, SvdupI(0));

                SV_FTYPE matr = svsel(select_flag, mat1r, mat0r);
                SV_FTYPE mati = svsel(select_flag, mat1i, mat0i);

                // select odd or even elements from two vectors
                SV_FTYPE cvalr = svuzp1(input0, input1);
                SV_FTYPE cvali = svuzp2(input0, input1);

                // perform complex multiplication
                SV_FTYPE resultr = svmul_x(pg, cvalr, matr);
                SV_FTYPE resulti = svmul_x(pg, cvali, matr);

                resultr = svmsb_x(pg, cvali, mati, resultr);
                resulti = svmad_x(pg, cvalr, mati, resulti);

                // interleave elements from low halves of two vectors
                SV_FTYPE output0 = svzip1(resultr, resulti);
                SV_FTYPE output1 = svzip2(resultr, resulti);

                // set values
                svst1(pg, (ETYPE *)&state[state_index], output0);
                svst1(
                    pg, (ETYPE *)&state[state_index + (vec_len >> 1)], output1);
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
#ifdef _OPENMP
        UINT threshold = 12;
        UINT default_thread_count = omp_get_max_threads();
        if (dim < (((ITYPE)1) << threshold)) omp_set_num_threads(1);
#endif

        const MPIutil m = get_mpiutil();
        const int rank = m->get_rank();
        const int pair_rank_bit = 1 << (target_qubit_index - inner_qc);

        _single_qubit_diagonal_matrix_gate_mpi(
            diagonal_matrix, state, dim, (rank & pair_rank_bit) != 0);
#ifdef _OPENMP
        omp_set_num_threads(default_thread_count);
#endif
    }
}

void _single_qubit_diagonal_matrix_gate_mpi(
    const CTYPE diagonal_matrix[2], CTYPE *state, ITYPE dim, int isone) {
    // loop variables
    ITYPE state_index;
#if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
    ITYPE vec_len =
        getVecLength();  // length of SVE registers (# of 64-bit elements)

    if (dim >= vec_len) {
        SV_PRED pg = Svptrue();  // this predicate register is all 1.

        SV_FTYPE matr, mati;

        matr = SvdupF(creal(diagonal_matrix[isone]));
        mati = SvdupF(cimag(diagonal_matrix[isone]));

#pragma omp parallel for
        for (state_index = 0; state_index < dim; state_index += vec_len) {
            // fetch values
            SV_FTYPE input0 = svld1(pg, (ETYPE *)&state[state_index]);
            SV_FTYPE input1 =
                svld1(pg, (ETYPE *)&state[state_index + (vec_len >> 1)]);

            // select odd or even elements from two vectors
            SV_FTYPE cvalr = svuzp1(input0, input1);
            SV_FTYPE cvali = svuzp2(input0, input1);

            // perform complex multiplication
            SV_FTYPE resultr = svmul_x(pg, cvalr, matr);
            SV_FTYPE resulti = svmul_x(pg, cvali, matr);

            resultr = svmsb_x(pg, cvali, mati, resultr);
            resulti = svmad_x(pg, cvalr, mati, resulti);

            // interleave elements from low halves of two vectors
            SV_FTYPE output0 = svzip1(resultr, resulti);
            SV_FTYPE output1 = svzip2(resultr, resulti);

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
#endif  //#ifdef _USE_MPI
