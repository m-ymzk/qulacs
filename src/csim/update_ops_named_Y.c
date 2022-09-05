#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "constant.h"
#include "update_ops.h"
#include "utility.h"
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

void Y_gate(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim, 13);
#endif

#if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
    Y_gate_sve(target_qubit_index, state, dim);
#elif defined(_USE_SIMD)
    Y_gate_simd(target_qubit_index, state, dim);
#else
    Y_gate_unroll(target_qubit_index, state, dim);
#endif

#ifdef _OPENMP
    omputil->reset_qulacs_num_threads();
#endif
}

void Y_gate_unroll(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    ITYPE state_index = 0;
    const CTYPE imag = 1.i;
    if (target_qubit_index == 0) {
        ITYPE basis_index;
#pragma omp parallel for
        for (basis_index = 0; basis_index < dim; basis_index += 2) {
            CTYPE temp0 = state[basis_index];
            state[basis_index] = -imag * state[basis_index + 1];
            state[basis_index + 1] = imag * temp0;
        }
    } else {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 =
                (state_index & mask_low) + ((state_index & mask_high) << 1);
            ITYPE basis_index_1 = basis_index_0 + mask;
            CTYPE temp0 = state[basis_index_0];
            CTYPE temp1 = state[basis_index_0 + 1];
            state[basis_index_0] = -imag * state[basis_index_1];
            state[basis_index_0 + 1] = -imag * state[basis_index_1 + 1];
            state[basis_index_1] = imag * temp0;
            state[basis_index_1 + 1] = imag * temp1;
        }
    }
}

#if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)
void Y_gate_sve(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    ITYPE state_index = 0;
    const CTYPE imag = 1.i;
    ITYPE vec_len = getVecLength();

    if (mask >= (vec_len >> 1)) {
        SV_PRED pg = Svptrue();
        SV_FTYPE minus_even = svzip1(SvdupF(1.0), SvdupF(-1.0));
        SV_FTYPE minus_odd = svzip1(SvdupF(-1.0), SvdupF(1.0));

#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim;
             state_index += (vec_len >> 1)) {
            SV_FTYPE input0, input1, output0, output1, cval_real, cval_imag;
            ITYPE basis_index_0 =
                (state_index & mask_low) + ((state_index & mask_high) << 1);
            ITYPE basis_index_1 = basis_index_0 + mask;

            input0 = svld1(pg, (ETYPE*)&state[basis_index_0]);
            input1 = svld1(pg, (ETYPE*)&state[basis_index_1]);

            cval_real = svuzp1(input0, input1);
            cval_imag = svuzp2(input0, input1);

            output0 = svzip1(cval_imag, cval_real);
            output1 = svzip2(cval_imag, cval_real);

            output0 = svmul_x(pg, output0, minus_odd);
            output1 = svmul_x(pg, output1, minus_even);

            if (5 <= target_qubit_index && target_qubit_index <= 10) {
                // L1 prefetch
                __builtin_prefetch(&state[basis_index_0 + mask * 4], 1, 3);
                __builtin_prefetch(&state[basis_index_1 + mask * 4], 1, 3);
                // L2 prefetch
                __builtin_prefetch(&state[basis_index_0 + mask * 8], 1, 2);
                __builtin_prefetch(&state[basis_index_1 + mask * 8], 1, 2);
            }

            svst1(pg, (ETYPE*)&state[basis_index_0], output1);
            svst1(pg, (ETYPE*)&state[basis_index_1], output0);
        }
    } else if (dim >= vec_len) {
        SV_PRED pg = Svptrue();
        SV_FTYPE minus_even = svzip1(SvdupF(1.0), SvdupF(-1.0));
        SV_FTYPE minus_odd = svzip1(SvdupF(-1.0), SvdupF(1.0));
        SV_FTYPE minus_half;
        SV_ITYPE vec_shuffle_table;

        minus_half = SvdupF(0.0);
        ITYPE len = 0;
        while (len < vec_len) {
            for (ITYPE i = 0; i < (1ULL << target_qubit_index); ++i)
                minus_half = svext(minus_half, minus_even, 2);
            len += (1ULL << (target_qubit_index + 1));

            for (ITYPE i = 0; i < (1ULL << target_qubit_index); ++i)
                minus_half = svext(minus_half, minus_odd, 2);
            len += (1ULL << (target_qubit_index + 1));
        }

        vec_shuffle_table = sveor_z(
            pg, SvindexI(0, 1), SvdupI(1ULL << (target_qubit_index + 1)));

#pragma omp parallel for
        for (state_index = 0; state_index < dim; state_index += vec_len) {
            SV_FTYPE input0, input1, output0, output1, cval_real, cval_imag,
                shuffle0, shuffle1;

            input0 = svld1(pg, (ETYPE*)&state[state_index]);
            input1 = svld1(pg, (ETYPE*)&state[state_index + (vec_len >> 1)]);

            // shuffle
            shuffle0 = svtbl(input0, vec_shuffle_table);
            shuffle1 = svtbl(input1, vec_shuffle_table);

            cval_real = svuzp1(shuffle0, shuffle1);
            cval_imag = svuzp2(shuffle0, shuffle1);

            output0 = svzip1(cval_imag, cval_real);
            output1 = svzip2(cval_imag, cval_real);

            shuffle0 = svmul_x(pg, output0, minus_half);
            shuffle1 = svmul_x(pg, output1, minus_half);

            svst1(pg, (ETYPE*)&state[state_index], shuffle0);
            svst1(pg, (ETYPE*)&state[state_index + (vec_len >> 1)], shuffle1);
        }
    } else {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 =
                (state_index & mask_low) + ((state_index & mask_high) << 1);
            ITYPE basis_index_1 = basis_index_0 + mask;
            CTYPE temp = state[basis_index_0];
            state[basis_index_0] = -imag * state[basis_index_1];
            state[basis_index_1] = imag * temp;
        }
    }
}
#endif  // #if defined(__ARM_FEATURE_SVE) && defined(_USE_SVE)

#ifdef _USE_SIMD
void Y_gate_simd(UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 2;
    const ITYPE mask = (1ULL << target_qubit_index);
    const ITYPE mask_low = mask - 1;
    const ITYPE mask_high = ~mask_low;
    ITYPE state_index = 0;
    // const CTYPE imag = 1.i;
    __m256d minus_even = _mm256_set_pd(1, -1, 1, -1);
    __m256d minus_odd = _mm256_set_pd(-1, 1, -1, 1);
    __m256d minus_half = _mm256_set_pd(1, -1, -1, 1);
    if (target_qubit_index == 0) {
        ITYPE basis_index = 0;
#pragma omp parallel for
        for (basis_index = 0; basis_index < dim; basis_index += 2) {
            double* ptr0 = (double*)(state + basis_index);
            __m256d data0 = _mm256_loadu_pd(ptr0);
            data0 = _mm256_permute4x64_pd(
                data0, 27);  // (3210) -> (0123) : 16+4*2+3=27
            data0 = _mm256_mul_pd(data0, minus_half);
            _mm256_storeu_pd(ptr0, data0);
        }
    } else {
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 =
                (state_index & mask_low) + ((state_index & mask_high) << 1);
            ITYPE basis_index_1 = basis_index_0 + mask;
            double* ptr0 = (double*)(state + basis_index_0);
            double* ptr1 = (double*)(state + basis_index_1);
            __m256d data0 = _mm256_loadu_pd(ptr0);
            __m256d data1 = _mm256_loadu_pd(ptr1);
            data0 = _mm256_permute_pd(data0, 5);  // (3210) -> (2301) : 4+1
            data1 = _mm256_permute_pd(data1, 5);
            data0 = _mm256_mul_pd(data0, minus_even);
            data1 = _mm256_mul_pd(data1, minus_odd);
            _mm256_storeu_pd(ptr1, data0);
            _mm256_storeu_pd(ptr0, data1);
        }
    }
}
#endif

#ifdef _USE_MPI
void Y_gate_mpi(
    UINT target_qubit_index, CTYPE* state, ITYPE dim, UINT inner_qc) {
    if (target_qubit_index < inner_qc) {
        Y_gate(target_qubit_index, state, dim);
    } else {
        const MPIutil m = get_mpiutil();
        const int rank = m->get_rank();
        ITYPE dim_work = dim;
        ITYPE num_work = 0;
        CTYPE* t = m->get_workarea(&dim_work, &num_work);
        assert(num_work > 0);
#ifdef _OPENMP
        OMPutil omputil = get_omputil();
        omputil->set_qulacs_num_threads(dim_work, 13);
#endif
        const int pair_rank_bit = 1 << (target_qubit_index - inner_qc);
        const int pair_rank = rank ^ pair_rank_bit;
        const CTYPE imag = 1.i;
        CTYPE* si = state;
        // printf("#debug dim,dim_work,num_work,t: %lld, %lld, %lld, %p\n", dim,
        // dim_work, num_work, t);
        for (ITYPE iter = 0; iter < num_work; ++iter) {
            m->m_DC_sendrecv(si, t, dim_work, pair_rank);
            ITYPE state_index = 0;
            if (rank & pair_rank_bit) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (state_index = 0; state_index < dim_work; ++state_index) {
                    si[state_index] = imag * t[state_index];
                }
            } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (state_index = 0; state_index < dim_work; ++state_index) {
                    si[state_index] = -imag * t[state_index];
                }
            }
            si += dim_work;
        }

#ifdef _OPENMP
        omputil->reset_qulacs_num_threads();
#endif
    }
}
#endif
