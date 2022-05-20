#include "memory_ops.h"

#include <stdio.h>
#include <stdlib.h>

#include "utility.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MSC_VER
#define aligned_free _aligned_free;
#define aligned_malloc _aligned_malloc;
#endif

#if defined(__ARM_FEATURE_SVE)
#include "arm_acle.h"
#include "arm_sve.h"
#endif

// memory allocation
CTYPE* allocate_quantum_state(ITYPE dim) {
    CTYPE* state = (CTYPE*)malloc((size_t)(sizeof(CTYPE) * dim));
    // CTYPE* state = (CTYPE*)_aligned_malloc((size_t)(sizeof(CTYPE)*dim), 32);

    if (!state) {
        fprintf(stderr, "Out of memory\n");
        fflush(stderr);
        exit(1);
    }
    return state;
}

void release_quantum_state(CTYPE* state) {
    free(state);
    //_aligned_free(state);
}

#if defined(__ARM_FEATURE_SVE)
typedef svfloat64_t SV_FTYPE
    __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));

#define ZFILL_DISTANCE 14
#define PRF_DISTANCE 5
#define CACHE_LINE_SIZE 256
// TODO: It targets CPU architectures that implement SVE for vector
//       registers with a size of 512 bits. When executing with a CPU
//       architecture that implements SVE of vector registers of other sizes, it
//       is necessary to take measures.
void memcpy_sve(double* Out, double* In, ITYPE Num) {
    ITYPE i;

    UINT threshold = 256;
    if (Num * sizeof(double) >= threshold * 1024) {
        ITYPE vec_len = svcntd();

#ifdef _OPENMP
        const UINT thread_count = omp_get_max_threads();
        const ITYPE block_size = Num / thread_count;
        const ITYPE residual = Num % thread_count;
#endif

#pragma omp parallel private(i)
        {
            ITYPE sub_num;
            ITYPE iteration_count;
#ifdef _OPENMP
            UINT thread_id = omp_get_thread_num();
            ITYPE start_index = block_size * thread_id +
                                (residual > thread_id ? thread_id : residual);
            ITYPE end_index =
                block_size * (thread_id + 1) +
                (residual > (thread_id + 1) ? (thread_id + 1) : residual);
            // end_index = (end_index < 0) ? 0 : end_index;
#else
            ITYPE start_index = 0;
            ITYPE end_index = Num;
#endif

            // If the number of revolutions divided by the increment value is
            // not divisible, the closing price of the loop is decremented by
            // one revolution.
            sub_num =
                (end_index < (vec_len << 2)) ? 0 : (end_index - (vec_len << 2));
            iteration_count =
                (end_index < start_index) ? 0 : (end_index - start_index);
            ITYPE end_kernel =
                (iteration_count % (vec_len << 2) != 0) ? sub_num : end_index;

            // Finish one cache line size earlier in consideration of alignment
            // zfill distance + 1 cache line size
            sub_num = (vec_len << 2) * ZFILL_DISTANCE +
                      (CACHE_LINE_SIZE / sizeof(double));
            end_kernel = (end_kernel < sub_num) ? 0 : end_kernel - sub_num;

            for (i = start_index; i < end_kernel; i += (vec_len << 2)) {
                SV_FTYPE val0, val1, val2, val3;
                __asm__("dc ZVA, %0\n\t"
                        :
                        : "r"(&Out[i + (vec_len << 2) * ZFILL_DISTANCE])
                        : "memory");
                val0 = svld1(svptrue_b64(), &In[i]);
                val1 = svld1(svptrue_b64(), &In[i + vec_len]);
                val2 = svld1(svptrue_b64(), &In[i + vec_len * 2]);
                val3 = svld1(svptrue_b64(), &In[i + vec_len * 3]);
                svst1(svptrue_b64(), &Out[i], val0);
                svst1(svptrue_b64(), &Out[i + vec_len], val1);
                svst1(svptrue_b64(), &Out[i + vec_len * 2], val2);
                svst1(svptrue_b64(), &Out[i + vec_len * 3], val3);
                __builtin_prefetch(
                    &Out[i + (vec_len << 2) * PRF_DISTANCE], 1, 3);
            }

            // If the number of revolutions divided by the increment value is
            // not divisible, the closing price of the loop is decremented by
            // one revolution.
            sub_num = (end_index < vec_len) ? 0 : (end_index - vec_len);
            iteration_count = (end_index < i) ? 0 : (end_index - i);
            ITYPE end_mod =
                (iteration_count % vec_len) != 0 ? sub_num : end_index;

            // Remainder loop of zero fill loop
            // This loop always issues a store instruction to an address
            // with embedded zeros.
            for (; i < end_mod; i += vec_len) {
                SV_FTYPE val;
                val = svld1(svptrue_b64(), &In[i]);
                svst1(svptrue_b64(), &Out[i], val);
            }

            // Remainder loop of sve loop (ACLE)
            for (; i < end_index; i++) {
                Out[i] = In[i];
            }
        }
    } else {
        for (i = 0; i < Num; i++) {
            Out[i] = In[i];
        }
    }
}
#endif
