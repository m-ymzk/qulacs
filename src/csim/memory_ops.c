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
typedef svfloat64_t SV_FTYPE __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));

void memcpy_sve(double* Out, double* In, ITYPE Num){
  ITYPE i;
  ITYPE vec_len = svcntd();

#if 0
  if((Num >= vec_len) && ((Num % vec_len) == 0)){
    SV_FTYPE val;

#pragma omp parallel for private(val)
    for(i = 0; i < Num; i+=vec_len){
      val = svld1(svptrue_b64(), &In[i]);
      svst1(svptrue_b64(), &Out[i], val);
    }

  }else{
#pragma omp parallel for
    for(i = 0; i < Num; i++){
      Out[i] = In[i];
    }
  }
#else

#define ZFILL_DISTANCE 2
#define PRF_DISTANCE ZFILL_DISTANCE
#define CACHE_LINE_SIZE 256

  if((Num >= (CACHE_LINE_SIZE/sizeof(double))) && ((Num % vec_len) == 0)){

		// 
    ITYPE aligned_start_index = 0;
    for(i = 0; i < Num; i++){
      if (((size_t)(&Out[i]) & (CACHE_LINE_SIZE-1)) == 0){
        aligned_start_index = i;
        break;
      }
      Out[i] = In[i];
    }

    const UINT thread_count = omp_get_max_threads();
    const ITYPE remain_num = Num - aligned_start_index;
		const ITYPE num_cachelines = remain_num / (CACHE_LINE_SIZE/sizeof(double));
		const ITYPE block_size = (num_cachelines + thread_count - 1) / thread_count;
    const ITYPE residual = remain_num % (CACHE_LINE_SIZE/sizeof(double));

    if (num_cachelines){

#pragma omp parallel private(i)
      {
        UINT thread_id = omp_get_thread_num();
				ITYPE start_idx = thread_id * block_size * (CACHE_LINE_SIZE/sizeof(double)) + aligned_start_index;
				ITYPE end_idx = (thread_id + 1) * block_size * (CACHE_LINE_SIZE/sizeof(double)) + aligned_start_index;
				end_idx = (end_idx > (num_cachelines * (CACHE_LINE_SIZE/sizeof(double)) + aligned_start_index)) 
										? num_cachelines * (CACHE_LINE_SIZE/sizeof(double)) + aligned_start_index : end_idx;

        // Finish one cache line size earlier in consideration of alignment
        // zfill distance + 1 cache line size
        for(i = start_idx; i < end_idx; i += (vec_len << 2)){
  				SV_FTYPE val0, val1, val2, val3;
  				//__asm__ __volatile__("dc ZVA, %0\n\t" : : "r" (&Out[i + (vec_len << 2) * ZFILL_DISTANCE]) :"memory");
  				__asm__ __volatile__("dc ZVA, %0\n\t" : : "r" (&Out[i]) :"memory");
  				__builtin_prefetch(&Out[i], 1, 3);
  				val0 = svld1(svptrue_b64(), &In[i]);
  				val1 = svld1(svptrue_b64(), &In[i + vec_len]);
  				val2 = svld1(svptrue_b64(), &In[i + vec_len * 2]);
  				val3 = svld1(svptrue_b64(), &In[i + vec_len * 3]);
  				svst1(svptrue_b64(), &Out[i], val0);
  				svst1(svptrue_b64(), &Out[i + vec_len], val1);
  				svst1(svptrue_b64(), &Out[i + vec_len * 2], val2);
  				svst1(svptrue_b64(), &Out[i + vec_len * 3], val3);
  				//__builtin_prefetch(&Out[i + (vec_len << 2) * PRF_DISTANCE], 1, 3);
        } 
      }
    }

		if (residual) {
      for(i = residual; i > 0; i--){
        Out[Num - i] = In[Num - i];
      }
		}
  }else{
#pragma omp parallel for private(i)
    for(i = 0; i < Num; i++){
      Out[i] = In[i];
    }
  }


#endif

}
#endif
