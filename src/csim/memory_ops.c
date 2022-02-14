#include <stdio.h>
#include <stdlib.h>
#include "memory_ops.h"
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
    CTYPE* state = (CTYPE*)malloc((size_t)(sizeof(CTYPE)*dim));
	//CTYPE* state = (CTYPE*)_aligned_malloc((size_t)(sizeof(CTYPE)*dim), 32);

    if (!state){
        fprintf(stderr,"Out of memory\n");
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

}
#endif

