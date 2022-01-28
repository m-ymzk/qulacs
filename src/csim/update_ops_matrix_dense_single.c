
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
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
#if defined(__ARM_FEATURE_SVE)
#include "arm_sve.h"
#include "arm_acle.h"
#endif // #if defined(__ARM_FEATURE_SVE)

//void single_qubit_dense_matrix_gate_old_single(UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim);
//void single_qubit_dense_matrix_gate_old_parallel(UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim);
//void single_qubit_dense_matrix_gate_single(UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim);
//void single_qubit_dense_matrix_gate_parallel(UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim);

void single_qubit_dense_matrix_gate(UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
	//single_qubit_dense_matrix_gate_old_single(target_qubit_index, matrix, state, dim);
	//single_qubit_dense_matrix_gate_old_parallel(target_qubit_index, matrix, state, dim);
	//single_qubit_dense_matrix_gate_single(target_qubit_index, matrix, state, dim);
	//single_qubit_dense_matrix_gate_single_unroll(target_qubit_index, matrix, state, dim);
	//single_qubit_dense_matrix_gate_single_simd(target_qubit_index, matrix, state, dim);
	//single_qubit_dense_matrix_gate_parallel_simd(target_qubit_index, matrix, state, dim);
	//return;

#ifdef _USE_SIMD
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
#if defined(__ARM_FEATURE_SVE)
	  single_qubit_dense_matrix_gate_single_sve(target_qubit_index, matrix, state, dim);
#else // #if defined(__ARM_FEATURE_SVE)
	  single_qubit_dense_matrix_gate_single_simd(target_qubit_index, matrix, state, dim);
#endif // #if defined(__ARM_FEATURE_SVE)
	}
	else {
		single_qubit_dense_matrix_gate_parallel_simd(target_qubit_index, matrix, state, dim);
	}
#else
#if defined(__ARM_FEATURE_SVE)
	single_qubit_dense_matrix_gate_single_sve(target_qubit_index, matrix, state, dim);
#else // #if defined(__ARM_FEATURE_SVE)
	single_qubit_dense_matrix_gate_single_simd(target_qubit_index, matrix, state, dim);
#endif // #if defined(__ARM_FEATURE_SVE)
#endif
#else
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
#if defined(__ARM_FEATURE_SVE)
	  single_qubit_dense_matrix_gate_single_sve(target_qubit_index, matrix, state, dim);
#else // #if defined(__ARM_FEATURE_SVE)
	  //single_qubit_dense_matrix_gate_single_unroll(target_qubit_index, matrix, state, dim);
	  single_qubit_dense_matrix_gate_single(target_qubit_index, matrix, state, dim);
#endif // #if defined(__ARM_FEATURE_SVE)
	}
	else {
		//single_qubit_dense_matrix_gate_parallel_unroll(target_qubit_index, matrix, state, dim);
		single_qubit_dense_matrix_gate_parallel(target_qubit_index, matrix, state, dim);
	}
#else

#if defined(__ARM_FEATURE_SVE)
	single_qubit_dense_matrix_gate_single_sve(target_qubit_index, matrix, state, dim);
#else // #if defined(__ARM_FEATURE_SVE)
	//single_qubit_dense_matrix_gate_single_unroll(target_qubit_index, matrix, state, dim);
	single_qubit_dense_matrix_gate_single(target_qubit_index, matrix, state, dim);
#endif // #if defined(__ARM_FEATURE_SVE)

#endif
#endif
}



void single_qubit_dense_matrix_gate_single(UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;

	ITYPE state_index = 0;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
		ITYPE basis_1 = basis_0 + mask;

		// fetch values
		CTYPE cval_0 = state[basis_0];
		CTYPE cval_1 = state[basis_1];

		// set values
		state[basis_0] = matrix[0] * cval_0 + matrix[1] * cval_1;
		state[basis_1] = matrix[2] * cval_0 + matrix[3] * cval_1;
	}
}

#if defined(__ARM_FEATURE_SVE)
void single_qubit_dense_matrix_gate_single_sve(UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;

  ITYPE state_index = 0;
  int vec_len = svcntw();       // length of SVE registers

  if( (loop_dim % (vec_len>>1)) == 0){
    svbool_t pg = svptrue_b64();  // this predicate register is all 1.
    
    // duplicate scalar masks
    svint64_t vec_mask = svdup_s64(mask);
    svint64_t vec_mask_low = svdup_s64(mask_low);
    svint64_t vec_mask_high = svdup_s64(mask_high);
  
    // Prepare a vector of state_index 
    svint64_t vec_index = svindex_s64(0, 1);
    vec_index = svlsl_n_s64_x(pg, vec_index, 1);
    svint64_t basis_0, basis_1;
  
    // offset for imag elements
    svint64_t vec_img_offset = svzip1(svdup_s64(0), svdup_s64(8));
  
    // SVE registers for matrix-vector products
    svfloat64_t input0, input1, output0, output1;
    svfloat64_t cal01_real, cal01_imag, cal10_real, cal10_imag;
    svfloat64_t result01_real, result01_imag;
    svfloat64_t mat02_real, mat02_imag, mat13_real, mat13_imag;
  
    // load matrix elements
    mat02_real = svuzp1_f64(svdup_f64(creal(matrix[0])), svdup_f64(creal(matrix[2])));
    mat02_imag = svuzp1_f64(svdup_f64(cimag(matrix[0])), svdup_f64(cimag(matrix[2])));
    mat13_real = svuzp1_f64(svdup_f64(creal(matrix[1])), svdup_f64(creal(matrix[3])));
    mat13_imag = svuzp1_f64(svdup_f64(cimag(matrix[1])), svdup_f64(cimag(matrix[3])));
  
  	for (state_index = 0; state_index < loop_dim; state_index += (vec_len>>1)) {
      // calculate indices
      basis_0 = svand_s64_x(pg, vec_index, vec_mask_high);
      basis_0 = svadd_s64_x(pg, 
                            svand_s64_x(pg, vec_index, vec_mask_low),
                            svlsl_n_s64_x(pg, basis_0, 1));
      basis_0 = svadd_s64_x(pg, basis_0, vec_img_offset);
  
      basis_1 = svand_s64_x(pg, basis_0, vec_mask);
  
      // get byte offset (index * 8 Bytes)
      basis_0 = svmul_s64_x(pg, basis_0, svdup_s64(8));
      basis_1 = svmul_s64_x(pg, basis_1, svdup_s64(8));
  
  		// fetch values
  		input0 = svld1_gather_s64offset_f64(pg, (double*)state, basis_0);
  		input1 = svld1_gather_s64offset_f64(pg, (double*)state, basis_1);
  
      // select odd or even elements from two vectors
      cal01_real = svuzp1_f64(input0, input1);
      cal01_imag = svuzp2_f64(input0, input1);
      cal10_real = svuzp1_f64(input1, input0);
      cal10_imag = svuzp2_f64(input1, input0);
  
      // perform matrix-vector product
      result01_real = svmul_f64_x(pg, cal01_real, mat02_real);
      result01_imag = svmul_f64_x(pg, cal01_imag, mat02_real);
      result01_real = svmsb_f64_x(pg, cal01_imag, mat02_imag, result01_real);
      result01_imag = svmad_f64_x(pg, cal01_real, mat02_imag, result01_imag);
      result01_real = svmad_f64_x(pg, cal10_real, mat13_real, result01_real);
      result01_imag = svmad_f64_x(pg, cal10_real, mat13_imag, result01_imag);
      result01_real = svmsb_f64_x(pg, cal10_imag, mat13_imag, result01_real);
      result01_imag = svmad_f64_x(pg, cal10_imag, mat13_real, result01_imag);
  
      // interleave elements from low halves of two vectors
      output0 = svzip1_f64(result01_real, result01_imag);
      output1 = svzip2_f64(result01_real, result01_imag);
  
  		// set values
  		svst1_scatter_s64offset_f64(pg, (double*)state, basis_0, output0);
  		svst1_scatter_s64offset_f64(pg, (double*)state, basis_1, output1);
  
      // increment vec_index
      vec_index = svadd_s64_x(pg, vec_index, svdup_s64(vec_len>>1));
  	}
  }else{
    for (state_index = 0; state_index < loop_dim; ++state_index) {
      ITYPE basis_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
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
#endif // #if defined(__ARM_FEATURE_SVE)

#ifdef _OPENMP
void single_qubit_dense_matrix_gate_parallel(UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;

	ITYPE state_index = 0;
#pragma omp parallel for
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
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


void single_qubit_dense_matrix_gate_single_unroll(UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;

	if (target_qubit_index == 0) {
		ITYPE basis = 0;
		for (basis = 0; basis < dim; basis+=2) {
			CTYPE val0a = state[basis];
			CTYPE val1a = state[basis + 1];
			CTYPE res0a = val0a * matrix[0] + val1a * matrix[1];
			CTYPE res1a = val0a * matrix[2] + val1a * matrix[3];
			state[basis] = res0a;
			state[basis + 1] = res1a;
		}
	}
	else {
		ITYPE state_index = 0;
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
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
void single_qubit_dense_matrix_gate_parallel_unroll(UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
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
	}
	else {
		ITYPE state_index = 0;
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
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

#ifdef _USE_SIMD
void single_qubit_dense_matrix_gate_single_simd(UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;

	if (target_qubit_index == 0) {
		ITYPE basis = 0;
		__m256d mv00 = _mm256_set_pd(-cimag(matrix[1]), creal(matrix[1]), -cimag(matrix[0]), creal(matrix[0]));
		__m256d mv01 = _mm256_set_pd(creal(matrix[1]), cimag(matrix[1]), creal(matrix[0]), cimag(matrix[0]));
		__m256d mv20 = _mm256_set_pd(-cimag(matrix[3]), creal(matrix[3]), -cimag(matrix[2]), creal(matrix[2]));
		__m256d mv21 = _mm256_set_pd(creal(matrix[3]), cimag(matrix[3]), creal(matrix[2]), cimag(matrix[2]));
		for (basis = 0; basis < dim; basis += 2) {
			double* ptr = (double*)(state + basis);
			__m256d data = _mm256_loadu_pd(ptr);

			__m256d data_u0 = _mm256_mul_pd(data, mv00);
			__m256d data_u1 = _mm256_mul_pd(data, mv01);
			__m256d data_u2 = _mm256_hadd_pd(data_u0, data_u1);
			data_u2 = _mm256_permute4x64_pd(data_u2, 216); // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216

			__m256d data_d0 = _mm256_mul_pd(data, mv20);
			__m256d data_d1 = _mm256_mul_pd(data, mv21);
			__m256d data_d2 = _mm256_hadd_pd(data_d0, data_d1);
			data_d2 = _mm256_permute4x64_pd(data_d2, 216); // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216

			__m256d data_r = _mm256_hadd_pd(data_u2, data_d2);

			data_r = _mm256_permute4x64_pd(data_r, 216); // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216
			_mm256_storeu_pd(ptr, data_r);
		}
	}
	else {
		ITYPE state_index = 0;
		__m256d mv00 = _mm256_set_pd(-cimag(matrix[0]), creal(matrix[0]), -cimag(matrix[0]), creal(matrix[0]));
		__m256d mv01 = _mm256_set_pd(creal(matrix[0]), cimag(matrix[0]), creal(matrix[0]), cimag(matrix[0]));
		__m256d mv10 = _mm256_set_pd(-cimag(matrix[1]), creal(matrix[1]), -cimag(matrix[1]), creal(matrix[1]));
		__m256d mv11 = _mm256_set_pd(creal(matrix[1]), cimag(matrix[1]), creal(matrix[1]), cimag(matrix[1]));
		__m256d mv20 = _mm256_set_pd(-cimag(matrix[2]), creal(matrix[2]), -cimag(matrix[2]), creal(matrix[2]));
		__m256d mv21 = _mm256_set_pd(creal(matrix[2]), cimag(matrix[2]), creal(matrix[2]), cimag(matrix[2]));
		__m256d mv30 = _mm256_set_pd(-cimag(matrix[3]), creal(matrix[3]), -cimag(matrix[3]), creal(matrix[3]));
		__m256d mv31 = _mm256_set_pd(creal(matrix[3]), cimag(matrix[3]), creal(matrix[3]), cimag(matrix[3]));
		for (state_index = 0; state_index < loop_dim; state_index+=2) {
			ITYPE basis_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
			ITYPE basis_1 = basis_0 + mask;
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

#ifdef _OPENMP

void single_qubit_dense_matrix_gate_parallel_simd(UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;

	if (target_qubit_index == 0) {
		ITYPE basis = 0;
		__m256d mv00 = _mm256_set_pd(-cimag(matrix[1]), creal(matrix[1]), -cimag(matrix[0]), creal(matrix[0]));
		__m256d mv01 = _mm256_set_pd(creal(matrix[1]), cimag(matrix[1]), creal(matrix[0]), cimag(matrix[0]));
		__m256d mv20 = _mm256_set_pd(-cimag(matrix[3]), creal(matrix[3]), -cimag(matrix[2]), creal(matrix[2]));
		__m256d mv21 = _mm256_set_pd(creal(matrix[3]), cimag(matrix[3]), creal(matrix[2]), cimag(matrix[2]));
#pragma omp parallel for
		for (basis = 0; basis < dim; basis += 2) {
			double* ptr = (double*)(state + basis);
			__m256d data = _mm256_loadu_pd(ptr);

			__m256d data_u0 = _mm256_mul_pd(data, mv00);
			__m256d data_u1 = _mm256_mul_pd(data, mv01);
			__m256d data_u2 = _mm256_hadd_pd(data_u0, data_u1);
			data_u2 = _mm256_permute4x64_pd(data_u2, 216); // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216

			__m256d data_d0 = _mm256_mul_pd(data, mv20);
			__m256d data_d1 = _mm256_mul_pd(data, mv21);
			__m256d data_d2 = _mm256_hadd_pd(data_d0, data_d1);
			data_d2 = _mm256_permute4x64_pd(data_d2, 216); // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216

			__m256d data_r = _mm256_hadd_pd(data_u2, data_d2);

			data_r = _mm256_permute4x64_pd(data_r, 216); // (3210) -> (3120) : 1*0 + 4*2 + 16*1 + 64*3 = 216
			_mm256_storeu_pd(ptr, data_r);
		}
	}
	else {
		ITYPE state_index = 0;
		__m256d mv00 = _mm256_set_pd(-cimag(matrix[0]), creal(matrix[0]), -cimag(matrix[0]), creal(matrix[0]));
		__m256d mv01 = _mm256_set_pd(creal(matrix[0]), cimag(matrix[0]), creal(matrix[0]), cimag(matrix[0]));
		__m256d mv10 = _mm256_set_pd(-cimag(matrix[1]), creal(matrix[1]), -cimag(matrix[1]), creal(matrix[1]));
		__m256d mv11 = _mm256_set_pd(creal(matrix[1]), cimag(matrix[1]), creal(matrix[1]), cimag(matrix[1]));
		__m256d mv20 = _mm256_set_pd(-cimag(matrix[2]), creal(matrix[2]), -cimag(matrix[2]), creal(matrix[2]));
		__m256d mv21 = _mm256_set_pd(creal(matrix[2]), cimag(matrix[2]), creal(matrix[2]), cimag(matrix[2]));
		__m256d mv30 = _mm256_set_pd(-cimag(matrix[3]), creal(matrix[3]), -cimag(matrix[3]), creal(matrix[3]));
		__m256d mv31 = _mm256_set_pd(creal(matrix[3]), cimag(matrix[3]), creal(matrix[3]), cimag(matrix[3]));
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
			ITYPE basis_1 = basis_0 + mask;
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
#endif

#ifdef _USE_MPI
void single_qubit_dense_matrix_gate_mpi(UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim, UINT inner_qc) {
    if (target_qubit_index < inner_qc){
		single_qubit_dense_matrix_gate(target_qubit_index, matrix, state, dim);
    }
	else {
        const MPIutil m = get_mpiutil();
        const int rank = m->get_rank();
        ITYPE dim_work = dim;
        ITYPE num_work = 0;
        CTYPE* t = m->get_workarea(&dim_work, &num_work);
        assert(num_work > 0);
        const int pair_rank_bit = 1 << (target_qubit_index - inner_qc);
        const int pair_rank = rank ^ pair_rank_bit;
		CTYPE* si = state;
		for (ITYPE iter=0; iter < num_work; ++iter) {
            m->m_DC_sendrecv(si, t, dim_work, pair_rank);
#ifdef _OPENMP
			UINT threshold = 13;
			if (dim < (((ITYPE)1) << threshold)) {
				single_qubit_dense_matrix_gate_single_mpi(t, matrix, si, dim_work, rank & pair_rank_bit);
			}
			else {
				single_qubit_dense_matrix_gate_parallel_mpi(t, matrix, si, dim_work, rank & pair_rank_bit);
#pragma omp barrier
			}
#else
			single_qubit_dense_matrix_gate_single_mpi(t, matrix, si, dim_work, rank & pair_rank_bit);
#endif
			si += dim_work;
		}
	}
}

void single_qubit_dense_matrix_gate_single_mpi(CTYPE *t, const CTYPE matrix[4], CTYPE *state, ITYPE dim, int flag) {
	for (ITYPE state_index = 0; state_index < dim; ++state_index) {
		if (flag){ //val=1
			// fetch values
			CTYPE cval_0 = t[state_index];
			CTYPE cval_1 = state[state_index];

			// set values
			state[state_index] = matrix[2] * cval_0 + matrix[3] * cval_1;
		}
		else { //val=0
			// fetch values
			CTYPE cval_0 = state[state_index];
			CTYPE cval_1 = t[state_index];

			// set values
			state[state_index] = matrix[0] * cval_0 + matrix[1] * cval_1;
		}
	}
}

#ifdef _OPENMP
void single_qubit_dense_matrix_gate_parallel_mpi(CTYPE *t, const CTYPE matrix[4], CTYPE *state, ITYPE dim, int flag) {
#pragma omp parallel for
	for (ITYPE state_index = 0; state_index < dim; ++state_index) {
		if (flag){ //val=1
			// fetch values
			CTYPE cval_0 = t[state_index];
			CTYPE cval_1 = state[state_index];

			// set values
			state[state_index] = matrix[2] * cval_0 + matrix[3] * cval_1;
		}
		else { //val=0
			// fetch values
			CTYPE cval_0 = state[state_index];
			CTYPE cval_1 = t[state_index];

			// set values
			state[state_index] = matrix[0] * cval_0 + matrix[1] * cval_1;
		}
	}
}
#endif
#endif //#ifdef _USE_MPI

/*

void single_qubit_dense_matrix_gate_old_single(UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {

	// target mask
	const ITYPE target_mask = 1ULL << target_qubit_index;

	// loop variables
	const ITYPE loop_dim = dim / 2;
	ITYPE state_index;

	for (state_index = 0; state_index < loop_dim; ++state_index) {
		// create index
		ITYPE basis_0 = insert_zero_to_basis_index(state_index, target_mask, target_qubit_index);

		// gather index
		ITYPE basis_1 = basis_0 ^ target_mask;

		// fetch values
		CTYPE cval_0 = state[basis_0];
		CTYPE cval_1 = state[basis_1];

		// set values
		state[basis_0] = matrix[0] * cval_0 + matrix[1] * cval_1;
		state[basis_1] = matrix[2] * cval_0 + matrix[3] * cval_1;
	}
}

#ifdef _OPENMP
void single_qubit_dense_matrix_gate_old_parallel(UINT target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim) {

	// target mask
	const ITYPE target_mask = 1ULL << target_qubit_index;

	// loop variables
	const ITYPE loop_dim = dim / 2;
	ITYPE state_index;

#pragma omp parallel for
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		// create index
		ITYPE basis_0 = insert_zero_to_basis_index(state_index, target_mask, target_qubit_index);

		// gather index
		ITYPE basis_1 = basis_0 ^ target_mask;

		// fetch values
		CTYPE cval_0 = state[basis_0];
		CTYPE cval_1 = state[basis_1];

		// set values
		state[basis_0] = matrix[0] * cval_0 + matrix[1] * cval_1;
		state[basis_1] = matrix[2] * cval_0 + matrix[3] * cval_1;
	}
}
#endif
*/
