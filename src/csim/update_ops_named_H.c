#include <stdio.h>
#include <assert.h>
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
#include <iostream>
#else
#include <x86intrin.h>
#endif
#endif

//void H_gate_old_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim);
//void H_gate_single(UINT target_qubit_index, CTYPE *state, ITYPE dim);
//void H_gate_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim);

void H_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	//H_gate_old_single(target_qubit_index, state, dim);
	//H_gate_old_parallel(target_qubit_index, state, dim);
	//H_gate_single(target_qubit_index, state, dim);
	//H_gate_single_simd(target_qubit_index, state, dim);
	//H_gate_single_unroll(target_qubit_index, state, dim);
	//H_gate_parallel(target_qubit_index, state, dim);
	//return;

#ifdef _USE_SIMD
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
		H_gate_single_simd(target_qubit_index, state, dim);
	}
	else {
		H_gate_parallel_simd(target_qubit_index, state, dim);
	}
#else
	H_gate_single_simd(target_qubit_index, state, dim);
#endif
#else
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
		H_gate_single_unroll(target_qubit_index, state, dim);
	}
	else {
		H_gate_parallel_unroll(target_qubit_index, state, dim);
	}
#else
	H_gate_single_unroll(target_qubit_index, state, dim);
#endif
#endif
}


void H_gate_single_unroll(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	const double sqrt2inv = 1. / sqrt(2.);
	ITYPE state_index = 0;
	if (target_qubit_index == 0) {
		ITYPE basis_index;
		for (basis_index = 0; basis_index < dim; basis_index += 2) {
			CTYPE temp0 = state[basis_index];
			CTYPE temp1 = state[basis_index + 1];
			state[basis_index] = (temp0 + temp1)*sqrt2inv;
			state[basis_index + 1] = (temp0 - temp1)*sqrt2inv;
		}
	}
	else {
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
			ITYPE basis_index_1 = basis_index_0 + mask;
			CTYPE temp_a0 = state[basis_index_0];
			CTYPE temp_a1 = state[basis_index_1];
			CTYPE temp_b0 = state[basis_index_0 + 1];
			CTYPE temp_b1 = state[basis_index_1 + 1];
			state[basis_index_0] = (temp_a0 + temp_a1)*sqrt2inv;
			state[basis_index_1] = (temp_a0 - temp_a1)*sqrt2inv;
			state[basis_index_0 + 1] = (temp_b0 + temp_b1)*sqrt2inv;
			state[basis_index_1 + 1] = (temp_b0 - temp_b1)*sqrt2inv;
		}
	}
}

#ifdef _OPENMP
void H_gate_parallel_unroll(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	const double sqrt2inv = 1. / sqrt(2.);
	ITYPE state_index = 0;
	if (target_qubit_index == 0) {
		ITYPE basis_index;
#pragma omp parallel for
		for (basis_index = 0; basis_index < dim; basis_index += 2) {
			CTYPE temp0 = state[basis_index];
			CTYPE temp1 = state[basis_index + 1];
			state[basis_index] = (temp0 + temp1)*sqrt2inv;
			state[basis_index + 1] = (temp0 - temp1)*sqrt2inv;
		}
	}
	else {
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
			ITYPE basis_index_1 = basis_index_0 + mask;
			CTYPE temp_a0 = state[basis_index_0];
			CTYPE temp_a1 = state[basis_index_1];
			CTYPE temp_b0 = state[basis_index_0 + 1];
			CTYPE temp_b1 = state[basis_index_1 + 1];
			state[basis_index_0] = (temp_a0 + temp_a1)*sqrt2inv;
			state[basis_index_1] = (temp_a0 - temp_a1)*sqrt2inv;
			state[basis_index_0 + 1] = (temp_b0 + temp_b1)*sqrt2inv;
			state[basis_index_1 + 1] = (temp_b0 - temp_b1)*sqrt2inv;
		}
	}
}
#endif

#ifdef _USE_SIMD
void H_gate_single_simd(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	//const CTYPE imag = 1.i;
	const double sqrt2inv = 1. / sqrt(2.);
	__m256d sqrt2inv_array = _mm256_set_pd(sqrt2inv, sqrt2inv, sqrt2inv, sqrt2inv);
	if (target_qubit_index == 0) {
		//__m256d sqrt2inv_array_half = _mm256_set_pd(sqrt2inv, sqrt2inv, -sqrt2inv, -sqrt2inv);
		ITYPE basis_index = 0;
		for (basis_index = 0; basis_index < dim; basis_index += 2) {
			double* ptr0 = (double*)(state + basis_index);
			__m256d data0 = _mm256_loadu_pd(ptr0);
			__m256d data1 = _mm256_permute4x64_pd(data0, 78); // (3210) -> (1032) : 1*2 + 4*3 + 16*0 + 64*1 = 2+12+64=78
			__m256d data2 = _mm256_add_pd(data0, data1);
			__m256d data3 = _mm256_sub_pd(data1, data0);
			__m256d data4 = _mm256_blend_pd(data3, data2, 3); // take data3 for latter half
			data4 = _mm256_mul_pd(data4, sqrt2inv_array);
			_mm256_storeu_pd(ptr0, data4);
		}
	}
	else {
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
			ITYPE basis_index_1 = basis_index_0 + mask;
			double* ptr0 = (double*)(state + basis_index_0);
			double* ptr1 = (double*)(state + basis_index_1);
			__m256d data0 = _mm256_loadu_pd(ptr0);
			__m256d data1 = _mm256_loadu_pd(ptr1);
			__m256d data2 = _mm256_add_pd(data0, data1);
			__m256d data3 = _mm256_sub_pd(data0, data1);
			data2 = _mm256_mul_pd(data2, sqrt2inv_array);
			data3 = _mm256_mul_pd(data3, sqrt2inv_array);
			_mm256_storeu_pd(ptr0, data2);
			_mm256_storeu_pd(ptr1, data3);
		}
	}
}

#ifdef _OPENMP

void H_gate_parallel_simd(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	//const CTYPE imag = 1.i;
	const double sqrt2inv = 1. / sqrt(2.);
	__m256d sqrt2inv_array = _mm256_set_pd(sqrt2inv, sqrt2inv, sqrt2inv, sqrt2inv);
	if (target_qubit_index == 0) {
		//__m256d sqrt2inv_array_half = _mm256_set_pd(sqrt2inv, sqrt2inv, -sqrt2inv, -sqrt2inv);
		ITYPE basis_index = 0;
#pragma omp parallel for
		for (basis_index = 0; basis_index < dim; basis_index += 2) {
			double* ptr0 = (double*)(state + basis_index);
			__m256d data0 = _mm256_loadu_pd(ptr0);
			__m256d data1 = _mm256_permute4x64_pd(data0, 78); // (3210) -> (1032) : 1*2 + 4*3 + 16*0 + 64*1 = 2+12+64=78
			__m256d data2 = _mm256_add_pd(data0, data1);
			__m256d data3 = _mm256_sub_pd(data1, data0);
			__m256d data4 = _mm256_blend_pd(data3, data2, 3); // take data3 for latter half
			data4 = _mm256_mul_pd(data4, sqrt2inv_array);
			_mm256_storeu_pd(ptr0, data4);
		}
	}
	else {
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
			ITYPE basis_index_1 = basis_index_0 + mask;
			double* ptr0 = (double*)(state + basis_index_0);
			double* ptr1 = (double*)(state + basis_index_1);
			__m256d data0 = _mm256_loadu_pd(ptr0);
			__m256d data1 = _mm256_loadu_pd(ptr1);
			__m256d data2 = _mm256_add_pd(data0, data1);
			__m256d data3 = _mm256_sub_pd(data0, data1);
			data2 = _mm256_mul_pd(data2, sqrt2inv_array);
			data3 = _mm256_mul_pd(data3, sqrt2inv_array);
			_mm256_storeu_pd(ptr0, data2);
			_mm256_storeu_pd(ptr1, data3);
		}
	}
}
#endif
#endif

#ifdef _USE_MPI
void H_gate_mpi(UINT target_qubit_index, CTYPE *state, ITYPE dim, UINT inner_qc) {
    if (target_qubit_index < inner_qc){
        H_gate(target_qubit_index, state, dim);
    } else {
        const MPIutil m = get_mpiutil();
        const int rank = m->get_rank();
        CTYPE* t = NULL;
        const int peer_rank_bit = 1 << (target_qubit_index - inner_qc);
        const int peer_rank = rank ^ peer_rank_bit;
        _MALLOC_AND_CHECK(t, CTYPE, dim);
        m->m_DC_sendrecv(state, t, dim, peer_rank);

		#ifdef _OPENMP
			UINT threshold = 13;
			if (dim < (((ITYPE)1) << threshold)) {
				H_gate_single_unroll_mpi(t, state, dim, rank & peer_rank_bit);
			}
			else {
				H_gate_parallel_unroll_mpi(t, state, dim, rank & peer_rank_bit);
			}
		#else
			H_gate_single_unroll_mpi(t, state, dim, rank & peer_rank_bit);
		#endif
        free(t);
    }
}

void H_gate_single_unroll_mpi(CTYPE *t, CTYPE *state, ITYPE dim, int flag) {
	const ITYPE loop_dim = dim;
	const double sqrt2inv = 1. / sqrt(2.);
	ITYPE state_index = 0;
	for (state_index = 0; state_index < loop_dim; state_index += 2) {
		CTYPE temp_a0 = state[state_index];
		CTYPE temp_a1 = t[state_index];
		CTYPE temp_b0 = state[state_index + 1];
		CTYPE temp_b1 = t[state_index + 1];
		if (flag) {
			state[state_index] = (t[state_index] - state[state_index])*sqrt2inv;
			state[state_index + 1] = (t[state_index + 1] - state[state_index + 1])*sqrt2inv;
		}
		else {
			state[state_index] = (temp_a0 + temp_a1)*sqrt2inv;
			state[state_index + 1] = (temp_b0 + temp_b1)*sqrt2inv;
		}
	}
}

#ifdef _OPENMP
void H_gate_parallel_unroll_mpi(CTYPE *t, CTYPE *state, ITYPE dim, int flag) {
	const ITYPE loop_dim = dim;
	const double sqrt2inv = 1. / sqrt(2.);
	ITYPE state_index = 0;
#pragma omp parallel for
	for (state_index = 0; state_index < loop_dim; state_index += 2) {
		CTYPE temp_a0 = state[state_index];
		CTYPE temp_a1 = state[state_index];
		CTYPE temp_b0 = state[state_index + 1];
		CTYPE temp_b1 = state[state_index + 1];
		if (flag) {
			state[state_index] = (temp_a0 - temp_a1)*sqrt2inv;
			state[state_index + 1] = (temp_b0 - temp_b1)*sqrt2inv;
		}
		else {
			state[state_index] = (temp_a0 + temp_a1)*sqrt2inv;
			state[state_index + 1] = (temp_b0 + temp_b1)*sqrt2inv;
		}
	}
}
#endif //#ifdef _OPENMP
#endif //#ifdef _USE_MPI

/*
#ifdef _OPENMP
void H_gate_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	const double sqrt2inv = 1. / sqrt(2.);
	//std::cout << dim << std::endl;
#pragma omp parallel for
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
		ITYPE basis_index_1 = basis_index_0 + mask;
		//std::cout << basis_index_0 << " " << basis_index_1 << std::endl;
		CTYPE temp0 = state[basis_index_0];
		CTYPE temp1 = state[basis_index_1];
		state[basis_index_0] = (temp0 + temp1) *sqrt2inv;
		state[basis_index_1] = (temp0 - temp1) *sqrt2inv;
	}
}
#endif


void H_gate_old_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	single_qubit_dense_matrix_gate(target_qubit_index, HADAMARD_MATRIX, state, dim);
}

void H_gate_single(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	const double sqrt2inv = 1. / sqrt(2.);
	ITYPE state_index = 0;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
		ITYPE basis_index_1 = basis_index_0 + mask;
		CTYPE temp0 = state[basis_index_0];
		CTYPE temp1 = state[basis_index_1];
		state[basis_index_0] = (temp0 + temp1)*sqrt2inv;
		state[basis_index_1] = (temp0 - temp1)*sqrt2inv;
	}
}

*/
