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
#else
#include <x86intrin.h>
#endif
#endif

//void X_gate_old(UINT target_qubit_index, CTYPE *state, ITYPE dim);
//void X_gate_single(UINT target_qubit_index, CTYPE *state, ITYPE dim);
//void X_gate_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim);

void X_gate(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	//UINT threshold = 13;
	//X_gate_old(target_qubit_index, state, dim);
	//X_gate_single(target_qubit_index, state, dim);
	//X_gate_single_simd(target_qubit_index, state, dim);
	//X_gate_single_unroll(target_qubit_index, state, dim);
	//X_gate_parallel(target_qubit_index, state, dim);
	//return;

#ifdef _USE_SIMD
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
		X_gate_single_simd(target_qubit_index, state, dim);
	}
	else {
		X_gate_parallel_simd(target_qubit_index, state, dim);
	}
#else
	X_gate_single_simd(target_qubit_index, state, dim);
#endif
#else
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
		X_gate_single_unroll(target_qubit_index, state, dim);
	}
	else {
		X_gate_parallel_unroll(target_qubit_index, state, dim);
	}
#else
	X_gate_single_unroll(target_qubit_index, state, dim);
#endif
#endif
}


void X_gate_single_unroll(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	if (target_qubit_index == 0) {
		ITYPE basis_index = 0;
		for (basis_index = 0; basis_index < dim; basis_index += 2) {
			CTYPE temp = state[basis_index];
			state[basis_index] = state[basis_index + 1];
			state[basis_index + 1] = temp;
		}
	}
	else {
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
			ITYPE basis_index_1 = basis_index_0 + mask;
			CTYPE temp0 = state[basis_index_0];
			CTYPE temp1 = state[basis_index_0 + 1];
			state[basis_index_0] = state[basis_index_1];
			state[basis_index_0 + 1] = state[basis_index_1 + 1];
			state[basis_index_1] = temp0;
			state[basis_index_1 + 1] = temp1;
		}
	}
}

#ifdef _OPENMP
void X_gate_parallel_unroll(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	if (target_qubit_index == 0) {
		ITYPE basis_index = 0;
#pragma omp parallel for
		for (basis_index = 0; basis_index < dim; basis_index += 2) {
			CTYPE temp = state[basis_index];
			state[basis_index] = state[basis_index + 1];
			state[basis_index + 1] = temp;
		}
	}
#ifdef __aarch64__
	else if (6 <= target_qubit_index && target_qubit_index <= 8) {
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 8) {
			ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
			ITYPE basis_index_1 = basis_index_0 + mask;
			ITYPE basis_index_2 = ((state_index + 2)&mask_low) + (((state_index + 2)&mask_high) << 1);
			ITYPE basis_index_3 = basis_index_2 + mask;
			ITYPE basis_index_4 = ((state_index + 4)&mask_low) + (((state_index + 4)&mask_high) << 1);
			ITYPE basis_index_5 = basis_index_4 + mask;
			ITYPE basis_index_6 = ((state_index + 6)&mask_low) + (((state_index + 6)&mask_high) << 1);
			ITYPE basis_index_7 = basis_index_6 + mask;
			CTYPE temp0 = state[basis_index_0];
			CTYPE temp1 = state[basis_index_0 + 1];
			CTYPE temp2 = state[basis_index_2];
			CTYPE temp3 = state[basis_index_2 + 1];
			CTYPE temp4 = state[basis_index_4];
			CTYPE temp5 = state[basis_index_4 + 1];
			CTYPE temp6 = state[basis_index_6];
			CTYPE temp7 = state[basis_index_6 + 1];

			// L1 prefetch
			__builtin_prefetch(&state[basis_index_0 + mask * 2], 1, 3);
			__builtin_prefetch(&state[basis_index_1 + mask * 2], 1, 3);
			// L2 prefetch
			__builtin_prefetch(&state[basis_index_0 + mask * 4], 1, 2);
			__builtin_prefetch(&state[basis_index_1 + mask * 4], 1, 2);

			state[basis_index_0] = state[basis_index_1];
			state[basis_index_0 + 1] = state[basis_index_1 + 1];
			state[basis_index_2] = state[basis_index_3];
			state[basis_index_2 + 1] = state[basis_index_3 + 1];
			state[basis_index_4] = state[basis_index_5];
			state[basis_index_4 + 1] = state[basis_index_5 + 1];
			state[basis_index_6] = state[basis_index_7];
			state[basis_index_6 + 1] = state[basis_index_7 + 1];
			state[basis_index_1] = temp0;
			state[basis_index_1 + 1] = temp1;
			state[basis_index_3] = temp2;
			state[basis_index_3 + 1] = temp3;
			state[basis_index_5] = temp4;
			state[basis_index_5 + 1] = temp5;
			state[basis_index_7] = temp6;
			state[basis_index_7 + 1] = temp7;
		}
	}
#endif
	else {
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
			ITYPE basis_index_1 = basis_index_0 + mask;
			CTYPE temp0 = state[basis_index_0];
			CTYPE temp1 = state[basis_index_0 + 1];
			state[basis_index_0] = state[basis_index_1];
			state[basis_index_0 + 1] = state[basis_index_1 + 1];
			state[basis_index_1] = temp0;
			state[basis_index_1 + 1] = temp1;
		}
	}
}
#endif

#ifdef _USE_SIMD
void X_gate_single_simd(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	//double* cast_state = (double*)state;
	if (target_qubit_index == 0) {
		ITYPE basis_index = 0;
		for (basis_index = 0; basis_index < dim; basis_index += 2) {
			double* ptr = (double*)(state + basis_index);
			__m256d data = _mm256_loadu_pd(ptr);
			data = _mm256_permute4x64_pd(data, 78); // (3210) -> (1032) : 1*2 + 4*3 + 16*0 + 64*1 = 2+12+64=78
			_mm256_storeu_pd(ptr, data);
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
			_mm256_storeu_pd(ptr1, data0);
			_mm256_storeu_pd(ptr0, data1);
		}
	}
}

#ifdef _OPENMP
void X_gate_parallel_simd(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	//double* cast_state = (double*)state;
	if (target_qubit_index == 0) {
		ITYPE basis_index = 0;
#pragma omp parallel for
		for (basis_index = 0; basis_index < dim; basis_index += 2) {
			double* ptr = (double*)(state + basis_index);
			__m256d data = _mm256_loadu_pd(ptr);
			data = _mm256_permute4x64_pd(data, 78); // (3210) -> (1032) : 1*2 + 4*3 + 16*0 + 64*1 = 2+12+64=78
			_mm256_storeu_pd(ptr, data);
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
			_mm256_storeu_pd(ptr1, data0);
			_mm256_storeu_pd(ptr0, data1);
		}
	}
}
#endif
#endif

#ifdef _USE_MPI
void X_gate_mpi(UINT target_qubit_index, CTYPE *state, ITYPE dim, UINT inner_qc) {
    if (target_qubit_index < inner_qc){
        X_gate(target_qubit_index, state, dim);
    } else {
        const MPIutil m = get_mpiutil();
        const int rank = m->get_rank();
        ITYPE dim_work = dim;
        ITYPE num_work = 0;
        CTYPE* t = m->get_workarea(&dim_work, &num_work);
        assert(num_work > 0);
        const int pair_rank_bit = 1 << (target_qubit_index - inner_qc);
        const int pair_rank = rank ^ pair_rank_bit;
        //ITYPE dim_work = get_min_ll(1 << nqubit_WORK, dim);
        //ITYPE num_work = get_max_ll(1, dim >> nqubit_WORK);
        //printf("#debug dim,dim_work,num_work,t: %lld, %lld, %lld, %p\n", dim, dim_work, num_work, t);
        CTYPE* si = state;
        for (ITYPE i=0; i < num_work; ++i) {
            m->m_DC_sendrecv(si, t, dim_work, pair_rank);
            memcpy(si, t, dim_work * sizeof(CTYPE));
            si += dim_work;
        }
    }
}
#endif

/*


void X_gate_old(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	ITYPE state_index;
#ifdef _OPENMP
//#pragma omp parallel for
#endif
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_index_0 = insert_zero_to_basis_index(state_index, mask, target_qubit_index);
		ITYPE basis_index_1 = basis_index_0 ^ mask;
		swap_amplitude(state, basis_index_0, basis_index_1);
	}
}

void X_gate_single(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
		ITYPE basis_index_1 = basis_index_0 + mask;
		CTYPE temp = state[basis_index_0];
		state[basis_index_0] = state[basis_index_1];
		state[basis_index_1] = temp;
	}
}

#ifdef _OPENMP
void X_gate_parallel(UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 2;
	const ITYPE mask = (1ULL << target_qubit_index);
	const ITYPE mask_low = mask - 1;
	const ITYPE mask_high = ~mask_low;
	ITYPE state_index = 0;
#pragma omp parallel for
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_index_0 = (state_index&mask_low) + ((state_index&mask_high) << 1);
		ITYPE basis_index_1 = basis_index_0 + mask;
		CTYPE temp = state[basis_index_0];
		state[basis_index_0] = state[basis_index_1];
		state[basis_index_1] = temp;
	}
}
#endif

*/
