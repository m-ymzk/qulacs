#include <stddef.h>
#include <stdio.h>
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

//void CNOT_gate_old_single(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim);
//void CNOT_gate_old_parallel(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim);
//void CNOT_gate_single(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim);

void CNOT_gate(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	//CNOT_gate_old_single(control_qubit_index, target_qubit_index, state, dim);
	//CNOT_gate_old_parallel(control_qubit_index, target_qubit_index, state, dim);
	//CNOT_gate_single(control_qubit_index, target_qubit_index, state, dim);
	//CNOT_gate_single_unroll(control_qubit_index, target_qubit_index, state, dim);
	//CNOT_gate_single_simd(control_qubit_index, target_qubit_index, state, dim);
	//CNOT_gate_parallel(control_qubit_index, target_qubit_index, state, dim);
	//return;

#ifdef _USE_SIMD
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
		CNOT_gate_single_simd(control_qubit_index, target_qubit_index, state, dim);
	}
	else {
		CNOT_gate_parallel_simd(control_qubit_index, target_qubit_index, state, dim);
	}
#else
	CNOT_gate_single_simd(control_qubit_index, target_qubit_index, state, dim);
#endif
#else
#ifdef _OPENMP
	UINT threshold = 13;
	if (dim < (((ITYPE)1) << threshold)) {
		CNOT_gate_single_unroll(control_qubit_index, target_qubit_index, state, dim);
	}
	else {
		CNOT_gate_parallel_unroll(control_qubit_index, target_qubit_index, state, dim);
	}
#else
	CNOT_gate_single_unroll(control_qubit_index, target_qubit_index, state, dim);
#endif
#endif
}



void CNOT_gate_single_unroll(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;

	const ITYPE target_mask = 1ULL << target_qubit_index;
	const ITYPE control_mask = 1ULL << control_qubit_index;

	const UINT min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index);
	const UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	ITYPE state_index = 0;
	if (control_qubit_index == IS_OUTER_QB) {
		for (state_index = 0; state_index < (loop_dim * 2); ++state_index) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&(~low_mask)) << 1);
			ITYPE basis_index_1 = basis_index_0 + target_mask;
			CTYPE temp = state[basis_index_0];
			state[basis_index_0] = state[basis_index_1];
			state[basis_index_1] = temp;
		}
	}
	else if (target_qubit_index == 0) {
		// swap neighboring two basis
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index = ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask;
			CTYPE temp = state[basis_index];
			state[basis_index] = state[basis_index + 1];
			state[basis_index + 1] = temp;
		}
	}
	else if (control_qubit_index == 0) {
		// no neighboring swap
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask;
			ITYPE basis_index_1 = basis_index_0 + target_mask;
			CTYPE temp = state[basis_index_0];
			state[basis_index_0] = state[basis_index_1];
			state[basis_index_1] = temp;
		}
	}
	else {
		// a,a+1 is swapped to a^m, a^m+1, respectively
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask;
			ITYPE basis_index_1 = basis_index_0 + target_mask;
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

void CNOT_gate_parallel_unroll(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;

	const ITYPE target_mask = 1ULL << target_qubit_index;
	const ITYPE control_mask = 1ULL << control_qubit_index;

	const UINT min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index);
	const UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	ITYPE state_index = 0;
	if (target_qubit_index == 0) {
		// swap neighboring two basis
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index = ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask;
			CTYPE temp = state[basis_index];
			state[basis_index] = state[basis_index + 1];
			state[basis_index + 1] = temp;
		}
	}
	else if (control_qubit_index == 0) {
		// no neighboring swap
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask;
			ITYPE basis_index_1 = basis_index_0 + target_mask;
			CTYPE temp = state[basis_index_0];
			state[basis_index_0] = state[basis_index_1];
			state[basis_index_1] = temp;
		}
	}
	else {
		// a,a+1 is swapped to a^m, a^m+1, respectively
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask;
			ITYPE basis_index_1 = basis_index_0 + target_mask;
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
void CNOT_gate_single_simd(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;

	const ITYPE target_mask = 1ULL << target_qubit_index;
	const ITYPE control_mask = 1ULL << control_qubit_index;

	const UINT min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index);
	const UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	ITYPE state_index = 0;
	if (target_qubit_index == 0) {
		// swap neighboring two basis
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index = ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask;
			double* ptr = (double*)(state + basis_index);
			__m256d data = _mm256_loadu_pd(ptr);
			data = _mm256_permute4x64_pd(data, 78); // (3210) -> (1032) : 1*2 + 4*3 + 16*0 + 64*1 = 2+12+64=78
			_mm256_storeu_pd(ptr, data);
		}
	}
	else if (control_qubit_index == 0) {
		// no neighboring swap
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask;
			ITYPE basis_index_1 = basis_index_0 + target_mask;
			CTYPE temp = state[basis_index_0];
			state[basis_index_0] = state[basis_index_1];
			state[basis_index_1] = temp;
		}
	}
	else {
		// a,a+1 is swapped to a^m, a^m+1, respectively
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask;
			ITYPE basis_index_1 = basis_index_0 + target_mask;
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

void CNOT_gate_parallel_simd(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;

	const ITYPE target_mask = 1ULL << target_qubit_index;
	const ITYPE control_mask = 1ULL << control_qubit_index;

	const UINT min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index);
	const UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	ITYPE state_index = 0;
	if (target_qubit_index == 0) {
		// swap neighboring two basis
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index = ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask;
			double* ptr = (double*)(state + basis_index);
			__m256d data = _mm256_loadu_pd(ptr);
			data = _mm256_permute4x64_pd(data, 78); // (3210) -> (1032) : 1*2 + 4*3 + 16*0 + 64*1 = 2+12+64=78
			_mm256_storeu_pd(ptr, data);
		}
	}
	else if (control_qubit_index == 0) {
		// no neighboring swap
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask;
			ITYPE basis_index_1 = basis_index_0 + target_mask;
			CTYPE temp = state[basis_index_0];
			state[basis_index_0] = state[basis_index_1];
			state[basis_index_1] = temp;
		}
	}
	else {
		// a,a+1 is swapped to a^m, a^m+1, respectively
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ ((state_index&high_mask) << 2)
				+ control_mask;
			ITYPE basis_index_1 = basis_index_0 + target_mask;
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
void CNOT_gate_mpi(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim, UINT inner_qc) {
    if (control_qubit_index < inner_qc){
        if (target_qubit_index < inner_qc){
            printf("#enter CNOT_gate_mpi, c-inner, t-inner\n");
            CNOT_gate(control_qubit_index, target_qubit_index, state, dim);
        } else {
            printf("#enter CNOT_gate_mpi, c-inner, t-outer\n");
            const MPIutil m = get_instance();
            const int rank = m->get_rank();
            CTYPE* t = NULL;
            const int peer_rank_bit = 1 << (target_qubit_index - inner_qc);
            const int peer_rank = rank ^ peer_rank_bit;
            _MALLOC_AND_CHECK(t, CTYPE, dim); // get all data(tmp.)
            m->mpisendrecv(state, t, dim * 2, peer_rank); // CTYPE = MPI_DOUBLE * 2
#ifdef _OPENMP
			UINT threshold = 13;
			if (dim < (((ITYPE)1) << threshold)) {
				CNOT_gate_single_unroll_mpi(control_qubit_index, state, t, dim);
			}
			else {
				CNOT_gate_parallel_unroll_mpi(control_qubit_index, state, t, dim);
			}
#else
			CNOT_gate_single_unroll_mpi(control_qubit_index, state, t, dim);
#endif
            free(t);
        }
    } else {
        if (target_qubit_index < inner_qc) {
            int target_rank_bit = 1 << (target_qubit_index - inner_qc);
            printf("#enter CNOT_gate_mpi, c-outer, t-inner, %d\n", target_qubit_index);
            MPIutil m = get_instance();
            int rank = m->get_rank();
            if (rank & target_rank_bit) {
                CNOT_gate(IS_OUTER_QB, target_qubit_index, state, dim);
            } // if else, nothing to do.
        } else {
            printf("#enter CNOT_gate_mpi, c-outer, t-outer\n");
            int control_rank_bit0 = 1 << (control_qubit_index - inner_qc);
            int target_rank_bit1 = 1 << (target_qubit_index - inner_qc);
            MPIutil m = get_instance();
            int rank = m->get_rank();
			if (rank & control_rank_bit0) {
                double* t = NULL;
                _MALLOC_AND_CHECK(t, double, dim * 2);
                int peer_rank_bit = 1 << (target_qubit_index - inner_qc);
                int peer_rank = rank ^ peer_rank_bit;
                printf("#%d: call mpisendrecv, dim*2 = %lld, peer_rank=%d\n", rank, dim*2, peer_rank);
                m->mpisendrecv(state, t, dim * 2, peer_rank);
                memcpy(state, t, dim * sizeof(CTYPE));
                free(t);
            }
			else {
				m->get_tag(); // dummy to count up tag
			}
        }
    }
}

void CNOT_gate_single_unroll_mpi(UINT control_qubit_index, CTYPE *state, CTYPE *tmp, ITYPE dim) {
    printf("#enter CNOT_gate_single_unroll_mpi\n");
	const ITYPE loop_dim = dim / 2;

	const ITYPE control_mask = 1ULL << control_qubit_index;

	const UINT min_qubit_index = control_qubit_index;
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = dim;
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;

	ITYPE state_index = 0;
	if (control_qubit_index == 0) {
		// no neighboring swap
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ control_mask;
			state[basis_index_0] = tmp[basis_index_0];
		}
	}
	else {
		// a,a+1 is swapped to a^m, a^m+1, respectively
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ control_mask;
			state[basis_index_0] = tmp[basis_index_0];
			state[basis_index_0 + 1] = tmp[basis_index_0 + 1];
		}
	}
}

#ifdef _OPENMP
void CNOT_gate_parallel_unroll_mpi(UINT control_qubit_index, CTYPE *state, CTYPE *tmp, ITYPE dim) {
    printf("#enter CNOT_gate_parallel_unroll_mpi\n");
	const ITYPE loop_dim = dim / 2;

	const ITYPE control_mask = 1ULL << control_qubit_index;

	const UINT min_qubit_index = control_qubit_index;
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = dim;
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;

	ITYPE state_index = 0;
	if (control_qubit_index == 0) {
		// no neighboring swap
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; ++state_index) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ control_mask;
			state[basis_index_0] = tmp[basis_index_0];
		}
	}
	else {
		// a,a+1 is swapped to a^m, a^m+1, respectively
#pragma omp parallel for
		for (state_index = 0; state_index < loop_dim; state_index += 2) {
			ITYPE basis_index_0 = (state_index&low_mask)
				+ ((state_index&mid_mask) << 1)
				+ control_mask;
			state[basis_index_0] = tmp[basis_index_0];
			state[basis_index_0 + 1] = tmp[basis_index_0 + 1];
		}
	}
}
#endif //#ifdef _OPENMP
#endif //#ifdef _USE_MPI

/*

#ifdef _OPENMP
void CNOT_gate_parallel(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;

	const ITYPE target_mask = 1ULL << target_qubit_index;
	const ITYPE control_mask = 1ULL << control_qubit_index;

	const UINT min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index);
	const UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	ITYPE state_index = 0;
#pragma omp parallel for
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_index_0 = (state_index&low_mask)
			+ ((state_index&mid_mask) << 1)
			+ ((state_index&high_mask) << 2)
			+ control_mask;
		ITYPE basis_index_1 = basis_index_0 + target_mask;
		CTYPE temp = state[basis_index_0];
		state[basis_index_0] = state[basis_index_1];
		state[basis_index_1] = temp;
	}
}
#endif

void CNOT_gate_old_single(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;
	const UINT min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index);
	const UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << max_qubit_index;
	const ITYPE control_mask = 1ULL << control_qubit_index;
	const ITYPE target_mask = 1ULL << target_qubit_index;
	ITYPE state_index;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_insert_only_min = insert_zero_to_basis_index(state_index, min_qubit_mask, min_qubit_index);
		ITYPE basis_c1t0 = insert_zero_to_basis_index(basis_insert_only_min, max_qubit_mask, max_qubit_index) ^ control_mask;
		ITYPE basis_c1t1 = basis_c1t0 ^ target_mask;
		swap_amplitude(state, basis_c1t0, basis_c1t1);
	}
}

#ifdef _OPENMP
void CNOT_gate_old_parallel(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;
	const UINT min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index);
	const UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << max_qubit_index;
	const ITYPE control_mask = 1ULL << control_qubit_index;
	const ITYPE target_mask = 1ULL << target_qubit_index;
	ITYPE state_index;

#pragma omp parallel for
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_insert_only_min = insert_zero_to_basis_index(state_index, min_qubit_mask, min_qubit_index);
		ITYPE basis_c1t0 = insert_zero_to_basis_index(basis_insert_only_min, max_qubit_mask, max_qubit_index) ^ control_mask;
		ITYPE basis_c1t1 = basis_c1t0 ^ target_mask;
		swap_amplitude(state, basis_c1t0, basis_c1t1);
	}
}
#endif


void CNOT_gate_single(UINT control_qubit_index, UINT target_qubit_index, CTYPE *state, ITYPE dim) {
	const ITYPE loop_dim = dim / 4;

	const ITYPE target_mask = 1ULL << target_qubit_index;
	const ITYPE control_mask = 1ULL << control_qubit_index;

	const UINT min_qubit_index = get_min_ui(control_qubit_index, target_qubit_index);
	const UINT max_qubit_index = get_max_ui(control_qubit_index, target_qubit_index);
	const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
	const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
	const ITYPE low_mask = min_qubit_mask - 1;
	const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
	const ITYPE high_mask = ~(max_qubit_mask - 1);

	ITYPE state_index = 0;
	for (state_index = 0; state_index < loop_dim; ++state_index) {
		ITYPE basis_index_0 = (state_index&low_mask)
			+ ((state_index&mid_mask) << 1)
			+ ((state_index&high_mask) << 2)
			+ control_mask;
		ITYPE basis_index_1 = basis_index_0 + target_mask;
		CTYPE temp = state[basis_index_0];
		state[basis_index_0] = state[basis_index_1];
		state[basis_index_1] = temp;
	}
}
*/
