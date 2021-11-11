#include <stdio.h>
#include <stdlib.h>
#include "init_ops.h"
#include "utility.h"
#include <time.h>
#include <limits.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// state initialization
void initialize_quantum_state_single(CTYPE *state, ITYPE dim);
void initialize_quantum_state_parallel(CTYPE *state, ITYPE dim);
void initialize_quantum_state_single_rank(CTYPE *state, ITYPE dim, ITYPE rank);
void initialize_quantum_state_parallel_rank(CTYPE *state, ITYPE dim, ITYPE rank);
void initialize_quantum_state(CTYPE *state, ITYPE dim) {
#ifdef _OPENMP
	UINT threshold = 15;
	if (dim < (((ITYPE)1) << threshold)) {
		initialize_quantum_state_single(state, dim);
	}
	else {
		initialize_quantum_state_parallel(state, dim);
	}
#else
	initialize_quantum_state_single(state, dim);
#endif
}
void initialize_quantum_state_rank(CTYPE *state, ITYPE dim, ITYPE rank) {
#ifdef _OPENMP
	UINT threshold = 15;
	if (dim < (((ITYPE)1) << threshold)) {
		initialize_quantum_state_single_rank(state, dim, rank);
	}
	else {
		initialize_quantum_state_parallel_rank(state, dim, rank);
	}
#else
	initialize_quantum_state_single(state, dim);
#endif
}
void initialize_quantum_state_single(CTYPE *state, ITYPE dim) {
	ITYPE index;
	for (index = 0; index < dim; ++index) {
		state[index] = 0;
	}
	state[0] = 1.0;
}
void initialize_quantum_state_single_rank(CTYPE *state, ITYPE dim, ITYPE rank) {
	ITYPE index;
	for (index = 0; index < dim; ++index) {
		state[index] = 0;
	}
	if (rank==0) state[0] = 1.0;
}

#ifdef _OPENMP
void initialize_quantum_state_parallel(CTYPE *state, ITYPE dim) {
    ITYPE index;
#pragma omp parallel for
    for(index=0; index < dim ; ++index){
        state[index]=0;
    }
    state[0] = 1.0;
}
void initialize_quantum_state_parallel_rank(CTYPE *state, ITYPE dim, ITYPE rank) {
    ITYPE index;
#pragma omp parallel for
    for(index=0; index < dim ; ++index){
        state[index]=0;
    }
    if (rank==0) state[0] = 1.0;
}
#endif
