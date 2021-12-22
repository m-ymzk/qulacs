#include <stdio.h>
#include <stdlib.h>
#include "init_ops.h"
#include "utility.h"
#include <time.h>
#include <limits.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _USE_MPI
#include "csim/MPIutil.h"
#endif

// state initialization
void initialize_quantum_state_single(CTYPE *state, ITYPE dim);
void initialize_quantum_state_parallel(CTYPE *state, ITYPE dim);
#ifdef _USE_MPI
void initialize_quantum_state_single_mpi(CTYPE *state, ITYPE dim, UINT outer_qc);
void initialize_quantum_state_parallel_mpi(CTYPE *state, ITYPE dim, UINT outer_qc);
#endif
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

void initialize_quantum_state_single(CTYPE *state, ITYPE dim) {
	ITYPE index;
	for (index = 0; index < dim; ++index) {
		state[index] = 0;
	}
	state[0] = 1.0;
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
#endif

#ifdef _USE_MPI
void initialize_quantum_state_mpi(CTYPE *state, ITYPE dim, UINT outer_qc) {
#ifdef _OPENMP
	UINT threshold = 15;
	if (dim < (((ITYPE)1) << threshold)) {
		initialize_quantum_state_single_mpi(state, dim, outer_qc);
	}
	else {
		initialize_quantum_state_parallel_mpi(state, dim, outer_qc);
	}
#else
	initialize_quantum_state_single(state, dim);
#endif
}

void initialize_quantum_state_single_mpi(CTYPE *state, ITYPE dim, UINT outer_qc) {
	ITYPE index;
	MPIutil m = get_mpiutil();
	for (index = 0; index < dim; ++index) {
		state[index] = 0;
	}
	if (outer_qc==0 || m->get_rank()==0) state[0] = 1.0;
}

#ifdef _OPENMP
void initialize_quantum_state_parallel_mpi(CTYPE *state, ITYPE dim, UINT outer_qc) {
    ITYPE index;
    MPIutil m = get_mpiutil();
#pragma omp parallel for
    for(index=0; index < dim ; ++index){
        state[index]=0;
    }
    if (outer_qc==0 || m->get_rank()==0) state[0] = 1.0;
}
#endif
#endif //#ifdef _USE_MPI
