#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "init_ops.h"
#include "utility.h"

// state initialization
void initialize_quantum_state(CTYPE *state, ITYPE dim) {
#ifdef _OPENMP
	OMPutil omputil = get_omputil();
	omputil->set_qulacs_num_threads(dim, 15);
#endif

    ITYPE index;
#pragma omp parallel for
    for (index = 0; index < dim; ++index) {
        state[index] = 0;
    }
    state[0] = 1.0;

#ifdef _OPENMP
	omputil->reset_qulacs_num_threads();
#endif
}

#ifdef _USE_MPI
void initialize_quantum_state_mpi(CTYPE *state, ITYPE dim, UINT outer_qc) {
#ifdef _OPENMP
	OMPutil omputil = get_omputil();
	omputil->set_qulacs_num_threads(dim, 15);
#endif

    ITYPE index;
    MPIutil m = get_mpiutil();
#pragma omp parallel for
    for (index = 0; index < dim; ++index) {
        state[index] = 0;
    }
    if (outer_qc == 0 || m->get_rank() == 0) state[0] = 1.0;

#ifdef _OPENMP
	omputil->reset_qulacs_num_threads();
#endif
}
#endif  //#ifdef _USE_MPI
