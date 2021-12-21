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
#include "MPIutil.h"
#endif

// state randomization
unsigned long xor128(unsigned long* state);
double random_uniform(unsigned long* state);
double random_normal(unsigned long* state);
void initialize_Haar_random_state_with_seed_single(CTYPE *state, ITYPE dim, UINT outer_qc, UINT seed);
void initialize_Haar_random_state_with_seed_parallel(CTYPE *state, ITYPE dim, UINT outer_qc, UINT seed);

void initialize_Haar_random_state(CTYPE *state, ITYPE dim, UINT outer_qc) {
    int seed = (int)time(NULL);
    //printf("# enter init-Haar-rand-stat, %lld\n", dim);
#ifdef _USE_MPI
    MPIutil m = get_mpiutil();
    int size = m->get_size();
    if (size > 1) {
        seed = m->s_i_bcast(seed);
    }
#endif //#ifdef _USE_MPI
	initialize_Haar_random_state_with_seed(state, dim, outer_qc, seed);
}
void initialize_Haar_random_state_with_seed(CTYPE *state, ITYPE dim, UINT outer_qc, UINT seed) {
    //printf("# enter init-Haar-rand-stat(seed), %lld, %d\n", dim, seed);
#ifdef _OPENMP
	UINT threshold = 8;
	if (dim < (((ITYPE)1) << threshold)) {
		initialize_Haar_random_state_with_seed_single(state, dim, outer_qc, seed);
	}
	else {
		initialize_Haar_random_state_with_seed_parallel(state, dim, outer_qc, seed);
	}
#else
	initialize_Haar_random_state_with_seed_single(state, dim, outer_qc, seed);
#endif
}

// single thread
void initialize_Haar_random_state_with_seed_single(CTYPE *state, ITYPE dim, UINT outer_qc, UINT seed) {
    const int ignore_first = 40;
    double norm = 0.;
    unsigned long random_state[4];
    srand(seed);
    random_state[0] = rand();
    random_state[1] = rand();
    random_state[2] = rand();
    random_state[3] = rand();
    for (int i = 0; i < ignore_first; ++i) xor128(random_state);
    for (ITYPE index = 0; index < dim; ++index) {
        double r1, r2;
        r1 = random_normal(random_state);
        r2 = random_normal(random_state);
        state[index] = r1 + 1.i * r2;
        norm += r1 * r1 + r2 * r2;
    }
#ifdef _USE_MPI
    MPIutil m = get_mpiutil();
    if (outer_qc > 0) m->s_D_allreduce(&norm);
#endif
    norm = sqrt(norm);
    for (ITYPE index = 0; index < dim; ++index) {
        state[index] /= norm;
    }
}
#ifdef _OPENMP
void initialize_Haar_random_state_with_seed_parallel(CTYPE *state, ITYPE dim, UINT outer_qc, UINT seed) {
	// multi thread
    const int ignore_first = 40;
    const UINT thread_count = omp_get_max_threads();
    const ITYPE block_size = dim / thread_count;
    const ITYPE residual = dim % thread_count;

    unsigned long* random_state_list = (unsigned long*)malloc(sizeof(unsigned long)*4*thread_count);
    srand(seed);
    for (UINT i = 0; i < 4*thread_count; ++i) {
        random_state_list[i] = rand();
    }

    double* norm_list = (double*)malloc(sizeof(double)*thread_count);
    for (UINT i = 0; i < thread_count; ++i) {
        norm_list[i] = 0;
    }
#pragma omp parallel
    {
        UINT thread_id = omp_get_thread_num();
        unsigned long* my_random_state = random_state_list +4 * thread_id;
        ITYPE start_index = block_size * thread_id + (residual>thread_id?thread_id:residual);
        ITYPE end_index = block_size * (thread_id+1) + (residual > (thread_id+1) ? (thread_id+1) : residual);
        ITYPE index;

        // ignore first randoms
        for (int i = 0; i < ignore_first; ++i) xor128(my_random_state);

        for (index = start_index; index < end_index; ++index) {
            double r1, r2;
            r1 = r2 = 1;
            r1 = random_normal(my_random_state);
            r2 = random_normal(my_random_state);
            state[index] = r1 + 1.i * r2;
            norm_list[thread_id] += r1 * r1 + r2 * r2;
        }
    }

    double normalizer = 0.;
    for (UINT i = 0; i < thread_count; ++i) {
        normalizer += norm_list[i];
    }
#ifdef _USE_MPI
    MPIutil m = get_mpiutil();
    if (outer_qc > 0) m->s_D_allreduce(&normalizer);
#endif
    normalizer = 1./sqrt(normalizer);

#pragma omp parallel for
    for (ITYPE index = 0; index < dim; ++index) {
        state[index] *= normalizer;
    }
    free(random_state_list);
    free(norm_list);
}
#endif

unsigned long xor128(unsigned long* state) {
	unsigned long t;
	t = (state[0] ^ (state[0] << 11));
	state[0] = state[1]; state[1] = state[2]; state[2] = state[3];
	return (state[3] = (state[3] ^ (state[3] >> 19)) ^ (t ^ (t >> 8)));
}
double random_uniform(unsigned long* state) {
	return xor128(state) / ((float)ULONG_MAX);
}
double random_normal(unsigned long* state) {
	return sqrt(-1.0*log(random_uniform(state))) * sin(2.0*M_PI*random_uniform(state));
}
