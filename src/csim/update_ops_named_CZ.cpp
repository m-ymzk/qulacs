
#include "constant.hpp"
#include "update_ops.hpp"
#include "utility.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _USE_SIMD
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

void CZ_gate(UINT control_qubit_index, UINT target_qubit_index, CTYPE* state,
    ITYPE dim) {
#ifdef _OPENMP
    OMPutil omputil = get_omputil();
    omputil->set_qulacs_num_threads(dim, 13);
#endif

#ifdef _USE_SIMD
    CZ_gate_parallel_simd(control_qubit_index, target_qubit_index, state, dim);
#else
    CZ_gate_parallel_unroll(
        control_qubit_index, target_qubit_index, state, dim);
#endif

#ifdef _OPENMP
    omputil->reset_qulacs_num_threads();
#endif
}

void CZ_gate_parallel_unroll(UINT control_qubit_index, UINT target_qubit_index,
    CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE target_mask = 1ULL << target_qubit_index;
    const ITYPE control_mask = 1ULL << control_qubit_index;

    const UINT min_qubit_index =
        get_min_ui(control_qubit_index, target_qubit_index);
    const UINT max_qubit_index =
        get_max_ui(control_qubit_index, target_qubit_index);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    const ITYPE mask = target_mask + control_mask;
    ITYPE state_index = 0;
    if (target_qubit_index == 0 || control_qubit_index == 0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + mask;
            state[basis_index] *= -1;
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + mask;
            state[basis_index] *= -1;
            state[basis_index + 1] *= -1;
        }
    }
}

#ifdef _USE_SIMD
void CZ_gate_parallel_simd(UINT control_qubit_index, UINT target_qubit_index,
    CTYPE* state, ITYPE dim) {
    const ITYPE loop_dim = dim / 4;

    const ITYPE target_mask = 1ULL << target_qubit_index;
    const ITYPE control_mask = 1ULL << control_qubit_index;

    const UINT min_qubit_index =
        get_min_ui(control_qubit_index, target_qubit_index);
    const UINT max_qubit_index =
        get_max_ui(control_qubit_index, target_qubit_index);
    const ITYPE min_qubit_mask = 1ULL << min_qubit_index;
    const ITYPE max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const ITYPE low_mask = min_qubit_mask - 1;
    const ITYPE mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const ITYPE high_mask = ~(max_qubit_mask - 1);

    const ITYPE mask = target_mask + control_mask;
    ITYPE state_index = 0;
    if (target_qubit_index == 0 || control_qubit_index == 0) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + mask;
            state[basis_index] *= -1;
        }
    } else {
        __m256d minus_one = _mm256_set_pd(-1, -1, -1, -1);
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index = (state_index & low_mask) +
                                ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + mask;
            double* ptr = (double*)(state + basis_index);
            __m256d data = _mm256_loadu_pd(ptr);
            data = _mm256_mul_pd(data, minus_one);
            _mm256_storeu_pd(ptr, data);
        }
    }
}
#endif
