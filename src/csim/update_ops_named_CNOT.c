#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "constant.h"
#include "memory_ops.h"
#include "update_ops.h"
#include "utility.h"

#ifdef _USE_SIMD
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

void CNOT_gate(UINT control_qubit_index, UINT target_qubit_index, CTYPE* state,
    ITYPE dim) {
#ifdef _OPENMP
	OMPutil omputil = get_omputil();
	omputil->set_qulacs_num_threads(dim, 13);
#endif

#ifdef _USE_SIMD
#ifdef _OPENMP
    CNOT_gate_parallel_simd(
        control_qubit_index, target_qubit_index, state, dim);
#else
    CNOT_gate_single_simd(control_qubit_index, target_qubit_index, state, dim);
#endif
#else
#ifdef _OPENMP
    CNOT_gate_parallel_unroll(
        control_qubit_index, target_qubit_index, state, dim);
#else
    CNOT_gate_single_unroll(
        control_qubit_index, target_qubit_index, state, dim);
#endif
#endif

#ifdef _OPENMP
	omputil->reset_qulacs_num_threads();
#endif
}

void CNOT_gate_single_unroll(UINT control_qubit_index, UINT target_qubit_index,
    CTYPE* state, ITYPE dim) {
    // printf("#enter CNOT_g_s_u, %lld, %d, %d\n", dim, control_qubit_index,
    // target_qubit_index);
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

    ITYPE state_index = 0;
    if (control_qubit_index == IS_OUTER_QB) {
        if (target_qubit_index == 0) {
            // swap neighboring two basis in ALL_AREA
            for (state_index = 0; state_index < (dim / 2); ++state_index) {
                ITYPE basis_index = (state_index << 1);
                CTYPE temp = state[basis_index];
                state[basis_index] = state[basis_index + 1];
                state[basis_index + 1] = temp;
            }
        } else {
            // a,a+1 is swapped to a^m, a^m+1, respectively in ALL_AREA
            for (state_index = 0; state_index < (dim / 2); ++state_index) {
                ITYPE basis_index_0 = (state_index & low_mask) +
                                      ((state_index & (~low_mask)) << 1);
                ITYPE basis_index_1 = basis_index_0 + target_mask;
                CTYPE temp = state[basis_index_0];
                state[basis_index_0] = state[basis_index_1];
                state[basis_index_1] = temp;
            }
        }
    } else if (target_qubit_index == 0) {
        // swap neighboring two basis
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index = ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + control_mask;
            CTYPE temp = state[basis_index];
#if defined(__ARM_FEATURE_SVE)
#ifdef __aarch64__
            if (5 <= control_qubit_index && control_qubit_index <= 8) {
// L1 prefetch
#undef _PRF_L1_ITR
#define _PRF_L1_ITR 16
                ITYPE basis_index_l1pf =
                    (((state_index + _PRF_L1_ITR) & mid_mask) << 1) +
                    (((state_index + _PRF_L1_ITR) & high_mask) << 2) +
                    control_mask;
                __builtin_prefetch(&state[basis_index_l1pf], 1, 3);
                __builtin_prefetch(&state[basis_index_l1pf + 1], 1, 3);
// L2 prefetch
#undef _PRF_L2_ITR
#define _PRF_L2_ITR 64
                ITYPE basis_index_l2pf =
                    (((state_index + _PRF_L2_ITR) & mid_mask) << 1) +
                    (((state_index + _PRF_L2_ITR) & high_mask) << 2) +
                    control_mask;
                __builtin_prefetch(&state[basis_index_l2pf], 1, 2);
                __builtin_prefetch(&state[basis_index_l2pf + 1], 1, 2);
            }
#endif  // #ifdef __aarch64__
#endif  // #if defined(__ARM_FEATURE_SVE)
            state[basis_index] = state[basis_index + 1];
            state[basis_index + 1] = temp;
        }
    } else if (control_qubit_index == 0) {
        // no neighboring swap
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask;
            ITYPE basis_index_1 = basis_index_0 + target_mask;
            CTYPE temp = state[basis_index_0];
#if defined(__ARM_FEATURE_SVE)
#ifdef __aarch64__
            if (5 <= target_qubit_index && target_qubit_index <= 8) {
// L2 prefetch
#undef _PRF_L2_ITR
#define _PRF_L2_ITR 128
                ITYPE basis_index_l2pf0 =
                    ((state_index + _PRF_L2_ITR) & low_mask) +
                    (((state_index + _PRF_L2_ITR) & mid_mask) << 1) +
                    (((state_index + _PRF_L2_ITR) & high_mask) << 2) +
                    control_mask;
                ITYPE basis_index_l2pf1 = basis_index_l2pf0 + target_mask;
                __builtin_prefetch(&state[basis_index_l2pf0], 1, 2);
                __builtin_prefetch(&state[basis_index_l2pf0 + 1], 1, 2);
                __builtin_prefetch(&state[basis_index_l2pf1], 1, 2);
                __builtin_prefetch(&state[basis_index_l2pf1 + 1], 1, 2);
            }
#endif  // #ifdef __aarch64__
#endif  // #if defined(__ARM_FEATURE_SVE)
            state[basis_index_0] = state[basis_index_1];
            state[basis_index_1] = temp;
        }
#if defined(__ARM_FEATURE_SVE)
#ifdef __aarch64__
    } else if ((5 <= control_qubit_index && control_qubit_index <= 8) ||
               (5 <= target_qubit_index && target_qubit_index <= 8)) {
        if (4 <= min_qubit_mask) {
            // SIMD + PREFETCH
            // a,a+1 is swapped to a^m, a^m+1, respectively
            for (state_index = 0; state_index < loop_dim; state_index += 4) {
                ITYPE basis_index_0 =
                    (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                    ((state_index & high_mask) << 2) + control_mask;
                ITYPE basis_index_1 = basis_index_0 + target_mask;
                ETYPE* restrict state0 = (ETYPE*)&state[basis_index_0];
                ETYPE* restrict state1 = (ETYPE*)&state[basis_index_1];
// L1 prefetch
#undef _PRF_L1_ITR
#define _PRF_L1_ITR 4
                ITYPE basis_index_l1pf0 =
                    ((state_index + 4 * _PRF_L1_ITR) & low_mask) +
                    (((state_index + 4 * _PRF_L1_ITR) & mid_mask) << 1) +
                    (((state_index + 4 * _PRF_L1_ITR) & high_mask) << 2) +
                    control_mask;
                ITYPE basis_index_l1pf1 = basis_index_l1pf0 + target_mask;
                ETYPE* restrict state0_l1pf = (ETYPE*)&state[basis_index_l1pf0];
                ETYPE* restrict state1_l1pf = (ETYPE*)&state[basis_index_l1pf1];
                __builtin_prefetch(&state0_l1pf[0], 1, 3);
                __builtin_prefetch(&state0_l1pf[7], 1, 3);
                __builtin_prefetch(&state1_l1pf[0], 1, 3);
                __builtin_prefetch(&state1_l1pf[7], 1, 3);
// L2 prefetch
#undef _PRF_L2_ITR
#define _PRF_L2_ITR 64
                ITYPE basis_index_l2pf0 =
                    ((state_index + 4 * _PRF_L2_ITR) & low_mask) +
                    (((state_index + 4 * _PRF_L2_ITR) & mid_mask) << 1) +
                    (((state_index + 4 * _PRF_L2_ITR) & high_mask) << 2) +
                    control_mask;
                ITYPE basis_index_l2pf1 = basis_index_l2pf0 + target_mask;
                ETYPE* restrict state0_l2pf = (ETYPE*)&state[basis_index_l2pf0];
                ETYPE* restrict state1_l2pf = (ETYPE*)&state[basis_index_l2pf1];
                __builtin_prefetch(&state0_l2pf[0], 1, 2);
                __builtin_prefetch(&state0_l2pf[7], 1, 2);
                __builtin_prefetch(&state1_l2pf[0], 1, 2);
                __builtin_prefetch(&state1_l2pf[7], 1, 2);
#pragma omp simd
                for (ITYPE i = 0; i < 8; ++i) {
                    ETYPE temp = state0[i];
                    state0[i] = state1[i];
                    state1[i] = temp;
                }
            }
        } else {
            // PREFETCH
            // a,a+1 is swapped to a^m, a^m+1, respectively
            for (state_index = 0; state_index < loop_dim; state_index += 2) {
                ITYPE basis_index_0 =
                    (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                    ((state_index & high_mask) << 2) + control_mask;
                ITYPE basis_index_1 = basis_index_0 + target_mask;
                CTYPE temp0 = state[basis_index_0];
                CTYPE temp1 = state[basis_index_0 + 1];
// L1 prefetch
#undef _PRF_L1_ITR
#define _PRF_L1_ITR 4
                ITYPE basis_index_l1pf0 =
                    ((state_index + 2 * _PRF_L1_ITR) & low_mask) +
                    (((state_index + 2 * _PRF_L1_ITR) & mid_mask) << 1) +
                    (((state_index + 2 * _PRF_L1_ITR) & high_mask) << 2) +
                    control_mask;
                ITYPE basis_index_l1pf1 = basis_index_l1pf0 + target_mask;
                __builtin_prefetch(&state[basis_index_l1pf0], 1, 3);
                __builtin_prefetch(&state[basis_index_l1pf0 + 1], 1, 3);
                __builtin_prefetch(&state[basis_index_l1pf1], 1, 3);
                __builtin_prefetch(&state[basis_index_l1pf1 + 1], 1, 3);
// L2 prefetch
#undef _PRF_L2_ITR
#define _PRF_L2_ITR 64
                ITYPE basis_index_l2pf0 =
                    ((state_index + 2 * _PRF_L2_ITR) & low_mask) +
                    (((state_index + 2 * _PRF_L2_ITR) & mid_mask) << 1) +
                    (((state_index + 2 * _PRF_L2_ITR) & high_mask) << 2) +
                    control_mask;
                ITYPE basis_index_l2pf1 = basis_index_l2pf0 + target_mask;
                __builtin_prefetch(&state[basis_index_l2pf0], 1, 2);
                __builtin_prefetch(&state[basis_index_l2pf0 + 1], 1, 2);
                __builtin_prefetch(&state[basis_index_l2pf1], 1, 2);
                __builtin_prefetch(&state[basis_index_l2pf1 + 1], 1, 2);

                state[basis_index_0] = state[basis_index_1];
                state[basis_index_0 + 1] = state[basis_index_1 + 1];
                state[basis_index_1] = temp0;
                state[basis_index_1 + 1] = temp1;
            }
        }
    } else if (4 <= min_qubit_mask && 9 <= control_qubit_index) {
        // a,a+1 is swapped to a^m, a^m+1, respectively
        for (state_index = 0; state_index < loop_dim; state_index += 4) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask;
            ITYPE basis_index_1 = basis_index_0 + target_mask;
            ETYPE* restrict state0 = (ETYPE*)&state[basis_index_0];
            ETYPE* restrict state1 = (ETYPE*)&state[basis_index_1];
#pragma omp simd
            for (ITYPE i = 0; i < 8; ++i) {
                ETYPE temp = state0[i];
                state0[i] = state1[i];
                state1[i] = temp;
            }
        }
#endif  // #ifdef __aarch64__
#endif  // #if defined(__ARM_FEATURE_SVE)
    } else {
        // a,a+1 is swapped to a^m, a^m+1, respectively
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask;
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
void CNOT_gate_parallel_unroll(UINT control_qubit_index,
    UINT target_qubit_index, CTYPE* state, ITYPE dim) {
    // printf("#enter CNOT_g_p_u, %lld, %d, %d\n", dim, control_qubit_index,
    // target_qubit_index);
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

    ITYPE state_index = 0;
    if (control_qubit_index == IS_OUTER_QB) {
        if (target_qubit_index == 0) {
            // swap neighboring two basi in ALL_AREAs
#pragma omp parallel for
            for (state_index = 0; state_index < (dim / 2); ++state_index) {
                ITYPE basis_index = (state_index << 1);
                CTYPE temp = state[basis_index];
                state[basis_index] = state[basis_index + 1];
                state[basis_index + 1] = temp;
            }
        } else {
            // a,a+1 is swapped to a^m, a^m+1, respectively in ALL_AREA
#pragma omp parallel for
            for (state_index = 0; state_index < (dim / 2); ++state_index) {
                ITYPE basis_index_0 = (state_index & low_mask) +
                                      ((state_index & (~low_mask)) << 1);
                ITYPE basis_index_1 = basis_index_0 + target_mask;
                CTYPE temp = state[basis_index_0];
                state[basis_index_0] = state[basis_index_1];
                state[basis_index_1] = temp;
            }
        }
    } else if (target_qubit_index == 0) {
        // swap neighboring two basis
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index = ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + control_mask;
            CTYPE temp = state[basis_index];
#ifdef __aarch64__
            if (5 <= control_qubit_index && control_qubit_index <= 8) {
// L1 prefetch
#undef _PRF_L1_ITR
#define _PRF_L1_ITR 16
                ITYPE basis_index_l1pf =
                    (((state_index + _PRF_L1_ITR) & mid_mask) << 1) +
                    (((state_index + _PRF_L1_ITR) & high_mask) << 2) +
                    control_mask;
                __builtin_prefetch(&state[basis_index_l1pf], 1, 3);
                __builtin_prefetch(&state[basis_index_l1pf + 1], 1, 3);
// L2 prefetch
#undef _PRF_L2_ITR
#define _PRF_L2_ITR 64
                ITYPE basis_index_l2pf =
                    (((state_index + _PRF_L2_ITR) & mid_mask) << 1) +
                    (((state_index + _PRF_L2_ITR) & high_mask) << 2) +
                    control_mask;
                __builtin_prefetch(&state[basis_index_l2pf], 1, 2);
                __builtin_prefetch(&state[basis_index_l2pf + 1], 1, 2);
            }
#endif  // #ifdef __aarch64__
            state[basis_index] = state[basis_index + 1];
            state[basis_index + 1] = temp;
        }
    } else if (control_qubit_index == 0) {
        // no neighboring swap
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask;
            ITYPE basis_index_1 = basis_index_0 + target_mask;
            CTYPE temp = state[basis_index_0];
#if defined(__ARM_FEATURE_SVE)
#ifdef __aarch64__
            if (5 <= target_qubit_index && target_qubit_index <= 8) {
// L2 prefetch
#undef _PRF_L2_ITR
#define _PRF_L2_ITR 128
                ITYPE basis_index_l2pf0 =
                    ((state_index + _PRF_L2_ITR) & low_mask) +
                    (((state_index + _PRF_L2_ITR) & mid_mask) << 1) +
                    (((state_index + _PRF_L2_ITR) & high_mask) << 2) +
                    control_mask;
                ITYPE basis_index_l2pf1 = basis_index_l2pf0 + target_mask;
                __builtin_prefetch(&state[basis_index_l2pf0], 1, 2);
                __builtin_prefetch(&state[basis_index_l2pf0 + 1], 1, 2);
                __builtin_prefetch(&state[basis_index_l2pf1], 1, 2);
                __builtin_prefetch(&state[basis_index_l2pf1 + 1], 1, 2);
            }
#endif  // #ifdef __aarch64__
#endif  // #if defined(__ARM_FEATURE_SVE)
            state[basis_index_0] = state[basis_index_1];
            state[basis_index_1] = temp;
        }
#ifdef __aarch64__
    } else if ((5 <= control_qubit_index && control_qubit_index <= 8) ||
               (5 <= target_qubit_index && target_qubit_index <= 8)) {
        if (4 <= min_qubit_mask) {
            // SIMD + PREFETCH
            // a,a+1 is swapped to a^m, a^m+1, respectively
#pragma omp parallel for
            for (state_index = 0; state_index < loop_dim; state_index += 4) {
                ITYPE basis_index_0 =
                    (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                    ((state_index & high_mask) << 2) + control_mask;
                ITYPE basis_index_1 = basis_index_0 + target_mask;
                ETYPE* restrict state0 = (ETYPE*)&state[basis_index_0];
                ETYPE* restrict state1 = (ETYPE*)&state[basis_index_1];
// L1 prefetch
#undef _PRF_L1_ITR
#define _PRF_L1_ITR 4
                ITYPE basis_index_l1pf0 =
                    ((state_index + 4 * _PRF_L1_ITR) & low_mask) +
                    (((state_index + 4 * _PRF_L1_ITR) & mid_mask) << 1) +
                    (((state_index + 4 * _PRF_L1_ITR) & high_mask) << 2) +
                    control_mask;
                ITYPE basis_index_l1pf1 = basis_index_l1pf0 + target_mask;
                ETYPE* restrict state0_l1pf = (ETYPE*)&state[basis_index_l1pf0];
                ETYPE* restrict state1_l1pf = (ETYPE*)&state[basis_index_l1pf1];
                __builtin_prefetch(&state0_l1pf[0], 1, 3);
                __builtin_prefetch(&state0_l1pf[7], 1, 3);
                __builtin_prefetch(&state1_l1pf[0], 1, 3);
                __builtin_prefetch(&state1_l1pf[7], 1, 3);
// L2 prefetch
#undef _PRF_L2_ITR
#define _PRF_L2_ITR 64
                ITYPE basis_index_l2pf0 =
                    ((state_index + 4 * _PRF_L2_ITR) & low_mask) +
                    (((state_index + 4 * _PRF_L2_ITR) & mid_mask) << 1) +
                    (((state_index + 4 * _PRF_L2_ITR) & high_mask) << 2) +
                    control_mask;
                ITYPE basis_index_l2pf1 = basis_index_l2pf0 + target_mask;
                ETYPE* restrict state0_l2pf = (ETYPE*)&state[basis_index_l2pf0];
                ETYPE* restrict state1_l2pf = (ETYPE*)&state[basis_index_l2pf1];
                __builtin_prefetch(&state0_l2pf[0], 1, 2);
                __builtin_prefetch(&state0_l2pf[7], 1, 2);
                __builtin_prefetch(&state1_l2pf[0], 1, 2);
                __builtin_prefetch(&state1_l2pf[7], 1, 2);
#pragma omp simd
                for (ITYPE i = 0; i < 8; ++i) {
                    ETYPE temp = state0[i];
                    state0[i] = state1[i];
                    state1[i] = temp;
                }
            }
        } else {
            // PREFETCH
            // a,a+1 is swapped to a^m, a^m+1, respectively
#pragma omp parallel for
            for (state_index = 0; state_index < loop_dim; state_index += 2) {
                ITYPE basis_index_0 =
                    (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                    ((state_index & high_mask) << 2) + control_mask;
                ITYPE basis_index_1 = basis_index_0 + target_mask;
                CTYPE temp0 = state[basis_index_0];
                CTYPE temp1 = state[basis_index_0 + 1];
#if defined(__ARM_FEATURE_SVE)
#ifdef __aarch64__
// L1 prefetch
#undef _PRF_L1_ITR
#define _PRF_L1_ITR 4
                ITYPE basis_index_l1pf0 =
                    ((state_index + 2 * _PRF_L1_ITR) & low_mask) +
                    (((state_index + 2 * _PRF_L1_ITR) & mid_mask) << 1) +
                    (((state_index + 2 * _PRF_L1_ITR) & high_mask) << 2) +
                    control_mask;
                ITYPE basis_index_l1pf1 = basis_index_l1pf0 + target_mask;
                __builtin_prefetch(&state[basis_index_l1pf0], 1, 3);
                __builtin_prefetch(&state[basis_index_l1pf0 + 1], 1, 3);
                __builtin_prefetch(&state[basis_index_l1pf1], 1, 3);
                __builtin_prefetch(&state[basis_index_l1pf1 + 1], 1, 3);
// L2 prefetch
#undef _PRF_L2_ITR
#define _PRF_L2_ITR 64
                ITYPE basis_index_l2pf0 =
                    ((state_index + 2 * _PRF_L2_ITR) & low_mask) +
                    (((state_index + 2 * _PRF_L2_ITR) & mid_mask) << 1) +
                    (((state_index + 2 * _PRF_L2_ITR) & high_mask) << 2) +
                    control_mask;
                ITYPE basis_index_l2pf1 = basis_index_l2pf0 + target_mask;
                __builtin_prefetch(&state[basis_index_l2pf0], 1, 2);
                __builtin_prefetch(&state[basis_index_l2pf0 + 1], 1, 2);
                __builtin_prefetch(&state[basis_index_l2pf1], 1, 2);
                __builtin_prefetch(&state[basis_index_l2pf1 + 1], 1, 2);
#endif // #ifdef __aarch64__
#endif // #if defined(__ARM_FEATURE_SVE)

                state[basis_index_0] = state[basis_index_1];
                state[basis_index_0 + 1] = state[basis_index_1 + 1];
                state[basis_index_1] = temp0;
                state[basis_index_1 + 1] = temp1;
            }
        }
    } else if (4 <= min_qubit_mask && 9 <= control_qubit_index) {
        // a,a+1 is swapped to a^m, a^m+1, respectively
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 4) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask;
            ITYPE basis_index_1 = basis_index_0 + target_mask;
            ETYPE* restrict state0 = (ETYPE*)&state[basis_index_0];
            ETYPE* restrict state1 = (ETYPE*)&state[basis_index_1];
#pragma omp simd
            for (ITYPE i = 0; i < 8; ++i) {
                ETYPE temp = state0[i];
                state0[i] = state1[i];
                state1[i] = temp;
            }
        }
#endif  // #ifdef __aarch64__
    } else {
        // a,a+1 is swapped to a^m, a^m+1, respectively
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask;
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
void CNOT_gate_single_simd(UINT control_qubit_index, UINT target_qubit_index,
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

    ITYPE state_index = 0;
    if (control_qubit_index == IS_OUTER_QB) {
        if (target_qubit_index == 0) {
            // swap neighboring two basis in ALL_AREA
            for (state_index = 0; state_index < (dim / 2); ++state_index) {
                ITYPE basis_index = (state_index << 1);
                CTYPE temp = state[basis_index];
                state[basis_index] = state[basis_index + 1];
                state[basis_index + 1] = temp;
            }
        } else {
            // a,a+1 is swapped to a^m, a^m+1, respectively in ALL_AREA
            for (state_index = 0; state_index < (dim / 2); ++state_index) {
                ITYPE basis_index_0 = (state_index & low_mask) +
                                      ((state_index & (~low_mask)) << 1);
                ITYPE basis_index_1 = basis_index_0 + target_mask;
                CTYPE temp = state[basis_index_0];
                state[basis_index_0] = state[basis_index_1];
                state[basis_index_1] = temp;
            }
        }
    } else if (target_qubit_index == 0) {
        // swap neighboring two basis
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index = ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + control_mask;
            double* ptr = (double*)(state + basis_index);
            __m256d data = _mm256_loadu_pd(ptr);
            data = _mm256_permute4x64_pd(data,
                78);  // (3210) -> (1032) : 1*2 + 4*3 + 16*0 + 64*1 = 2+12+64=78
            _mm256_storeu_pd(ptr, data);
        }
    } else if (control_qubit_index == 0) {
        // no neighboring swap
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask;
            ITYPE basis_index_1 = basis_index_0 + target_mask;
            CTYPE temp = state[basis_index_0];
            state[basis_index_0] = state[basis_index_1];
            state[basis_index_1] = temp;
        }
    } else {
        // a,a+1 is swapped to a^m, a^m+1, respectively
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask;
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
void CNOT_gate_parallel_simd(UINT control_qubit_index, UINT target_qubit_index,
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

    ITYPE state_index = 0;
    if (control_qubit_index == IS_OUTER_QB) {
        if (target_qubit_index == 0) {
            // swap neighboring two basis in ALL_AREA
#pragma omp parallel for
            for (state_index = 0; state_index < (dim / 2); ++state_index) {
                ITYPE basis_index = (state_index << 1);
                CTYPE temp = state[basis_index];
                state[basis_index] = state[basis_index + 1];
                state[basis_index + 1] = temp;
            }
        } else {
            // a,a+1 is swapped to a^m, a^m+1, respectively in ALL_AREA
#pragma omp parallel for
            for (state_index = 0; state_index < (dim / 2); ++state_index) {
                ITYPE basis_index_0 = (state_index & low_mask) +
                                      ((state_index & (~low_mask)) << 1);
                ITYPE basis_index_1 = basis_index_0 + target_mask;
                CTYPE temp = state[basis_index_0];
                state[basis_index_0] = state[basis_index_1];
                state[basis_index_1] = temp;
            }
        }
    } else if (target_qubit_index == 0) {
        // swap neighboring two basis
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index = ((state_index & mid_mask) << 1) +
                                ((state_index & high_mask) << 2) + control_mask;
            double* ptr = (double*)(state + basis_index);
            __m256d data = _mm256_loadu_pd(ptr);
            data = _mm256_permute4x64_pd(data,
                78);  // (3210) -> (1032) : 1*2 + 4*3 + 16*0 + 64*1 = 2+12+64=78
            _mm256_storeu_pd(ptr, data);
        }
    } else if (control_qubit_index == 0) {
        // no neighboring swap
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; ++state_index) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask;
            ITYPE basis_index_1 = basis_index_0 + target_mask;
            CTYPE temp = state[basis_index_0];
            state[basis_index_0] = state[basis_index_1];
            state[basis_index_1] = temp;
        }
    } else {
        // a,a+1 is swapped to a^m, a^m+1, respectively
#pragma omp parallel for
        for (state_index = 0; state_index < loop_dim; state_index += 2) {
            ITYPE basis_index_0 =
                (state_index & low_mask) + ((state_index & mid_mask) << 1) +
                ((state_index & high_mask) << 2) + control_mask;
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
void CNOT_gate_mpi(UINT control_qubit_index, UINT target_qubit_index,
    CTYPE* state, ITYPE dim, UINT inner_qc) {
    if (control_qubit_index < inner_qc) {
        if (target_qubit_index < inner_qc) {
            // printf("#enter CNOT_gate_mpi, c-inner, t-inner %d, %d, %d,
            // %lld\n", control_qubit_index, target_qubit_index, inner_qc, dim);
            CNOT_gate(control_qubit_index, target_qubit_index, state, dim);
        } else {
            // printf("#enter CNOT_gate_mpi, c-inner, t-outer %d, %d, %d,
            // %lld\n", control_qubit_index, target_qubit_index, inner_qc, dim);
            const MPIutil m = get_mpiutil();
            const UINT rank = m->get_rank();
            const UINT pair_rank_bit = 1 << (target_qubit_index - inner_qc);
            const UINT pair_rank = rank ^ pair_rank_bit;

            CNOT_gate_single_unroll_cin_tout(
                control_qubit_index, pair_rank, state, dim);
        }
    } else {  // (control_qubit_index >= inner_qc)
        const MPIutil m = get_mpiutil();
        const int rank = m->get_rank();
        const int control_rank_bit = 1 << (control_qubit_index - inner_qc);
        if (target_qubit_index < inner_qc) {
            // printf("#enter CNOT_gate_mpi, c-outer, t-inner %d, %d, %d,
            // %lld\n", control_qubit_index, target_qubit_index, inner_qc, dim);
            if (rank & control_rank_bit) {
                CNOT_gate(IS_OUTER_QB, target_qubit_index, state, dim);
            }  // if else, nothing to do.
        } else {
            // printf("#enter CNOT_gate_mpi, c-outer, t-outer %d, %d, %d,
            // %lld\n", control_qubit_index, target_qubit_index, inner_qc, dim);
            const int pair_rank_bit = 1 << (target_qubit_index - inner_qc);
            const int pair_rank = rank ^ pair_rank_bit;
            ITYPE dim_work = dim;
            ITYPE num_work = 0;
            CTYPE* t = m->get_workarea(&dim_work, &num_work);
            CTYPE* si = state;
            for (ITYPE i = 0; i < num_work; ++i) {
                if (rank & control_rank_bit) {
                    // printf("#%d: call m_DC_sendrecv, dim = %lld,
                    // pair_rank=%d\n", rank, dim, pair_rank);
                    m->m_DC_sendrecv(si, t, dim_work, pair_rank);
#if defined(__ARM_FEATURE_SVE)
                    memcpy_sve((double*)si, (double*)t, dim_work * 2);
#else
                    memcpy(si, t, dim_work * sizeof(CTYPE));
#endif
                    si += dim_work;
                } else {
                    m->get_tag();  // dummy to count up tag
                }
            }
        }  // (target_qubit_index < inner_qc)
    }      // (control_qubit_index >= inner_qc)
}

// CNOT_gate_mpi, control_qubit_index is inner, target_qubit_index is outer.
void CNOT_gate_single_unroll_cin_tout(
    UINT control_qubit_index, UINT pair_rank, CTYPE* state, ITYPE dim) {
    const MPIutil m = get_mpiutil();
    ITYPE dim_work = dim;
    ITYPE num_work = 0;
    CTYPE* t = m->get_workarea(&dim_work, &num_work);
    assert(num_work > 0);
    assert(dim_work > 0);

    const ITYPE control_isone_offset = 1ULL << control_qubit_index;

    if (control_isone_offset < dim_work) {
        dim_work >>= 1;  // 1/2: for send, 1/2: for recv
        CTYPE* t_send = t;
        CTYPE* t_recv = t + dim_work;
        const ITYPE num_control_block = (dim / dim_work) >> 1;
        assert(num_control_block > 0);
        const ITYPE num_elem_block = dim_work >> control_qubit_index;
        assert(num_elem_block > 0);
        CTYPE* si0 = state + control_isone_offset;
        for (ITYPE i = 0; i < num_control_block; ++i) {
            // gather
            CTYPE* si = si0;
            CTYPE* ti = t_send;
            for (ITYPE k = 0; k < num_elem_block; ++k) {
                memcpy(ti, si, control_isone_offset * sizeof(CTYPE));
                si += (control_isone_offset << 1);
                ti += control_isone_offset;
            }

            // sendrecv
            m->m_DC_sendrecv(t_send, t_recv, dim_work, pair_rank);

            // scatter
            si = t_recv;
            ti = si0;
            for (ITYPE k = 0; k < num_elem_block; ++k) {
                memcpy(ti, si, control_isone_offset * sizeof(CTYPE));
                si += control_isone_offset;
                ti += (control_isone_offset << 1);
            }
            si0 += (dim_work << 1);
        }
    } else {  // transfar unit size >= dim_work
        // printf("#enter CNOT_gate_single_unroll_cin_tout(w/o gather), %d,
        // %lld, %lld\n", control_qubit_index, dim_work, dim);
        const ITYPE num_control_block = dim >> (control_qubit_index + 1);
        const ITYPE num_work_block = control_isone_offset / dim_work;

        CTYPE* si = state + control_isone_offset;
        for (ITYPE i = 0; i < num_control_block; ++i) {
            for (ITYPE j = 0; j < num_work_block; ++j) {
                m->m_DC_sendrecv(si, t, dim_work, pair_rank);
#if defined(__ARM_FEATURE_SVE)
                memcpy_sve((double*)si, (double*)t, dim_work * 2);
#else
                memcpy(si, t, dim_work * sizeof(CTYPE));
#endif
                si += dim_work;
            }
            si += control_isone_offset;
        }
    }
}
#endif  //#ifdef _USE_MPI
