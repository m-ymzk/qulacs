#pragma once

#ifndef _MSC_VER
extern "C" {
#include <csim/update_ops.h>
}
#else
#include <csim/update_ops.h>
#endif

#include "gate_named.hpp"
#include "state.hpp"

/**
 * \~japanese-en CNOTゲート
 */
class ClsCNOTGate : public QuantumGate_OneControlOneTarget {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param control_qubit_index コントロール量子ビットの添え字
     * @param target_qubit_index ターゲット量子ビットの添え字
     */
    ClsCNOTGate(UINT control_qubit_index, UINT target_qubit_index) {
        this->_update_func = CNOT_gate;
        this->_update_func_dm = dm_CNOT_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = CNOT_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = CNOT_gate_mpi;
#endif
        this->_name = "CNOT";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_X_COMMUTE));
        this->_control_qubit_list.push_back(
            ControlQubitInfo(control_qubit_index, 1));
        this->_gate_property = FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 0, 1, 1, 0;
    }
};

/**
 * \~japanese-en Control-Zゲート
 */
class ClsCZGate : public QuantumGate_OneControlOneTarget {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param control_qubit_index コントロール量子ビットの添え字
     * @param target_qubit_index ターゲット量子ビットの添え字
     */
    ClsCZGate(UINT control_qubit_index, UINT target_qubit_index) {
        this->_update_func = CZ_gate;
        this->_update_func_dm = dm_CZ_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = CZ_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = CZ_gate_mpi;
#endif
        this->_name = "CZ";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_Z_COMMUTE));
        this->_control_qubit_list.push_back(
            ControlQubitInfo(control_qubit_index, 1));
        this->_gate_property = FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 1, 0, 0, -1;
    }
};

/**
 * \~japanese-en SWAPゲート
 */
class ClsSWAPGate : public QuantumGate_TwoQubit {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param target_qubit_index1 ターゲット量子ビットの添え字
     * @param target_qubit_index2 もう一つのターゲット量子ビットの添え字
     */
    ClsSWAPGate(UINT target_qubit_index1, UINT target_qubit_index2) {
        this->_update_func = SWAP_gate;
        this->_update_func_dm = dm_SWAP_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = SWAP_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = SWAP_gate_mpi;
#endif
        this->_name = "SWAP";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index1, 0));
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index2, 0));
        this->_gate_property = FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(4, 4);
        this->_matrix_element << 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1;
    }
};

/**
 * \~japanese-en FusedSWAPゲート
 */
class ClsFusedSWAPGate : public QuantumGate_NQubitpair {
public:
    /**
     * \~japanese-en コンストラクタ
     *
     * @param target_qubit_index1 ターゲット量子ビットの添え字の先頭
     * @param target_qubit_index2 もう一つのターゲット量子ビットの添え字の先頭
     * @param num_qubits ターゲット量子ビット数
     */
    ClsFusedSWAPGate(
        UINT target_qubit_index1, UINT target_qubit_index2, UINT num_qubits) {
        this->_update_func = FusedSWAP_gate;
        // this->_update_func_dm = dm_SWAP_gate;
#ifdef _USE_GPU
        // this->_update_func_gpu = SWAP_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = FusedSWAP_gate_mpi;
#endif
        this->_name = "FusedSWAP";

        // 以下の順序でtarget_qubit_listに追加
        // [target_qubit_index1, target_qubit_index1+1, ..., target_qubit_index1+(num_qubits-1),
        //  target_qubit_index2, target_qubit_index2+1, ..., target_qubit_index2+(num_qubits-1)]
        for (UINT i = 0; i < num_qubits; i++) {
            this->_target_qubit_list.push_back(
                TargetQubitInfo(target_qubit_index1 + i, 0));
        }
        for (UINT i = 0; i < num_qubits; i++) {
            this->_target_qubit_list.push_back(
                TargetQubitInfo(target_qubit_index2 + i, 0));
        }
        this->_num_qubits = num_qubits;
        this->_gate_property = FLAG_CLIFFORD;

        // matrix生成
        // FYI fmergeで生成する方法
        // for (UINT i = 0; i < num_qubits; i++) {
        //     QuantumGateBase *swap_gate = gate::SWAP(target_qubit_index1 + i, target_qubit_index2 + i);
        //     ComplexMatrix matrix;
        //     get_extended_matrix(swap_gate, this->_target_qubit_list, this->_control_qubit_list, matrix);
        //     if (i == 0) {
        //         this->_matrix_element = matrix;
        //     } else {
        //         this->_matrix_element = matrix * this->_matrix_element;
        //     }
        // }
        const ITYPE pow2_nq = 1ULL << num_qubits;
        const ITYPE pow2_2nq = 1ULL << (num_qubits * 2);
        this->_matrix_element = SparseComplexMatrix(pow2_2nq, pow2_2nq);
        this->_matrix_element.reserve(pow2_2nq);
        for (ITYPE i = 0; i < pow2_nq; i++) {
            for (ITYPE j = 0; j < pow2_nq; j++) {
                this->_matrix_element.insert(i * pow2_nq + j, i + j * pow2_nq) = 1;
            }
        }
    }
};
