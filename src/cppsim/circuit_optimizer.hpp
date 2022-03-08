
#pragma once

#include "type.hpp"
#include <unordered_set>

class QuantumCircuit;
class QuantumGateBase;
class QuantumGateMatrix;

/**
 * \~japanese-en 量子回路の圧縮を行うクラス
 *
 * 量子回路の圧縮を行う。
 * 与えらえた量子回路を適切なサイズまで圧縮したり、まとめたゲートに変換するなどの処理を行う。
 */
class DllExport QuantumCircuitOptimizer {
private:
    QuantumCircuit* circuit;
    UINT get_rightmost_commute_index(UINT gate_index);
    UINT get_leftmost_commute_index(UINT gate_index);
    UINT get_merged_gate_size(UINT gate_index1, UINT gate_index2);
    bool is_neighboring(UINT gate_index1, UINT gate_index2);
    void insert_fswap(void);
    bool use_outer_qubits(UINT gate_index, std::vector<UINT> &qubit_table);
    std::unordered_set<UINT> find_next_inner_qubits(UINT start_gate_idx, const std::vector<UINT> cur_qubit_table);
    UINT insert_swaps(const UINT gate_idx, const std::vector<UINT> cur_qubit_table, std::unordered_set<UINT> next_innder_qubits, std::vector<UINT>& next_qubit_table);
    void rewrite_qubits_index(const UINT gate_idx, std::vector<UINT> &qubit_table);
    void add_swaps_to_reorder(std::vector<UINT> &qubit_table);
public:
    /**
     * \~japanese-en コンストラクタ
     */
    QuantumCircuitOptimizer(){};

    /**
     * \~japanese-en デストラクタ
     */
    virtual ~QuantumCircuitOptimizer(){};

    /**
     * \~japanese-en 与えられた量子回路のゲートを指定されたブロックまで纏める。
     *
     * 与えられた量子回路において、若い添え字から任意の二つのゲートを選び、二つが他のゲートに影響を与えず合成可能なら合成を行う。
     * これを合成可能なペアがなくなるまで繰り返す。
     * 二つのゲートが合成可能であるとは、二つのゲートそれぞれについて隣接するゲートとの交換を繰り返し、二つのゲートが隣接した位置まで移動できることを指す。
     *
     * @param[in] circuit 量子回路のインスタンス
     * @param[in] max_block_size 合成後に許されるブロックの最大サイズ
     */
    void optimize(QuantumCircuit* circuit, UINT max_block_size = 2);

    /**
     * \~japanese-en 与えられた量子回路のゲートを指定されたブロックまで纏める。
     *
     * 与えられた量子回路において、若い添え字から任意の二つのゲートを選び、二つが他のゲートに影響を与えず合成可能なら合成を行う。
     * これを合成可能なペアがなくなるまで繰り返す。
     * 二つのゲートが合成可能であるとは、二つのゲートそれぞれについて隣接するゲートとの交換を繰り返し、二つのゲートが隣接した位置まで移動できることを指す。
     *
     * @param[in] circuit 量子回路のインスタンス
     */
    void optimize_light(QuantumCircuit* circuit);

    /**
     * \~japanese-en 量子回路を纏めて一つの巨大な量子ゲートにする
     *
     * @param[in] circuit 量子回路のインスタンス
     * @return 変換された量子ゲート
     */
    QuantumGateMatrix* merge_all(const QuantumCircuit* circuit);
};
