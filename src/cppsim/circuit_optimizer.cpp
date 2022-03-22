#include "circuit_optimizer.hpp"

#include <stdio.h>

#include <algorithm>
#include <iterator>
#include <unordered_set>

#include "circuit.hpp"
#include "gate.hpp"
#include "gate_factory.hpp"
#include "gate_matrix.hpp"
#include "gate_merge.hpp"

UINT QuantumCircuitOptimizer::get_rightmost_commute_index(UINT gate_index) {
    UINT cursor = gate_index + 1;
    for (; cursor < circuit->gate_list.size(); ++cursor) {
        if (!circuit->gate_list[gate_index]->is_commute(
                circuit->gate_list[cursor]))
            break;
    }
    return cursor - 1;
}

UINT QuantumCircuitOptimizer::get_leftmost_commute_index(UINT gate_index) {
    // be careful for underflow of unsigned value
    int cursor = (signed)(gate_index - 1);
    for (; cursor >= 0; cursor--) {
        if (!circuit->gate_list[gate_index]->is_commute(
                circuit->gate_list[cursor]))
            break;
    }
    return cursor + 1;
}

UINT QuantumCircuitOptimizer::get_merged_gate_size(
    UINT gate_index1, UINT gate_index2) {
    auto fetch_target_index =
        [](const std::vector<TargetQubitInfo>& info_list) {
            std::vector<UINT> index_list;
            for (auto val : info_list) index_list.push_back(val.index());
            return index_list;
        };
    auto fetch_control_index =
        [](const std::vector<ControlQubitInfo>& info_list) {
            std::vector<UINT> index_list;
            for (auto val : info_list) index_list.push_back(val.index());
            return index_list;
        };

    auto target_index_list1 =
        fetch_target_index(circuit->gate_list[gate_index1]->target_qubit_list);
    auto target_index_list2 =
        fetch_target_index(circuit->gate_list[gate_index2]->target_qubit_list);
    auto control_index_list1 = fetch_control_index(
        circuit->gate_list[gate_index1]->control_qubit_list);
    auto control_index_list2 = fetch_control_index(
        circuit->gate_list[gate_index2]->control_qubit_list);

    std::sort(target_index_list1.begin(), target_index_list1.end());
    std::sort(target_index_list2.begin(), target_index_list2.end());
    std::sort(control_index_list1.begin(), control_index_list1.end());
    std::sort(control_index_list2.begin(), control_index_list2.end());

    std::vector<UINT> target_index_merge, control_index_merge, whole_index;
    std::set_union(target_index_list1.begin(), target_index_list1.end(),
        target_index_list2.begin(), target_index_list2.end(),
        std::back_inserter(target_index_merge));
    std::set_union(control_index_list1.begin(), control_index_list1.end(),
        control_index_list2.begin(), control_index_list2.end(),
        std::back_inserter(control_index_merge));
    std::set_union(target_index_merge.begin(), target_index_merge.end(),
        control_index_merge.begin(), control_index_merge.end(),
        std::back_inserter(whole_index));
    return (UINT)(whole_index.size());
}

/////////////////////////////////////

bool QuantumCircuitOptimizer::is_neighboring(
    UINT gate_index1, UINT gate_index2) {
    assert(gate_index1 != gate_index2);
    if (gate_index1 > gate_index2) std::swap(gate_index1, gate_index2);
    UINT ind1_right = this->get_rightmost_commute_index(gate_index1);
    UINT ind2_left = this->get_leftmost_commute_index(gate_index2);
    return ind2_left <= ind1_right + 1;
}

void QuantumCircuitOptimizer::optimize(
    QuantumCircuit* circuit_, UINT max_block_size, UINT swap_level) {
    circuit = circuit_;

    if (max_block_size > 0 && swap_level >= 1) {
        std::cerr
            << "Warning: QuantumCircuit::QuantumCircuitOptimizer::optimize(circuit, max_block_size, swap_level) "
               ": using both gate merge and swap optimization is not tested"
            << std::endl;
    }

    insert_fswap(swap_level);

    bool merged_flag = true;
    while (merged_flag) {
        merged_flag = false;
        for (UINT ind1 = 0; ind1 < circuit->gate_list.size(); ++ind1) {
            for (UINT ind2 = ind1 + 1; ind2 < circuit->gate_list.size();
                 ++ind2) {
                // parametric gate cannot be merged
                if (circuit->gate_list[ind1]->is_parametric() ||
                    circuit->gate_list[ind2]->is_parametric())
                    continue;

                // if merged block size is larger than max_block_size, we cannot
                // merge them
                if (this->get_merged_gate_size(ind1, ind2) > max_block_size)
                    continue;

                // if they are separated by not-commutive gate, we cannot merge
                // them
                // TODO: use cache for merging
                if (!this->is_neighboring(ind1, ind2)) continue;

                // generate merged gate
                auto merged_gate = gate::merge(
                    circuit->gate_list[ind1], circuit->gate_list[ind2]);

                // remove merged two gates, and append new one
                UINT ind2_left = this->get_leftmost_commute_index(ind2);
                // insert at max(just after first_applied_gate, just before
                // left-most gate commuting with later_applied_gate ) Insertion
                // point is always later than the first, and always earlier than
                // the second.
                UINT insert_point = std::max(ind2_left, ind1 + 1);

                // Not to change index with removal, process from later ones to
                // earlier ones.
                circuit->remove_gate(ind2);
                circuit->add_gate(merged_gate, insert_point);
                circuit->remove_gate(ind1);

                ind2--;
                merged_flag = true;
            }
        }
    }
}
/*
void QuantumCircuitOptimizer::optimize_light(QuantumCircuit* circuit) {
        UINT qubit_count = circuit->qubit_count;
        std::vector<std::pair<int, std::vector<UINT>>> current_step(qubit_count,
std::make_pair(-1,std::vector<UINT>())); for (UINT ind1 = 0; ind1 <
circuit->gate_list.size(); ++ind1) { QuantumGateBase* gate =
circuit->gate_list[ind1]; std::vector<UINT> target_qubits; for (auto val :
gate->get_target_index_list()) target_qubits.push_back(val); for (auto val :
gate->get_control_index_list()) target_qubits.push_back(val);
                std::sort(target_qubits.begin(), target_qubits.end());
                if (target_qubits.size() == 1) {
                        // merge
                        UINT target_qubit_index = target_qubits[0];
                        UINT target_gate_index =
current_step[target_qubit_index].first; if (target_gate_index != -1) { auto
merged_gate = gate::merge(circuit->gate_list[target_gate_index], gate);
                                circuit->remove_gate(ind1);
                                circuit->add_gate(merged_gate,
target_gate_index+1); circuit->remove_gate(target_gate_index); ind1--;
                        }
                        else {
                                current_step[target_qubit_index] =
std::make_pair(ind1, target_qubits);
                        }
                }
                else {
                        for (auto target_qubit : target_qubits) {
                                current_step[target_qubit] = make_pair(ind1,
target_qubits);
                        }
                }
        }
}
*/

void QuantumCircuitOptimizer::optimize_light(QuantumCircuit* circuit, UINT swap_level) {
    this->circuit = circuit;

    if (swap_level >= 1) {
        std::cerr
            << "Warning: QuantumCircuit::QuantumCircuitOptimizer::optimize_light(circuit, swap_level) "
               ": using both gate merge and swap optimization is not tested"
            << std::endl;
    }

    insert_fswap(swap_level);

    UINT qubit_count = circuit->qubit_count;
    std::vector<std::pair<int, std::vector<UINT>>> current_step(
        qubit_count, std::make_pair(-1, std::vector<UINT>()));
    for (UINT ind1 = 0; ind1 < circuit->gate_list.size(); ++ind1) {
        QuantumGateBase* gate = circuit->gate_list[ind1];
        std::vector<UINT> target_qubits;
        std::vector<UINT> parent_qubits;

        for (auto val : gate->get_target_index_list())
            target_qubits.push_back(val);
        for (auto val : gate->get_control_index_list())
            target_qubits.push_back(val);
        std::sort(target_qubits.begin(), target_qubits.end());

        int pos = -1;
        int hit = -1;
        for (UINT target_qubit : target_qubits) {
            if (current_step[target_qubit].first > pos) {
                pos = current_step[target_qubit].first;
                hit = target_qubit;
            }
        }
        if (hit != -1) parent_qubits = current_step[hit].second;
        if (std::includes(parent_qubits.begin(), parent_qubits.end(),
                target_qubits.begin(), target_qubits.end())) {
            auto merged_gate = gate::merge(circuit->gate_list[pos], gate);
            circuit->remove_gate(ind1);
            circuit->add_gate(merged_gate, pos + 1);
            circuit->remove_gate(pos);
            ind1--;

            // std::cout << "merge ";
            // for (auto val : target_qubits) std::cout << val << " ";
            // std::cout << " into ";
            // for (auto val : parent_qubits) std::cout << val << " ";
            // std::cout << std::endl;
        } else {
            for (auto target_qubit : target_qubits) {
                current_step[target_qubit] = make_pair(ind1, target_qubits);
            }
        }
    }
}

QuantumGateMatrix* QuantumCircuitOptimizer::merge_all(
    const QuantumCircuit* circuit_) {
    QuantumGateBase* identity = gate::Identity(0);
    QuantumGateMatrix* current_gate = gate::to_matrix_gate(identity);
    QuantumGateMatrix* next_gate = NULL;
    delete identity;

    for (auto gate : circuit_->gate_list) {
        next_gate = gate::merge(current_gate, gate);
        delete current_gate;
        current_gate = next_gate;
    }
    return current_gate;
}


QuantumCircuitOptimizer::QubitTable::QubitTable(const UINT nc): _nc(nc),
                                                                _p2l_table(nc), _l2p_table(nc),
                                                                p2l(_p2l_table), l2p(_l2p_table)
{
    std::iota(_p2l_table.begin(), _p2l_table.end(), 0);
    std::iota(_l2p_table.begin(), _l2p_table.end(), 0);
}

QuantumCircuitOptimizer::QubitTable::QubitTable(const QubitTable& qt): _nc(qt._nc),
                                                                       p2l(_p2l_table), l2p(_l2p_table)
{
    std::copy(qt.p2l.begin(), qt.p2l.end(), back_inserter(_p2l_table));
    std::copy(qt.l2p.begin(), qt.l2p.end(), back_inserter(_l2p_table));
}

QuantumCircuitOptimizer::QubitTable& QuantumCircuitOptimizer::QubitTable::operator=(const QubitTable& rhs) {
    _nc = rhs._nc;
    _p2l_table.clear();
    _l2p_table.clear();
    std::copy(rhs.p2l.begin(), rhs.p2l.end(), back_inserter(_p2l_table));
    std::copy(rhs.l2p.begin(), rhs.l2p.end(), back_inserter(_l2p_table));
    return *this;
}

bool QuantumCircuitOptimizer::QubitTable::swap(const UINT i, const UINT j) {
    if (i >= _nc || j >= _nc) {
        std::cerr << "QubitTable::swap() out of qubit range" << std::endl;
        return false;
    }
    if (i == j) {
        return true;
    }
    std::swap(_p2l_table[i], _p2l_table[j]);
    std::swap(_l2p_table[_p2l_table[i]], _l2p_table[_p2l_table[j]]);
    return true;
}

bool QuantumCircuitOptimizer::QubitTable::fswap(const UINT i, const UINT j, const UINT width) {
   if (i + width > _nc || j + width > _nc) {
       std::cerr << "QubitTable::fswap() out of qubit range" << std::endl;
       return false;
   }
   if ((i <= j && i + width > j) || (j < i && j + width > i)) {
       std::cerr << "QubitTable::fswap() overlap range" << std::endl;
       return false;
   }

   for (UINT w = 0; w < width; w++) {
       swap(i + w, j + w);
   }
   return true;
}

std::vector<UINT> QuantumCircuitOptimizer::get_comm_qubits(UINT gate_index) {
    // 通信を必要とするqubitのリストを取得する
    
    auto& gate = circuit->gate_list[gate_index];

    // CZはouterのtarget_qubitを使用しても通信不要
    auto gate_name = gate->get_name();
    if (gate_name == "I" ||
        gate_name == "Z" || gate_name == "Z-rotation" || gate_name == "CZ" ||
        gate_name == "Projection-0" || gate_name == "Projection-1" ||
        gate_name == "S" || gate_name == "Sdag" ||
        gate_name == "T" || gate_name == "Tdag" ||
        gate_name == "DiagonalMatrix") {
        return std::vector<UINT>();
    }
    
    return gate->get_target_index_list();
}

bool QuantumCircuitOptimizer::need_comm(UINT gate_index, QubitTable& qt){
    for (auto target_idx : get_comm_qubits(gate_index)) {
        UINT phy_target_idx = qt.l2p[target_idx];
        if (phy_target_idx >= inner_qc) {
            return true;
        }
    }

    return false;
}

std::unordered_set<UINT> QuantumCircuitOptimizer::find_next_inner_qubits(UINT start_gate_idx)
{
#ifndef NDEBUG
    MPIutil m = get_mpiutil();
    int rank = m->get_rank();
#endif

    // 使用予定のqubitの集合 (logical index)
    std::unordered_set<UINT> used_idx;

    for (UINT gate_idx = start_gate_idx; gate_idx < circuit->gate_list.size(); gate_idx++) {
        auto target_idx_list = get_comm_qubits(gate_idx);

        UINT additional_idx_count = 0;
        // 現在のgateを処理することでqubit集合に追加されるqubitの個数を数える
        // target_idx_listは重複がない前提
        for (auto idx : target_idx_list) {
            if (used_idx.find(idx) == used_idx.end()) {
                additional_idx_count++;
            }
        }
        if (used_idx.size() + additional_idx_count <= inner_qc) {
            // 現在のgateを追加で処理してもinner qubit内に抑えられる
            used_idx.insert(target_idx_list.begin(), target_idx_list.end());
        } else {
            // 現在のgateを追加するとinner qubit内に抑えられないので終了
            break;
        }
    }

#ifndef NDEBUG
    if(rank==0){
        std::cout<<"next inner qubits: ";
        for(UINT i = 0; i < circuit->qubit_count; i++){
            if (used_idx.find(i) != used_idx.end()) {
                std::cout<<i<<",";
            }
        }
        std::cout << std::endl;
    }
#endif

    return used_idx;
}

void QuantumCircuitOptimizer::add_swap_gate(UINT idx0, UINT idx1, UINT width, QubitTable &qt) {
    add_swap_gate(idx0, idx1, width, qt, circuit->gate_list.size());
}
void QuantumCircuitOptimizer::add_swap_gate(UINT idx0, UINT idx1, UINT width, QubitTable &qt, UINT gate_pos) {
#ifndef NDEBUG
    MPIutil m = get_mpiutil();
    int rank = m->get_rank();
    if (rank == 0) std::cout << "add_swap_gate(" << idx0 << "," << idx1 << "," << width << ")" << std::endl;
#endif

    if (width == 1) {
        circuit->add_gate(gate::SWAP(idx0, idx1), gate_pos);
    } else {
        circuit->add_gate(gate::BSWAP(idx0, idx1, width), gate_pos);
    }
    qt.fswap(idx0, idx1, width);
}

UINT QuantumCircuitOptimizer::insert_swaps(const UINT gate_idx, std::unordered_set<UINT> next_inner_qubits, QubitTable &qt) {
    UINT num_inserted_gates = 0;

#ifndef NDEBUG
    MPIutil m = get_mpiutil();
    int rank = m->get_rank();
#endif

    std::unordered_set<UINT> cur_inner_qubits(qt.p2l.begin(), qt.p2l.begin()+inner_qc);
    std::unordered_set<UINT> import_qubits;
    std::unordered_set<UINT> exportable_qubits;

    // import_qubits = next_inner_qubits - cur_inner_qubits
    for (UINT i : next_inner_qubits) {
        if (cur_inner_qubits.find(i) == cur_inner_qubits.end()) {
            import_qubits.insert(i);
        }
    }

    // exportable_qubits = cur_inner_qubits - next_inner_qubits
    for (UINT i : cur_inner_qubits) {
        if (next_inner_qubits.find(i) == next_inner_qubits.end()) {
            exportable_qubits.insert(i);
        }
    }


    UINT fswap_width = import_qubits.size();
    if (fswap_width > exportable_qubits.size()) {
        std::cerr << "Error: QuantumCircuitOptimizer::insert_swaps() "
                     "import_qubits.size() > export_qubits.size()"
                  << std::endl;
        return 0;
    }
    if (inner_qc < fswap_width) {
        std::cerr << "Error. QuantumCircuitOptimizer::insert_swaps() "
                     "inner_qubits < fswap_width"
                  << std::endl;
        return 0;
    }


#ifndef NDEBUG
    if(rank==0){
        std::cout<<"import qubits: ";
        for(UINT i = 0; i < circuit->qubit_count; i++){
            if (import_qubits.find(i) != import_qubits.end()) {
                std::cout<<i<<",";
            }
        }
        std::cout << std::endl;
        std::cout<<"exportable qubits: ";
        for(UINT i = 0; i < circuit->qubit_count; i++){
            if (exportable_qubits.find(i) != exportable_qubits.end()) {
                std::cout<<i<<",";
            }
        }
        std::cout << std::endl;
    }
#endif


    /// importするouter qubitのリストを作成
    std::vector<UINT> fswap_outer_list;
    std::vector<UINT> fswap_width_list;
    for (UINT i = inner_qc; i < circuit->qubit_count; i++) {
        if(import_qubits.find(qt.p2l[i]) != import_qubits.end()) {
            fswap_outer_list.push_back(i);
            fswap_width_list.push_back(1);
        }
    }

    // outer qubitをまとめる
    // TODO outer qubit同士のswapが高速化されたら連続となるようにswapするように変更
    UINT available_ext_inner_qc = exportable_qubits.size() - fswap_width;
    for (UINT i = 0; i < fswap_outer_list.size(); i++) {
        for (UINT j = i + 1; j < fswap_outer_list.size(); ) {
            if (fswap_outer_list[i] + fswap_width_list[i] + available_ext_inner_qc >= fswap_outer_list[j]) {
                UINT ext_inner_qc = fswap_outer_list[j] - (fswap_outer_list[i] + fswap_width_list[i]);
                fswap_width_list[i] = fswap_width_list[i] + fswap_width_list[j] + ext_inner_qc;
                available_ext_inner_qc -= ext_inner_qc;
                fswap_outer_list.erase(fswap_outer_list.begin() + j);
                fswap_width_list.erase(fswap_width_list.begin() + j);
            } else {
                j++;
            }
        }
    }

#ifndef NDEBUG
    if (rank == 0) std::cout << "available_ext_inner_qc = " << available_ext_inner_qc << std::endl;
#endif
    UINT use_ext_inner_qc = exportable_qubits.size() - fswap_width - available_ext_inner_qc;
    fswap_width = fswap_width + use_ext_inner_qc;
    UINT fswap_inner = inner_qc - fswap_width;
    // export するqubitをinnerの最後にまとめる
    for (int i = inner_qc - 1; i >= (int)fswap_inner; i--) {
        if (exportable_qubits.find(qt.p2l[i]) == exportable_qubits.end()) {
            // fswap対象のqubitがexportableでないのでswapを入れる
            bool swap_target_found = false;
            for (int j = inner_qc - fswap_width - 1; j >= 0; j--) {
                if (exportable_qubits.find(qt.p2l[j]) != exportable_qubits.end()) {
                    // swapを行う
                    swap_target_found = true;
                    add_swap_gate(j, i, 1, qt, gate_idx + num_inserted_gates);
                    num_inserted_gates++;
                    break;
                }
            }
            if (swap_target_found == false) {
                std::cerr << "Error: QuantumCircuitOptimizer::insert_swaps() "
                             "no enougth exportable qubits"
                          << std::endl;
            }
        }
    }

    UINT part_fswap_inner = fswap_inner;
    for (UINT i = 0; i < fswap_outer_list.size(); i++) {
        UINT part_fswap_outer = fswap_outer_list[i];
        UINT part_fswap_width = fswap_width_list[i];

        add_swap_gate(part_fswap_inner, part_fswap_outer, part_fswap_width, qt, gate_idx + num_inserted_gates);
        num_inserted_gates++;
        part_fswap_inner += part_fswap_width;
    }

#ifndef NDEBUG
    if(rank==0){
        std::cout<<"next: " << qt << std::endl;
    }
#endif

    return num_inserted_gates;
}

void QuantumCircuitOptimizer::add_swaps_to_reorder_at(QubitTable &qt, const UINT i, const UINT v) {
#ifndef NDEBUG
    MPIutil m = get_mpiutil();
    int rank = m->get_rank();
    if (rank == 0) std::cout << "add_swaps_to_reorder_at(" << i<< "," << v <<")" << std::endl;
#endif

    if (qt.p2l[i] != v) {
        auto it = std::find(qt.p2l.begin(), qt.p2l.end(), v);
        if (it == qt.p2l.end()) {
            std::cerr << "Error: QuantumCircuitOptimizer::add_swaps_to_reorder_at "
                         "invalid qubit_order"
                      << std::endl;
        }
        auto j = std::distance(qt.p2l.begin(), it);

        add_swap_gate(i, j, 1, qt);
    }
}

void QuantumCircuitOptimizer::add_swaps_to_reorder(QubitTable &qt) {
    // qubitを昇順に並び替えるためのswapをcircuitの最後に追加する

    // outer qubitで交換が必要な物を探す
    UINT fswap_outer = 0;
    UINT fswap_width = 0;
    for (UINT i = inner_qc; i < circuit->qubit_count; i++) {
        if (qt.p2l[i] != i) {
            if (fswap_outer == 0) {
                fswap_outer = i;
            }
            fswap_width = i - fswap_outer + 1;
        }
    }

    if (fswap_width > 0) {
        // outer qubitのreorder

        UINT fswap_inner = inner_qc - fswap_width;
        //// fswapのouter側に[fswap_outer, fswap_outer+fswap_width)のqubitがあるかチェック
        bool outerqc_in_outer = false;
        for (UINT i = fswap_outer; i < fswap_outer + fswap_width; i++) {
            if (qt.p2l[i] >= fswap_outer && qt.p2l[i] < fswap_outer + fswap_width) {
                outerqc_in_outer = true;
                break;
            }
        }

        if (!outerqc_in_outer || inner_qc >= fswap_width * 2) {
            // 最大2回のfswapで並べ替える

            if (outerqc_in_outer) {
                // ある場合、fswapのinner側に[fswap_outer, fswap_outer+fswap_width)のqubitが現れないようにinner同士でswap
                int swap_idx = inner_qc - fswap_width - 1;
                for (UINT i = inner_qc - fswap_width; i < inner_qc; i++) {
                    if (qt.p2l[i] >= inner_qc) {
                        if (swap_idx < 0) {
                            std::cerr << "invalid index at QuantumCircuitOptimizer::add_swaps_to_reorder" << std::endl;
                        }
                        add_swap_gate(i, swap_idx, 1, qt);
                        swap_idx--;
                    }
                }
                // fswap
                add_swap_gate(fswap_inner, fswap_outer, fswap_width, qt);
            }

            // fswapのinner側にinner_qc~(qc-1)を集める
            for (UINT i = inner_qc - fswap_width, v = fswap_outer; i < inner_qc; i++, v++) {
                add_swaps_to_reorder_at(qt, i, v);
            }

            // fswap
            add_swap_gate(fswap_inner, fswap_outer, fswap_width, qt);
        } else {
            // ナイーブな1qubitずつの並べ替え
            for (UINT i = inner_qc; i < circuit->qubit_count; i++) {
                add_swaps_to_reorder_at(qt, i, i);
            }
        }
    }

    // inner qubitのreorder
    for (UINT i = 0; i < inner_qc; i++) {
        add_swaps_to_reorder_at(qt, i, i);
    }
}

void QuantumCircuitOptimizer::insert_fswap(UINT level) {
    if (level == 0) {
        return;
    }

#ifdef _USE_MPI
    MPIutil mpiutil = get_mpiutil();
    UINT mpisize = mpiutil->get_size();
#ifndef NDEBUG
    UINT mpirank = mpiutil->get_rank();
#endif

    assert(!(mpisize & (mpisize - 1)));  // mpi-size must be power of 2

    UINT log_nodes = std::log2(mpisize);
    inner_qc = circuit->qubit_count - log_nodes;
    outer_qc = log_nodes;
#else
    std::cerr
        << "Error: QuantumCircuit::QuantumCircuitOptimizer::insert_fswap(level) "
        ": insert_swap is no effect to non MPI build"
        << std::endl;
    return;
#endif

    if (outer_qc == 0 || inner_qc == 0) {
        std::cerr
            << "Error: QuantumCircuit::QuantumCircuitOptimizer::insert_fswap(level) "
               ": insert_swap is no effect when MPI size = 1 or 2^inner_qc"
            << std::endl;
        return;
    }

    if (level > 1) {
        std::cerr
            << "Error: QuantumCircuit::QuantumCircuitOptimizer::insert_fswap(level) "
               ": invalid level. currently supports only level <= 1"
            << std::endl;
        return;
    }

#ifndef NDEBUG
    if (mpirank == 0) {
        std::cout << "insert_fswap" << std::endl;
    }
#endif

    UINT num_gates = circuit->gate_list.size();
    QubitTable qt(circuit->qubit_count);

    for (UINT gate_idx = 0; gate_idx < num_gates; gate_idx++) {
#ifndef NDEBUG
        if (mpirank == 0) std::cout << "processing gate #" << gate_idx << std::endl;
#endif
        if (need_comm(gate_idx, qt)) {
#ifndef NDEBUG
            if(mpirank==0) std::cout<< "cur: " << qt << std::endl;
#endif
            std::unordered_set<UINT> next_inner_qubits = find_next_inner_qubits(gate_idx);
            UINT num_inserted_gates = insert_swaps(gate_idx, next_inner_qubits, qt);
            gate_idx += num_inserted_gates;
            num_gates += num_inserted_gates;

        }
#ifndef NDEBUG
        if (mpirank == 0) std::cout << "rewrite_qubits_index #" << gate_idx << std::endl;
#endif
        circuit->gate_list[gate_idx]->rewrite_qubits_index(qt.l2p);
    }

    //最初の順序に戻す
    add_swaps_to_reorder(qt);
}
