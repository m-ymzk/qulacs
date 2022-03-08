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

    if (max_block_size > 0 && swap_level > 1) {
        std::cerr
            << "Warning: QuantumCircuit::QuantumCircuitOptimizer::optimize(circuit, max_block_size, swap_level) "
               ": using both gate merge and swap optimization is not tested well"
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

void QuantumCircuitOptimizer::optimize_light(QuantumCircuit* circuit) {
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


bool QuantumCircuitOptimizer::use_outer_qubits(UINT gate_index, std::vector<UINT> &qubit_table){
    for (auto target_idx : circuit->gate_list[gate_index]->get_target_index_list()) {
        #if 0
        UINT phy_target_idx = qubit_table[target_idx];
        #else
        auto it = std::find(qubit_table.begin(), qubit_table.end(), target_idx);
        UINT phy_target_idx = std::distance(qubit_table.begin(), it);
        #endif
        if (phy_target_idx >= inner_qc) {
            return true;
        }
    }

    return false;
}

std::unordered_set<UINT> QuantumCircuitOptimizer::find_next_inner_qubits(UINT start_gate_idx, const std::vector<UINT> cur_qubit_table)
{
    MPIutil m = get_mpiutil();
    int rank = m->get_rank();

    if(rank==0){
        std::cout<<"cur table: ";
        for(auto idx : cur_qubit_table){
            std::cout<<idx<<",";
        }
        std::cout << std::endl;
    }

    // 使用予定のqubitの集合 (logical index)
    std::unordered_set<UINT> used_idx;

    for (UINT gate_idx = start_gate_idx; gate_idx < circuit->gate_list.size(); gate_idx++) {
        auto target_idx_list = circuit->gate_list[gate_idx]->get_target_index_list();

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

    if(rank==0){
        std::cout<<"next inner qubits: ";
        for(UINT i = 0; i < circuit->qubit_count; i++){
            if (used_idx.find(i) != used_idx.end()) {
                std::cout<<i<<",";
            }
        }
        std::cout << std::endl;
    }

    return used_idx;
}

UINT QuantumCircuitOptimizer::insert_swaps(const UINT gate_idx, const std::vector<UINT> cur_qubit_table, std::unordered_set<UINT> next_inner_qubits, std::vector<UINT>& next_qubit_table) {
    UINT num_inserted_gates = 0;

    // 現在のqubit table をコピー
    next_qubit_table.clear();
    next_qubit_table.insert(next_qubit_table.end(), cur_qubit_table.begin(), cur_qubit_table.end());


    MPIutil m = get_mpiutil();
    int rank = m->get_rank();
    //ここでcurからnextのtableに変わるように複数swapと1fswapを入れる

    UINT cur_gate_idx = gate_idx;


    std::unordered_set<UINT> cur_inner_qubits(cur_qubit_table.begin(), cur_qubit_table.begin()+inner_qc);
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
        if(rank == 0) std::cout << "error. import_qubits.size() > export_qubits.size()" << std::endl;
    }
    if (inner_qc < fswap_width) {
        if(rank == 0) std::cout << "error. inner_qubits < fswap_width" << std::endl;
    }

    UINT fswap_outer = 0;
    for (UINT i = inner_qc; i < circuit->qubit_count; i++) {
        if(import_qubits.find(next_qubit_table[i]) != import_qubits.end()) {
            fswap_outer = i;
            break;
        }
    }
    if (fswap_outer == 0) {
        if(rank == 0) std::cout << "error fswap_outer=0" << std::endl;
    }
    bool is_fswap_outer_contiguous = true;
    for (UINT i = fswap_outer; i < fswap_outer + fswap_width; i++){
        if(import_qubits.find(next_qubit_table[i]) == import_qubits.end()) {
            is_fswap_outer_contiguous = false;
        }
    }
    if (is_fswap_outer_contiguous == false) {
        if(rank == 0) std::cout << "error. currently fswap outer qubits must be contiguous." << std::endl;

        //TODO outer qubit間での並び替えに対応したらswapを入れる
    }

    if(rank == 0) std::cout << "debug 4. " << import_qubits.size() << " " << exportable_qubits.size() << std::endl;
    if(rank==0){
        std::cout<<"import qubits: ";
        for(UINT i = 0; i < circuit->qubit_count; i++){
            if (import_qubits.find(i) != import_qubits.end()) {
                std::cout<<i<<",";
            }
        }
        std::cout << std::endl;
    }
    if(rank==0){
        std::cout<<"exportable qubits: ";
        for(UINT i = 0; i < circuit->qubit_count; i++){
            if (exportable_qubits.find(i) != exportable_qubits.end()) {
                std::cout<<i<<",";
            }
        }
        std::cout << std::endl;
    }
    
    UINT fswap_inner = inner_qc - fswap_width;
    // export するqubitをinnerの最後にまとめる
    for (int i = inner_qc - 1; i >= (int)fswap_inner; i--) {
        if (exportable_qubits.find(next_qubit_table[i]) == exportable_qubits.end()) {
            // fswap対象のqubitがexportableでないのでswapを入れる
            bool swap_target_found = false;
            for (int j = inner_qc - fswap_width - 1; j >= 0; j--) {
                if (exportable_qubits.find(next_qubit_table[j]) != exportable_qubits.end()) {
                    // swapを行う
                    swap_target_found = true;
                    UINT tmp = next_qubit_table[j];
                    next_qubit_table[j] = next_qubit_table[i];
                    next_qubit_table[i] = tmp;
                    circuit->add_gate(gate::SWAP(j, i), cur_gate_idx);
                    cur_gate_idx++;
                    num_inserted_gates++;
                    break;
                }
            }
            if (swap_target_found == false) {
                if(rank == 0) std::cout << "error. not enougth exportable qubits" << std::endl;
            }
        }
    }

    // fswap追加
    circuit->add_gate(gate::BSWAP(fswap_inner, fswap_outer, fswap_width), cur_gate_idx);
    for (UINT i = 0; i < fswap_width; i++) {
        UINT tmp = next_qubit_table[fswap_inner + i];
        next_qubit_table[fswap_inner + i] = next_qubit_table[fswap_outer + i];
        next_qubit_table[fswap_outer + i] = tmp;
    }
    cur_gate_idx++;
    num_inserted_gates++;

    if(rank==0){
        std::cout<<"next table: ";
        for(auto idx : next_qubit_table){
            std::cout<<idx<<",";
        }
        std::cout << std::endl;
    }

    return num_inserted_gates;
}

void QuantumCircuitOptimizer::rewrite_qubits_index(const UINT gate_idx, std::vector<UINT> &qubit_order) {
    // indexを書き換える
    std::cout << "rewrite_qubits_index #" << gate_idx << std::endl;

    std::vector<UINT> qubit_table;
    for (UINT i = 0; i < circuit->qubit_count; i++) {
        auto it = std::find(qubit_order.begin(), qubit_order.end(), i);
        UINT phy_target_idx = std::distance(qubit_order.begin(), it);
        qubit_table.push_back(phy_target_idx);
    }

    circuit->gate_list[gate_idx]->rewrite_qubits_index(qubit_table);
}

void QuantumCircuitOptimizer::add_swaps_to_reorder(std::vector<UINT> &qubit_table) {
    // qubitを昇順に並び替えるためのswapをcircuitの最後に追加する

    // TODO fswapを活用する。現在はswapのみのナイーブ実装
    for (UINT i = 0; i < circuit->qubit_count; i++) {
        if (qubit_table[i] != i) {
            auto it = std::find(qubit_table.begin()+i, qubit_table.end(), i);
            if (it == qubit_table.end()) {
                std::cout << "invalid qubit_table" << std::endl;
            }
            auto j = std::distance(qubit_table.begin(), it);

            UINT tmp = qubit_table[j];
            qubit_table[j] = qubit_table[i];
            qubit_table[i] = tmp;
            circuit->add_gate(gate::SWAP(i, j));
        }
    }
}

void QuantumCircuitOptimizer::insert_fswap(UINT level) {
    if (level == 0) {
        return;
    }

#ifdef _USE_MPI
    MPIutil mpiutil = get_mpiutil();
    UINT mpisize = mpiutil->get_size();
    UINT mpirank = mpiutil->get_rank();

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

    if (outer_qc == 0) {
        std::cerr
            << "Error: QuantumCircuit::QuantumCircuitOptimizer::insert_fswap(level) "
               ": insert_swap is no effect when MPI size = 1"
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

    if (mpirank == 0) {
        std::cout << "insert_fswap" << std::endl;
    }

    UINT num_gates = circuit->gate_list.size();

    std::vector<UINT> initial_qubit_table(circuit->qubit_count);
    for (UINT i = 0; i < circuit->qubit_count; i++) {
        initial_qubit_table[i] = i;
    }

    std::vector<UINT> cur_qubit_table = initial_qubit_table;

    for (UINT gate_idx = 0; gate_idx < num_gates; gate_idx++) {
        if (mpirank == 0) std::cout << "processing gate #" << gate_idx << std::endl;
        if (use_outer_qubits(gate_idx, cur_qubit_table)) {
            std::unordered_set<UINT> next_inner_qubits = find_next_inner_qubits(gate_idx, cur_qubit_table);
            std::vector<UINT> next_qubit_table;
            UINT num_inserted_gates = insert_swaps(gate_idx, cur_qubit_table, next_inner_qubits, next_qubit_table);
            cur_qubit_table = next_qubit_table;
            gate_idx += num_inserted_gates;
            num_gates += num_inserted_gates;

        }
        if (mpirank == 0) std::cout << "rewrite_qubits_index #" << gate_idx << std::endl;
        //circuit->gate_list[gate_idx]->rewrite_qubits_index(cur_qubit_table);
        rewrite_qubits_index(gate_idx, cur_qubit_table);
    }

    //最初の順序に戻す
    add_swaps_to_reorder(cur_qubit_table);
}
