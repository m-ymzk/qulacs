#include <iostream>
#include <iomanip>
#include <time.h>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/circuit_optimizer.hpp>

double get_realtime(void)
{
    struct timespec t;
    //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    clock_gettime(CLOCK_REALTIME, &t);
    return t.tv_sec + (double)t.tv_nsec*1e-9;
}

void bench(int nqubits, int depth, int lv_opt, int num_dmy, int num_meas){
    double dt;
    double tsum = 0;
    double tsum2 = 0;

    for (int i = 0; i < (num_dmy + num_meas); ++i) {
        dt = -1*get_realtime();

        QuantumState state(nqubits);
        state.set_Haar_random_state();
    
        // Build Circuit
        QuantumCircuit circuit(nqubits);
        for (int i=0; i<nqubits; ++i){
            circuit.add_RX_gate(i, std::rand());
            circuit.add_RZ_gate(i, std::rand());
        }
        for (int i=0; i<nqubits; ++i)
            circuit.add_CNOT_gate(i, (i+1)%nqubits);
        for (int j=0; j<depth; ++j){
            for (int i=0; i<nqubits; ++i){
                circuit.add_RZ_gate(i, std::rand());
                circuit.add_RX_gate(i, std::rand());
                circuit.add_RZ_gate(i, std::rand());
                circuit.add_CNOT_gate(i, (i+1)%nqubits);
            }
        }
        for (int i=0; i<nqubits; ++i){
            circuit.add_RZ_gate(i, std::rand());
            circuit.add_RX_gate(i, std::rand());
        }
    
        // Output information of quantum circuit
        // std::cout << circuit << std::endl;
    
        // Circuit Optimize
        if (lv_opt >= 0){
            QuantumCircuitOptimizer opt;
            if (lv_opt == 0) opt.optimize_light(&circuit);
            else {
                opt.optimize(&circuit, lv_opt);
            }
        }
    
        // Update State
        circuit.update_quantum_state(&state);
    
        dt += get_realtime();
        if (i >= num_dmy){
            tsum += dt;
            tsum2 += dt * dt;
        }
    }
    double tavg = tsum / num_meas;
    double tstd = sqrt(tsum2 / num_meas - tavg * tavg);

    std::cout << "opt= " << lv_opt << ", q= " << nqubits << ", time[s] " << tavg << " +- " << tstd << std::endl;
    return;
}

int main(int argc, char *argv[]){
    int num_dmy = 3;
    int num_meas = 5;
    int st_nq = 10;
    int ed_nq = 20;
    int depth = 9;
    if (argc > 1) st_nq = atoi(argv[1]);
    if (argc > 2) ed_nq = atoi(argv[2]);
    std::cout << "# nqubits: " << st_nq << " ~ " << ed_nq << std::endl;

    std::cout << std::scientific << std::setprecision(9);
    int i = std::max(st_nq, ed_nq);
    std::cout << "# dummy" << std::endl;
    bench(i, 9, -1, 0, 3);

    std::cout << "# start meas" << std::endl;
    int stp=1;
    if (st_nq > ed_nq) stp=-1;
    ed_nq+=stp;
    for (int lv_opt=-1; lv_opt<5; ++lv_opt){
        for (int i=st_nq; i!=ed_nq; i+=stp){
            bench(i, 9, lv_opt, num_dmy, num_meas);
        }
    }
}

