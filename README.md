# mpi-qulacs General info

## Base
- qulacs v0.3.0
    - [code(original, github)](https://github.com/qulacs/qulacs.git)
    - [document](http://docs.qulacs.org/en/latest/index.html)

## Functionality
- Quantum state generation & gate simulation with multi-process and multi-nodes
- MPI-Qulacs distributes a state (QuantumState) when it is instantiated and flag "use_multi_cpu=true" is enabled.
  - However, in the case ${N-k} \leqq log_2S$, the flag is ignored.
  - $S$ is MPI rank, $N$ is the number of qubits, $k$ is the min number of qubit per process （$k=1$ constant）
- Please also see Limitation

<hr>

## Limitation

- The number of MPI rank (WORLD_SIZE) should be $2^n$
- Unsupported gates/functions may cause severe error.
- "device=gpu" not supported

- The following items are supported. MPI-Qulacs does not support any other items.
  - QuantumState
      - Constructor
      - get_device_name
      - sampling
      - set_computational_basis
      - set_Haar_random_state
      - to_string
  - gate
      - X / Y / Z
      - CNOT / CZ / SWAP
      - Identity / H
      - P0 / P1
      - RX / RY / RZ
      - S / Sdag / T / Tdag
      - SqrtX / SqrtXdag / SqrtY / SqrtYdag
      - U1 / U2 / U3
      - DenseMatrix(single target)
      - DenseMatrix(single control, single target)
      - DiagonalMatrix(single target)

- To be supported after March (T.B.D.)
  - gate
      - Measurement
      - Pauli
      - PauliRotation
      - DiagonalMatrix(with control)
      - to_matrix_gate
  - Observable
  - QuantumCircuit
  - QuantumState
      - normalize
      - copy
      - load
      - get_vector
  - ParametricQuantumCircuit
  - PauliOperator

## Additional info
- To be supported after April (T.B.D.)
  - gate
      - TOFFOLI
      - FREDKIN
      - DenseMatrix(double target)
      - DenseMatrix(multi control, single target)
      - DiagonalMatrix(multi target)
      - merge
      - CPTP
      - Instrument
      - Adaptive
  - QuantumCircuitOptimizer
  - QuantumCircuitSimulator
  - state
      - inner_product
      - tensor_product
      - permutate_qubit
      - drop_qubit
      - partial_trace

- Might be supported in future (T.B.D.)
  - gate
      - DenseMatrix(multi target)
      - DenseMatrix(single control, multi target)
      - DenseMatrix(multi control, multi target)
      - SparseMatrix
      - RandomUnitary
      - ReversibleBoolean
      - StateReflection
      - BitFlipNoise
      - DephasingNoise
      - IndependentXZNoise
      - DepolarizingNoise
      - TwoQubitDepolarizingNoise
      - AmplitudeDampingNoise
      - add
      - Probabilistic
      - ProbabilisticInstrument
      - CP
  - DensityMatrix (simulation)
  - GeneralQuantumOperator
  - QuantumGateBase
  - QuantumGateMatrix
  - QuantumGate_SingleParameter

- API which has different functionality from the original
  - Instantiation of QuantumState
    - QuantumState state(qubits, use_multi_cpu)
      - use_multi_cpu = false
        -  Generate state vector in a node (same as the original)
      - use_multi_cpu = true
        -  Generate a state vector in multiple nodes if possible.
        -  qubits are divided into inner_qc + outer_qc internally.
            - inner_qc: qubits in one node
            - outer_qc: qubits in multiple nodes (=log2(#rank))
    - state.get_device()
      - return the list of devices having the state vector.

        | ret value | explanation|
        | -------- | -------- |
        | "cpu"   | state vector generated in a cpu |
        | "multi-cpu" | state vector generated in multi cpu |
        | ("gpu") | Not supported in mpi-qulacs |

    - state.to_string()
      Output state info
        ```
        to_string() example
        -- rank 0 --------------------------------------
         *** Quantum State ***
         * MPI rank / size : 0 / 2
         * Qubit Count : 20 (inner / outer : 19 / 1 )
         * Dimension   : 262144
         * state vector is too long, so the (128 x 2) elements are output.
         * State vector (rank 0):
          (0,0)
          ...
        -- rank 1 --------------------------------------
         * State vector (rank 1):
          (0,0)
          ...
        ```
  - state.set_Haar_random_state()
    - Initialize each item with random value
    - In the case state vector distributed in multi nodes
      - If the seed is not specified, random value in rank0 is broadcasted in all ranks.
      - Based on the specified or broadcasted seed, each rank uses (seed + rank) as a seed. Even if the same seed is set in a distributed state vector, the random created states are different if the number of divisions is different.

  - state.sample( number_sampling [, seed])
    - As the same as gate operation, you must call it in all ranks.
    - Even if a seed is not specified, the random value in rank0 is shared (bcast) and used as a seed.
    - If you specify a seed, use the same one in all ranks.

<hr>

## build/install
### use python3-venv (qenv)
```shell
$ cd [mpi-qulacs]
$ python3 -m venv qenv
$ . ./qenv/bin/activate
$ pip install -U pip wheel (*1)
$ pip install pytest numpy mpi4py (*1, *2)
$ python setup_mpi.py install (*1)

*1 internet required
*2 if you use fcc, Fujitsu C compiler, run below instead (memo)
   $ MPICC=mpifcc pip install mpi4py
```
### test
```
$ pytest
```

## c++/c library build
### GCC
- Prerequisites (Verified version)
    - gcc 11.2
    - openmpi 4.1 (gcc 11.2)
        - configure-option: --with-openib
```shell
<lib. build>
$ cd [mpi-qulacs]
$ . ./setenv
$ ./script/build_mpicc.sh

<test>
$ cd build
$ make test
$ mpirun -n 2 ../bin/csim_test
$ mpirun -n 2 ../bin/cppsim_test
$ mpirun -n 2 ../bin/vqcsim_test

<sample>
$ cd ict
$ make
$ mpirun -n 4 mpiqtest -1 20 0
(USAGE: mpiqtest debug-flag n-qubits target-qubit)
(USAGE: mpiqbench [start n-qubits] [end n-qubit])
```

### fcc/FCC
```shell
$ cd [mpi-qulacs]
$ ./script/build_fcc.sh

<c++ program sample>
$ cd [mpi-qulacs]/ict
$ usefcc=1 make
$ mpirun -n 4 mpiqtest -1 20 0

<python script sample>
$ cd [mpi-qulacs]/ict/python
$ mpirun -n 4 python test.py -n 20
```

<hr>

## Example
### Python sample code
```python=
from qulacs import Observable, QuantumCircuit, QuantumState
from qulacs.gate import Y,CNOT,merge
from mpi4py import MPI

#state = QuantumState(3) # use cpu
state = QuantumState(3, use_multi_cpu=True)
state.set_Haar_random_state()

circuit = QuantumCircuit(3)

circuit.add_X_gate(0)
merged_gate = merge(CNOT(0,1),Y(1))
circuit.add_gate(merged_gate)
circuit.add_RX_gate(1,0.5)
circuit.update_quantum_state(state)

#observable = Observable(3)
#observable.add_operator(2.0, "X 2 Y 1 Z 0")
#observable.add_operator(-3.0, "Z 2")
#value = observable.get_expectation_value(state)
#print(value)
```

### C++ sample code
```cpp=
#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //QuantumState state(3); // same as original-qulacs
    QuantumState state(3, 1); // 1(ture): use_multi_cpu
    //QuantumState state(3, 0); // 0(false): don't use_multi_cpu
    state.set_Haar_random_state();

    QuantumCircuit circuit(3);
    circuit.add_X_gate(0);

    auto merged_gate = gate::merge(gate::CNOT(0, 1),gate::Y(1));
    circuit.add_gate(merged_gate);
    circuit.add_RX_gate(1, 0.5);
    circuit.update_quantum_state(&state);

    // sampling
    //   1st param. is number of sampling.
    //   2nd param. is random-seed.
    // You must call state.sampling on every mpi-ranks
    // with the same random seed.
    std::vector<ITYPE> sample = state.sampling(50, 2021);
    if (rank==0) {
        std::cout << "#result_state.sampling: ";
        for (const auto& e : sample) std::cout << e << " ";
        std::cout << std::endl;
    }

    //
    // observable function is not available in mpi.
    //
    //Observable observable(3);
    //observable.add_operator(2.0, "X 2 Y 1 Z 0");
    //observable.add_operator(-3.0, "Z 2");
    //auto value = observable.get_expectation_value(&state);
    //std::cout << value << std::endl;
    return 0;
}
```
