# mpi-qulacs 概説

## base
- qulacs v0.3.0
    - [code(original, github)](https://github.com/qulacs/qulacs.git)
    - [document](http://docs.qulacs.org/ja/latest/index.html)

## 機能
- マルチプロセス、マルチノードで量子状態(state)生成、gateシミュレーション
- state(QuantumState型)インスタンス生成時に、flag "use_multi_cpu=ture"とすることで、起動しているMPIノードで分散配置させる。ただし、 ${N-k} \leqq log_2S$ の場合は分散配置しない。ここで $S$ はMPIランク数、 $N$ は qubit数、 $k$ は、1プロセスあたりの最少qubit数（定数 $k=1$ ）
- 対応関数及び範囲は、制限事項を参照

<hr>

## build/install
### use python3-venv (qenv)
```shell
$ python3 -m venv qenv
$ . ./qenv/bin/activate
$ pip install -U pip wheel (*1)
$ pip install mpi4py (*1, *2)
$ cd [mpi-qulacs]
$ python setup.py install (*1)

*1 要internet接続
*2 fccを使う場合(参考)
   $ MPICC=mpifcc pip install mpi4py
```

## c++/c library build
#### GCC
- 前提条件 (確認済みバージョン)
    - gcc 11.2.0
    - openmpi 4.1.2 (gcc 11.2)
        - configure-option: --with-openib
```shell
<lib. build>
$ cd [mpi-qulacs]
$ ./script/build_gcc.sh

<test>
$ cd build
$ make test

<sample>
$ cd ict
$ make
$ mpirun -n 4 mpiqtest -1 20 0
(USAGE: mpiqtest debug-flag n-qubits target-qubit)
(USAGE: mpiqbench [start n-qubits] [end n-qubit])
```

#### fcc/FCC
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

## 制限事項

- ビルド オプション（fx700で選択される
  | ビルドオプション | MPI-qulacs対応値 | 説明 |
  | -------- | -------- | -------- |
  | _MSC_VAR | False    | windows環境には未対応 |
  | _USE_SIMD | False   | avx2を想定しているため使用しない |
  | _OPENMP  | Ture     | OpenMP有効 |
  | _USE_MPI | Ture     | MPI対応で追加 |

- mpi実行時のランク数（WORLD_SIZE）は2のべき数とすること
- X-gate, CNOT-gate処理において、ノード内stateと同量のメモリを一時的に確保する仮方式となっているため、ノード当たりの最大qubit数は1bit少ない、29 qubit(ComplexTYPE 512M = 8GiB)が最大
- オリジナルqulacsとの機能に差があるAPI
  - QuantumStateインスタンスの作成
    - QuantumState state(qubits)
　    ノード内にstate vectorを作成する。（従来動作）
    - QuantumState state(qubits, use_multi_cpu)
      - use_multi_cpu = false
          ノード内にstate vectorを作成する。（従来動作）
      - use_multi_cpu = true
          可能であれば分散してstate vectorを作成する。
          qubits を内部で inner_qc + outer_qc に分割
        - inner_qc: １ノード内のqubits
        - outer_qc: 分散配置されたqubits (=log2(rank数))
    - state.get_device()
    state vectorの配置されているデバイスを返す。

        | 返り値 | 説明 |
        | -------- | -------- |
        | "cpu"   | ノード内に作成されたstate vector |
        | ("gpu") | mpi-qulacsではサポートしない |
        | "multi-cpu" | 分散配置されたstate vector |

    - state.to_string()
      state情報を出力
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
    - 各要素を乱数で初期化する。
    - 分散配置されたstate vectorの場合
      - seedを指定しない場合でも、rank0の乱数値が全ランクで共有され、seedとして使用する。
      - 指定されたseedもしくは共有されたseedを基に、各rankは (seed + rank番号) をseedとして使用する。
      そのため、分散配置されたstate vectorでは同じseedを設定しても、 **分割数が異なると、作成される状態は異なる。**

  - state.sample( number_sampling [, seed])
    - 他のgate等の操作と同様に、必ず全ランクでcallすること。
    - seedを指定しない場合でも、rank0での乱数値が全ランクで共有(bcast)され、seedとして使用される。
    - seedを指定する場合、全ランクで共通の値を指定すること。

- 動作確認済み機能
  - QuantumState
      - Constructor
      - get_device_name
      - sampling
      - set_computational_basis
      - set_Haar_random_state
      - to_string
  - gate
      - X
      - CNOT
      - Identity / H
      - S / Sdag
      - T / Tdag
      - RX / RY  RZ
      - DenseMatrix(w/o control, single only)

## Example
### Python sample code
```python=
from qulacs import Observable, QuantumCircuit, QuantumState
from qulacs.gate import Y,CNOT,merge

state = QuantumState(3)
state.set_Haar_random_state()

#circuit = QuantumCircuit(3)
circuit = QuantumCircuit(3, use_multi_cpu=True)

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

    //
    // merge gate function is not available in mpi.
    //
    //auto merged_gate = gate::merge(gate::CNOT(0,1),gate::Y(1));
    //circuit.add_gate(merged_gate);
    circuit.add_RX_gate(1,0.5);
    circuit.update_quantum_state(&state);

    // sampling
    //   1st param. is number of sampling.
    //   2nd param. is random-seed.
    // You must call state.sampling on every mpi-ranks.
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

- 対応予定関数・機能
  - gate
      - Y / Z
      - SqrtX / SqrtXdag
      - SqrtY /SqrtYdag
      - P0 / P1
      (gate_named_one.hpp)
      - U1 / U2 / U3
      (QuantumGateMatrix)
      - CZ
      - SWAP
      - DenseMatrix(multi target-qubits)
      - DenseMatrix(with control)
      - Pauli
      - PauliRotation
      - DenseMatrix
      - DiagonalMatrix
      - TOFFOLI
  - QuantumState
      - normalize
      - copy
      - load
      - get_vector
  - Observable
  - QuantumCircuit
  - QuantumCircuitOptimizer
  - QuantumCircuitSimulator
  - state
      - inner_product
      - tensor_product
      - permutate_qubit
      - drop_qubit
      - partial_trace

- 対応予定が未定な関数・機能
  - gate
      - FREDKIN
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
      - Measurement
      - merge
      - add
      - to_matrix_gate
      - Probabilistic
      - ProbabilisticInstrument
      - CPTP
      - CP
      - Instrument
      - Adaptive
  - DensityMatrix (simulation)
  - GeneralQuantumOperator
  - ParametricQuantumCircuit
  - PauliOperator
  - QuantumGateBase
  - QuantumGateMatrix
  - QuantumGate_SingleParameter

- [github(mpi-qulacs)](https://github.com/m-ymzk/mpi-qulacs.git)

