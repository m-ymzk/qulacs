# mpi-qulacs 概説

## base
- qulacs v0.3.0
    - [code(original, github)](https://github.com/qulacs/qulacs.git)
    - [code(mpi-qulacs, github)](https://github.com/m-ymzk/mpi-qulacs.git)
    - [document](http://docs.qulacs.org/ja/latest/index.html)

## 機能
- マルチプロセス、マルチノードで量子状態(state)生成、gateシミュレーション
- state(QuantumState型)のインスタンス生成時に、MPI communicatorを渡し、全MPIランクで分散配置させる。MPIランク数( $S$ )よりも小さなqubit( $N$ )の場合、すなわち ${N-k} \leqq log_2S$ の場合は分散配置しない。ここで $k$ は、1プロセスあたりの最少qubit数で、ソースで定義された定数（現状：$k=1$ ）
- 対応関数及び範囲は、制限事項を参照

<hr>

## build/install
### pip, setup.py (現状fcc and/or MPI版は未サポート)
```bash
$ python setup.py install
```

### source build (gcc)
- 前提条件 (確認済みバージョン)
    - gcc 11.2.0 (source-build)
    - openmpi 4.1.2 (source-build)
        - configure-option: --with-openib
```
<lib. build>
$ cd qulacs
$ ./script/build_gcc.sh

<test>
$ cd build
$ make test

<samples>
$ cd ict
$ usefcc=1 make
$ mpirun -n 4 mpiqtest -1 20 0
```

### source build (fcc)
```bash
## make venv
$ python3 -m venv venv
$ . ./venv/bin/activate
$ pip install -U pip wheel

（富士通コンパイラ環境の有効化）
$ export TCSDS_PATH="/opt/FJSVstclanga/v1.1.0"
$ export PATH=${TCSDS_PATH}/bin:$PATH
$ export LD_LIBRARY_PATH="${TCSDS_PATH}/lib64:${LD_LIBRARY_PATH}"

(MPI4PYのインストール)
$ MPICC=mpifcc pip install mpi4py

## qulacs ライブラリのbiuld
$ git clone https://github.com/qulacs/qulacs.git
$ cd qulacs
$ ./script/build_fcc.sh

## sample by ict-labs
$ cd ict
$ usefcc=1 make
$ mpirun -n 4 mpiqtest -1 20 0
```
<hr>

## 制限事項

- ビルド時のオプション：
  | ビルドオプション | MPI-qulacs対応値 | 説明 |
  | -------- | -------- | -------- |
  | _MSC_VAR | False    | windows環境には未対応 |
  | _USE_SIMD | False   | avx2を想定しているため使用しない |
  | _OPENMP  | Ture     | OpenMP有効 |
  | _USE_MPI | Ture     | MPI対応で追加 |

- 現状、pythonからのインタフェースには未対応（MPI-Communicator型を渡すとエラーになる。参考になりそうなOSSが見つかっているが、未着手）
- mpiexecでの実行時に指定できるランク数は2のべき数のみに対応
- X-gate, CNOT-gate処理において、ノード内stateと同量のメモリを一時的に確保する仮方式となっているため、ノード当たりの最大qubit数は1bit少ない、29 qubit(ComplexTYPE 512M = 8GiB)が最大
- 現状の対応範囲は以下の通り
  - QuantumStateインスタンスの作成
    - QuantumState state(qubits)
    プロセス内のメモリに作成
    - QuantumState state(qubits, MPI_COMM)
    可能な限りMPIプロセス数で分散して作成する
    qubits => inner_qb + outer_qb に分割
       - inner_qb: １ノード内のqubits
       - outer_qb: 分散配置されたqubits (=log2(rank数))

    - state.to_string()
      state情報を出力
        ```
        ex.
        *** Quantum State ***
        * MPI rank / size : 0 / 4
        * Qubit Count : 20 (inner / outer : 18 / 2 )
        * Dimension   : 262144
        * state vector is too long, so the first 128 elements are output.
        * State vector :
          (0,0)
          ...
        ```
  - state.set_Haar_random_state() の注意事項
各要素を乱数で初期化する。
注意事項：内部で乱数をseedとして設定。ただし、マルチプロセス実行の場合、プロセス間で同一となる可能性があるため、使用は推奨しない。
  - state.set_Haar_random_state(seed) の注意事項
各要素を乱数で初期化する。
注意事項：内部で、(seed + rank)を設定。よって、分割数が異なる場合、同じseedを設定しても、作成される状態は異なる。

- 対応済みリスト（動作するもの全てではありません）
  - QuantumState
      - Constructor (with MPI-Communicator)
      - get_device_name : "cpu" or "cpu-multi"
      - sampling
      - set_computational_basis
      - set_Haar_random_state
      - to_string (each rank output)
  - gate
      - CNOT
      - H
      - Identity
      - S / Sdag
      - T / Tdag
      - X
      - RX / RY  RZ
      - DenseMatrix(w/o control, single only)

- 未対応、未確認関数リスト（全てではありません）
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
 （ここまでで、tk_to_qulacsサポートOK?）
      - TOFFOLI
      - FREDKIN
      - Pauli
      - PauliRotation
      - DenseMatrix
      - SparseMatrix
      - DiagonalMatrix
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
  - QuantumState
      - normalize
      - copy
      - load
      - get_vector
      - sampling
  - DensityMatrix
  - GeneralQuantumOperator
  - Observable
  - ParametricQuantumCircuit
  - PauliOperator
  - QuantumGateBase
  - QuantumGateMatrix
  - QuantumGate_SingleParameter
  - QuantumCircuit
  - QuantumCircuitOptimizer
  - QuantumCircuitSimulator
  - state
      - inner_product
      - tensor_product
      - permutate_qubit
      - drop_qubit
      - partial_trace

## Example
### Python sample code
### C++ sample code
```cpp
#include <iostream>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>

int main(){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    QuantumState state(3, MPI_COMM_WORLD);
    state.set_Haar_random_state();

    QuantumCircuit circuit(3);
    circuit.add_X_gate(0);
    //auto merged_gate = gate::merge(gate::CNOT(0,1),gate::Y(1));
    //circuit.add_gate(merged_gate);
    circuit.add_RX_gate(1,0.5);
    circuit.update_quantum_state(&state);

    //Observable observable(3);
    //observable.add_operator(2.0, "X 2 Y 1 Z 0");
    //observable.add_operator(-3.0, "Z 2");
    //auto value = observable.get_expectation_value(&state);
    //std::cout << value << std::endl;
    return 0;
}
```
