# mpi-qulacs 概説 [or fast-qulacs, distibuted-qulacs]
## base
- qulacs v0.3.0 https://github.com/qulacs/qulacs.git

## 機能
- マルチプロセス、マルチノードで量子状態(state)生成、gateシミュレーション
- state(QuantumState型)のインスタンス生成時に、MPI communicatorを渡し、全MPIランクで分散配置させる。MPIランク数よりも小さなqubit(N)の場合、すなわち 2^(N - k) < MPI_sizeの場合は分散配置しない。ここでkは、1プロセスあたりの最少qubit数で、定数（現状：k=0）
- 対応関数及び範囲は、制限事項を参照

<hr>

## build/install
### pip, setup.py (TBD)
```bash
$ python setup.py install
```

### source build
```bash
（proxyが必要な環境の場合）
$ export https_proxy=${PROXY}
$ export http_proxy=${https_proxy}

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
$ make
$ mpirun -n 4 mpiqtest 20

（qulacsのpython libraryとしてのインストールの場合：現状fcc and/or MPI版は未サポート）
$ cd [qulacs-rep. home]
$ python setup.py install
```
<hr>

## 制限事項
- 上記以外の関数は未対応または動作未確認
- mpiexecでの実行時に指定できるランク数は2のべき数のみに対応
- 現状の対応範囲は以下の通り
  - QuantumStateインスタンスの作成
    - QuantumState state(qubits)
    プロセス内のメモリに作成
    - QuantumState state(qubits, MPI_COMM)
    可能な限りMPIプロセス数で分散して作成
       - inner_qubits: １ノード内のqubits
       - outer_qubits: 分散配置されたqubits (=log2(rank数))

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
  - state.set_Haar_random_state();
各要素を乱数で初期化する。(norm=1)
ToDo: MPI実行時、rank毎にnormを出してしまっている。
ToDo: 乱数をseedとして設定しているが、rank間で被る可能性あり。（内部でrankを足してしまうような仕様が良いかも？）
  - state.set_Haar_random_state(seed);
各要素を乱数で初期化する。(norm=1)
ToDo: MPI実行時、rank毎にnormを出してしまっている。
ToDo: mpiの各ランクで同じseedとならないように、seed + rankを推奨。（内部でrankを足してしまうような仕様が良いかも？）

  - gate_X
    ex. circuit.add_X_gate(qb);
      - ノード内qubit(qb<=inner_qubits)は従来と同じ動きとなる
      - 分散配置されたqubit(qb>inner_qubits)
      rank番号の2bit表記：0b0000_0000

  - gate_S, gate_Sdag, gate_T, gate_Tdag


