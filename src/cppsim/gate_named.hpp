
#pragma once

#include "gate.hpp"
#include "state.hpp"
#include "utility.hpp"

#ifdef _USE_MPI
#include "csim/MPIutil.h"
#endif


/**
 * \~japanese-en 1量子ビットを対象とする回転角固定のゲートのクラス
 */
class QuantumGate_OneQubit : public QuantumGateBase{
protected:
    typedef void (T_UPDATE_FUNC)(UINT, CTYPE*, ITYPE);
    typedef void (T_UPDATE_FUNC_MPI)(UINT, CTYPE*, ITYPE, UINT);
	typedef void (T_GPU_UPDATE_FUNC)(UINT, void*, ITYPE, void*, UINT);
	T_UPDATE_FUNC* _update_func;
	T_UPDATE_FUNC* _update_func_dm;
	T_UPDATE_FUNC_MPI* _update_func_mpi;
	T_GPU_UPDATE_FUNC* _update_func_gpu;
    ComplexMatrix _matrix_element;

    QuantumGate_OneQubit() {};
public:
    /**
     * \~japanese-en 量子状態を更新する
     * 
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override{
		if (state->is_state_vector()) {
#ifdef _USE_GPU
			if (state->get_device_name() == "gpu") {
				_update_func_gpu(this->target_qubit_list[0].index(), state->data(), state->dim, state->get_cuda_stream(), state->device_number);
			}
			else {
				_update_func(this->_target_qubit_list[0].index(), state->data_c(), state->dim);
			}
#else
#ifdef _USE_MPI
            // index, state, dim
            if (state->inner_qc > this->_target_qubit_list[0].index()) {
                std::cout << "#update qstate-1qubit, inner " << state->inner_qc << ", " << this->_target_qubit_list[0].index() << std::endl;
			    _update_func(this->_target_qubit_list[0].index(), state->data_c(), state->dim);
            } else {
                std::cout << "#update qstate-1qubit, outer " << state->inner_qc << ", " << this->_target_qubit_list[0].index() << std::endl;
                _update_func_mpi(this->_target_qubit_list[0].index(), state->data_c(), state->dim, state->inner_qc);
            }
#else //#ifdef _USE_MPI
			_update_func(this->_target_qubit_list[0].index(), state->data_c(), state->dim);
#endif //#ifdef _USE_MPI
#endif
		}
		else {
            std::cout << "#update qstate-1qubit, dm " << state->inner_qc << ", " << this->_target_qubit_list[0].index() << std::endl;
			_update_func_dm(this->_target_qubit_list[0].index(), state->data_c(), state->dim);
		}
    };
    /**
     * \~japanese-en 自身のディープコピーを生成する
     * 
     * @return 自身のディープコピー
     */
    virtual QuantumGateBase* copy() const override{
        return new QuantumGate_OneQubit(*this);
    };
    /**
     * \~japanese-en 自身のゲート行列をセットする
     * 
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override{
        matrix = this->_matrix_element;
    }
};

/**
 * \~japanese-en 2量子ビットを対象とする回転角固定のゲートのクラス
 */
class QuantumGate_TwoQubit : public QuantumGateBase{
protected:
    typedef void (T_UPDATE_FUNC)(UINT, UINT, CTYPE*, ITYPE);
	typedef void (T_GPU_UPDATE_FUNC)(UINT, UINT, void*, ITYPE, void*, UINT);
	T_UPDATE_FUNC* _update_func;
	T_UPDATE_FUNC* _update_func_dm;
	T_GPU_UPDATE_FUNC* _update_func_gpu;
    ComplexMatrix _matrix_element;

    QuantumGate_TwoQubit() {};
public:
    /**
     * \~japanese-en 量子状態を更新する
     * 
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override{
		if (state->is_state_vector()) {
#ifdef _USE_GPU
			if (state->get_device_name() == "gpu") {
				_update_func_gpu(this->_target_qubit_list[0].index(), this->_target_qubit_list[1].index(), state->data(), state->dim, state->get_cuda_stream(), state->device_number);
			}
			else {
				_update_func(this->_target_qubit_list[0].index(), this->_target_qubit_list[1].index(), state->data_c(), state->dim);
			}
#else //#ifdef _USE_GPU
#ifdef _USE_MPI
            // index x 2, state, dim
            if (state->inner_qc > this->_target_qubit_list[0].index()) {
			    _update_func(this->_target_qubit_list[0].index(), this->_target_qubit_list[1].index(), state->data_c(), state->dim);
            }
#else //#ifdef _USE_MPI
			_update_func(this->_target_qubit_list[0].index(), this->_target_qubit_list[1].index(), state->data_c(), state->dim);
#endif //#ifdef _USE_MPI
#endif //#ifdef _USE_GPU
		}
		else {
			_update_func_dm(this->_target_qubit_list[0].index(), this->_target_qubit_list[1].index(), state->data_c(), state->dim);
		}
	};
    /**
     * \~japanese-en 自身のディープコピーを生成する
     * 
     * @return 自身のディープコピー
     */
    virtual QuantumGateBase* copy() const override{
        return new QuantumGate_TwoQubit(*this);
    };
    /**
     * \~japanese-en 自身のゲート行列をセットする
     * 
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        matrix = this->_matrix_element;
    }
};

/**
 * \~japanese-en 1量子ビットを対象とし1量子ビットにコントロールされる回転角固定のゲートのクラス
 */
class QuantumGate_OneControlOneTarget : public QuantumGateBase {
protected:
    typedef void (T_UPDATE_FUNC)(UINT, UINT, CTYPE*, ITYPE);
	typedef void (T_GPU_UPDATE_FUNC)(UINT, UINT, void*, ITYPE, void*, UINT);
	T_UPDATE_FUNC* _update_func;
	T_UPDATE_FUNC* _update_func_dm;
	T_GPU_UPDATE_FUNC* _update_func_gpu;
    ComplexMatrix _matrix_element;

    QuantumGate_OneControlOneTarget() {};
public:
    /**
     * \~japanese-en 量子状態を更新する
     * 
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override {
		if (state->is_state_vector()) {
#ifdef _USE_GPU
			if (state->get_device_name() == "gpu") {
				_update_func_gpu(this->_control_qubit_list[0].index(), this->_target_qubit_list[0].index(), state->data(), state->dim, state->get_cuda_stream(), state->device_number);
			}
			else {
				_update_func(this->_control_qubit_list[0].index(), this->_target_qubit_list[0].index(), state->data_c(), state->dim);
			}
#else
            // control-index, target-index, data, dim
			_update_func(this->_control_qubit_list[0].index(), this->_target_qubit_list[0].index(), state->data_c(), state->dim);
#endif
		}
		else {
			_update_func_dm(this->_control_qubit_list[0].index(), this->_target_qubit_list[0].index(), state->data_c(), state->dim);
		}
    };
    /**
     * \~japanese-en 自身のディープコピーを生成する
     * 
     * @return 自身のディープコピー
     */
    virtual QuantumGateBase* copy() const override {
        return new QuantumGate_OneControlOneTarget(*this);
    };
    /**
     * \~japanese-en 自身のゲート行列をセットする
     * 
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        matrix = this->_matrix_element;
    }
};

/**
 * \~japanese-en 1量子ビットを対象とする回転ゲートのクラス
 */
class QuantumGate_OneQubitRotation : public QuantumGateBase{
protected:
	typedef void (T_UPDATE_FUNC)(UINT, double, CTYPE*, ITYPE);
	typedef void (T_GPU_UPDATE_FUNC)(UINT, double, void*, ITYPE, void*, UINT);
	T_UPDATE_FUNC* _update_func;
	T_UPDATE_FUNC* _update_func_dm;
	T_GPU_UPDATE_FUNC* _update_func_gpu;
    ComplexMatrix _matrix_element;
    double _angle;

    QuantumGate_OneQubitRotation(double angle) : _angle(angle) {};
public:
    /**
     * \~japanese-en 量子状態を更新する
     * 
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override{
		if (state->is_state_vector()) {
#ifdef _USE_GPU
			if (state->get_device_name() == "gpu") {
				_update_func_gpu(this->_target_qubit_list[0].index(), _angle, state->data(), state->dim, state->get_cuda_stream(), state->device_number);
            }
			else {
				_update_func(this->_target_qubit_list[0].index(), _angle, state->data_c(), state->dim);
			}
#else
            // index, angle, data, dim
			_update_func(this->_target_qubit_list[0].index(), _angle, state->data_c(), state->dim);
#endif
		}
		else {
			_update_func_dm(this->_target_qubit_list[0].index(), _angle, state->data_c(), state->dim);
		}
    };
    /**
     * \~japanese-en 自身のディープコピーを生成する
     * 
     * @return 自身のディープコピー
     */
    virtual QuantumGateBase* copy() const override{
        return new QuantumGate_OneQubitRotation(*this);
    };
    /**
     * \~japanese-en 自身のゲート行列をセットする
     * 
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        matrix = this->_matrix_element;
    }
};


