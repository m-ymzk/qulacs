#ifdef _USE_MPI
#include <gtest/gtest.h>
#include "../util/util.h"

#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/pauli_operator.hpp>

TEST(PauliOperatorTest_multicpu,BasicTest) {
	int n = 10;
	double coef = 2.0;
	std::string Pauli_string = "X 0 Y 1 Z 2 I 3 X 4 Y 6 Z 5 I 8 X 7 Y 9";
	PauliOperator pauli = PauliOperator(Pauli_string, coef);
	QuantumState state = QuantumState(n, 1);
	state.set_Haar_random_state();
	CPPCTYPE value = pauli.get_expectation_value(&state);
	ASSERT_NE(value, CPPCTYPE(0,0));
}
#endif // #ifdef _USE_MPI
