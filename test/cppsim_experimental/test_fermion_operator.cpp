#include <gtest/gtest.h>

#include <cppsim_experimental/fermion_operator.hpp>
#include <cppsim_experimental/observable.hpp>

TEST(FermionOperatorTest, GetTermCountTest) {
    FermionOperator fermion_operator;
    fermion_operator.add_term(1.0, "2^ 1");
    EXPECT_EQ(1, fermion_operator.get_term_count());

    fermion_operator.add_term(2.0, "5^ 4 3^");
    EXPECT_EQ(2, fermion_operator.get_term_count());
}

TEST(FermionOperatorTest, AddGetTermTest) {
    FermionOperator fermion_operator;
    fermion_operator.add_term(1.0, "2^ 1");
    fermion_operator.add_term(-2.0, "5^ 4 3^");
    SingleFermionOperator op("5^ 4 3^");
    fermion_operator.add_term(-2.0, op);

    // AddしたものとGetしたものが一致することを確認する
    auto term1 = fermion_operator.get_term(0);
    EXPECT_EQ(1.0, term1.first);
    EXPECT_EQ(2, term1.second.get_target_index_list().at(0));
    EXPECT_EQ(ACTION_CREATE_ID, term1.second.get_action_id_list().at(0));
    EXPECT_EQ(1, term1.second.get_target_index_list().at(1));
    EXPECT_EQ(ACTION_DESTROY_ID, term1.second.get_action_id_list().at(1));

    auto term2 = fermion_operator.get_term(1);
    EXPECT_EQ(-2.0, term2.first);
    EXPECT_EQ(5, term2.second.get_target_index_list().at(0));
    EXPECT_EQ(ACTION_CREATE_ID, term2.second.get_action_id_list().at(0));
    EXPECT_EQ(4, term2.second.get_target_index_list().at(1));
    EXPECT_EQ(ACTION_DESTROY_ID, term2.second.get_action_id_list().at(1));
    EXPECT_EQ(3, term2.second.get_target_index_list().at(2));
    EXPECT_EQ(ACTION_CREATE_ID, term2.second.get_action_id_list().at(2));

    // SingleFermionOperatorを用いてAddした場合と、文字列を用いてAddした場合で同じ結果
    // が得られることを確認する
    auto term3 = fermion_operator.get_term(2);
    EXPECT_EQ(term3.first, term2.first);
    EXPECT_EQ(term3.second.get_target_index_list().at(0), term2.second.get_target_index_list().at(0));
    EXPECT_EQ(term3.second.get_action_id_list().at(0), term2.second.get_action_id_list().at(0));
    EXPECT_EQ(term3.second.get_target_index_list().at(1), term2.second.get_target_index_list().at(1));
    EXPECT_EQ(term3.second.get_action_id_list().at(1), term2.second.get_action_id_list().at(1));
    EXPECT_EQ(term3.second.get_target_index_list().at(2), term2.second.get_target_index_list().at(2));
    EXPECT_EQ(term3.second.get_action_id_list().at(2), term2.second.get_action_id_list().at(2));
}

TEST(FermionOperatorTest, RemoveTermTest) {
    FermionOperator fermion_operator;
    fermion_operator.add_term(1.0, "2^ 1");
    fermion_operator.add_term(-2.0, "5^ 4 3^");
    fermion_operator.add_term(3.0, "6 7^");

    EXPECT_EQ(3, fermion_operator.get_term_count());
    
    // Removeした結果、Termは1個減る
    fermion_operator.remove_term(1);
    EXPECT_EQ(2, fermion_operator.get_term_count());

    auto term = fermion_operator.get_term(1);
    EXPECT_EQ(3.0, term.first);
    EXPECT_EQ(6, term.second.get_target_index_list().at(0));
    EXPECT_EQ(ACTION_DESTROY_ID, term.second.get_action_id_list().at(0));
    EXPECT_EQ(7, term.second.get_target_index_list().at(1));
    EXPECT_EQ(ACTION_CREATE_ID, term.second.get_action_id_list().at(1));
}

