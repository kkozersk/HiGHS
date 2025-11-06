#include "Benders.h"
#include "HConst.h"
#include "Highs.h"
#include "HighsInt.h"
#include "HighsStatus.h"
#include "catch.hpp"
#include <cmath>
#include <set>
#include <vector>

const double inf = kHighsInf;

TEST_CASE("test-find-column-index", "[highs_benders]") {
    /*
      Matrix [
        1 1 0
        0 0 1
        0 0 0
        1 0 1
        1 1 1
      ]
    */
    std::vector<HighsInt> csr_starts {0,2,3,3,5,8};
    std::vector<HighsInt> column_indices {0, 0, 1, 3, 3, 4, 4, 4};
    std::vector<HighsInt> results;
    for (int i = 0; i < 8; ++i)
      results.push_back(find_row_index(csr_starts, i));
    REQUIRE(results == column_indices);
}
  
TEST_CASE("test-division", "[highs_benders]") {
  /*
    Matrix [
      1 0
      0 1
      1 1
    ], with first column being a complicating variable
  */
  std::vector<HighsInt> csr_index {0, 1, 0, 1};
  std::vector<HighsInt> csr_starts {0, 1, 2, 4};
  std::set<HighsInt> master_indices {0};
  auto result = divide_rows(csr_index, csr_starts, master_indices);
  REQUIRE(result.master_rows == std::set<HighsInt>{0, 2});
  REQUIRE(result.subproblem_rows == std::set<HighsInt>{1, 2});
  REQUIRE(result.mixed_rows == std::set<HighsInt>{2});
}
  
TEST_CASE("test-division-v2", "[highs_benders]") {
    /*
      Matrix [
        1 1 0
        0 0 1
        0 0 0
        1 0 1
        1 1 1
      ], with complicating variables 0, 1
    */
    std::vector<HighsInt> csr_index {0,1,2,0,2,0,1,2};
    std::vector<HighsInt> csr_starts {0,2,3,3,5,8};
    std::set<HighsInt> master_indices {0, 1};
    auto result = divide_rows(csr_index, csr_starts, master_indices);
    REQUIRE(result.master_rows == std::set<HighsInt>{0, 3, 4});
    REQUIRE(result.subproblem_rows == std::set<HighsInt>{1, 3, 4});
    REQUIRE(result.mixed_rows == std::set<HighsInt>{3, 4});
}

TEST_CASE("test-division-v3", "[highs_benders]") {
    /*
      Matrix [
        ​101001​
         010100​
         100010​
         010101​
         001010​
         100101​
      ], with complicating variables 1, 2
    */
    std::vector<HighsInt> csr_index {0,2,5,1,3,0,4,1,3,5,2,4,0,3,5};
    std::vector<HighsInt> csr_starts {0,3,5,7,10,12,15};
    std::set<HighsInt> master_indices {1, 2};
    auto result = divide_rows(csr_index, csr_starts, master_indices);
    REQUIRE(result.master_rows == std::set<HighsInt>{0, 1, 3, 4});
    REQUIRE(result.subproblem_rows == std::set<HighsInt>{0, 1, 2, 3, 4, 5});
    REQUIRE(result.mixed_rows == std::set<HighsInt>{0, 1, 3, 4});
}

TEST_CASE("test-division-empty-master", "[highs_benders]") {
    /*
      Matrix [
        ​101001​
         010100​
         100010​
         010101​
         001010​
         100101​
      ], with no complicating variables 
    */
    std::vector<HighsInt> csr_index {0,2,5,1,3,0,4,1,3,5,2,4,0,3,5};
    std::vector<HighsInt> csr_starts {0,3,5,7,10,12,15};
    std::set<HighsInt> master_indices {};
    auto result = divide_rows(csr_index, csr_starts, master_indices);
    REQUIRE(result.master_rows == std::set<HighsInt>{});
    REQUIRE(result.subproblem_rows == std::set<HighsInt>{0, 1, 2, 3, 4, 5});
    REQUIRE(result.mixed_rows == std::set<HighsInt>{});
}

TEST_CASE("test-division-full-master", "[highs_benders]") {
    /*
      Matrix [
        ​101001​
         010100​
         100010​
         010101​
         001010​
         100101​
      ], with every variable being complicating
    */
    std::vector<HighsInt> csr_index {0,2,5,1,3,0,4,1,3,5,2,4,0,3,5};
    std::vector<HighsInt> csr_starts {0,3,5,7,10,12,15};
    std::set<HighsInt> master_indices {0, 1, 2, 3, 4, 5};
    auto result = divide_rows(csr_index, csr_starts, master_indices);
    REQUIRE(result.master_rows == std::set<HighsInt>{0, 1, 2, 3, 4, 5});
    REQUIRE(result.subproblem_rows == std::set<HighsInt>{});
    REQUIRE(result.mixed_rows == std::set<HighsInt>{});
}

TEST_CASE("test-division-disjoint", "[highs_benders]") {
    /*
      Matrix [
         1 0
         0 1
      ], with 0 as a complicating variable
    */
    std::vector<HighsInt> csr_index {0, 1};
    std::vector<HighsInt> csr_starts {0, 1, 2};
    std::set<HighsInt> master_indices {0};
    auto result = divide_rows(csr_index, csr_starts, master_indices);
    REQUIRE(result.master_rows == std::set<HighsInt>{0});
    REQUIRE(result.subproblem_rows == std::set<HighsInt>{1});
}

TEST_CASE("test-set-master-values-highs-instance", "[highs-benders]") {
  /*
    Matrix [
      1 0
      0 1
      1 1
    ], with first column being a complicating variable
  */
  std::vector<HighsInt> csr_index {0, 1, 0, 1};
  std::vector<double> csr_values = {1, 1, 1, 1};
  std::vector<HighsInt> csr_starts {0, 1, 2, 4};
  std::set<HighsInt> master_indices {0};
  std::vector<double> master_values {1};
  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 3;
  lp.col_lower_.assign(lp.num_col_, 0);
  lp.col_upper_.assign(lp.num_col_, inf);
  lp.col_cost_.assign(lp.num_col_, 0);
  lp.row_lower_.assign(lp.num_row_, 0);
  lp.row_upper_.assign(lp.num_row_, 0);
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.start_ = csr_starts;
  lp.a_matrix_.index_ = csr_index;
  lp.a_matrix_.value_ = csr_values;
  Highs instance;
  instance.passModel(lp);
  auto result = fix_master_variables(instance, master_indices, master_values);
  auto res_lp  = instance.getLp();
  REQUIRE(result);
  REQUIRE(res_lp.col_lower_ == std::vector<double> {1, 0});
  REQUIRE(res_lp.col_upper_ == std::vector<double> {1, inf});

}

TEST_CASE("test-set-complement", "[highs-benders]") {
  REQUIRE(sequence_complement({0, 1, 2}, 5) == std::set<HighsInt>{3, 4});
  REQUIRE(sequence_complement({0}, 5) == std::set<HighsInt>{1, 2, 3, 4});
  REQUIRE(sequence_complement({}, 5) == std::set<HighsInt>{0, 1, 2, 3, 4});
  REQUIRE(sequence_complement({0, 3}, 2) == std::set<HighsInt>{1});
  REQUIRE(sequence_complement({0, 3}, 0) == std::set<HighsInt>{});
}

TEST_CASE("test-create-master", "[highs-benders]") {
  /*
    Matrix [
      1 1 0
      0 0 1
      0 0 0
      1 0 1
      1 1 1
    ], with complicating variables 0, 1
  */
  std::vector<HighsInt> csr_index {0,1,2,0,2,0,1,2};
  std::vector<double> csr_values(8, 1);
  std::vector<HighsInt> csr_starts {0,2,3,3,5,8};
  HighsLp lp;
  lp.offset_ = 5;
  lp.num_col_ = 3;
  lp.num_row_ = 5;
  lp.col_lower_ = {1, 2, 3};
  lp.col_upper_ = {1, 2, 3};
  lp.col_cost_ = {1, 2, 3};
  lp.row_lower_ = {1, 2, 3, 4, 5};
  lp.row_upper_ = {1, 2, 3, 4, 5};
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.start_ = csr_starts;
  lp.a_matrix_.index_ = csr_index;
  lp.a_matrix_.value_ = csr_values;
  lp.a_matrix_.num_row_ = 5;
  lp.a_matrix_.num_col_ = 3;
  std::set<HighsInt> master_variables {0, 1};

  /*
    We expect:
    min 5 + x0 + 2x1 + z
    s.t.
    [ 1 1 0    [ x0  = 1
      0 0 0 ]    x1  = 3
                 z ] 
      1 <= x0 <=1, 2 <= x1 <= 2, z free
  */
  HighsLp expected;
  expected.offset_ = 5;
  expected.num_col_ = 3;
  expected.num_row_ = 2;
  expected.col_lower_ = {1, 2, -inf};
  expected.col_upper_= {1, 2, inf};
  expected.col_cost_ = {1, 2, 1};
  expected.row_upper_ = {1, 3};
  expected.row_lower_= {1, 3};
  expected.a_matrix_.format_ = MatrixFormat::kRowwise; //TODO: can we leave it this way?
  expected.a_matrix_.start_ = {0, 2, 2};
  expected.a_matrix_.index_ = {0, 1};
  expected.a_matrix_.value_ = {1, 1};
  expected.a_matrix_.num_row_ = 2; //TODO: delete the empty row?
  expected.a_matrix_.num_col_ = 3;


  RowDivision rd;
  rd.master_rows = {0};
  rd.subproblem_rows = {1, 3, 4};
  rd.mixed_rows = {3, 4};
  Highs master;
  create_master_problem(master, lp, master_variables, rd.subproblem_rows);
  auto result = master.getLp();
  result.ensureRowwise();
  REQUIRE(result.a_matrix_ == expected.a_matrix_);
  REQUIRE(result.col_cost_  == expected.col_cost_);
  REQUIRE(result.col_upper_  == expected.col_upper_);
  REQUIRE(result.col_lower_ == expected.col_lower_);
  REQUIRE(result == expected);
}

TEST_CASE("test-create-subproblem", "[highs-benders]") {
  /*
    Matrix [
      1 1 0
      0 0 1
      0 0 0
      1 0 1
      1 1 1
    ], with complicating variables 0, 1
  */
  std::vector<HighsInt> csr_index {0,1,2,0,2,0,1,2};
  std::vector<double> csr_values(8, 1);
  std::vector<HighsInt> csr_starts {0,2,3,3,5,8};
  HighsLp lp;
  lp.offset_ = 5;
  lp.num_col_ = 3;
  lp.num_row_ = 5;
  lp.col_lower_.assign(lp.num_col_, 0);
  lp.col_upper_.assign(lp.num_col_, inf);
  lp.col_cost_.assign(lp.num_col_, 1);
  lp.row_lower_.assign(lp.num_row_, 1);
  lp.row_upper_.assign(lp.num_row_, 1);
  lp.a_matrix_.start_ = csr_starts;
  lp.a_matrix_.index_ = csr_index;
  lp.a_matrix_.value_ = csr_values;
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.num_col_ = 3;
  lp.a_matrix_.num_row_ = 5;
  std::set<HighsInt> master_variables {0, 1};

  HighsLp expected = lp;
  expected.offset_ = 0;
  expected.col_cost_ = {0, 0, 1};
  Highs subproblem;
  create_subproblem(subproblem, lp, master_variables);
  auto result = subproblem.getLp();
  result.ensureRowwise();
  REQUIRE(result  == expected);
}

HighsLp get_simple_test_problem() {
  /*
    Problem
    min 5 + [1 2 -1]^T x
    s.t.
      [1 1 0         = 1
      0 0 1  [x0     <= 1
      0 0 0   x1     = 0
      1 0 1   x2]    <= 2
      1 1 1]         >= 1.5
      x >= 0
  */
  std::vector<HighsInt> csr_index {0,1,2,0,2,0,1,2};
  std::vector<double> csr_values(8, 1);
  std::vector<HighsInt> csr_starts {0,2,3,3,5,8};
  HighsLp lp;
  lp.offset_ = 5;
  lp.num_col_ = 3;
  lp.num_row_ = 5;
  lp.col_lower_ = {0, 0, 0};
  lp.col_upper_ = {inf, inf, inf};
  lp.col_cost_ = {1, 2, -1};
  lp.row_lower_ = {1, -inf, 0, -inf, 1.5};
  lp.row_upper_ = {1, 1, 0, 2, inf};
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.start_ = csr_starts;
  lp.a_matrix_.index_ = csr_index;
  lp.a_matrix_.value_ = csr_values;
  lp.a_matrix_.num_row_ = 5;
  lp.a_matrix_.num_col_ = 3;
  return lp;
}

TEST_CASE("test-get-multipliers", "[highs-benders]") {
  auto lp = get_simple_test_problem();
  std::set<HighsInt> master_variables {0}; //  with complicating variables 0

  Highs subproblem;
  create_subproblem(subproblem, lp, master_variables);

  fix_master_variables(subproblem, master_variables, {0});
  auto status = subproblem.run();
  REQUIRE(status == HighsStatus::kOk);
  REQUIRE(subproblem.getModelStatus() == HighsModelStatus::kOptimal);
  auto multipliers = get_all_multipliers(subproblem);
  REQUIRE(multipliers == std::vector<double> {2, 2, -1});
  auto master_multipliers = get_master_multipliers(subproblem, master_variables);
  REQUIRE(master_multipliers == std::vector<double> {2});

  fix_master_variables(subproblem, master_variables, {2});
  status = subproblem.run();
  REQUIRE(status == HighsStatus::kOk);
  REQUIRE(subproblem.getModelStatus() == HighsModelStatus::kInfeasible);
  bool has_dual_ray;
  double dual_ray[5];
  subproblem.getDualRay(has_dual_ray, dual_ray);
  REQUIRE(has_dual_ray);
  REQUIRE(std::vector<double>(dual_ray, dual_ray + 5) == std::vector<double> {-1, 0, 0, 0, 0});
}

TEST_CASE("test-add-objective-cut", "[highs-benders]") {
  auto lp = get_simple_test_problem();
  std::set<HighsInt> master_variables {0}; //  with complicating variables 0
  BendersProblems problems;
  decompose_problem(problems, lp, master_variables);
  auto & master = problems.master;
  auto & subproblem = problems.subproblem;

  auto num_col = master.getLp().num_col_;
  auto num_row = master.getLp().num_row_;

  std::vector<double> master_values {0};
  fix_master_variables(subproblem, master_variables, master_values);
  auto status = subproblem.run();
  REQUIRE(status == HighsStatus::kOk);
  REQUIRE(subproblem.getModelStatus() == HighsModelStatus::kOptimal);

  add_objective_cut(master, subproblem, master_variables, master_values);
  auto new_master = master.getLp();
  /*
    We want the new master to be in form:
      min 5 + x_0 + z
      0 x_0 + 0 z = 0
      2 x_0 + z >= 1

      x_0 >= 0
    */
  HighsLp expected;
  expected.offset_ = 5;
  expected.num_col_ = 2;
  expected.num_row_ = 2;
  expected.col_lower_ = {0, -inf};
  expected.col_upper_ = {inf, inf};
  expected.col_cost_ = {1, 1};
  expected.row_lower_ = {0, 1};
  expected.row_upper_ = {0, inf};
  expected.a_matrix_.format_ = MatrixFormat::kRowwise;
  expected.a_matrix_.start_ = {0,0,2};
  expected.a_matrix_.index_ = {0,1};
  expected.a_matrix_.value_ = {2,1};
  expected.a_matrix_.num_row_ = 2;
  expected.a_matrix_.num_col_ = 2;
  REQUIRE(new_master.num_row_ == num_row + 1);
  REQUIRE(new_master == expected);
}

TEST_CASE("test-create_nonzero-vector", "[highs-benders]") {
  REQUIRE(create_nonzero_vector({1, 2}) == NonZeroVector {2, {0, 1}, {1, 2}});
  REQUIRE(create_nonzero_vector({1, 1, 1}) == NonZeroVector {3, {0, 1, 2}, {1, 1, 1}});
  REQUIRE(create_nonzero_vector({1, 0, 2, 0, 0, 3, 0}) == NonZeroVector {3, {0, 2, 5}, {1, 2, 3}});
  REQUIRE(create_nonzero_vector({0, 0}) == NonZeroVector {0, {}, {}});
  REQUIRE(create_nonzero_vector({}) == NonZeroVector {0, {}, {}});
 }
