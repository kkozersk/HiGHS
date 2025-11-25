#include "Benders.h"
#include "HCheckConfig.h"
#include "FilereaderLp.h"
#include "HConst.h"
#include "Highs.h"
#include "HighsInt.h"
#include "HighsSolution.h"
#include "HighsStatus.h"
#include "catch.hpp"
#include <cmath>
#include <set>
#include <vector>

const double inf = kHighsInf;

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

HighsLp get_simple_multi_test_problem() {
  /*
    Problem
    min 5 + [1 2 -1 -3]^T x
    s.t.
      [1 1 0 0         = 1
      0 0 1 0  [x0     <= 1
      0 0 0 0  x1      = 0
      1 0 1 0  x2      <= 2
      1 1 1 0  x3]     >= 1.5
      0 0 0 1          <= 1
      1 0 0 1]         = 1.5
      ]          
      x >= 0
  */
  std::vector<HighsInt> csr_index {0,1,2,0,2,0,1,2,3,0,3};
  std::vector<double> csr_values(11, 1);
  std::vector<HighsInt> csr_starts {0,2,3,3,5,8,9,11};
  HighsLp lp;
  lp.offset_ = 5;
  lp.num_col_ = 4;
  lp.num_row_ = 7;
  lp.col_lower_ = {0, 0, 0, 0};
  lp.col_upper_ = {inf, inf, inf, inf};
  lp.col_cost_ = {1, 2, -1, -3};
  lp.row_lower_ = {1, -inf, 0, -inf, 1.5, -inf, 1.5};
  lp.row_upper_ = {1, 1, 0, 2, inf, 1, 1.5};
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.start_ = csr_starts;
  lp.a_matrix_.index_ = csr_index;
  lp.a_matrix_.value_ = csr_values;
  lp.a_matrix_.num_row_ = 7;
  lp.a_matrix_.num_col_ = 4;
  return lp;
}

HighsLp get_second_test_problem() {
  /*
  min  [1 5 3 4 5 -6]^T x
  [
  	1 1  0  0 0 0      <=  4  
  	1 0 -1 0 0 0       <=  0
  	0 2  1 0 1 0   x   >=  3
  	1 1  0 1 0 0       ==  5 
  	0 0  0 0 1 0       <=  2
  	1 0  0 0 0 1       ==  1
  ]
      x >= 0
  */
  std::vector<HighsInt> csr_index {0,1, 0,2, 1,2,4, 0,1,3, 4, 0,5};
  std::vector<double> csr_values {1,1, 1,-1, 2,1,1, 1,1,1, 1, 1,1 };
  std::vector<HighsInt> csr_starts {0,2,4,7,10,11,13};
  HighsLp lp;
  lp.offset_ = 0;
  lp.num_col_ = 6;
  lp.num_row_ = 6;
  lp.col_lower_ = {0, 0, 0, 0, 0, 0};
  lp.col_upper_ = {inf, inf, inf, inf, inf, inf};
  lp.col_cost_ = {1, 5, 3, 4, 5, -6};
  lp.row_lower_ = {-inf, -inf, 3, 5, -inf, 1};
  lp.row_upper_ = {4, 0, inf, 5, 2, 1};
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.start_ = csr_starts;
  lp.a_matrix_.index_ = csr_index;
  lp.a_matrix_.value_ = csr_values;
  lp.a_matrix_.num_row_ = 6;
  lp.a_matrix_.num_col_ = 6;
  return lp;
}
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
  REQUIRE(result.inset_only_rows == std::set<HighsInt>{0, 2});
  REQUIRE(result.other_rows == std::set<HighsInt>{1, 2});
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
    REQUIRE(result.inset_only_rows == std::set<HighsInt>{0, 3, 4});
    REQUIRE(result.other_rows == std::set<HighsInt>{1, 3, 4});
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
    REQUIRE(result.inset_only_rows == std::set<HighsInt>{0, 1, 3, 4});
    REQUIRE(result.other_rows == std::set<HighsInt>{0, 1, 2, 3, 4, 5});
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
    REQUIRE(result.inset_only_rows== std::set<HighsInt>{});
    REQUIRE(result.other_rows == std::set<HighsInt>{0, 1, 2, 3, 4, 5});
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
    REQUIRE(result.inset_only_rows == std::set<HighsInt>{0, 1, 2, 3, 4, 5});
    REQUIRE(result.other_rows == std::set<HighsInt>{});
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
    REQUIRE(result.inset_only_rows == std::set<HighsInt>{0});
    REQUIRE(result.other_rows == std::set<HighsInt>{1});
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
  expected.col_lower_ = {1, 2, mu_lb};
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
  rd.inset_only_rows = {0};
  rd.other_rows = {1, 3, 4};
  rd.mixed_rows = {3, 4};
  Highs master;
  create_master_problem(master, lp, master_variables, rd.other_rows);
  auto result = master.getLp();
  result.ensureRowwise();
  REQUIRE(result.a_matrix_ == expected.a_matrix_);
  REQUIRE(result.col_cost_  == expected.col_cost_);
  REQUIRE(result.col_upper_  == expected.col_upper_);
  REQUIRE(result.col_lower_ == expected.col_lower_);
  REQUIRE(result == expected);
}

// TEST_CASE("test-create-subproblem", "[highs-benders]") {
//   auto lp = get_simple_test_problem();
//   std::set<HighsInt> master_variables {0, 1};

//   HighsLp expected = lp;
//   expected.offset_ = 0;
//   expected.col_cost_ = {0, 0, -1};
//   Highs subproblem;
//   create_subproblem(subproblem, lp, master_variables);
//   auto result = subproblem.getLp();
//   result.ensureRowwise();
//   REQUIRE(result  == expected);
// }

TEST_CASE("test-create-feas-subproblem", "[highs-benders]") {
  /*
    We expect:
    min alpha1_- + alpha2_+
    s.t.
      [1 1 0  0   0            = 1
       0 0 1  0   0  [x0       <= 1
       0 0 0  0   0   x1       = 0
       1 0 1 -1   0   x2       <= 2
       1 1 1  0  1]   alpha_+  >= 1.5
                   alpha_-]
      x >= 0, alpha >= 0
  */
  auto lp = get_simple_test_problem();
  std::set<HighsInt> master_variables {0, 1};

  std::vector<HighsInt> csr_index {0, 1, 2, 0, 2, 3, 0, 1, 2, 4};
  std::vector<double> csr_values {1, 1, 1, 1, 1, -1, 1, 1, 1, 1};
  std::vector<HighsInt> csr_starts {0, 2, 3, 3, 6, 10};
  HighsLp expected;
  expected.offset_ = 0;
  expected.num_col_ = 5;
  expected.num_row_ = 5;
  expected.col_lower_ = {0, 0, 0, 0, 0};
  expected.col_upper_ = {inf, inf, inf, inf, inf};
  expected.col_cost_ = {0, 0, 0, 1, 1};
  expected.row_lower_ = {1, -inf, 0, -inf, 1.5};
  expected.row_upper_ = {1, 1, 0, 2, inf};
  expected.a_matrix_.format_ = MatrixFormat::kRowwise;
  expected.a_matrix_.start_ = csr_starts;
  expected.a_matrix_.index_ = csr_index;
  expected.a_matrix_.value_ = csr_values;
  expected.a_matrix_.num_row_ = 5;
  expected.a_matrix_.num_col_ = 5;

  Highs subproblem;
  subproblem.passModel(lp);
  auto division = divide_rows(lp.a_matrix_, master_variables);
  REQUIRE(division.mixed_rows == std::set<HighsInt> {3, 4});
  create_feasibility_subproblem(subproblem, lp, master_variables, division.mixed_rows);
  auto result = subproblem.getLp();
  result.ensureRowwise();
  REQUIRE(result.a_matrix_.index_  == expected.a_matrix_.index_);
  REQUIRE(result.a_matrix_.start_ == expected.a_matrix_.start_);
  REQUIRE(result.a_matrix_.value_ == expected.a_matrix_.value_);
  REQUIRE(result.a_matrix_  == expected.a_matrix_);
  REQUIRE(result.col_cost_  == expected.col_cost_);
  REQUIRE(result.col_lower_  == expected.col_lower_);
  REQUIRE(result.row_lower_  == expected.row_lower_);
  REQUIRE(result.col_upper_  == expected.col_upper_);
  REQUIRE(result.row_upper_  == expected.row_upper_);
  REQUIRE(result  == expected);
}

TEST_CASE("test-create-feas-subproblem-2", "[highs-benders]") {
  /*
    We expect:
    min 0 + 0^T x + alpha1_+ + alpha1_- +  alpha2_- +  alpha3_+ 
    s.t.
      [1 1 0 1 -1  0 0               =  1
       0 0 1 0  0  0 0     [x        <= 1
       0 0 0 0  0  0 0      alpha]   =  0
       1 0 1 0  0 -1 0               <= 2
       1 1 1 0  0  0 1  ]            >= 1.5

      ]
      x >= 0, alpha >= 0
  */
  auto lp = get_simple_test_problem();
  std::set<HighsInt> master_variables {0};

  std::vector<HighsInt> csr_index {0, 1, 3, 4, 2, 0, 2, 5, 0, 1, 2, 6};
  std::vector<double> csr_values {1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1};
  std::vector<HighsInt> csr_starts {0, 4, 5, 5, 8, 12};
  HighsLp expected;
  expected.offset_ = 0;
  expected.num_col_ = 7;
  expected.num_row_ = 5;
  expected.col_lower_ = {2, 0, 0, 0, 0, 0, 0};
  expected.col_upper_ = {2, inf, inf, inf, inf, inf, inf};
  expected.col_cost_ = {0, 0, 0, 1, 1, 1, 1};
  expected.row_lower_ = {1, -inf, 0, -inf, 1.5};
  expected.row_upper_ = {1, 1, 0, 2, inf};
  expected.a_matrix_.format_ = MatrixFormat::kRowwise;
  expected.a_matrix_.start_ = csr_starts;
  expected.a_matrix_.index_ = csr_index;
  expected.a_matrix_.value_ = csr_values;
  expected.a_matrix_.num_row_ = 5;
  expected.a_matrix_.num_col_ = 7;

  Highs subproblem;
  subproblem.passModel(lp);
  auto division = divide_rows(lp.a_matrix_, master_variables);
  create_feasibility_subproblem(subproblem, lp, master_variables, division.mixed_rows);
  fix_master_variables(subproblem, master_variables, {2});
  auto result = subproblem.getLp();
  result.ensureRowwise();
  REQUIRE(result.a_matrix_.start_  == expected.a_matrix_.start_);
  REQUIRE(result.a_matrix_.index_  == expected.a_matrix_.index_);
  REQUIRE(result.a_matrix_.value_ == expected.a_matrix_.value_);
  REQUIRE(result.col_lower_ == expected.col_lower_);
  REQUIRE(result.col_upper_ == expected.col_upper_);
  REQUIRE(result  == expected);
  subproblem.run();
  REQUIRE(subproblem.getModelStatus() == HighsModelStatus::kOptimal);
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
  // auto multipliers = get_all_multipliers(subproblem);
  // REQUIRE(multipliers == std::vector<double> {2, 2, -1});
  // REQUIRE(multipliers == std::vector<double> {2, 0, 0});
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
  
  add_cut(master, subproblem, master_variables, master_values, CutType::Objective);
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
  expected.col_lower_ = {0, mu_lb};
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

TEST_CASE("test-discover-master-variables", "[highs-benders]") {
  REQUIRE(discover_master_variables({"EC1", "EC2", "EG1", "EH2", "ECC", "EC4"}, "EC\\d") == std::set<HighsInt> {0, 1, 5});
  REQUIRE(discover_master_variables({"EC1", "EC2", "EG1", "EH2", "ECC", "EC4"}, "EX") == std::set<HighsInt> {});
  REQUIRE(discover_master_variables({}, "EC\\d") == std::set<HighsInt> {});
}

TEST_CASE("test-add-feasibility-cut", "[highs-benders]") {
  auto lp = get_simple_test_problem();
  std::set<HighsInt> master_variables {0}; //  with complicating variables 0
  BendersProblems problems;
  decompose_problem(problems, lp, master_variables);
  auto & master = problems.master;
  auto & subproblem = problems.subproblem;
  auto & feas_subproblem = problems.feas_subproblem;

  auto num_col = master.getLp().num_col_;
  auto num_row = master.getLp().num_row_;

  std::vector<double> master_values {2};
  fix_master_variables(subproblem, master_variables, master_values);
  auto status = HighsStatus::kOk;
  // auto status = subproblem.run();
  REQUIRE(status == HighsStatus::kOk);
  // REQUIRE(subproblem.getModelStatus() == HighsModelStatus::kInfeasible);
  fix_master_variables(feas_subproblem, master_variables, master_values);
  status = feas_subproblem.run();
  REQUIRE(status == HighsStatus::kOk);
  REQUIRE(feas_subproblem.getModelStatus() == HighsModelStatus::kOptimal);

  double dual_objective;
  feas_subproblem.getDualObjectiveValue(dual_objective);
  auto multipliers = get_master_multipliers(feas_subproblem, master_variables);
  auto old_value_multiple = std::inner_product(multipliers.begin(), multipliers.end(), master_values.begin(), 0.0);
  auto nonzero_multipliers = create_nonzero_vector(multipliers);
  add_cut(master, feas_subproblem, master_variables, master_values, CutType::Feasibility);
  auto new_master = master.getLp();
  /*
    We want the new master to be in form:
      min 5 + x_0 + z
      0 x_0 + 0 z = 0
      -1 x_0 + 0z >= -1

      x_0 >= 0
    */
  HighsLp expected;
  expected.offset_ = 5;
  expected.num_col_ = 2;
  expected.num_row_ = 2;
  expected.col_lower_ = {0, mu_lb};
  expected.col_upper_ = {inf, inf};
  expected.col_cost_ = {1, 1};
  expected.row_lower_ = {0, -1};
  expected.row_upper_ = {0, inf};
  expected.a_matrix_.format_ = MatrixFormat::kRowwise;
  expected.a_matrix_.start_ = {0,0,1};
  expected.a_matrix_.index_ = {0};
  expected.a_matrix_.value_ = {-1}; //?
  expected.a_matrix_.num_row_ = 2;
  expected.a_matrix_.num_col_ = 2;
  REQUIRE(new_master.num_row_ == num_row + 1);
  REQUIRE(expected.a_matrix_.start_  ==  new_master.a_matrix_.start_);
  REQUIRE(expected.a_matrix_.index_ ==  new_master.a_matrix_.index_);
  REQUIRE(expected.a_matrix_.value_ ==  new_master.a_matrix_.value_);
  REQUIRE(expected.a_matrix_ ==  new_master.a_matrix_);
  REQUIRE(expected.col_lower_ ==  new_master.col_lower_);
  REQUIRE(expected.col_upper_ ==  new_master.col_upper_);
  REQUIRE(expected.col_cost_ == new_master.col_cost_ );
  REQUIRE(expected.row_lower_ ==  new_master.row_lower_);
  REQUIRE(expected.row_upper_ ==  new_master.row_upper_);
  REQUIRE(new_master == expected);
}

TEST_CASE("test-solve-simple-system", "[highs-benders]") {
  auto lp = get_simple_test_problem();
  lp.col_names_ = {"m1", "s1", "s2"};
  Highs nodecomp;
  nodecomp.passModel(lp);
  nodecomp.run();
  auto expected = nodecomp.getObjectiveValue();
  auto res = benders(lp, "m\\d", {2});
  REQUIRE(expected == 5);
  REQUIRE(res == 5);
  REQUIRE(std::abs(res - expected) < 1e-3);

}

TEST_CASE("test-solve-simple-system-2", "[highs-benders]") {
  auto lp = get_simple_test_problem();
  lp.col_names_ = {"m1", "s1", "s2"};
  Highs nodecomp;
  nodecomp.passModel(lp);
  nodecomp.run();
  auto expected = nodecomp.getObjectiveValue();
  // auto res = benders(lp, "m\\d", {2});
  auto res = benders(lp, std::set<HighsInt>{0, 2}, {2, 0});
  REQUIRE(res == expected);
  REQUIRE(std::abs(res - expected) < 1e-3);
}

TEST_CASE("test-solve-second-system", "[highs-benders]") {
  auto lp = get_second_test_problem();
  lp.col_names_ = {"m1", "m2", "m3", "s1", "s2", "s3"};
  Highs nodecomp;
  nodecomp.passModel(lp);
  nodecomp.run();
  auto expected = nodecomp.getObjectiveValue();
  // auto res = benders(lp, "m\\d", {0, 1.5, 0});
  auto res = benders(lp, "m\\d", {0, 0, 0});
  REQUIRE(std::abs(res - expected) < 1e-3);
}

TEST_CASE("test-solve-blending", "[highs-benders]") {
  auto path = std::string(HIGHS_DIR) + "/check/instances/blending.mps";
  Highs nodecomp;
  nodecomp.readModel(path);
  auto lp = nodecomp.getLp();
  nodecomp.run();
  auto expected = nodecomp.getObjectiveValue();
  lp.ensureRowwise();
  auto res = benders(lp, "P0", std::vector<double> (40, 0));
  REQUIRE(std::abs(res - expected) < 1e-3);
}

TEST_CASE("test-solve-afiro", "[highs-benders]") {
  auto path = std::string(HIGHS_DIR) + "/check/instances/afiro.mps";
  Highs nodecomp;
  nodecomp.readModel(path);
  auto lp = nodecomp.getLp();
  nodecomp.run();
  auto expected = nodecomp.getObjectiveValue();
  lp.ensureRowwise();
  auto res = benders(lp, "X0\\d", std::vector<double> (40, 0));
  REQUIRE(std::abs(res - expected) < 1e-3);
}

TEST_CASE("2test-solve-simple-system", "[highs-benders]") {
  auto lp = get_simple_test_problem();
  lp.col_names_ = {"m1", "s1", "s2"};
  Highs nodecomp;
  nodecomp.passModel(lp);
  nodecomp.run();
  auto expected = nodecomp.getObjectiveValue();
  auto res = benders2(lp, "m\\d", {2});
  REQUIRE(expected == 5);
  REQUIRE(res == 5);
  REQUIRE(std::abs(res - expected) < 1e-3);

}

TEST_CASE("2test-solve-simple-system-2", "[highs-benders]") {
  auto lp = get_simple_test_problem();
  lp.col_names_ = {"m1", "s1", "s2"};
  Highs nodecomp;
  nodecomp.passModel(lp);
  nodecomp.run();
  auto expected = nodecomp.getObjectiveValue();
  // auto res = benders(lp, "m\\d", {2});
  auto res = benders2(lp, std::set<HighsInt>{0, 2}, {2, 0});
  REQUIRE(res == expected);
  REQUIRE(std::abs(res - expected) < 1e-3);
}

TEST_CASE("2test-solve-second-system", "[highs-benders]") {
  auto lp = get_second_test_problem();
  lp.col_names_ = {"m1", "m2", "m3", "s1", "s2", "s3"};
  Highs nodecomp;
  nodecomp.passModel(lp);
  nodecomp.run();
  auto expected = nodecomp.getObjectiveValue();
  // auto res = benders(lp, "m\\d", {0, 1.5, 0});
  auto res = benders2(lp, "m\\d", {0, 0, 0});
  REQUIRE(std::abs(res - expected) < 1e-3);
}

TEST_CASE("2test-solve-blending", "[highs-benders]") {
  auto path = std::string(HIGHS_DIR) + "/check/instances/blending.mps";
  Highs nodecomp;
  nodecomp.readModel(path);
  auto lp = nodecomp.getLp();
  nodecomp.run();
  auto expected = nodecomp.getObjectiveValue();
  lp.ensureRowwise();
  auto res = benders2(lp, "P0", std::vector<double> (40, 0));
  REQUIRE(std::abs(res - expected) < 1e-3);
}

TEST_CASE("2test-solve-afiro", "[highs-benders]") {
  auto path = std::string(HIGHS_DIR) + "/check/instances/afiro.mps";
  Highs nodecomp;
  nodecomp.readModel(path);
  auto lp = nodecomp.getLp();
  nodecomp.run();
  auto expected = nodecomp.getObjectiveValue();
  lp.ensureRowwise();
  auto res = benders2(lp, "X0\\d", std::vector<double> (40, 0));
  REQUIRE(std::abs(res - expected) < 1e-3);
}

TEST_CASE("test-solve-multi-simple-system-", "[highs-benders]") {
  auto lp = get_simple_multi_test_problem();
  Highs nodecomp;
  nodecomp.passModel(lp);
  nodecomp.run();
  auto expected = nodecomp.getObjectiveValue();
  // auto res = benders(lp, "m\\d", {2});
  auto res = multi_benders(lp, {0}, {{1,2}, {3}}, {2, 0});
  REQUIRE(res == expected);
  REQUIRE(std::abs(res - expected) < 1e-3);
}
