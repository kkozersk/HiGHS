#include "Benders.h"
#include "HighsInt.h"
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
  auto result = divide_overlapping_rows(csr_index, csr_starts, master_indices);
  REQUIRE(result.master_rows == std::set<HighsInt>{0, 2});
  REQUIRE(result.subproblem_rows == std::set<HighsInt>{1, 2});

  auto disj_result = divide_disjoint_rows(csr_index, csr_starts, master_indices);
  REQUIRE(disj_result.master_only_rows == std::vector<HighsInt>{0});
  REQUIRE(disj_result.subproblem_only_rows == std::vector<HighsInt>{1});
  REQUIRE(disj_result.mixed_rows == std::vector<HighsInt>{2});
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
    auto result = divide_overlapping_rows(csr_index, csr_starts, master_indices);
    REQUIRE(result.master_rows == std::set<HighsInt>{0, 3, 4});
    REQUIRE(result.subproblem_rows == std::set<HighsInt>{1, 3, 4});

    auto disj_results = divide_disjoint_rows(csr_index, csr_starts, master_indices);
    REQUIRE(disj_results.master_only_rows == std::vector<HighsInt>{0});
    REQUIRE(disj_results.subproblem_only_rows == std::vector<HighsInt>{1});
    REQUIRE(disj_results.mixed_rows == std::vector<HighsInt>{3, 4});
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
    auto result = divide_overlapping_rows(csr_index, csr_starts, master_indices);
    REQUIRE(result.master_rows == std::set<HighsInt>{0, 1, 3, 4});
    REQUIRE(result.subproblem_rows == std::set<HighsInt>{0, 1, 2, 3, 4, 5});

    auto disj_results = divide_disjoint_rows(csr_index, csr_starts, master_indices);
    REQUIRE(disj_results.master_only_rows == std::vector<HighsInt>{});
    REQUIRE(disj_results.subproblem_only_rows == std::vector<HighsInt>{2, 5});
    REQUIRE(disj_results.mixed_rows == std::vector<HighsInt>{0, 1, 3, 4});
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
    auto result = divide_overlapping_rows(csr_index, csr_starts, master_indices);
    REQUIRE(result.master_rows == std::set<HighsInt>{});
    REQUIRE(result.subproblem_rows == std::set<HighsInt>{0, 1, 2, 3, 4, 5});

    auto disj_results = divide_disjoint_rows(csr_index, csr_starts, master_indices);
    REQUIRE(disj_results.master_only_rows == std::vector<HighsInt>{});
    REQUIRE(disj_results.subproblem_only_rows == std::vector<HighsInt>{0, 1, 2, 3, 4, 5});
    REQUIRE(disj_results.mixed_rows == std::vector<HighsInt>{});
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
    auto result = divide_overlapping_rows(csr_index, csr_starts, master_indices);
    REQUIRE(result.master_rows == std::set<HighsInt>{0, 1, 2, 3, 4, 5});
    REQUIRE(result.subproblem_rows == std::set<HighsInt>{});

    auto disj_results = divide_disjoint_rows(csr_index, csr_starts, master_indices);
    REQUIRE(disj_results.master_only_rows == std::vector<HighsInt>{0, 1, 2, 3, 4, 5});
    REQUIRE(disj_results.subproblem_only_rows == std::vector<HighsInt>{});
    REQUIRE(disj_results.mixed_rows == std::vector<HighsInt>{});
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
    auto result = divide_overlapping_rows(csr_index, csr_starts, master_indices);
    REQUIRE(result.master_rows == std::set<HighsInt>{0});
    REQUIRE(result.subproblem_rows == std::set<HighsInt>{1});

    auto disj_results = divide_disjoint_rows(csr_index, csr_starts, master_indices);
    REQUIRE(disj_results.master_only_rows == std::vector<HighsInt>{0});
    REQUIRE(disj_results.subproblem_only_rows == std::vector<HighsInt>{1});
}
TEST_CASE("test-index-collection-from-set", "[highs_benders]") {
  auto index_collection = index_collection_from_set(std::set<HighsInt> {1, 2, 4, 7});
  REQUIRE(!index_collection.is_interval_);
  REQUIRE(!index_collection.is_mask_);
  REQUIRE(index_collection.is_set_);
  REQUIRE(index_collection.set_num_entries_ == 4);
  REQUIRE(index_collection.dimension_ == 4);
  REQUIRE(index_collection.set_ == std::vector<HighsInt> {1, 2, 4, 7});
}

TEST_CASE("test-index-collection-from-set-empty", "[highs_benders]") {
  auto index_collection = index_collection_from_set(std::set<HighsInt> {});
  REQUIRE(!index_collection.is_interval_);
  REQUIRE(!index_collection.is_mask_);
  REQUIRE(index_collection.is_set_);
  REQUIRE(index_collection.set_num_entries_ == 0);
  REQUIRE(index_collection.dimension_ == 0);
  REQUIRE(index_collection.set_ == std::vector<HighsInt> {});
}

TEST_CASE("test-set-master-values", "[highs-benders]") {
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
  lp.a_matrix_.start_ = csr_starts;
  lp.a_matrix_.index_ = csr_index;
  lp.a_matrix_.value_ = csr_values;
  auto result = fix_master_variables(lp, master_indices, master_values);
  REQUIRE(result);
  REQUIRE(lp.col_lower_ == std::vector<double> {1, 0});
  REQUIRE(lp.col_upper_ == std::vector<double> {1, inf});

}

TEST_CASE("test-set-master-values-invalid-sizes", "[highs-benders]") {
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
  lp.a_matrix_.start_ = csr_starts;
  lp.a_matrix_.index_ = csr_index;
  lp.a_matrix_.value_ = csr_values;
  auto result = fix_master_variables(lp, std::set<HighsInt>{0}, std::vector<double>{});
  REQUIRE(!result);
  REQUIRE(lp.col_lower_ == std::vector<double> {0, 0});
  REQUIRE(lp.col_upper_ == std::vector<double> {inf, inf});

  result = fix_master_variables(lp, std::set<HighsInt>{}, std::vector<double>{1});
  REQUIRE(!result);
  REQUIRE(lp.col_lower_ == std::vector<double> {0, 0});
  REQUIRE(lp.col_upper_ == std::vector<double> {inf, inf});
}
