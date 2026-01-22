#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include "io/SMPS.h"
#include "HCheckConfig.h"
#include "catch.hpp"
#include "lp_data/HConst.h"
#include "lp_data/HighsOptions.h"
#include "util/HighsInt.h"
#include "util/HighsSparseMatrix.h"

struct Matrix {
  int n, m;
  std::vector<double> vals;
  double operator[](std::pair<int, int> idx) {return vals.at(idx.first * n + idx.second); }
};

HighsSparseMatrix to_csr(Matrix mat) {
    int n = mat.n, m = mat.m;
    std::vector<double> values;
    std::vector<int> indices, starts {0};
    double val;
    for (int r = 0; r < n; ++r) {
      for (int c = 0; c < m; ++c) {
        if ((val = mat[{r,c}])) {
            values.push_back(val);
            indices.push_back(c);
        }
      }
      starts.push_back(indices.size());
    }
    HighsSparseMatrix A;
    A.format_ = MatrixFormat::kRowwise;
    A.value_ = values;
    A.index_ = indices;
    A.start_ = starts;
    A.num_row_ = n;
    A.num_col_ = m;
    return A;
}


const double inf = kHighsInf;
HighsLp get_test_problem() {
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

TEST_CASE("test-read-invalid-time-file", "[highs_smps]") {
  std::istringstream empty_file ("");
  SmpsTimeStructure smps(empty_file);
  REQUIRE(!smps.is_valid());

  std::istringstream only_header ("TIME NAME");
  smps = SmpsTimeStructure(only_header);
  REQUIRE(!smps.is_valid());

  std::istringstream no_periods("TIME NAME\n"
                                "C1 C2 T1\n"
                              "ENDATA");
  smps = SmpsTimeStructure(no_periods);
  REQUIRE(!smps.is_valid());

  std::istringstream malformed_entries("TIME NAME\n"
                                      "PERIODS\n"
                                "C1 C2 T1\n"
                                "C1 T1\n"
                              "ENDATA");
  smps = SmpsTimeStructure(malformed_entries);
  REQUIRE(!smps.is_valid());

  std::istringstream malformed_entries2("TIME NAME\n"
                                      "PERIODS\n"
                                "\n"
                                "C1 T1\n"
                              "ENDATA");
  smps = SmpsTimeStructure(malformed_entries2);
  REQUIRE(!smps.is_valid());

  std::istringstream malformed_entries3("TIME NAME\n"
                                      "PERIODS\n"
                                "T1\n"
                              "ENDATA");
  smps = SmpsTimeStructure(malformed_entries3);
  REQUIRE(!smps.is_valid());

  std::istringstream no_entries("TIME NAME\n"
                                      "PERIODS\n"
                              "ENDATA");
  smps = SmpsTimeStructure(no_entries);
  REQUIRE(!smps.is_valid());

  std::istringstream missing_end("TIME NAME\n"
                                      "PERIODS\n"
                                "C1 C2 T1\n"
                                "C1 R3 T1");
  smps = SmpsTimeStructure(missing_end);
  REQUIRE(!smps.is_valid());
}

TEST_CASE("test-read-valid-time-file", "[highs_smps]") {
  std::istringstream data("TIME NAME\n"
                                      "PERIODS\n"
                                "C1 R1 T1\n"
                                "C1 R3 T2\n"
                              "C3 R4 T3\n"
                            "ENDATA");
  SmpsTimeStructure smps(data);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_entries() == std::vector<TimeStageEntry>  {{"R1","C1","T1"}, {"R3","C1", "T2"}, {"R4", "C3", "T3"}});
  REQUIRE(smps.get_stage_names() == std::vector<std::string> {"T1", "T2", "T3"});
  REQUIRE(smps.get_problem_name() == "NAME");
  REQUIRE(smps.get_stage_index("T1") == 0);
  REQUIRE(smps.get_stage_index("T2") == 1);
  REQUIRE(smps.get_stage_index("T3") == 2);

  auto path = std::string(HIGHS_DIR) + "/check/instances/fxm2.tim";
  smps = SmpsTimeStructure(path);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_entries() == std::vector<TimeStageEntry> {{".COSTA", "1D1IK","TIME1"}, {"1DT019", "SCCOL1", "TIME2"}});
  REQUIRE(smps.get_stage_names() == std::vector<std::string> {"TIME1", "TIME2"});
  REQUIRE(smps.get_problem_name() == "SCFXM1");
  REQUIRE(smps.get_stage_index("TIME1") == 0);
  REQUIRE(smps.get_stage_index("TIME2") == 1);
  REQUIRE(smps.get_stage_index("NOTIME") == 2);
}

TEST_CASE("test-load-core-from-file", "[highs_smps]") {
  auto path = std::string(HIGHS_DIR) + "/check/instances/fxm.cor";
  HighsOptions opt;
  SmpsCoreStructure smps(opt, path);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.num_col_ == 457);
  REQUIRE(smps.num_col_ == smps.a_matrix_.num_col_);
  REQUIRE(smps.num_row_ == 330);
  REQUIRE(smps.num_row_ == smps.a_matrix_.num_row_);
}

TEST_CASE("test-load-malformed-core", "[highs_smps]") {
  auto path = std::string(HIGHS_DIR) + "/check/instances/afiro_malf.cor";
  HighsOptions opt;
  SmpsCoreStructure smps(opt, path);
  REQUIRE(!smps.is_valid());
  REQUIRE(smps.num_col_ == 0);
  REQUIRE(smps.num_col_ == smps.a_matrix_.num_col_);
  REQUIRE(smps.num_row_ == 0);
  REQUIRE(smps.num_row_ == smps.a_matrix_.num_row_);
}

TEST_CASE("test-assign-time-to-core", "[highs_smps]") {
  auto path = std::string(HIGHS_DIR) + "/check/instances/fxm";
  HighsOptions opt;
  SmpsCoreStructure smps_core(opt, path + ".cor");
  REQUIRE(smps_core.is_valid());
  SmpsTimeStructure smps_time(path + "2.tim");
  REQUIRE(smps_time.is_valid());
  REQUIRE(smps_core.load_time_stages(smps_time));
  auto matrix = smps_core.stage_submatrix;
  REQUIRE(matrix == std::map<std::string, SubMatrixRange> {
            {"TIME1", {0, 92, 0, 114}},
            {"TIME2", {92, smps_core.num_row_, 114, smps_core.num_col_}}
          });
}

TEST_CASE("test-assign-invalid-time-to-core", "[highs_smps]") {
  HighsOptions opt;
  SmpsCoreStructure smps_core(opt, std::string(HIGHS_DIR) + "/check/instances/afiro.cor");
  REQUIRE(smps_core.is_valid());
  SmpsTimeStructure smps_time(std::string(HIGHS_DIR)+ "/check/instances/fxm2.tim");
  REQUIRE(smps_time.is_valid());
  REQUIRE(!smps_core.load_time_stages(smps_time));

  smps_core = SmpsCoreStructure(opt, std::string(HIGHS_DIR) + "/check/instances/afiro_malf.cor");
  REQUIRE(!smps_core.is_valid());
  REQUIRE(!smps_core.load_time_stages(smps_time));

  smps_core = SmpsCoreStructure(opt, std::string(HIGHS_DIR) + "/check/instances/afiro.cor");
  REQUIRE(smps_core.is_valid());
  std::istringstream missing_end("TIME NAME\n"
                                      "PERIODS\n"
                                "C1 C2 T1\n"
                                "C1 R3 T1");
  smps_time = SmpsTimeStructure(missing_end);
  REQUIRE(!smps_time.is_valid());
  REQUIRE(!smps_core.load_time_stages(smps_time));

  //TODO more intricate tests depending on missing rows/cols
}

TEST_CASE("test-create-dummy-stochastic-tree", "[highs_smps]") {
  auto root = new Node ("TIME1");
  root->add_child(std::unique_ptr<Node>(new Node("TIME2", 0.5)));
  root->add_child(std::unique_ptr<Node>(new Node("TIME2", 0.3)));
  REQUIRE(!root->verify_children_probabilities());
  root->add_child(std::unique_ptr<Node>(new Node("TIME2", 0.2)));
  REQUIRE(root->verify_children_probabilities());
  root->get_child(0)->add_child(std::unique_ptr<Node>(new Node("TIME3", 0.7)));
  REQUIRE(root->verify_children_probabilities());

  REQUIRE(root->get_child(0)->get_parent() == root);
  REQUIRE(root->get_child(1)->get_parent() == root);
  REQUIRE(root->get_child(2)->get_parent() == root);
  REQUIRE(root->get_child(0)->get_child(0)->get_parent() == root->get_child(0).get());

  StochasticTree tree {std::unique_ptr<Node>(root)};
  REQUIRE(tree.root.get() == root);
}

TEST_CASE("test-create-simple-indep-structure", "[highs_smps]") {
  std::istringstream data("STOCH NAME\n"
                          "INDEP DISCRETE\n"
                          "RHS R1 50 TIME2 0.5\n"
                          "RHS R1 40 TIME2 0.3\n"
                          "RHS R1 60 TIME2 0.2\n"
                          "ENDATA");
  IndepStructure smps(data);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_no_timestage_random_entries() == 1);
  auto rvt = smps.get_timestage_random_entry(0);
  REQUIRE(rvt.timestage == "TIME2");
  REQUIRE(rvt.rvs.size() == 1);
  TimestageRandomVariables rvt1 {
    {
      {"RHS", "R1", {{0.5, 50}, {0.3, 40}, {0.2, 60}}},
    },"TIME2"};
  REQUIRE(rvt == rvt1);

  auto vec = rvt.combine_variables();
  RandomVector expected {
    {0.5, {{"R1", "RHS", 50}}},
    {0.3, {{"R1", "RHS", 40}}},
    {0.2, {{"R1", "RHS", 60}}},
   };
  REQUIRE(vec == expected);
}

TEST_CASE("test-create-single-val-indep-structure", "[highs_smps]") {
  std::istringstream data("STOCH NAME\n"
                          "INDEP DISCRETE\n"
                          "RHS R1 50 TIME2 1\n"
                          "ENDATA");
  IndepStructure smps(data);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_no_timestage_random_entries() == 1);
  auto rvt = smps.get_timestage_random_entry(0);
  REQUIRE(rvt.timestage == "TIME2");
  REQUIRE(rvt.rvs.size() == 1);
  TimestageRandomVariables rvt1 {
    {
      {"RHS", "R1", {{1., 50}}},
    },"TIME2"};
  REQUIRE(rvt == rvt1);

  auto vec = rvt.combine_variables();
  RandomVector expected {
    {1., {{"R1", "RHS", 50}}},
   };
  REQUIRE(vec == expected);
}

TEST_CASE("test-create-complex-indep-structure", "[highs_smps]") {
  std::istringstream data("STOCH NAME\n"
                          "INDEP DISCRETE\n"
                          "RHS R1 50 TIME2 0.5\n"
                          "RHS R1 40 TIME2 0.3\n"
                          "RHS R1 60 TIME2 0.2\n"
                          "C1 R1 50 TIME2 0.7\n"
                          "C1 R1 30 TIME2 0.3\n"
                          "C1 R1 50 TIME3 0.5\n"
                          "C1 R1 30 TIME3 0.5\n"
                          "ENDATA");
  IndepStructure smps(data);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_no_timestage_random_entries() == 2);
  TimestageRandomVariables rvt1 {
    {
      {"RHS", "R1", {{0.5, 50}, {0.3, 40}, {0.2, 60}}},
      {"C1", "R1", {{0.7, 50}, {0.3, 30}}},
    },"TIME2"};
  REQUIRE(smps.get_timestage_random_entry(0) == rvt1);
  TimestageRandomVariables rvt2 {
    {
      {"C1", "R1", {{0.5, 50}, {0.5, 30}}},
    },"TIME3"};
  REQUIRE(smps.get_timestage_random_entry(1) == rvt2);
  RandomVector expected {
    {0.5 * 0.7, {{"R1", "RHS", 50}, {"R1", "C1", 50}}},
    {0.3 * 0.7, {{"R1", "RHS", 40}, {"R1", "C1", 50}}},
    {0.2 * 0.7, {{"R1", "RHS", 60}, {"R1", "C1", 50}}},
    {0.5 * 0.3, {{"R1", "RHS", 50}, {"R1", "C1", 30}}},
    {0.3 * 0.3, {{"R1", "RHS", 40}, {"R1", "C1", 30}}},
    {0.2 * 0.3, {{"R1", "RHS", 60}, {"R1", "C1", 30}}},
   };
  REQUIRE(rvt1.combine_variables() == expected);
  RandomVector expected2 {
    {0.5, {{"R1", "C1", 50}}},
    {0.5, {{"R1", "C1", 30}}},
   };
  REQUIRE(rvt2.combine_variables() == expected2);
}

TEST_CASE("test-create-indep-structure-with-comments", "[highs_smps]") {
  std::istringstream data("STOCH NAME\n"
                          "INDEP DISCRETE\n"
                          "*RHS R1 10 TIME2 0.5\n"
                          "RHS R1 50 TIME2 0.5\n"
                          "*\n"
                          "*\n"
                          "RHS R1 40 TIME2 0.3\n"
                          "RHS R1 60 TIME2 0.2\n"
                          "*\n"
                          "C1 R1 50 TIME2 0.7\n"
                          "C1 R1 30 TIME2 0.3\n"
                          "C1 R1 50 TIME3 0.5\n"
                          "C1 R1 30 TIME3 0.5\n"
                          "*\n"
                          "ENDATA");
  IndepStructure smps(data);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_no_timestage_random_entries() == 2);
  TimestageRandomVariables rvt1 {
    {
      {"RHS", "R1", {{0.5, 50}, {0.3, 40}, {0.2, 60}}},
      {"C1", "R1", {{0.7, 50}, {0.3, 30}}},
    },"TIME2"};
  REQUIRE(smps.get_timestage_random_entry(0) == rvt1);
  TimestageRandomVariables rvt2 {
    {
      {"C1", "R1", {{0.5, 50}, {0.5, 30}}},
    },"TIME3"};
  REQUIRE(smps.get_timestage_random_entry(1) == rvt2);
  RandomVector expected {
    {0.5 * 0.7, {{"R1", "RHS", 50}, {"R1", "C1", 50}}},
    {0.3 * 0.7, {{"R1", "RHS", 40}, {"R1", "C1", 50}}},
    {0.2 * 0.7, {{"R1", "RHS", 60}, {"R1", "C1", 50}}},
    {0.5 * 0.3, {{"R1", "RHS", 50}, {"R1", "C1", 30}}},
    {0.3 * 0.3, {{"R1", "RHS", 40}, {"R1", "C1", 30}}},
    {0.2 * 0.3, {{"R1", "RHS", 60}, {"R1", "C1", 30}}},
   };
  REQUIRE(rvt1.combine_variables() == expected);
  RandomVector expected2 {
    {0.5, {{"R1", "C1", 50}}},
    {0.5, {{"R1", "C1", 30}}},
   };
  REQUIRE(rvt2.combine_variables() == expected2);
}

TEST_CASE("test-create-malformed-indep-structure", "[highs_smps]") {
  std::istringstream missing_timeperiods("STOCH NAME\n"
                          "INDEP DISCRETE\n"
                          "RHS R1 50 0.5\n"
                          "RHS R1 40 0.3\n"
                          "RHS R1 60 0.2\n"
                          "C1 R1 50 0.7\n"
                          "C1 R1 30 0.3\n"
                          "C1 R1 50 0.5\n"
                          "C1 R1 30 0.5\n"
                          "ENDATA");
  IndepStructure smps(missing_timeperiods);
  REQUIRE(!smps.is_valid());

  std::istringstream single_missing_timeperiod("STOCH NAME\n"
                          "INDEP DISCRETE\n"
                          "RHS R1 50 TIME2 0.5\n"
                          "RHS R1 40 TIME2 0.3\n"
                          "RHS R1 60 TIME2 0.2\n"
                          "C1 R1 50 TIME2 0.7\n"
                          "C1 R1 30 TIME2 0.3\n"
                          "C1 R1 50 TIME3 0.5\n"
                          "C1 R1 30  0.5\n"
                          "ENDATA");
  smps = IndepStructure(single_missing_timeperiod);
  REQUIRE(!smps.is_valid());

  std::istringstream missing_cols("STOCH NAME\n"
                          "INDEP DISCRETE\n"
                          "R1 50 0.5\n"
                          "R1 40 0.3\n"
                          "R1 60 0.2\n"
                          "R1 50 0.7\n"
                          "R1 30 0.3\n"
                          "R1 50 0.5\n"
                          "R1 30 0.5\n"
                          "ENDATA");
  smps = IndepStructure(missing_cols);
  REQUIRE(!smps.is_valid());

  std::istringstream single_missing_col("STOCH NAME\n"
                          "INDEP DISCRETE\n"
                          "RHS R1 50 TIME2 0.5\n"
                          "RHS R1 40 TIME2 0.3\n"
                          "R1 60 TIME2 0.2\n"
                          "C1 R1 50 TIME2 0.7\n"
                          "C1 R1 30 TIME2 0.3\n"
                          "C1 R1 50 TIME3 0.5\n"
                          "C1 R1 30 TIME3 0.5\n"
                          "ENDATA");
  smps = IndepStructure(single_missing_col);
  REQUIRE(!smps.is_valid());


  std::istringstream too_much_data("STOCH NAME\n"
                          "INDEP DISCRETE\n"
                          "RHS R1 50 TIME2 0.5\n"
                          "RHS R1 40 TIME2 0.3 67\n"
                          "RHS R1 60 TIME2 0.2\n"
                          "C1 R1 50 TIME2 0.7\n"
                          "C1 R1 30 TIME2 0.3\n"
                          "C1 R1 50 TIME3 0.5\n"
                          "C1 R1 30 TIME3 0.5\n"
                          "ENDATA");
  smps = IndepStructure(too_much_data);
  REQUIRE(!smps.is_valid());
  
  std::istringstream missing_end("STOCH NAME\n"
                          "INDEP DISCRETE\n"
                          "RHS R1 50 TIME2 0.5\n"
                          "RHS R1 40 TIME2 0.3\n"
                          "RHS R1 60 TIME2 0.2\n"
                          "C1 R1 50 TIME2 0.7\n"
                          "C1 R1 30 TIME2 0.3\n"
                          "C1 R1 50 TIME3 0.5\n"
                          "C1 R1 30 TIME3 0.5\n"
                          );
  smps = IndepStructure(missing_end);
  REQUIRE(!smps.is_valid());
  
  std::istringstream malformed_end("STOCH NAME\n"
                          "INDEP DISCRETE\n"
                          "RHS R1 50 TIME2 0.5\n"
                          "RHS R1 40 TIME2 0.3\n"
                          "RHS R1 60 TIME2 0.2\n"
                          "C1 R1 50 TIME2 0.7\n"
                          "C1 R1 30 TIME2 0.3\n"
                          "C1 R1 50 TIME3 0.5\n"
                          "C1 R1 30 TIME3 0.5\n"
                          "ENDATA END VAL");
  smps = IndepStructure(malformed_end);
  REQUIRE(!smps.is_valid());

  std::istringstream nonnumeric_data("STOCH NAME\n"
                          "INDEP DISCRETE\n"
                          "RHS R1 50ALPHA TIME2 0.5\n"
                          "RHS R1 40 TIME2 0.3\n"
                          "RHS R1 60 TIME2 0.2\n"
                          "C1 R1 50 TIME2 0.7\n"
                          "C1 R1 30 TIME2 0.3\n"
                          "C1 R1 50 TIME3 0.5\n"
                          "C1 R1 30 TIME3 0.5\n"
                          "ENDATA");
  smps = IndepStructure(nonnumeric_data);
  REQUIRE(!smps.is_valid());

  std::istringstream nonnumeric_data2("STOCH NAME\n"
                          "INDEP DISCRETE\n"
                          "RHS R1 ALPHA TIME2 BAU\n"
                          "RHS R1 40 TIME2 0.3\n"
                          "RHS R1 60 TIME2 0.2\n"
                          "C1 R1 50 TIME2 0.7\n"
                          "C1 R1 30 TIME2 0.3\n"
                          "C1 R1 50 TIME3 0.5\n"
                          "C1 R1 30 TIME3 0.5\n"
                          "ENDATA");
  smps = IndepStructure(nonnumeric_data2);
  REQUIRE(!smps.is_valid());


  std::istringstream invalid_probability("STOCH NAME\n"
                          "INDEP DISCRETE\n"
                          "RHS R1 50 TIME2 -0.3\n"
                          "RHS R1 40 TIME2 0.3\n"
                          "RHS R1 60 TIME2 0.2\n"
                          "C1 R1 50 TIME2 0.7\n"
                          "C1 R1 30 TIME2 0.3\n"
                          "C1 R1 50 TIME3 0.5\n"
                          "C1 R1 30 TIME3 0.5\n"
                          "ENDATA");
  smps = IndepStructure(invalid_probability);
  REQUIRE(!smps.is_valid());

  std::istringstream invalid_probability2("STOCH NAME\n"
                          "INDEP DISCRETE\n"
                          "RHS R1 50 TIME2 0.5\n"
                          "RHS R1 40 TIME2 0.3\n"
                          "RHS R1 60 TIME2 0.2\n"
                          "C1 R1 50 TIME2 1.7\n"
                          "C1 R1 30 TIME2 0.3\n"
                          "C1 R1 50 TIME3 0.5\n"
                          "C1 R1 30 TIME3 0.5\n"
                          "ENDATA");
  smps = IndepStructure(invalid_probability2);
  REQUIRE(!smps.is_valid());

  std::istringstream malformed_header("STOCH NAME 3\n"
                          "INDEP DISCRETE\n"
                          "RHS R1 50 TIME2 0.5\n"
                          "RHS R1 40 TIME2 0.3\n"
                          "RHS R1 60 TIME2 0.2\n"
                          "C1 R1 50 TIME2 0.7\n"
                          "C1 R1 30 TIME2 0.3\n"
                          "C1 R1 50 TIME3 0.5\n"
                          "C1 R1 30 TIME3 0.5\n"
                          "ENDATA");
  smps = IndepStructure(malformed_header);
  REQUIRE(!smps.is_valid());

  std::istringstream malformed_header2("NAME NAME2\n"
                          "INDEP DISCRETE\n"
                          "RHS R1 50 TIME2 0.5\n"
                          "RHS R1 40 TIME2 0.3\n"
                          "RHS R1 60 TIME2 0.2\n"
                          "C1 R1 50 TIME2 0.7\n"
                          "C1 R1 30 TIME2 0.3\n"
                          "C1 R1 50 TIME3 0.5\n"
                          "C1 R1 30 TIME3 0.5\n"
                          "ENDATA");
  smps = IndepStructure(malformed_header2);
  REQUIRE(!smps.is_valid());

  std::istringstream malformed_header3("NAME NAME2\n"
                          "INDEP\n"
                          "RHS R1 50 TIME2 0.5\n"
                          "RHS R1 40 TIME2 0.3\n"
                          "RHS R1 60 TIME2 0.2\n"
                          "C1 R1 50 TIME2 0.7\n"
                          "C1 R1 30 TIME2 0.3\n"
                          "C1 R1 50 TIME3 0.5\n"
                          "C1 R1 30 TIME3 0.5\n"
                          "ENDATA");
  smps = IndepStructure(malformed_header3);
  REQUIRE(!smps.is_valid());

  std::istringstream malformed_header4("NAME NAME2\n"
                          "RHS R1 50 TIME2 0.5\n"
                          "RHS R1 40 TIME2 0.3\n"
                          "RHS R1 60 TIME2 0.2\n"
                          "C1 R1 50 TIME2 0.7\n"
                          "C1 R1 30 TIME2 0.3\n"
                          "C1 R1 50 TIME3 0.5\n"
                          "C1 R1 30 TIME3 0.5\n"
                          "ENDATA");
  smps = IndepStructure(malformed_header4);
  REQUIRE(!smps.is_valid());

  std::istringstream end_comments("STOCH NAME\n"
                          "INDEP DISCRETE\n"
                          "RHS R1 50 TIME2 0.5\n"
                          "RHS R1 40 TIME2 0.3\n"
                          "RHS R1 60 TIME2 0.2\n"
                          "C1 R1 50 TIME2 0.7\n"
                          "C1 R1 30 TIME2 0.3\n"
                          "C1 R1 50 TIME3 0.5\n"
                          "C1 R1 30 TIME3 0.5\n"
                          "*\n"
                          "*"
                          );
  smps = IndepStructure(end_comments);
  REQUIRE(!smps.is_valid());
}

TEST_CASE("test-create-simple-block-structure", "[highs_smps]") {
  std::istringstream data("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 1.0\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "ENDATA");
  BlockStructure smps(data);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_no_timestage_random_vectors() == 1);
  auto rvt = smps.get_timestage_random_vector(0);
  TimestageRandomVectors rvt1 {
    {
      {{1.0, {{"R1","RHS",50}, {"R2", "RHS", 40}}}}
    },"TIME2"};
  REQUIRE(rvt == rvt1);

}

TEST_CASE("test-create-complex-block-structure", "[highs_smps]") {
  std::istringstream data("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 0.3\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 0.7\n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 50\n"
                          "RHS R4 40\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 40\n"
                          "RHS R4 50\n"
                          "ENDATA");
  BlockStructure smps(data);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_no_timestage_random_vectors() == 2);
  auto rvt = smps.get_timestage_random_vector(0);
  TimestageRandomVectors rvt1 {
    {
      {{0.3, {{"R1","RHS",50}, {"R2", "RHS", 40}}},
      {0.7, {{"R1","RHS",40}, {"R2", "RHS", 50}}}}
  },"TIME2"};
  REQUIRE(smps.get_timestage_random_vector(0) == rvt1);
  TimestageRandomVectors rvt2 {
    {
      {{0.5, {{"R3","RHS",50}, {"R4", "RHS", 40}}},
      {0.5, {{"R3","RHS",40}, {"R4", "RHS", 50}}}}
  },"TIME3"};
  REQUIRE(smps.get_timestage_random_vector(1) == rvt2);
}

TEST_CASE("test-create-block-structure-with-comments", "[highs_smps]") {
  std::istringstream data("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "* values\n"
                          "**\n"
                          "BL BLOCK01 TIME2 0.3\n"
                          "*\n"
                          "RHS R1 50\n"
                          "* entry\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 0.7\n"
                          "RHS R2 50\n"
                          "* next one is assumed from the top block\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 50\n"
                          "RHS R4 40\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 40\n"
                          "RHS R4 50\n"
                          "*ENDATA\n"
                          "ENDATA");
  BlockStructure smps(data);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_no_timestage_random_vectors() == 2);
  auto rvt = smps.get_timestage_random_vector(0);
  TimestageRandomVectors rvt1 {
    {
      {{0.3, {{"R1","RHS",50}, {"R2", "RHS", 40}}},
      {0.7, {{"R1","RHS",50}, {"R2", "RHS", 50}}}}
  },"TIME2"};
  REQUIRE(smps.get_timestage_random_vector(0) == rvt1);
  TimestageRandomVectors rvt2 {
    {
      {{0.5, {{"R3","RHS",50}, {"R4", "RHS", 40}}},
      {0.5, {{"R3","RHS",40}, {"R4", "RHS", 50}}}}
  },"TIME3"};
  REQUIRE(smps.get_timestage_random_vector(1) == rvt2);
}

TEST_CASE("test-create-simple-block-structure-with-missing-entries", "[highs_smps]") {
  std::istringstream data("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 0.3\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 0.2\n"
                          "RHS R2 50\n"
                          "BL BLOCK01 TIME2 0.5\n"
                          "RHS R1 40\n"
                          "ENDATA");
  BlockStructure smps(data);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_no_timestage_random_vectors() == 1);
  TimestageRandomVectors rvt1 {
    {
      {{0.3, {{"R1","RHS",50}, {"R2", "RHS", 40}}},
      {0.2, {{"R1","RHS",50}, {"R2", "RHS", 50}}},
      {0.5, {{"R1","RHS",40}, {"R2", "RHS", 40}}},}
  },"TIME2"};
  REQUIRE(smps.get_timestage_random_vector(0) == rvt1);
}

TEST_CASE("test-create-block-structure-with-missing-entries", "[highs_smps]") {
  std::istringstream data("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 0.3\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 0.2\n"
                          "RHS R2 50\n"
                          "BL BLOCK01 TIME2 0.5\n"
                          "RHS R1 40\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 50\n"
                          "RHS R4 40\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R4 50\n"
                          "BL BLOCK03 TIME4 1\n"
                          "RHS R5 50\n"
                          "ENDATA");
  BlockStructure smps(data);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_no_timestage_random_vectors() == 3);
  TimestageRandomVectors rvt1 {
    {
      {{0.3, {{"R1","RHS",50}, {"R2", "RHS", 40}}},
      {0.2, {{"R1","RHS",50}, {"R2", "RHS", 50}}},
      {0.5, {{"R1","RHS",40}, {"R2", "RHS", 40}}},}
  },"TIME2"};
  REQUIRE(smps.get_timestage_random_vector(0) == rvt1);
  TimestageRandomVectors rvt2 {
    {
      {{0.5, {{"R3","RHS",50}, {"R4", "RHS", 40}}},
      {0.5, {{"R3","RHS",50}, {"R4", "RHS", 50}}}}
  },"TIME3"};
  REQUIRE(smps.get_timestage_random_vector(1) == rvt2);
  TimestageRandomVectors rvt3 {
    {
      {{1., {{"R5","RHS",50}}},}
  },"TIME4"};
  REQUIRE(smps.get_timestage_random_vector(2) == rvt3);
}

TEST_CASE("test-create-block-with-multiple-blocks-per-timestage", "[highs_smps]") {
  std::istringstream data("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 0.3\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 0.2\n"
                          "RHS R2 50\n"
                          "BL BLOCK01 TIME2 0.5\n"
                          "RHS R1 40\n"
                          "BL BLOCK02 TIME2 0.5\n"
                          "RHS R3 50\n"
                          "RHS R4 40\n"
                          "BL BLOCK02 TIME2 0.5\n"
                          "RHS R4 50\n"
                          "BL BLOCK03 TIME2 1\n"
                          "RHS R5 50\n"
                          "BL BLOCK04 TIME3 0.5\n"
                          "RHS R6 50\n"
                          "RHS R7 40\n"
                          "BL BLOCK04 TIME3 0.5\n"
                          "RHS R7 30\n"
                          "ENDATA");
  BlockStructure smps(data);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_no_timestage_random_vectors() == 2);
  TimestageRandomVectors rvt1 {
    {
      {
        {0.3,  {{"R1","RHS",50}, {"R2", "RHS", 40}}},
        {0.2,  {{"R1","RHS",50}, {"R2", "RHS", 50}}},
        {0.5,  {{"R1","RHS",40}, {"R2", "RHS", 40}}}
      },
      {
        {0.5,  {{"R3","RHS",50}, {"R4", "RHS", 40}}},
        {0.5,  {{"R3","RHS",50}, {"R4", "RHS", 50}}}
      },
      {
        {1.,   {{"R5","RHS",50}}}
      }
  },"TIME2"};
  REQUIRE(smps.get_timestage_random_vector(0) == rvt1);
  RandomVector combined {
    { 0.3 * 0.5, { {"R1","RHS",50}, {"R2", "RHS", 40}, {"R3","RHS",50}, {"R4", "RHS", 40}, {"R5","RHS",50} }, },    
    { 0.3 * 0.5, { {"R1","RHS",50}, {"R2", "RHS", 40}, {"R3","RHS",50}, {"R4", "RHS", 50}, {"R5","RHS",50} }, },    
    { 0.2 * 0.5, { {"R1","RHS",50}, {"R2", "RHS", 50}, {"R3","RHS",50}, {"R4", "RHS", 40}, {"R5","RHS",50} }, },    
    { 0.2 * 0.5, { {"R1","RHS",50}, {"R2", "RHS", 50}, {"R3","RHS",50}, {"R4", "RHS", 50}, {"R5","RHS",50} }, },    
    { 0.5 * 0.5, { {"R1","RHS",40}, {"R2", "RHS", 40}, {"R3","RHS",50}, {"R4", "RHS", 40}, {"R5","RHS",50} }, },    
    { 0.5 * 0.5, { {"R1","RHS",40}, {"R2", "RHS", 40}, {"R3","RHS",50}, {"R4", "RHS", 50}, {"R5","RHS",50} }, },    
  };
  REQUIRE(rvt1.combine_vectors() == combined);
  TimestageRandomVectors rvt2 {
    {
      {
        {0.5,  {{"R6","RHS",50}, {"R7", "RHS", 40}}},
        {0.5,  {{"R6","RHS",50}, {"R7", "RHS", 30}}}
      },
  },"TIME3"};
  REQUIRE(smps.get_timestage_random_vector(1) == rvt2);
  REQUIRE(rvt2.combine_vectors() == rvt2.rvs.at(0));
}

TEST_CASE("test-malformed-block-structure", "[highs_smps]") {
  std::istringstream missing_block_data("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 0.3\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 \n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 50\n"
                          "RHS R4 40\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 40\n"
                          "RHS R4 50\n"
                          "ENDATA");
  BlockStructure smps(missing_block_data);
  REQUIRE(!smps.is_valid());

  std::istringstream too_much_block_data("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 0.3\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 0.7\n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "BL BLOCK02 TIME3 0.5 RHS R3\n"
                          "RHS R3 50\n"
                          "RHS R4 40\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 40\n"
                          "RHS R4 50\n"
                          "ENDATA");
  smps = BlockStructure(too_much_block_data);
  REQUIRE(!smps.is_valid());

  std::istringstream missing_entry_data("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 0.3\n"
                          "R1 50\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 0.7\n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 50\n"
                          "RHS R4 40\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 40\n"
                          "RHS R4 50\n"
                          "ENDATA");
  smps = BlockStructure(missing_entry_data);
  REQUIRE(!smps.is_valid());

  std::istringstream too_much_entry_data("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 0.3\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 0.7\n"
                          "RHS R1 40 BLOCK01\n"
                          "RHS R2 50\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 50\n"
                          "RHS R4 40\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 40\n"
                          "RHS R4 50\n"
                          "ENDATA");
  smps = BlockStructure(too_much_entry_data);
  REQUIRE(!smps.is_valid());

  std::istringstream missing_end("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 0.3\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 0.7\n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 50\n"
                          "RHS R4 40\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 40\n"
                          "RHS R4 50\n"
                          );
  smps = BlockStructure(missing_end);
  REQUIRE(!smps.is_valid());

  std::istringstream malformed_end("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 0.3\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 0.7\n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 50\n"
                          "RHS R4 40\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 40\n"
                          "RHS R4 50\n"
                          "ENDDATA");
  smps = BlockStructure(malformed_end);
  REQUIRE(!smps.is_valid());

  std::istringstream nonnumeric_data("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 0.3+0.2\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 0.7\n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 50\n"
                          "RHS R4 40\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 40\n"
                          "RHS R4 50\n"
                          "ENDATA");
  smps = BlockStructure(nonnumeric_data);
  REQUIRE(!smps.is_valid());

  std::istringstream nonnumeric_data2("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 0.3\n"
                          "RHS R1 ABC\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 0.7\n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 50\n"
                          "RHS R4 40\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 40\n"
                          "RHS R4 50\n"
                          "ENDATA");
  smps = BlockStructure(nonnumeric_data2);
  REQUIRE(!smps.is_valid());

  std::istringstream invalid_prob("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 -4\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 0.7\n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 50\n"
                          "RHS R4 40\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 40\n"
                          "RHS R4 50\n"
                          "ENDATA");
  smps = BlockStructure(invalid_prob);
  REQUIRE(!smps.is_valid());

  std::istringstream invalid_prob2("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 0.3\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 1.1\n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 50\n"
                          "RHS R4 40\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 40\n"
                          "RHS R4 50\n"
                          "ENDATA");
  smps = BlockStructure(invalid_prob2);
  REQUIRE(!smps.is_valid());

  std::istringstream malformed_header("STOCH NAME\n"
                          "INDEP DISCRETE\n"
                          "BL BLOCK01 TIME2 0.3\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 0.7\n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 50\n"
                          "RHS R4 40\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 40\n"
                          "RHS R4 50\n"
                          "ENDATA");
  smps = BlockStructure(malformed_header);
  REQUIRE(!smps.is_valid());

  std::istringstream malformed_header2("STOCHASTIC NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 0.3\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 0.7\n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 50\n"
                          "RHS R4 40\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 40\n"
                          "RHS R4 50\n"
                          "ENDATA");
  smps = BlockStructure(malformed_header2);
  REQUIRE(!smps.is_valid());

  std::istringstream empty_block("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 0.3\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 0.7\n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 50\n"
                          "RHS R4 40\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "ENDATA");
  smps = BlockStructure(empty_block);
  REQUIRE(!smps.is_valid());

  std::istringstream incorrect_entry_ordering("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 0.3\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 0.7\n"
                          "RHS R2 50\n"
                          "RHS R1 40\n"
                          "ENDATA");
  smps = BlockStructure(incorrect_entry_ordering);
  REQUIRE(!smps.is_valid());

  std::istringstream end_comments("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 0.3\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 0.7\n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 50\n"
                          "RHS R4 40\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 40\n"
                          "RHS R4 50\n"
                          "*\n"
                          "*"
                          );
  smps = BlockStructure(end_comments);
  REQUIRE(!smps.is_valid());
}

TEST_CASE("test-create-simple-scenario-structure", "[highs_smps]") {
   std::istringstream timedata("TIME NAME\n"
                                       "PERIODS\n"
                                 "C1 R1 PERIOD2\n"
                             "ENDATA");
   SmpsTimeStructure time(timedata);
   REQUIRE(time.is_valid());
   auto path = std::string(HIGHS_DIR) + "/check/instances/simple.cor";
   SmpsCoreStructure core({}, path);
   REQUIRE(core.is_valid());

  std::istringstream data("SCENARIOS DISCRETE\n"
                          "SC S01 ROOT 0.5 PERIOD2\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "SC S02 ROOT 0.3 PERIOD2\n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "ENDATA");

  ScenarioStructure smps(data, time, core);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_no_scenarios() == 2);
  ScenarioModifications scen1 {
      {{"R1","RHS",50}, {"R2", "RHS", 40}},
      0.5, "PERIOD2", "S01", "ROOT" };
  REQUIRE(smps.get_scenario(0) == scen1);
  REQUIRE(smps.get_scenario("S01") == scen1);
  ScenarioModifications scen2 {
      {{"R1","RHS",40}, {"R2", "RHS", 50}},
      0.3, "PERIOD2", "S02", "ROOT" };
  REQUIRE(smps.get_scenario(1) == scen2);
  REQUIRE(smps.get_scenario("S02") == scen2);
}

TEST_CASE("test-create-complex-scenario-structure", "[highs_smps]") {
   std::istringstream timedata("TIME NAME\n"
                                       "PERIODS\n"
                                 "C1 R1 PERIOD2\n"
                             "ENDATA");
   SmpsTimeStructure time(timedata);
   REQUIRE(time.is_valid());
   auto path = std::string(HIGHS_DIR) + "/check/instances/simple.cor";
   SmpsCoreStructure core({}, path);
   REQUIRE(core.is_valid());
  std::istringstream data("SCENARIOS DISCRETE\n"
                          "SC S01 ROOT 0.5 PERIOD2\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "SC S02 ROOT 0.3 PERIOD2\n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "SC S03 S02 0.6 PERIOD3\n"
                          "RHS R3 30\n"
                          "SC S04 S03 0.3 PERIOD4\n"
                          "RHS R4 25\n"
                          "ENDATA");
  ScenarioStructure smps(data, time, core);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_no_scenarios() == 4);
  ScenarioModifications scen1 {
      {{"R1","RHS",50}, {"R2", "RHS", 40}},
      0.5, "PERIOD2", "S01", "ROOT" };
  REQUIRE(smps.get_scenario(0) == scen1);
  REQUIRE(smps.get_scenario("S01") == scen1);
  ScenarioModifications scen2 {
      {{"R1","RHS",40}, {"R2", "RHS", 50}},
      0.3, "PERIOD2", "S02", "ROOT" };
  REQUIRE(smps.get_scenario(1) == scen2);
  REQUIRE(smps.get_scenario("S02") == scen2);
  ScenarioModifications scen3 {
      {{"R3","RHS",30}},
      0.6, "PERIOD3", "S03", "S02" };
  REQUIRE(smps.get_scenario(2) == scen3);
  REQUIRE(smps.get_scenario("S03") == scen3);
  ScenarioModifications scen4 {
      {{"R4","RHS",25}},
      0.3, "PERIOD4", "S04", "S03" };
  REQUIRE(smps.get_scenario(3) == scen4);
  REQUIRE(smps.get_scenario("S04") == scen4);
}

TEST_CASE("test-create-scenario-structure-with-comments", "[highs_smps]") {
   std::istringstream timedata("TIME NAME\n"
                                       "PERIODS\n"
                                 "C1 R1 PERIOD2\n"
                             "ENDATA");
   SmpsTimeStructure time(timedata);
   REQUIRE(time.is_valid());
   auto path = std::string(HIGHS_DIR) + "/check/instances/simple.cor";
   SmpsCoreStructure core({}, path);
   REQUIRE(core.is_valid());
  std::istringstream data("SCENARIOS DISCRETE\n"
                          "*\n"
                          "*\n"
                          "*\n"
                          "SC S01 ROOT 0.5 PERIOD2\n"
                          "*\n"
                          "RHS R1 50\n"
                          "*\n"
                          "RHS R2 40\n"
                          "**\n"
                          "SC S02 ROOT 0.3 PERIOD2\n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "SC S03 S02 0.6 PERIOD3\n"
                          "RHS R3 30\n"
                          "SC S04 S03 0.3 PERIOD4\n"
                          "RHS R4 25\n"
                          "*\n"
                          "ENDATA");
  ScenarioStructure smps(data, time, core);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_no_scenarios() == 4);
  ScenarioModifications scen1 {
      {{"R1","RHS",50}, {"R2", "RHS", 40}},
      0.5, "PERIOD2", "S01", "ROOT" };
  REQUIRE(smps.get_scenario(0) == scen1);
  REQUIRE(smps.get_scenario("S01") == scen1);
  ScenarioModifications scen2 {
      {{"R1","RHS",40}, {"R2", "RHS", 50}},
      0.3, "PERIOD2", "S02", "ROOT" };
  REQUIRE(smps.get_scenario(1) == scen2);
  REQUIRE(smps.get_scenario("S02") == scen2);
  ScenarioModifications scen3 {
      {{"R3","RHS",30}},
      0.6, "PERIOD3", "S03", "S02" };
  REQUIRE(smps.get_scenario(2) == scen3);
  REQUIRE(smps.get_scenario("S03") == scen3);
  ScenarioModifications scen4 {
      {{"R4","RHS",25}},
      0.3, "PERIOD4", "S04", "S03" };
  REQUIRE(smps.get_scenario(3) == scen4);
  REQUIRE(smps.get_scenario("S04") == scen4);
}

TEST_CASE("test-create-malformed-stochastic-structure", "[highs_smps]") {
   std::istringstream timedata("TIME NAME\n"
                                       "PERIODS\n"
                                 "C1 R1 PERIOD2\n"
                             "ENDATA");
   SmpsTimeStructure time(timedata);
   REQUIRE(time.is_valid());
   REQUIRE(time.is_valid());
   auto path = std::string(HIGHS_DIR) + "/check/instances/simple.cor";
   SmpsCoreStructure core({}, path);
   REQUIRE(core.is_valid());
  std::istringstream malformed_scenario_header("SCENARIOS DISCRETE\n"
                          "SC S01 ROOT 0.5 PERIOD2\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "SC S02 ROOT 0.3 PERIOD2\n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "SC 0.6 PERIOD3\n"
                          "RHS R3 30\n"
                          "SC S04 S03 0.3 PERIOD4\n"
                          "RHS R4 25\n"
                          "ENDATA");
  ScenarioStructure smps(malformed_scenario_header, time, core);
  REQUIRE(!smps.is_valid());
  {
    std::istringstream malformed_scenario_header2("SCENARIOS DISCRETE\n"
                            "SC S01 ROOT 0.5 PERIOD2\n"
                            "RHS R1 50\n"
                            "RHS R2 40\n"
                            "SC S02 ROOT 0.3 PERIOD2 R2\n"
                            "RHS R1 40\n"
                            "RHS R2 50\n"
                            "SC S03 S02 0.6 PERIOD3\n"
                            "RHS R3 30\n"
                            "SC S04 S03 0.3 PERIOD4\n"
                            "RHS R4 25\n"
                            "ENDATA");
    auto smps = ScenarioStructure(malformed_scenario_header2, time, core);
    REQUIRE(!smps.is_valid());
  }
  {
    std::istringstream malformed_scenario_entry("SCENARIOS DISCRETE\n"
                            "SC S01 ROOT 0.5 PERIOD2\n"
                            "RHS R1 50\n"
                            "RHS R2 40\n"
                            "SC S02 ROOT 0.3 PERIOD2\n"
                            "RHS R1 40\n"
                            "R2 50\n"
                            "SC S03 S02 0.6 PERIOD3\n"
                            "RHS R3 30\n"
                            "SC S04 S03 0.3 PERIOD4\n"
                            "RHS R4 25\n"
                            "ENDATA");
    auto smps = ScenarioStructure(malformed_scenario_entry, time, core);
    REQUIRE(!smps.is_valid());
  }
  {
    std::istringstream malformed_scenario_entry2("SCENARIOS DISCRETE\n"
                            "SC S01 ROOT 0.5 PERIOD2\n"
                            "RHS R1 50\n"
                            "RHS R2 40\n"
                            "SC S02 ROOT 0.3 PERIOD2\n"
                            "RHS R1 40 PERIOD2\n"
                            "RHS R2 50\n"
                            "SC S03 S02 0.6 PERIOD3\n"
                            "RHS R3 30\n"
                            "SC S04 S03 0.3 PERIOD4\n"
                            "RHS R4 25\n"
                            "ENDATA");
    auto smps = ScenarioStructure(malformed_scenario_entry2, time, core);
    REQUIRE(!smps.is_valid());
  }
  {
    std::istringstream empty_scenario("SCENARIOS DISCRETE\n"
                            "SC S01 ROOT 0.5 PERIOD2\n"
                            "RHS R1 50\n"
                            "RHS R2 40\n"
                            "SC S02 ROOT 0.3 PERIOD2\n"
                            "SC S03 S02 0.6 PERIOD3\n"
                            "RHS R3 30\n"
                            "SC S04 S03 0.3 PERIOD4\n"
                            "RHS R4 25\n"
                            "ENDATA");
    auto smps = ScenarioStructure(empty_scenario, time, core);
    REQUIRE(!smps.is_valid());
  }
  {
    std::istringstream missing_end("SCENARIOS DISCRETE\n"
                            "SC S01 ROOT 0.5 PERIOD2\n"
                            "RHS R1 50\n"
                            "RHS R2 40\n"
                            "SC S02 ROOT 0.3 PERIOD2\n"
                            "RHS R1 40\n"
                            "RHS R2 50\n"
                            "SC S03 S02 0.6 PERIOD3\n"
                            "RHS R3 30\n"
                            "SC S04 S03 0.3 PERIOD4\n"
                            "RHS R4 25"
                            );
    auto smps = ScenarioStructure(missing_end, time, core);
    REQUIRE(!smps.is_valid());
  }
  {
    std::istringstream malformed_end("SCENARIOS DISCRETE\n"
                            "SC S01 ROOT 0.5 PERIOD2\n"
                            "RHS R1 50\n"
                            "RHS R2 40\n"
                            "SC S02 ROOT 0.3 PERIOD2\n"
                            "RHS R1 40\n"
                            "RHS R2 50\n"
                            "SC S03 S02 0.6 PERIOD3\n"
                            "RHS R3 30\n"
                            "SC S04 S03 0.3 PERIOD4\n"
                            "RHS R4 25\n"
                            "ENDDATA");
    auto smps = ScenarioStructure(malformed_end, time, core);
    REQUIRE(!smps.is_valid());
  }
  {
    std::istringstream nonnumeric("SCENARIOS DISCRETE\n"
                            "SC S01 ROOT 0.5 PERIOD2\n"
                            "RHS R1 50\n"
                            "RHS R2 40\n"
                            "SC S02 ROOT 0.3A PERIOD2\n"
                            "RHS R1 40\n"
                            "RHS R2 50\n"
                            "SC S03 S02 0.6 PERIOD3\n"
                            "RHS R3 30\n"
                            "SC S04 S03 0.3 PERIOD4\n"
                            "RHS R4 25\n"
                            "ENDATA");
    auto smps = ScenarioStructure(nonnumeric, time, core);
    REQUIRE(!smps.is_valid());
  }
  {
    std::istringstream nonumeric2("SCENARIOS DISCRETE\n"
                            "SC S01 ROOT 0.5 PERIOD2\n"
                            "RHS R1 50\n"
                            "RHS R2 RHS\n"
                            "SC S02 ROOT 0.3 PERIOD2\n"
                            "RHS R1 40\n"
                            "RHS R2 50\n"
                            "SC S03 S02 0.6 PERIOD3\n"
                            "RHS R3 30\n"
                            "SC S04 S03 0.3 PERIOD4\n"
                            "RHS R4 25\n"
                            "ENDATA");
    auto smps = ScenarioStructure(nonumeric2, time, core);
    REQUIRE(!smps.is_valid());
  }
  {
    std::istringstream invalid_probability("SCENARIOS DISCRETE\n"
                            "SC S01 ROOT -0.5 PERIOD2\n"
                            "RHS R1 50\n"
                            "RHS R2 40\n"
                            "SC S02 ROOT 0.3 PERIOD2\n"
                            "RHS R1 40\n"
                            "RHS R2 50\n"
                            "SC S03 S02 0.6 PERIOD3\n"
                            "RHS R3 30\n"
                            "SC S04 S03 0.3 PERIOD4\n"
                            "RHS R4 25\n"
                            "ENDATA");
    auto smps = ScenarioStructure(invalid_probability, time, core);
    REQUIRE(!smps.is_valid());
  }
  {
    std::istringstream invalid_probability2("SCENARIOS DISCRETE\n"
                            "SC S01 ROOT 0.5 PERIOD2\n"
                            "RHS R1 50\n"
                            "RHS R2 40\n"
                            "SC S02 ROOT 0.3 PERIOD2\n"
                            "RHS R1 40\n"
                            "RHS R2 50\n"
                            "SC S03 S02 2.6 PERIOD3\n"
                            "RHS R3 30\n"
                            "SC S04 S03 0.3 PERIOD4\n"
                            "RHS R4 25\n"
                            "ENDATA");
    auto smps = ScenarioStructure(invalid_probability2, time, core);
    REQUIRE(!smps.is_valid());
  }
  {
    std::istringstream malformed_header("STOCH NAME\n"
                                        "SCENARIOS DISCRETE\n"
                            "SC S01 ROOT 0.5 PERIOD2\n"
                            "RHS R1 50\n"
                            "RHS R2 40\n"
                            "SC S02 ROOT 0.3 PERIOD2\n"
                            "RHS R1 40\n"
                            "RHS R2 50\n"
                            "SC S03 S02 0.6 PERIOD3\n"
                            "RHS R3 30\n"
                            "SC S04 S03 0.3 PERIOD4\n"
                            "RHS R4 25\n"
                            "ENDATA");
    auto smps = ScenarioStructure(malformed_header, time, core);
    REQUIRE(!smps.is_valid());
  }
  {
    std::istringstream malformed_header2(
                                        "SCENARIO DISCRETE\n"
                            "SC S01 ROOT 0.5 PERIOD2\n"
                            "RHS R1 50\n"
                            "RHS R2 40\n"
                            "SC S02 ROOT 0.3 PERIOD2\n"
                            "RHS R1 40\n"
                            "RHS R2 50\n"
                            "SC S03 S02 0.6 PERIOD3\n"
                            "RHS R3 30\n"
                            "SC S04 S03 0.3 PERIOD4\n"
                            "RHS R4 25\n"
                            "ENDATA");
    auto smps = ScenarioStructure(malformed_header2, time, core);
    REQUIRE(!smps.is_valid());
  }
  {
    std::istringstream malformed_header3(
                            "SC S01 ROOT 0.5 PERIOD2\n"
                            "RHS R1 50\n"
                            "RHS R2 40\n"
                            "SC S02 ROOT 0.3 PERIOD2\n"
                            "RHS R1 40\n"
                            "RHS R2 50\n"
                            "SC S03 S02 0.6 PERIOD3\n"
                            "RHS R3 30\n"
                            "SC S04 S03 0.3 PERIOD4\n"
                            "RHS R4 25\n"
                            "ENDATA");
    auto smps = ScenarioStructure(malformed_header3, time, core);
    REQUIRE(!smps.is_valid());
  }
  {
    std::istringstream end_comments(
                            "SCENARIOS DISCRETE\n"
                            "SC S01 ROOT 0.5 PERIOD2\n"
                            "RHS R1 50\n"
                            "RHS R2 40\n"
                            "SC S02 ROOT 0.3 PERIOD2\n"
                            "RHS R1 40\n"
                            "RHS R2 50\n"
                            "SC S03 S02 0.6 PERIOD3\n"
                            "RHS R3 30\n"
                            "SC S04 S03 0.3 PERIOD4\n"
                            "RHS R4 25\n"
                            "*\n"
                            "*");
    auto smps = ScenarioStructure(end_comments, time, core);
    REQUIRE(!smps.is_valid());
  }
}

TEST_CASE("test-indep-to-stochastic-tree", "[highs_smps]") {
  std::istringstream data("STOCH NAME\n"
                          "INDEP DISCRETE\n"
                          "RHS R1 50 TIME2 0.5\n"
                          "RHS R1 40 TIME2 0.3\n"
                          "RHS R1 60 TIME2 0.2\n"
                          "C1 R1 50 TIME2 0.7\n"
                          "C1 R1 30 TIME2 0.3\n"

                          "C1 R1 50 TIME3 0.5\n"
                          "C1 R1 30 TIME3 0.5\n"
                          "ENDATA");
  IndepStructure smps(data);
  REQUIRE(smps.is_valid());
  auto tree = smps.constructTree();
  REQUIRE(tree.root->get_parent() == nullptr);
  REQUIRE(tree.root->verify_children_probabilities());
  REQUIRE(tree.root->get_no_children() == 6);
  REQUIRE(tree.root->get_lp_modifications() == std::vector<LpEntry> {});
  REQUIRE(tree.root->get_node_probability() == 1.);

  std::vector<double> child_probabilities = {
    0.5 * 0.7,
    0.3 * 0.7,
    0.2 * 0.7,
    0.5 * 0.3,
    0.3 * 0.3,
    0.2 * 0.3,
  };
  std::vector<std::vector<LpEntry>> child_mods {
    {{"R1", "RHS", 50}, {"R1", "C1", 50}},
    {{"R1", "RHS", 40}, {"R1", "C1", 50}},
    {{"R1", "RHS", 60}, {"R1", "C1", 50}},
    {{"R1", "RHS", 50}, {"R1", "C1", 30}},
    {{"R1", "RHS", 40}, {"R1", "C1", 30}},
    {{"R1", "RHS", 60}, {"R1", "C1", 30}},
  };
  std::vector<std::vector<LpEntry>> sub_mods {
    {{"R1", "C1", 50}}, {{"R1", "C1", 30}}
  };
  for (int i = 0; i < 6; ++i) {
    auto & child = tree.root->get_child(i);
    REQUIRE(child->get_parent() == tree.root.get());
    REQUIRE(child->verify_children_probabilities());
    REQUIRE(child->get_no_children() == 2);
    REQUIRE(child->get_node_probability() == child_probabilities.at(i));
    REQUIRE(child->get_lp_modifications() == child_mods.at(i));
    for (int j = 0; j < 2; ++j) {
      auto & subchild = child->get_child(j);
      REQUIRE(subchild->get_parent() == child.get());
      REQUIRE(subchild->get_no_children() == 0);
      REQUIRE(subchild->get_node_probability() == 0.5);
      REQUIRE(subchild->get_lp_modifications() == sub_mods.at(j));
    }
  }
}

TEST_CASE("test-block-to-stochastic-tree", "[highs_smps]") {
  std::istringstream data("STOCH NAME\n"
                          "BLOCKS DISCRETE\n"
                          "BL BLOCK01 TIME2 0.3\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "BL BLOCK01 TIME2 0.7\n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 50\n"
                          "RHS R4 40\n"
                          "BL BLOCK02 TIME3 0.5\n"
                          "RHS R3 40\n"
                          "RHS R4 50\n"
                          "ENDATA");
  BlockStructure smps(data);
  REQUIRE(smps.is_valid());
  auto tree = smps.constructTree();
  REQUIRE(tree.root->get_parent() == nullptr);
  REQUIRE(tree.root->verify_children_probabilities());
  REQUIRE(tree.root->get_no_children() == 2);
  REQUIRE(tree.root->get_lp_modifications() == std::vector<LpEntry> {});
  REQUIRE(tree.root->get_node_probability() == 1.);

  std::vector<double> child_probabilities = { 0.3, 0.7 };
  std::vector<std::vector<LpEntry>> child_mods {
    {{"R1", "RHS", 50}, {"R2", "RHS", 40}},
    {{"R1", "RHS", 40}, {"R2", "RHS", 50}},
  };
  std::vector<std::vector<LpEntry>> sub_mods {
    {{"R3", "RHS", 50}, {"R4", "RHS", 40}},
    {{"R3", "RHS", 40}, {"R4", "RHS", 50}},
  };
  for (int i = 0; i < 2; ++i) {
    auto & child = tree.root->get_child(i);
    REQUIRE(child->get_parent() == tree.root.get());
    REQUIRE(child->verify_children_probabilities());
    REQUIRE(child->get_no_children() == 2);
    REQUIRE(child->get_node_probability() == child_probabilities.at(i));
    REQUIRE(child->get_lp_modifications() == child_mods.at(i));
    for (int j = 0; j < 2; ++j) {
      auto & subchild = child->get_child(j);
      REQUIRE(subchild->get_parent() == child.get());
      REQUIRE(subchild->get_no_children() == 0);
      REQUIRE(subchild->get_node_probability() == 0.5);
      REQUIRE(subchild->get_lp_modifications() == sub_mods.at(j));
    }
  }
}

TEST_CASE("test-scenario-structure-tree", "[highs_smps]") {
   std::string filename = std::tmpnam(nullptr);
   std::ofstream(filename) << R"(
    NAME PROBLEM
    ROWS
    	N .COST
    	E R1
    	E R2
    	E R3
    	E R4
    COLUMNS
    	C1 R1 1 R2 2
    	C1  .COST -1
    	C2  R2 4
    	C2 .COST -2
    	C2 R3 4 R4 4
    	C3 R3 4
    	C4 R4 5
    RHS
    	RHS R1 6 R2 6
    	RHS R3 6 R4 6
    ENDATA
   )";
   SmpsCoreStructure core({}, filename);
   REQUIRE(core.is_valid());

   std::istringstream timedata("TIME NAME\n"
                                       "PERIODS\n"
                                 "C1 R1 PERIOD1\n"
                                 "C3 R3 PERIOD2\n"
                                 "C4 R4 PERIOD3\n"
                             "ENDATA");
   SmpsTimeStructure time(timedata);
   REQUIRE(time.is_valid());

   core.load_time_stages(time);

   std::istringstream data("SCENARIOS DISCRETE\n"
                           "SC S01 ROOT 0.25 PERIOD1\n"
                           "RHS R1 50\n"
                           "RHS R2 40\n"
                           "SC S02 ROOT 0.25 PERIOD1\n"
                           "RHS R1 40\n"
                           "RHS R2 50\n"
                           "SC S03 S02 0.25 PERIOD2\n"
                           "RHS R3 30\n"
                           "SC S04 S03 0.25 PERIOD3\n"
                           "RHS R4 25\n"
                           "ENDATA");
   ScenarioStructure smps(data, time, core);
   REQUIRE(smps.is_valid());
   auto tree = smps.constructTree();

   REQUIRE(tree.root->get_parent() == nullptr);
   REQUIRE(tree.root->verify_children_probabilities());
   REQUIRE(tree.root->get_no_children() == 2);
   REQUIRE(tree.root->get_lp_modifications() == std::vector<LpEntry> {});
   REQUIRE(tree.root->get_node_probability() == 1.);
   REQUIRE(tree.root->get_in_tree_probability() == 1.);

   {
     auto & child = tree.root->get_child(0);
     REQUIRE(child->get_parent() == tree.root.get());
     REQUIRE(child->verify_children_probabilities());
     REQUIRE(child->get_no_children() == 1);
     REQUIRE(child->get_node_probability() == 0.25);
     REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {{"R1", "RHS", 50}, {"R2", "RHS", 40}});
     REQUIRE(child->get_in_tree_probability() == 0.25);
   }
   {
     auto & parent = tree.root->get_child(0);
     auto & child = parent->get_child(0);
     REQUIRE(child->get_parent() == parent.get());
     REQUIRE(child->verify_children_probabilities());
     REQUIRE(child->get_no_children() == 1);
     REQUIRE(child->get_node_probability() == 1.);
     REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {});
     REQUIRE(child->get_in_tree_probability() == 0.25);
   }
   {
     auto & parent = tree.root->get_child(0)->get_child(0);
     auto & child = parent->get_child(0);
     REQUIRE(child->get_parent() == parent.get());
     REQUIRE(child->verify_children_probabilities());
     REQUIRE(child->get_no_children() == 0);
     REQUIRE(child->get_node_probability() == 1.);
     REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {});
     REQUIRE(child->get_in_tree_probability() == 0.25);
   }
   {
     auto & child = tree.root->get_child(1);
     REQUIRE(child->get_parent() == tree.root.get());
     REQUIRE(child->verify_children_probabilities());
     REQUIRE(child->get_no_children() == 2);
     REQUIRE(child->get_node_probability() == 0.75);
     REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {{"R1", "RHS", 40}, {"R2", "RHS", 50}});
     REQUIRE(child->get_in_tree_probability() == 0.75);
   }
   {
     auto & parent = tree.root->get_child(1);
     auto & child = parent->get_child(0);
     REQUIRE(child->get_parent() == parent.get());
     REQUIRE(child->verify_children_probabilities());
     REQUIRE(child->get_no_children() == 1);
     REQUIRE(std::abs(child->get_node_probability() -  1./3.) < 1e-6);
     REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {});
     REQUIRE(child->get_in_tree_probability() == 0.25);
   }
   {
     auto & parent = tree.root->get_child(1)->get_child(0);
     auto & child = parent->get_child(0);
     REQUIRE(child->get_parent() == parent.get());
     REQUIRE(child->verify_children_probabilities());
     REQUIRE(child->get_no_children() == 0);
     REQUIRE(child->get_node_probability() == 1.);
     REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {});
     REQUIRE(child->get_in_tree_probability() == 0.25);
   }
   {
     auto & parent = tree.root->get_child(1);
     auto & child = parent->get_child(1);
     REQUIRE(child->get_parent() == parent.get());
     REQUIRE(child->verify_children_probabilities());
     REQUIRE(child->get_no_children() == 2);
     REQUIRE(std::abs(child->get_node_probability() - 2./3.) < 1e-6);
     REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {{"R3", "RHS", 30}});
     REQUIRE(child->get_in_tree_probability() == 0.5);
   }
   {
     auto & parent = tree.root->get_child(1)->get_child(1);
     auto & child = parent->get_child(0);
     REQUIRE(child->get_parent() == parent.get());
     REQUIRE(child->verify_children_probabilities());
     REQUIRE(child->get_no_children() == 0);
     REQUIRE(child->get_node_probability() == 0.5);
     REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {});
     REQUIRE(child->get_in_tree_probability() == 0.25);
   }
   {
     auto & parent = tree.root->get_child(1)->get_child(1);
     auto & child = parent->get_child(1);
     REQUIRE(child->get_parent() == parent.get());
     REQUIRE(child->verify_children_probabilities());
     REQUIRE(child->get_no_children() == 0);
     REQUIRE(child->get_node_probability() == 0.5);
     REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {{"R4", "RHS", 25}});
     REQUIRE(child->get_in_tree_probability() == 0.25);
   }
 }

TEST_CASE("test-scenario-structure-tree-v2", "[highs_smps]") {
   std::string filename = std::tmpnam(nullptr);
   std::ofstream(filename) << R"(
    NAME PROBLEM
    ROWS
    	N .COST
    	E R1
    	E R2
    	E R3
    	E R4
    	E R5
    	E R6
    COLUMNS
    	C1 R1 1 R2 2
    	C1  .COST -1
    	C2  R2 4
    	C2 .COST -2
    	C2 R3 4 R4 4
    	C3 R3 4
    	C4 R4 4
    	C5 R5 5 R6 5
    RHS
    	RHS R1 6 R2 6
    	RHS R3 6 R4 6
    	RHS R5 6 R6 6
    ENDATA
   )";
   SmpsCoreStructure core({}, filename);
   REQUIRE(core.is_valid());

   std::istringstream timedata("TIME NAME\n"
                                       "PERIODS\n"
                                 "C1 R1 PERIOD0\n"
                                 "C2 R2 PERIOD1\n"
                                 "C3 R3 PERIOD2\n"
                                 "C4 R4 PERIOD3\n"
                             "ENDATA");
   SmpsTimeStructure time(timedata);
   REQUIRE(time.is_valid());

   core.load_time_stages(time);

   std::istringstream data("SCENARIOS DISCRETE\n"
                           "SC S01 ROOT 0.1 PERIOD0\n"
                           "RHS R1 10\n"
                           "RHS R2 10\n"
                           "RHS R3 10\n"
                           "RHS R4 10\n"
                           "RHS R5 10\n"
                           "RHS R6 10\n"
                           "SC S02 S01 0.3 PERIOD2\n"
                           "RHS R3 20\n"
                           "RHS R4 20\n"
                           "RHS R5 20\n"
                           "SC S03 S02 0.45 PERIOD3\n"
                           "RHS R4 30\n"
                           "SC S04 S02 0.15 PERIOD3\n"
                           "RHS R4 40\n"
                           "RHS R5 40\n"
                           "ENDATA");
   ScenarioStructure smps(data, time, core);
   REQUIRE(smps.is_valid());
   auto tree = smps.constructTree();

   REQUIRE(tree.root->get_parent() == nullptr);
   REQUIRE(tree.root->verify_children_probabilities());
   REQUIRE(tree.root->get_no_children() == 1);
   REQUIRE(tree.root->get_lp_modifications() == std::vector<LpEntry> {});
   REQUIRE(tree.root->get_node_probability() == 1.);
   REQUIRE(tree.root->get_in_tree_probability() == 1.);

   {
     auto & child = tree.root->get_child(0);
     REQUIRE(child->get_parent() == tree.root.get());
     REQUIRE(child->verify_children_probabilities());
     REQUIRE(child->get_no_children() == 1);
     REQUIRE(child->get_node_probability() == 1.);
     auto mod = child->get_lp_modifications().at(0);
     REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {{"R1", "RHS", 10}});
     REQUIRE(child->get_in_tree_probability() == 1.);
   }
   {
     auto & parent = tree.root->get_child(0);
     auto & child = parent->get_child(0);
     REQUIRE(child->get_parent() == parent.get());
     REQUIRE(child->verify_children_probabilities());
     REQUIRE(child->get_no_children() == 2);
     REQUIRE(child->get_node_probability() == 1.0);
     REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {{"R2", "RHS", 10}});
     REQUIRE(child->get_in_tree_probability() == 1.);
   }
   {
     auto & parent = tree.root->get_child(0)->get_child(0);
     auto & child = parent->get_child(0);
     REQUIRE(child->get_parent() == parent.get());
     REQUIRE(child->verify_children_probabilities());
     REQUIRE(child->get_no_children() == 1);
     REQUIRE(child->get_node_probability() == 0.1);
     REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {{"R3", "RHS", 10}});
     REQUIRE(child->get_in_tree_probability() == 0.1);
   }
   {
     auto & parent = tree.root->get_child(0)->get_child(0);
     auto & child = parent->get_child(1);
     REQUIRE(child->get_parent() == parent.get());
     REQUIRE(child->verify_children_probabilities());
     REQUIRE(child->get_no_children() == 3);
     REQUIRE(child->get_node_probability() == 0.9);
     REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {{"R3", "RHS", 20}});
     REQUIRE(child->get_in_tree_probability() == 0.9);
   }
   {
     auto & parent = tree.root->get_child(0)->get_child(0)->get_child(1);
     auto & child = parent->get_child(0);
     REQUIRE(child->get_parent() == parent.get());
     REQUIRE(child->verify_children_probabilities());
     REQUIRE(child->get_no_children() == 0);
     REQUIRE(child->get_node_probability() == 1./3.);
     REQUIRE(child->get_lp_modifications() == std::vector<LpEntry>
             {{"R4", "RHS", 20}, {"R5", "RHS", 20}, {"R6", "RHS", 10}});
     REQUIRE(child->get_in_tree_probability() == 0.3);
   }
   {
     auto & parent = tree.root->get_child(0)->get_child(0)->get_child(1);
     auto & child = parent->get_child(1);
     REQUIRE(child->get_parent() == parent.get());
     REQUIRE(child->verify_children_probabilities());
     REQUIRE(child->get_no_children() == 0);
     REQUIRE(child->get_node_probability() == 0.5);
     REQUIRE(child->get_lp_modifications() == std::vector<LpEntry>
             {{"R4", "RHS", 30}, {"R5", "RHS", 20}, {"R6", "RHS", 10}});
     REQUIRE(child->get_in_tree_probability() == 0.45);
   }
   {
     auto & parent = tree.root->get_child(0)->get_child(0)->get_child(1);
     auto & child = parent->get_child(2);
     REQUIRE(child->get_parent() == parent.get());
     REQUIRE(child->verify_children_probabilities());
     REQUIRE(child->get_no_children() == 0);
     REQUIRE(child->get_node_probability() == 1./6.);
     REQUIRE(child->get_lp_modifications() == std::vector<LpEntry>
             {{"R4", "RHS", 40}, {"R5", "RHS", 40}, {"R6", "RHS", 10}});
     REQUIRE(child->get_in_tree_probability() == 0.15);
   }
}

TEST_CASE("test-empty-sparse-vector", "[highs_smps]") {
  SparseVector vec {};
  REQUIRE(vec[0] == 0);
  REQUIRE(vec[7] == 0);
  vec.set(7, 3);
  REQUIRE(vec[0] == 0);
  REQUIRE(vec[7] == 3);
  REQUIRE(vec.num_nz() == 1);
}

TEST_CASE("test-nonempty-sparse-vector", "[highs_smps]") {
  SparseVector vec {{0, 2, 3}, {-2, 1.5, 4}};
  REQUIRE(vec[0] == -2);
  REQUIRE(vec[1] == 0);
  REQUIRE(vec[2] == 1.5);
  REQUIRE(vec[3] == 4);
  REQUIRE(vec[4] == 0);
  REQUIRE(vec.num_nz() == 3);

  vec.set(0, 0);
  vec.set(1, 3);
  vec.set(3, 3);
  vec.set(5, 1);

  REQUIRE(vec[0] == 0);
  REQUIRE(vec[1] == 3);
  REQUIRE(vec[2] == 1.5);
  REQUIRE(vec[3] == 3);
  REQUIRE(vec[4] == 0);
  REQUIRE(vec[5] == 1);
  REQUIRE(vec.num_nz() == 4);

  for (int i = 0; i < 6; ++i) vec.set(i, 0);
  for (int i = 0; i < 6; ++i) REQUIRE(vec[i] == 0);
  REQUIRE(vec.num_nz() == 0);
}

TEST_CASE("test-truncate-sparse-vector", "[highs_smps]") {
  SparseVector vec {{0, 2, 3}, {-2, 1.5, 4}};
  vec.truncate(2);
  REQUIRE(vec.num_nz() == 2);
  REQUIRE(vec[0] == -2);
  REQUIRE(vec[1] == 0);
  REQUIRE(vec[2] == 1.5);
  REQUIRE(vec[3] == 0);

  vec.truncate(5);
  REQUIRE(vec.num_nz() == 2);

  vec.truncate(0);
  REQUIRE(vec.num_nz() == 0);
}

TEST_CASE("test-sparse-vector-out-of-matrix", "[highs_smps]") {
    auto A = get_test_problem().a_matrix_;
    auto vec = SparseVector::get_matrix_row(A, 2);
    REQUIRE(vec.num_nz() == 3);
    REQUIRE(vec[0] == 0);
    REQUIRE(vec[1] == 2);
    REQUIRE(vec[2] == 1);
    REQUIRE(vec[3] == 0);
    REQUIRE(vec[4] == 1);
}

TEST_CASE("test-annotate-lp-entry", "[highs_smps]") {
  auto path = std::string(HIGHS_DIR) + "/check/instances/simple.cor";
  HighsOptions opt;
  SmpsCoreStructure smps(opt, path);
  REQUIRE(smps.is_valid());
  std::vector<LpEntry> entries {{"R1", "C1", 5}, {"R2", "RHS", 4}, {".COST", "C2", 3}};
  auto annotated = smps.annotate_lp_entries(entries);
  REQUIRE(annotated == std::vector<LpIdxEntry> {
          {entries[0], false, false, 0, 0},
          {entries[1], false, true, 1, -1},
          {entries[2], true, false, -1, 1},
  });
}

TEST_CASE("test-stage-submatrix", "[highs_smps]") {
  auto path = std::string(HIGHS_DIR) + "/check/instances/simple.cor";
  HighsOptions opt;
  SmpsCoreStructure smps(opt, path);
  REQUIRE(smps.is_valid());

  
  auto time_path = std::string(HIGHS_DIR) + "/check/instances/simple.tim";
  SmpsTimeStructure time(time_path);
  REQUIRE(time.is_valid());
  REQUIRE(time.get_stage_names() == std::vector<std::string> {"TIME1", "TIME2"});


  REQUIRE(smps.load_time_stages(time));
  REQUIRE(smps.stage_submatrix.size() == 2);

  REQUIRE(smps.stage_submatrix.at("TIME1") == SubMatrixRange {0, 1, 0, 1});
  REQUIRE(smps.stage_submatrix == std::map<std::string, SubMatrixRange>{
            {"TIME1", {0, 1, 0, 1}},
            {"TIME2", {1, 2, 1, 2}}
          });
}

TEST_CASE("test-create-submatrix", "[highs_smps]") {
  SubMatrixRange range {1, 3, 4, 18};
  REQUIRE(range.num_rows() == 2);
  REQUIRE(range.num_cols() == 14);

  SubMatrixRange range2 {0, 3, 4, 4};
  REQUIRE(range2.num_rows() == 3);
  REQUIRE(range2.num_cols() == 0);
}

TEST_CASE("test-create-range-in-problem", "[highs_smps]") {
    auto lp = get_test_problem();
    Highs highs;
    highs.passModel(lp);
    SubMatrixRange range { 2, 5, 3, 7};
    REQUIRE(range.create_in_problem_range(highs) == SubMatrixRange {6, 9, 6, 10});
}

//TODO verification of A dimensionality (if the appropriate num_col are updated)
TEST_CASE("test-expand-problem-by-range", "[highs_smps]") {
    auto lp = get_test_problem();
    Highs highs;
    highs.passModel(lp);
    SubMatrixRange range { 2, 5, 3, 7};
    range.expand_problem_by_range_vars(highs, {0, 0, 0, 1, 2, 3, 4}, {0, 0, 0, 5, 6, 7, 8});
    REQUIRE(highs.getNumCol() == 10);
    std::vector<double> col_lower = {0, 0, 0, 0, 0, 0, 1, 2, 3, 4};
    std::vector<double> col_upper = {inf, inf, inf, inf, inf, inf, 5, 6, 7, 8};
    auto new_lp = highs.getModel().lp_;
    REQUIRE(new_lp.col_lower_ == col_lower);
    REQUIRE(new_lp.col_upper_ == col_upper);
    REQUIRE(new_lp.a_matrix_.start_ == new_lp.a_matrix_.start_);
    REQUIRE(new_lp.a_matrix_.index_ == new_lp.a_matrix_.index_);
    REQUIRE(new_lp.a_matrix_.value_ == new_lp.a_matrix_.value_);
}

//TODO verify bounds
TEST_CASE("test-add-node-entry", "[highs_smps]") {
  auto instance = std::string(HIGHS_DIR) + "/check/instances/simple";
  HighsOptions opt;
  SmpsCoreStructure core(opt, instance + ".cor");
  REQUIRE(core.is_valid());

  SmpsTimeStructure time(instance + ".tim");
  REQUIRE(time.is_valid());

  IndepStructure stoch(instance + ".sto");
  REQUIRE(stoch.is_valid());
  auto tree = stoch.constructTree();

  core.load_time_stages(time);
  auto const & node = tree.root->get_child(0);
  Highs highs;
  add_node_entry(core, *node, highs);

  REQUIRE(highs.getNumCol() == 1);
  REQUIRE(highs.getNumRow() == 1);
  auto expectedA = to_csr({1, 1, { 50, }});
  auto lp = highs.getModel().lp_;
  auto A = lp.a_matrix_;
  A.ensureRowwise();
  REQUIRE(A == expectedA);
  
}

TEST_CASE("test-add-node-tree-entry", "[highs_smps]") {
  auto instance = std::string(HIGHS_DIR) + "/check/instances/simple";
  HighsOptions opt;
  SmpsCoreStructure core(opt, instance + ".cor");
  REQUIRE(core.is_valid());

  SmpsTimeStructure time(instance + ".tim");
  REQUIRE(time.is_valid());

  IndepStructure stoch(instance + ".sto");
  REQUIRE(stoch.is_valid());
  auto tree = stoch.constructTree();

  core.load_time_stages(time);
  auto const & node = tree.root->get_child(0);
  Highs highs;
  add_node_tree_entries(core, *node, highs);

  REQUIRE(highs.getNumCol() == 3);
  auto expectedA = to_csr({3, 3, {
            50,0,0,
            2,100,0,
            2,0,200,
                          }});
  
  auto lp = highs.getModel().lp_;
  auto A = lp.a_matrix_;
  A.ensureRowwise();
  REQUIRE(A == expectedA);
}

TEST_CASE("test-add-tree-entry", "[highs_smps]") {
  auto instance = std::string(HIGHS_DIR) + "/check/instances/simple";
  Highs highs;
  build_stochastic_problem(highs, instance);

  REQUIRE(highs.getNumCol() == 9);

  auto expectedA = to_csr({9, 9, {
            50,0,0,0,0,0,0,0,0,
            2,100,0,0,0,0,0,0,0,
            2,0,200,0,0,0,0,0,0,
            0,0,0,30,0,0,0,0,0,
            0,0,0,2,100,0,0,0,0,
            0,0,0,2,0,200,0,0,0,
            0,0,0,0,0,0,10,0,0,
            0,0,0,0,0,0,2,100,0,
            0,0,0,0,0,0,2,0,200,
                          }});
  
  auto lp = highs.getModel().lp_;
  auto A = lp.a_matrix_;
  A.ensureRowwise();
  REQUIRE(A == expectedA);
}

TEST_CASE("test-translate-index-in-problem", "[highs_smps]") {
  Timestage2Range in_core {
    {"TIME1", {0, 0, 0, 5 }},
    {"TIME2", {0, 0, 5, 8}},
    {"TIME3", {0, 0, 8, 9}},
  };
  Timestage2Range in_problem {
    {"TIME1", {0, 0, 10, 15 }},
    {"TIME2", {0, 0, 40, 43}},
    {"TIME3", {0, 0, 108, 109}},
  };
  IdxTranslator translator {in_core, in_problem};
  REQUIRE(translator(0) == 10);
  REQUIRE(translator(1) == 11);
  REQUIRE(translator(5) == 40);
  REQUIRE(translator(7) == 42);
  REQUIRE(translator(8) == 108);
  REQUIRE(translator(9) == -1);
}

TEST_CASE("test-translate-vector-indices-in-problem", "[highs_smps]") {
  Timestage2Range in_core {
    {"TIME1", {0, 0, 0, 5 }},
    {"TIME2", {0, 0, 5, 8}},
    {"TIME3", {0, 0, 8, 9}},
  };
  Timestage2Range in_problem {
    {"TIME1", {0, 0, 10, 15 }},
    {"TIME2", {0, 0, 40, 43}},
    {"TIME3", {0, 0, 108, 109}},
  };
  SparseVector vec {{0, 1, 5, 7, 8, 9}, {1, 2, 3, 4, 5, 6}};
  IdxTranslator translator {in_core, in_problem};
  vec.translate_to_in_problem(translator);
  REQUIRE(vec.num_nz() == 6);
  std::vector<int> proper_indices {10, 11, 40, 42, 108, -1};
  for (int i = 0; i < 6; ++i)
    REQUIRE(vec[proper_indices.at(i)] == i + 1);
}

TEST_CASE("test-update-bounds", "[highs_smps]") {
  REQUIRE(update_ub(3, 0) == 0);
  REQUIRE(update_lb(-inf, 0) == -inf);

  REQUIRE(update_ub(3, inf) == inf);
  REQUIRE(update_lb(-inf, inf) == -inf);

  REQUIRE(update_ub(inf, 0) == inf);
  REQUIRE(update_lb(-2, 0) == 0);

  REQUIRE(update_ub(inf, -inf) == inf);
  REQUIRE(update_lb(-2, -inf) == -inf);

  REQUIRE(update_ub(3, 0) == 0);
  REQUIRE(update_lb(3, 0) == 0);
}


TEST_CASE("test-build-dummy-example", "[highs_smps]") {
  Highs highs;
  auto path = std::string(HIGHS_DIR) + "/check/instances/less_simple";
  auto res = build_stochastic_problem(highs, path);
  REQUIRE(res);

  REQUIRE(highs.getNumCol() == 12);
  REQUIRE(highs.getNumRow() == 12);

  auto expectedA = to_csr({12, 12, {
                          1,1,0,0,0,0,0,0,0,0,0,0
                          ,1,0,0,0,0,0,0,0,0,0,0,0
                          ,-1,0,2,0,0,0,0,0,0,0,0,0
                          ,-1,0,0,2,0,0,0,0,0,0,0,0
                          ,-1,0,0,0,3,0,0,0,0,0,0,0
                          ,-1,0,0,0,0,3,0,0,0,0,0,0
                          ,0,0,0,0,0,0,1,1,0,0,0,0
                          ,0,0,0,0,0,0,1,0,0,0,0,0
                          ,0,0,0,0,0,0,-1,0,2,0,0,0
                          ,0,0,0,0,0,0,-1,0,0,2,0,0
                          ,0,0,0,0,0,0,-1,0,0,0,3,0
                          ,0,0,0,0,0,0,-1,0,0,0,0,3
                          }});
  
  auto lp = highs.getModel().lp_;
  auto A = lp.a_matrix_;
  A.ensureRowwise();
  REQUIRE(A.value_ == expectedA.value_);
  REQUIRE(A == expectedA);

  REQUIRE(lp.row_lower_ == std::vector<double> {-inf, 1, 0, 0, 0, 0, -inf, 1, 0, 0, 0,0 });
  REQUIRE(lp.row_upper_ == std::vector<double> {5, inf, 0, 0, 0, 0, 7, inf, 0, 0, 0,0 });

   REQUIRE(lp.col_cost_ == std::vector<double> {
           0.5, 0.5, 0.5*0.5*0.2, 0.5*0.5*0.8*-5, 0.5*0.5*0.2, 0.5*0.5*0.8*-5,
           0.5, 0.5, 0.5*0.5*0.2, 0.5*0.5*0.8*-5, 0.5*0.5*0.2, 0.5*0.5*0.8*-5,
         });

}

TEST_CASE("test-filter-mods", "[highs_smps]") {
   std::string filename = std::tmpnam(nullptr);
   std::ofstream(filename) << R"(
    NAME PROBLEM
    ROWS
    	N .COST
    	E R1
    	E R2
    	E R3
    	E R4
    COLUMNS
    	C1 R1 1 R2 2
    	C1  .COST -1
    	C2  R2 4
    	C2 .COST -2
    	C2 R3 4 
    	C3 R3 4
    	C4 R4 0
    RHS
    	RHS R1 6 R2 6
    	RHS R3 6 
    ENDATA
   )";
   SmpsCoreStructure core({}, filename);
   REQUIRE(core.is_valid());

   std::istringstream timedata("TIME NAME\n"
                                       "PERIODS\n"
                                 "C1 R1 TIME1\n"
                                 "C3 R2 TIME2\n"
                                 "C4 R4 TIME3\n"
                             "ENDATA");
   SmpsTimeStructure time(timedata);
   REQUIRE(time.is_valid());

   core.load_time_stages(time);

  ScenarioModifications scen({
                               {"R1", "C1", 1},
                               {"R1", "C2", 2},
                               {"R2", "C1", 3},
                               {"R3", "C3", 4},
                               {"R4", "C4", 5},

                               {".COST", "C1", 1},
                               {".COST", "C2", 2},
                               {".COST", "C3", 4},

                               {"R1", "RHS", 1},
                               {"R1", "RHS", 2},
                               {"R2", "RHS", 3},
                               {"R3", "RHS", 4},
                             }, 1., "TIME1", "S1", "ROOT" );
  REQUIRE(scen.filter_by_proper_timestage(core, "TIME1") == BlockLpEntry {
                               {"R1", "C1", 1},
                               {"R1", "C2", 2},
                               {".COST", "C1", 1},
                               {".COST", "C2", 2},
                               {"R1", "RHS", 1},
                               {"R1", "RHS", 2},
            
          });
  REQUIRE(scen.filter_by_proper_timestage(core, "TIME2") == BlockLpEntry {
                               {"R2", "C1", 3},
                               {"R3", "C3", 4},
                               {".COST", "C3", 4},
                               {"R2", "RHS", 3},
                               {"R3", "RHS", 4},
            
          });
  REQUIRE(scen.filter_by_proper_timestage(core, "TIME3") == BlockLpEntry {
                               {"R4", "C4", 5},
            
          });
}

TEST_CASE("test-rescale-to-child", "[highs_smps]") {
  {
    Node node("t1", 0);
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0.3)));
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0.2)));
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0.1)));
    node.get_child(0)->add_child(std::unique_ptr<Node>(new Node("t3", 0.1)));
    REQUIRE(node.rescale_to_children_probability());
    REQUIRE(node.get_node_probability() == 0.6);
    REQUIRE(node.get_child(0)->get_node_probability() == 0.5);
    REQUIRE(std::abs(node.get_child(1)->get_node_probability() - 1./3.) < 1e-6);
    REQUIRE(std::abs(node.get_child(2)->get_node_probability() - 1./6.) < 1e-6);
    REQUIRE(node.get_child(0)->get_child(0)->get_node_probability() == 0.1);
  }
  {
    Node node("t1", 0);
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0.3)));
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0.6)));
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0.1)));
    node.get_child(0)->add_child(std::unique_ptr<Node>(new Node("t3", 0.1)));
    REQUIRE(node.rescale_to_children_probability());
    REQUIRE(std::abs(node.get_node_probability() - 1.));
    REQUIRE(std::abs(node.get_child(0)->get_node_probability() - 0.3) < 1e-6);
    REQUIRE(std::abs(node.get_child(1)->get_node_probability() - 0.6) < 1e-6);
    REQUIRE(std::abs(node.get_child(2)->get_node_probability() - 0.1) < 1e-6);
    REQUIRE(node.get_child(0)->get_child(0)->get_node_probability() == 0.1);
  }
  {
    Node node("t1", 0);
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0.3)));
    node.get_child(0)->add_child(std::unique_ptr<Node>(new Node("t3", 0.1)));
    REQUIRE(node.rescale_to_children_probability());
    REQUIRE(std::abs(node.get_node_probability() - 0.3) < 1e-6);
    REQUIRE(std::abs(node.get_child(0)->get_node_probability() - 1.) < 1e-6);
    REQUIRE(node.get_child(0)->get_child(0)->get_node_probability() == 0.1);
  }
  {
    Node node("t1", 0.5);
    REQUIRE(node.rescale_to_children_probability());
    REQUIRE(node.get_node_probability() == 0.5);
  }
  {
    Node node("t1", 0);
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0)));
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0)));
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0)));
    node.get_child(0)->add_child(std::unique_ptr<Node>(new Node("t3", 0.1)));
    REQUIRE(!node.rescale_to_children_probability());
  }
}

TEST_CASE("test-rescale-to-leaves", "[highs_smps]") {
  {
    Node node("t1", 0);
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0.3)));
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0.2)));
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0.1)));
    node.get_child(0)->add_child(std::unique_ptr<Node>(new Node("t3", 0.1)));
    node.get_child(0)->add_child(std::unique_ptr<Node>(new Node("t3", 0.3)));
    node.get_child(1)->add_child(std::unique_ptr<Node>(new Node("t3", 0.3)));
    REQUIRE(node.rescale_tree_to_leaf_probability());

    REQUIRE(std::abs(node.get_node_probability() - 0.8) < 1e-6);
    REQUIRE(std::abs(node.get_child(0)->get_node_probability() - 0.5) < 1e-6);
    REQUIRE(std::abs(node.get_child(0)->get_child(0)->get_node_probability() - 0.25) < 1e-6);
    REQUIRE(std::abs(node.get_child(0)->get_child(1)->get_node_probability() - 0.75) < 1e-6);
    REQUIRE(std::abs(node.get_child(1)->get_node_probability() - 0.3 / 0.8) < 1e-6);
    REQUIRE(std::abs(node.get_child(1)->get_child(0)->get_node_probability() - 1.) < 1e-6);
    REQUIRE(std::abs(node.get_child(2)->get_node_probability() - 0.125) < 1e-6);
  }
  {
    Node node("t1", 0);
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0)));
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0)));
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0.3)));
    node.get_child(0)->add_child(std::unique_ptr<Node>(new Node("t3", 0.1)));
    node.get_child(0)->add_child(std::unique_ptr<Node>(new Node("t3", 0.3)));
    node.get_child(1)->add_child(std::unique_ptr<Node>(new Node("t3", 0.3)));
    REQUIRE(node.rescale_tree_to_leaf_probability());

    REQUIRE(std::abs(node.get_node_probability() - 1.) < 1e-6);
    REQUIRE(std::abs(node.get_child(0)->get_node_probability() - 0.4) < 1e-6);
    REQUIRE(std::abs(node.get_child(0)->get_child(0)->get_node_probability() - 0.25) < 1e-6);
    REQUIRE(std::abs(node.get_child(0)->get_child(1)->get_node_probability() - 0.75) < 1e-6);
    REQUIRE(std::abs(node.get_child(1)->get_node_probability() - 0.3) < 1e-6);
    REQUIRE(std::abs(node.get_child(1)->get_child(0)->get_node_probability() - 1.) < 1e-6);
    REQUIRE(std::abs(node.get_child(2)->get_node_probability() - 0.3) < 1e-6);
  }
  {
    Node node("t1", 0);
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0.3)));
    node.get_child(0)->add_child(std::unique_ptr<Node>(new Node("t3", 0.1)));
    REQUIRE(node.rescale_tree_to_leaf_probability());

    REQUIRE(std::abs(node.get_node_probability() - 0.1) < 1e-6);
    REQUIRE(std::abs(node.get_child(0)->get_node_probability() - 1.) < 1e-6);
    REQUIRE(std::abs(node.get_child(0)->get_child(0)->get_node_probability() - 1.) < 1e-6);
  }
  {
    Node node("t1", 0.5);
    REQUIRE(node.rescale_tree_to_leaf_probability());
    REQUIRE(std::abs(node.get_node_probability() - 0.5) < 1e-6);
  }
  {
    Node node("t1", 0);
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0.3)));
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0.2)));
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0)));
    node.get_child(0)->add_child(std::unique_ptr<Node>(new Node("t3", 0.1)));
    node.get_child(0)->add_child(std::unique_ptr<Node>(new Node("t3", 0.3)));
    node.get_child(1)->add_child(std::unique_ptr<Node>(new Node("t3", 0.3)));
    REQUIRE(!node.rescale_tree_to_leaf_probability());

  }
  {
    Node node("t1", 0);
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0.3)));
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0.2)));
    node.add_child(std::unique_ptr<Node>(new Node("t2", 0.1)));
    node.get_child(0)->add_child(std::unique_ptr<Node>(new Node("t3", 0)));
    node.get_child(0)->add_child(std::unique_ptr<Node>(new Node("t3", 0)));
    node.get_child(1)->add_child(std::unique_ptr<Node>(new Node("t3", 0.3)));
    REQUIRE(!node.rescale_tree_to_leaf_probability());

  }
}

TEST_CASE("test-create-stochastic-path-translation", "[highs_smps]") {
  Node root("root");
  root.set_in_problem_range({});
  root.add_child(std::unique_ptr<Node>(new Node("t1")));
  root.get_child(0)->set_in_problem_range({2, 3, 4, 5});
  root.add_child(std::unique_ptr<Node>(new Node("t1")));
  root.get_child(1)->set_in_problem_range({3, 4, 5, 6});
  root.get_child(0)->add_child(std::unique_ptr<Node>(new Node("t2")));
  root.get_child(0)->get_child(0)->set_in_problem_range({4, 5, 6, 7});
  root.get_child(0)->add_child(std::unique_ptr<Node>(new Node("t2")));
  root.get_child(0)->get_child(1)->set_in_problem_range({5, 6, 7, 8});
  root.get_child(1)->add_child(std::unique_ptr<Node>(new Node("t2")));
  root.get_child(1)->get_child(0)->set_in_problem_range({6, 7, 8, 9});

  REQUIRE(create_stochastic_path_translation(root) == Timestage2Range {
          {"root", {}},
  });
  REQUIRE(create_stochastic_path_translation(*root.get_child(0)) == Timestage2Range {
          {"root", {}},
          {"t1", {2, 3, 4, 5}},
  });
  REQUIRE(create_stochastic_path_translation(*root.get_child(0)->get_child(0)) == Timestage2Range {
          {"root", {}},
          {"t1", {2, 3, 4, 5}},
          {"t2", {4, 5, 6, 7}},
  });
  REQUIRE(create_stochastic_path_translation(*root.get_child(1)->get_child(0)) == Timestage2Range {
          {"root", {}},
          {"t1", {3, 4, 5, 6}},
          {"t2", {6, 7, 8, 9}},
  });
}
