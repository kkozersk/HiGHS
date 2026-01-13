#include <memory>
#include <sstream>
#include "io/SMPS.h"
#include "HCheckConfig.h"
#include "catch.hpp"
#include "lp_data/HighsOptions.h"
#include "util/HighsSparseMatrix.h"

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
  // REQUIRE(smps.get_row_stages() == std::vector<IndexStage> {{"R1","T1"}, {"R3", "T2"}, {"R4", "T3"}});
  REQUIRE(smps.get_stage_names() == std::vector<std::string> {"T1", "T2", "T3"});
  REQUIRE(smps.get_problem_name() == "NAME");
  REQUIRE(smps.get_stage_index("T1") == 0);
  REQUIRE(smps.get_stage_index("T2") == 1);
  REQUIRE(smps.get_stage_index("T3") == 2);

  auto path = std::string(HIGHS_DIR) + "/check/instances/fxm2.tim";
  smps = SmpsTimeStructure(path);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_entries() == std::vector<TimeStageEntry> {{".COSTA", "1D1IK","TIME1"}, {"1DT019", "SCCOL1", "TIME2"}});
  // REQUIRE(smps.get_col_stages() == std::vector<IndexStage> {{"1D1IK","TIME1"}, {"SCCOL1", "TIME2"}});
  // REQUIRE(smps.get_row_stages() == std::vector<IndexStage> {{".COSTA","TIME1"}, {"1DT019", "TIME2"}});
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
  // for (int i = 0; i < smps_core.row_time_stage.size(); ++i)
    // REQUIRE(smps_core.row_time_stage.at(i) == (i < 92 ? "TIME1" : "TIME2"));
  // for (int i = 0; i < smps_core.col_time_stage.size(); ++i)
    // REQUIRE(smps_core.col_time_stage.at(i) == (i < 114 ? "TIME1" : "TIME2"));
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
  std::istringstream data("SCENARIOS DISCRETE\n"
                          "SC S01 ROOT 0.5 PERIOD2\n"
                          "RHS R1 50\n"
                          "RHS R2 40\n"
                          "SC S02 ROOT 0.3 PERIOD2\n"
                          "RHS R1 40\n"
                          "RHS R2 50\n"
                          "ENDATA");
  ScenarioStructure smps(data);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_no_scenarios() == 2);
  ScenarioModifications scen1 {
      {{"R1","RHS",50}, {"R2", "RHS", 40}},
      0.5, "PERIOD2", "S01", "ROOT" };
  REQUIRE(smps.get_scenario(0) == scen1);
  ScenarioModifications scen2 {
      {{"R1","RHS",40}, {"R2", "RHS", 50}},
      0.3, "PERIOD2", "S02", "ROOT" };
  REQUIRE(smps.get_scenario(1) == scen2);
}

TEST_CASE("test-create-complex-scenario-structure", "[highs_smps]") {
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
  ScenarioStructure smps(data);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_no_scenarios() == 4);
  ScenarioModifications scen1 {
      {{"R1","RHS",50}, {"R2", "RHS", 40}},
      0.5, "PERIOD2", "S01", "ROOT" };
  REQUIRE(smps.get_scenario(0) == scen1);
  ScenarioModifications scen2 {
      {{"R1","RHS",40}, {"R2", "RHS", 50}},
      0.3, "PERIOD2", "S02", "ROOT" };
  REQUIRE(smps.get_scenario(1) == scen2);
  ScenarioModifications scen3 {
      {{"R3","RHS",30}},
      0.6, "PERIOD3", "S03", "S02" };
  REQUIRE(smps.get_scenario(2) == scen3);
  ScenarioModifications scen4 {
      {{"R4","RHS",25}},
      0.3, "PERIOD4", "S04", "S03" };
  REQUIRE(smps.get_scenario(3) == scen4);
}

TEST_CASE("test-create-scenario-structure-with-comments", "[highs_smps]") {
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
  ScenarioStructure smps(data);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_no_scenarios() == 4);
  ScenarioModifications scen1 {
      {{"R1","RHS",50}, {"R2", "RHS", 40}},
      0.5, "PERIOD2", "S01", "ROOT" };
  REQUIRE(smps.get_scenario(0) == scen1);
  ScenarioModifications scen2 {
      {{"R1","RHS",40}, {"R2", "RHS", 50}},
      0.3, "PERIOD2", "S02", "ROOT" };
  REQUIRE(smps.get_scenario(1) == scen2);
  ScenarioModifications scen3 {
      {{"R3","RHS",30}},
      0.6, "PERIOD3", "S03", "S02" };
  REQUIRE(smps.get_scenario(2) == scen3);
  ScenarioModifications scen4 {
      {{"R4","RHS",25}},
      0.3, "PERIOD4", "S04", "S03" };
  REQUIRE(smps.get_scenario(3) == scen4);
}

TEST_CASE("test-create-malformed-stochastic-structure", "[highs_smps]") {
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
  ScenarioStructure smps(malformed_scenario_header);
  REQUIRE(!smps.is_valid());

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
  smps = ScenarioStructure(malformed_scenario_header2);
  REQUIRE(!smps.is_valid());

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
  smps = ScenarioStructure(malformed_scenario_entry);
  REQUIRE(!smps.is_valid());

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
  smps = ScenarioStructure(malformed_scenario_entry2);
  REQUIRE(!smps.is_valid());

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
  smps = ScenarioStructure(empty_scenario);
  REQUIRE(!smps.is_valid());

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
  smps = ScenarioStructure(missing_end);
  REQUIRE(!smps.is_valid());

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
  smps = ScenarioStructure(malformed_end);
  REQUIRE(!smps.is_valid());

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
  smps = ScenarioStructure(nonnumeric);
  REQUIRE(!smps.is_valid());

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
  smps = ScenarioStructure(nonumeric2);
  REQUIRE(!smps.is_valid());

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
  smps = ScenarioStructure(invalid_probability);
  REQUIRE(!smps.is_valid());

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
  smps = ScenarioStructure(invalid_probability2);
  REQUIRE(!smps.is_valid());

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
  smps = ScenarioStructure(malformed_header);
  REQUIRE(!smps.is_valid());

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
  smps = ScenarioStructure(malformed_header2);
  REQUIRE(!smps.is_valid());

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
  smps = ScenarioStructure(malformed_header3);
  REQUIRE(!smps.is_valid());

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
  smps = ScenarioStructure(end_comments);
  REQUIRE(!smps.is_valid());
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

TEST_CASE("test-scenario-structure-tree-without-fill", "[highs_smps]") {
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
  ScenarioStructure smps(data);
  REQUIRE(smps.is_valid());
  auto tree = smps.constructTree();

  REQUIRE(tree.root->get_parent() == nullptr);
  REQUIRE(!tree.root->verify_children_probabilities());
  REQUIRE(tree.root->get_no_children() == 2);
  REQUIRE(tree.root->get_lp_modifications() == std::vector<LpEntry> {});
  REQUIRE(tree.root->get_node_probability() == 1.);

  {
    auto & child = tree.root->get_child(0);
    REQUIRE(child->get_parent() == tree.root.get());
    REQUIRE(child->get_no_children() == 0);
    REQUIRE(child->get_node_probability() == 0.5);
    REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {{"R1", "RHS", 50}, {"R2", "RHS", 40}});
  }
  {
    auto & child = tree.root->get_child(1);
    REQUIRE(child->get_parent() == tree.root.get());
    REQUIRE(!child->verify_children_probabilities());
    REQUIRE(child->get_no_children() == 1);
    REQUIRE(child->get_node_probability() == 0.3);
    REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {{"R1", "RHS", 40}, {"R2", "RHS", 50}});
  }
  {
    auto & parent = tree.root->get_child(1);
    auto & child = parent->get_child(0);
    REQUIRE(child->get_parent() == parent.get());
    REQUIRE(!child->verify_children_probabilities());
    REQUIRE(child->get_no_children() == 1);
    REQUIRE(child->get_node_probability() == 0.6);
    REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {{"R3", "RHS", 30}});
  }
  {
    auto & parent = tree.root->get_child(1)->get_child(0);
    auto & child = parent->get_child(0);
    REQUIRE(child->get_parent() == parent.get());
    REQUIRE(!child->verify_children_probabilities());
    REQUIRE(child->get_no_children() == 0);
    REQUIRE(child->get_node_probability() == 0.3);
    REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {{"R4", "RHS", 25}});
  }
}

TEST_CASE("test-scenario-structure-tree-with-fill", "[highs_smps]") {
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
  ScenarioStructure smps(data);
  REQUIRE(smps.is_valid());
  auto tree = smps.constructTree();
  tree.fill_tree();

  REQUIRE(tree.root->get_parent() == nullptr);
  REQUIRE(tree.root->verify_children_probabilities());
  REQUIRE(tree.root->get_no_children() == 3);
  REQUIRE(tree.root->get_lp_modifications() == std::vector<LpEntry> {});
  REQUIRE(tree.root->get_node_probability() == 1.);

  {
    auto & child = tree.root->get_child(0);
    REQUIRE(child->get_parent() == tree.root.get());
    REQUIRE(child->get_no_children() == 0);
    REQUIRE(child->get_node_probability() == 0.5);
    REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {{"R1", "RHS", 50}, {"R2", "RHS", 40}});
  }
  {
    auto & child = tree.root->get_child(1);
    REQUIRE(child->get_parent() == tree.root.get());
    REQUIRE(child->verify_children_probabilities());
    REQUIRE(child->get_no_children() == 2);
    REQUIRE(child->get_node_probability() == 0.3);
    REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {{"R1", "RHS", 40}, {"R2", "RHS", 50}});
  }
  {
    auto & child = tree.root->get_child(2);
    REQUIRE(child->get_parent() == tree.root.get());
    REQUIRE(child->get_no_children() == 0);
    REQUIRE(std::abs(child->get_node_probability()-0.2) < 1e-6 );
    REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {});
  }
  {
    auto & parent = tree.root->get_child(1);
    auto & child = parent->get_child(0);
    REQUIRE(child->get_parent() == parent.get());
    REQUIRE(child->verify_children_probabilities());
    REQUIRE(child->get_no_children() == 2);
    REQUIRE(child->get_node_probability() == 0.6);
    REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {{"R3", "RHS", 30}});
  }
  {
    auto & parent = tree.root->get_child(1)->get_child(0);
    auto & child = parent->get_child(0);
    REQUIRE(child->get_parent() == parent.get());
    REQUIRE(child->get_no_children() == 0);
    REQUIRE(child->get_node_probability() == 0.3);
    REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {{"R4", "RHS", 25}});
  }
  {
    auto & parent = tree.root->get_child(1)->get_child(0);
    auto & child = parent->get_child(1);
    REQUIRE(child->get_parent() == parent.get());
    REQUIRE(child->get_no_children() == 0);
    REQUIRE(child->get_node_probability() == 0.7);
    REQUIRE(child->get_lp_modifications() == std::vector<LpEntry> {});
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

TEST_CASE("test-shift-sparse-vector", "[highs_smps]") {
  SparseVector vec {{0, 2, 3}, {-2, 1.5, 4}};
  vec.shift_indices(2);
  REQUIRE(vec.num_nz() == 3);
  REQUIRE(vec[0] == 0);
  REQUIRE(vec[1] == 0);
  REQUIRE(vec[2] == -2);
  REQUIRE(vec[3] == 0);
  REQUIRE(vec[4] == 1.5);
  REQUIRE(vec[5] == 4);
  REQUIRE(vec[6] == 0);

  vec.shift_indices(0);
  REQUIRE(vec.num_nz() == 3);
  REQUIRE(vec[0] == 0);
  REQUIRE(vec[1] == 0);
  REQUIRE(vec[2] == -2);
  REQUIRE(vec[3] == 0);
  REQUIRE(vec[4] == 1.5);
  REQUIRE(vec[5] == 4);
  REQUIRE(vec[6] == 0);
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
}

TEST_CASE("test-add-tree-entry", "[highs_smps]") {
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
  Highs highs;
  add_tree_entries(core, tree, highs);

  REQUIRE(highs.getNumCol() == 9);
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
  REQUIRE(translate_index_to_in_problem(0, in_core, in_problem) == 10);
  REQUIRE(translate_index_to_in_problem(1, in_core, in_problem) == 11);
  REQUIRE(translate_index_to_in_problem(5, in_core, in_problem) == 40);
  REQUIRE(translate_index_to_in_problem(7, in_core, in_problem) == 42);
  REQUIRE(translate_index_to_in_problem(8, in_core, in_problem) == 108);
  REQUIRE(translate_index_to_in_problem(9, in_core, in_problem) == -1);
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
  vec.translate_to_in_problem(in_core, in_problem);
  REQUIRE(vec.num_nz() == 6);
  std::vector<int> proper_indices {10, 11, 40, 42, 108, -1};
  for (int i = 0; i < 6; ++i)
    REQUIRE(vec[proper_indices.at(i)] == i + 1);
}
