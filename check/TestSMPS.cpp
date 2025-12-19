#include <memory>
#include <sstream>
#include "io/SMPS.h"
#include "HCheckConfig.h"
#include "catch.hpp"
#include "lp_data/HighsOptions.h"

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
  REQUIRE(smps.get_col_stages() == std::vector<IndexStage> {{"C1","T1"}, {"C1", "T2"}, {"C3", "T3"}});
  REQUIRE(smps.get_row_stages() == std::vector<IndexStage> {{"R1","T1"}, {"R3", "T2"}, {"R4", "T3"}});
  REQUIRE(smps.get_stage_names() == std::vector<std::string> {"T1", "T2", "T3"});
  REQUIRE(smps.get_problem_name() == "NAME");
  REQUIRE(smps.get_stage_index("T1") == 0);
  REQUIRE(smps.get_stage_index("T2") == 1);
  REQUIRE(smps.get_stage_index("T3") == 2);

  auto path = std::string(HIGHS_DIR) + "/check/instances/fxm2.tim";
  smps = SmpsTimeStructure(path);
  REQUIRE(smps.is_valid());
  REQUIRE(smps.get_col_stages() == std::vector<IndexStage> {{"1D1IK","TIME1"}, {"SCCOL1", "TIME2"}});
  REQUIRE(smps.get_row_stages() == std::vector<IndexStage> {{".COSTA","TIME1"}, {"1DT019", "TIME2"}});
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
  for (int i = 0; i < smps_core.row_time_stage.size(); ++i)
    REQUIRE(smps_core.row_time_stage.at(i) == (i < 92 ? "TIME1" : "TIME2"));
  for (int i = 0; i < smps_core.col_time_stage.size(); ++i)
    REQUIRE(smps_core.col_time_stage.at(i) == (i < 114 ? "TIME1" : "TIME2"));
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

  auto vec = rvt.generate_vector();
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

  auto vec = rvt.generate_vector();
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
  REQUIRE(rvt1.generate_vector() == expected);
  RandomVector expected2 {
    {0.5, {{"R1", "C1", 50}}},
    {0.5, {{"R1", "C1", 30}}},
   };
  REQUIRE(rvt2.generate_vector() == expected2);
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
  REQUIRE(rvt1.generate_vector() == expected);
  RandomVector expected2 {
    {0.5, {{"R1", "C1", 50}}},
    {0.5, {{"R1", "C1", 30}}},
   };
  REQUIRE(rvt2.generate_vector() == expected2);
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
