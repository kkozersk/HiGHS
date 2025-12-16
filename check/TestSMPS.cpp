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
  auto & children = root->get_children();
  children.at(0)->add_child(std::unique_ptr<Node>(new Node("TIME3", 0.7)));
  REQUIRE(root->verify_children_probabilities());

  REQUIRE(children.at(0)->get_parent() == root);
  REQUIRE(children.at(1)->get_parent() == root);
  REQUIRE(children.at(2)->get_parent() == root);
  REQUIRE(children.at(0)->get_children().at(0)->get_parent() == children.at(0).get());

  StochasticTree tree {std::unique_ptr<Node>(root)};
  REQUIRE(tree.root.get() == root);
}

TEST_CASE("test-create-simple-indep-structure", "[highs_smps]") {
  //TODO missing periods
  std::istringstream data("STOCH NAME\n"
                          "INDEP DISCRETE\n"
                          "RHS R1 50 TIME2 0.5\n"
                          "RHS R1 40 TIME2 0.3\n"
                          "RHS R1 60 TIME2 0.2\n"
                          "ENDATA");
  IndepStructure smps("NAME", data);
  REQUIRE(smps.is_valid());
  auto mods = smps.get_modifications();
  REQUIRE(mods.size() == 1);
  auto rvt = mods.back();
  REQUIRE(rvt.timestage == "TIME2");
  REQUIRE(rvt.rvs.size() == 1);
  auto rv = rvt.rvs.back();
  REQUIRE(rv.col == "RHS");
  REQUIRE(rv.row == "R1");
  REQUIRE(rv.values.size() == 3);

  REQUIRE(rv.values.at(0).probability == 0.5);
  REQUIRE(rv.values.at(1).probability == 0.3);
  REQUIRE(rv.values.at(2).probability == 0.2);

  REQUIRE(rv.values.at(0).value == 50);
  REQUIRE(rv.values.at(1).value == 40);
  REQUIRE(rv.values.at(2).value == 60);

  auto vec = rvt.generate_vector();
  REQUIRE(vec.size() == 3);
  REQUIRE(vec.at(0).probability == 0.5);
  REQUIRE(vec.at(1).probability == 0.3);
  REQUIRE(vec.at(2).probability == 0.2);

  REQUIRE(vec.at(0).modifications == BlockEntry{{"R1", "RHS", 50}});
  REQUIRE(vec.at(1).modifications == BlockEntry{{"R1", "RHS", 40}});
  REQUIRE(vec.at(2).modifications == BlockEntry{{"R1", "RHS", 60}});
}
