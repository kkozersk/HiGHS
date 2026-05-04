#include "benders/Benders.h"
#include "HCheckConfig.h"
#include "io/SMPS.h"
#include "io/FilereaderLp.h"
#include "io/FilereaderMps.h"
#include "Highs.h"
#include "catch.hpp"
#include <cmath>
// #include <fstream>
#include <numeric>
#include <set>
#include <vector>

const double inf = kHighsInf;
const int max_iters = 200;

void modify(Highs & m) {
  double ac = 5;
  m.setOptionValue("solver", kHipoString);
  // m.setOptionValue("solver", kSimplexString);
  m.setOptionValue("optimality_tolerance", ac);
  m.setOptionValue("ipm_optimality_tolerance", ac);
  m.setOptionValue("run_crossover", kHighsOffString);
  m.setOptionValue("max_centring_steps", 1000);
  m.setOptionValue("presolve", kHighsOnString);
  m.setOptionValue("centring_gamma", 1-1e-5);
}

std::pair<std::vector<OptionValue>, MasterAdaptationParams> getMasterOpts(bool use_simplex=true) {
  double ipm_acc = 1;
  double ipm_feas = 10;
  std::string solver = use_simplex ? kSimplexString : kHipoString;
  ipm_feas = solver == kHipoString ? ipm_feas : 1e-8;
  return {
    {
      OptionValue("dual_feasibility_tolerance", ipm_feas),
      OptionValue("primal_feasibility_tolerance", ipm_feas),
      OptionValue("solver", solver),
      OptionValue("optimality_tolerance", ipm_acc),
      OptionValue("ipm_optimality_tolerance", ipm_acc),
      OptionValue("run_crossover", kHighsOffString),
      OptionValue("max_centring_steps", 1000),
      OptionValue("presolve", kHighsOnString),
      OptionValue("centring_gamma", 1-1e-4),

      OptionValue("recentring_step", 0.8),
      // OptionValue("parallel", kHighsOffString)
      // OptionValue("ipm_iteration_limit", 0),
    },
    MasterAdaptationParams(ipm_acc, ipm_feas)
  };
}

auto dir = std::string(HIGHS_DIR) + "/check/instances/stoch/";
struct SmpsTestCase {
  std::string corefile, timefile, stochfile, note;
  double sub_lb;
  bool aggregate;
  double expected;
};

enum Start {
  obj_start,
  feas_start,
  core_obj_start,
  core_feas_start
};

std::vector<SmpsTestCase> tests {
  // {dir + "fxm/fxm.cor", dir + "fxm/fxm2.tim", dir + "fxm/fxm2_6.sto", "indep-2", 0, false, 18417.066},
  // {dir + "fxm/fxm.cor", dir + "fxm/fxm2.tim", dir + "fxm/fxm2_16.sto", "indep-2-bigger", 0, false, 18416.759},
  // {dir + "fxm/fxm.cor", dir + "fxm/fxm3.tim", dir + "fxm/fxm3_6.sto", "indep-3", 0, false, 18615.74},
  // {dir + "fxm/fxm.cor", dir + "fxm/fxm3.tim", dir + "fxm/fxm3_16.sto", "indep-3-bigger", 0, false, 18438.995},
  // {dir + "fxm/fxm.cor", dir + "fxm/fxm4.tim", dir + "fxm/fxm4_6.sto", "indep-4", 0, false, 18616.05},
  // {dir + "fxm/fxm.cor", dir + "fxm/fxm4.tim", dir + "fxm/fxm4_16.sto", "indep-4-bigger", 0, false, 18438.995},

  // {dir + "pltexp/pltexpA2.cor", dir + "pltexp/pltexpA2.tim", dir + "pltexp/pltexpA2_6.sto", "block-2", -100, false, -9.47935},
  // {dir + "pltexp/pltexpA2.cor", dir + "pltexp/pltexpA2.tim", dir + "pltexp/pltexpA2_16.sto", "block-2-bigger", -100, false, -9.66331},
  // {dir + "pltexp/pltexpA3.cor", dir + "pltexp/pltexpA3.tim", dir + "pltexp/pltexpA3_6.sto", "block-3", -100, false, -13.9694},
  // {dir + "pltexp/pltexpA3.cor", dir + "pltexp/pltexpA3.tim", dir + "pltexp/pltexpA3_16.sto", "block-3-bigger", -100, false, -14.2675},
  // {dir + "pltexp/pltexpA4.cor", dir + "pltexp/pltexpA4.tim", dir + "pltexp/pltexpA4_6.sto", "block-4", -100, false, -19.5994},
  
//   //TODO fix

//   // {dir + "pltexp/pltexpA4.cor", dir + "pltexp/pltexpA4.tim", dir + "pltexp/pltexpA4_16.sto", "block-4-bigger", -100, false, kHighsInf},
//   // {dir + "pltexp/pltexpA5.cor", dir + "pltexp/pltexpA5.tim", dir + "pltexp/pltexpA5_6.sto", "block-5", -100, false, kHighsInf},
//   // {dir + "pltexp/pltexpA5.cor", dir + "pltexp/pltexpA5.tim", dir + "pltexp/pltexpA5_16.sto", "block-5-bigger", -100, false, kHighsInf},
//   // {dir + "pltexp/pltexpA6.cor", dir + "pltexp/pltexpA6.tim", dir + "pltexp/pltexpA6_6.sto", "block-6", -100, false, kHighsInf},
//   // {dir + "pltexp/pltexpA6.cor", dir + "pltexp/pltexpA6.tim", dir + "pltexp/pltexpA6_16.sto", "block-6-bigger", -100, false, kHighsInf},

//   // {dir + "pltexp/pltexpA3.cor", dir + "pltexp/pltexpA3.tim", dir + "pltexp/pltexpB3_6.sto", "blockB-3", -100, false, -13.6432},
//   // {dir + "pltexp/pltexpA4.cor", dir + "pltexp/pltexpA4.tim", dir + "pltexp/pltexpB4_6.sto", "blockB-4", -100, false, -17.9282},
//   // {dir + "pltexp/pltexpA5.cor", dir + "pltexp/pltexpA5.tim", dir + "pltexp/pltexpB5_6.sto", "blockB-5", -100, false, -23.8434},

  // {dir + "storm/stormG2.cor", dir + "storm/stormG2.tim", dir + "storm/stormG2_8.sto", "storm-8", 0, false, 15535235.73},
  // {dir + "storm/stormG2.cor", dir + "storm/stormG2.tim", dir + "storm/stormG2_27.sto", "storm-27", 0, false, 15508982.306},
  // {dir + "storm/stormG2.cor", dir + "storm/stormG2.tim", dir + "storm/stormG2_125.sto", "storm-125", 0, false, 15512091.185},
  // {dir + "storm/stormG2.cor", dir + "storm/stormG2.tim", dir + "storm/stormG2_1000.sto", "storm-1000", 0, false, 15802590.244},

//   {dir + "sg/sgpf5y3.cor", dir + "sg/sgpf5y3.tim", dir + "sg/sgpf5y3.sce", "scen-3", -1e4, false, -3084.234},
//   {dir + "sg/sgpf5y4.cor", dir + "sg/sgpf5y4.tim", dir + "sg/sgpf5y4.sce", "scen-4", -1e4, false, -4248.131},
//   {dir + "sg/sgpf5y5.cor", dir + "sg/sgpf5y5.tim", dir + "sg/sgpf5y5.sce", "scen-5", -1e4, false, -5593.008},
//   // {dir + "sg/sgpf5y6.cor", dir + "sg/sgpf5y6.tim", dir + "sg/sgpf5y6.sce", "scen-6", -1e4, false, kHighsInf},
// // };

// // std::vector<SmpsTestCase> slptests {

//   {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.1", "4node-1", 0, false, kHighsInf},
//   {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.2", "4node-2", 0, false, kHighsInf},
//   {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.4", "4node-4", 0, false, kHighsInf},
//   {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.8", "4node-8", 0, false, kHighsInf},
//   {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.16", "4node-16", 0, false, kHighsInf},
//   {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.32", "4node-32", 0, false, kHighsInf},
//   {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.64", "4node-64", 0, false, kHighsInf},
//   {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.128", "4node-128", 0, false, kHighsInf},
//   {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.256", "4node-256", 0, false, kHighsInf},
//   {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.512", "4node-512", 0, false, kHighsInf},
//   // {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.1024", "4node-1024", 0, false, kHighsInf},
//   // {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.2048", "4node-2048", 0, false, kHighsInf},
//   // {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.4096", "4node-4096", 0, false, kHighsInf},
//   // {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.8192", "4node-8192", 0, false, kHighsInf},
//   // {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.16384", "4node-16384", 0, false, kHighsInf},
//   // {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.32768", "4node-32768", 0, false, kHighsInf},

//   {dir + "chem/chem.cor", dir + "chem/chem.tim", dir + "chem/chem.sto", "chem", -1e6, false, -13009.167},
//   {dir + "chem/chem.cor.base", dir + "chem/chem.tim", dir + "chem/chem.sto", "chem-base", -1e6, false, -13009.167},
//   {dir + "airlift/AIRL.cor", dir + "airlift/AIRL.tim", dir + "airlift/AIRL.sto.first", "airlift1", -1e6, false, 249101.672}, 
//   {dir + "airlift/AIRL.cor", dir + "airlift/AIRL.tim", dir + "airlift/AIRL.sto.second", "airlift2", -1e6, false, 269665.498},
//   // {dir + "phone/phone.cor", dir + "phone/phone.tim", dir + "phone/phone.sto", "phone", 0, false, 36.9}, 
//   // {dir + "phone/phone.cor", dir + "phone/phone.tim", dir + "phone/phone.sto.1", "phone1", 0, false, 36.9}, 
//   {dir + "stocfor2/stocfor2.cor", dir + "stocfor2/stocfor2.tim", dir + "stocfor2/stocfor2.sto", "stocfor", -1e6, false, -39772.448},
//   // {dir + "assets/assets.cor", dir + "assets/assets.tim", dir + "assets/assets.sto.small", "assets-small", -1e4, false, -723.839},
//   // {dir + "assets/assets.cor", dir + "assets/assets.tim", dir + "assets/assets.sto.large", "assets-large", -1e4, false, -695.963},
};

void zero_costs(Highs & highs) {
  std::vector<double> zeros (highs.getNumCol(), 0);
  highs.changeColsCost(0, highs.getNumCol()-1, zeros.data());
}

std::pair<int, int> core_size(std::string corefile) {
  FilereaderMps mps;
  HighsModel model;
  REQUIRE(mps.readModelFromFile(HighsOptions(), corefile, model) == FilereaderRetcode::kOk);
  Highs highs;
  highs.passModel(model);
  return {highs.getNumCol(), highs.getNumRow()};
}

void run(std::string corefile, std::string timefile, std::string stochfile, std::string note, double subproblem_lb) {
  Highs highs;
  auto core_and_tree = build_stochastic_tree(corefile, timefile, stochfile);
  auto & tree = core_and_tree.second;
  auto core = core_and_tree.first;
  auto & stage_1st = tree.root->get_child(0);
  auto & stage_2nd = stage_1st->get_child(0);
  int no_scenarios = stage_1st->get_no_children();
  REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
  auto const & master_range = core.stage_submatrix.at(stage_1st->get_timestage());
  int no_master_vars = master_range.col_idx_end - master_range.col_idx_begin;
  int no_master_rows = master_range.row_idx_end - master_range.row_idx_begin;

  auto const & sub_range = core.stage_submatrix.at(stage_2nd->get_timestage());
  int no_sub_vars = sub_range.col_idx_end - sub_range.col_idx_begin;
  int no_sub_rows = sub_range.row_idx_end - sub_range.row_idx_begin;
  // REQUIRE(highs.getNumCol() == no_master_vars + no_sub_vars * no_subs);
  // REQUIRE(highs.getNumRow() == no_master_rows + no_sub_rows * no_subs);
  highs.run();
  auto expected = highs.getObjectiveValue();
  auto lp = highs.getModel().lp_;
  // lp.ensureRowwise();
  std::set<int> master_vars;
  for (int i = 0; i < no_master_vars; ++i) master_vars.emplace(i);
  Highs highs2;
  // highs2.passModel(lp);
  // highs2.passModel(core_and_tree.first);
  // zero_costs(highs);
  // modify(highs2);
  // highs2.run();
  // std::vector<double> starting_point;
  auto starting_point = highs2.getSolution().col_value;
  std::ofstream("/tmp/iteration.csv", std::ios::app) << note << "," << std::endl;
  std::ofstream("/tmp/cut_distances.csv", std::ios::app) << note << "," << std::endl;
  std::ofstream("/tmp/ubd_lbd.csv", std::ios::app) << note << "," << std::endl;

  auto opts = getMasterOpts();
  auto res = benders(lp, master_vars, starting_point, subproblem_lb, 1e-3, max_iters, opts.first, opts.second);
  // res.note(0, note);
  res.note(expected, note);
  REQUIRE(std::fabs(res.result - expected) < 1e-3);
}

void multi_run(SmpsTestCase smps, bool aggregate, int max_scenarios, Start start, bool use_simplex) {
  auto core_and_tree = build_stochastic_tree(smps.corefile, smps.timefile, smps.stochfile);
  auto & tree = core_and_tree.second;
  auto core = core_and_tree.first;
  auto & stage_1st = tree.root->get_child(0);
  auto & stage_2nd = stage_1st->get_child(0);
  int no_scenarios = stage_1st->get_no_children();
  if (no_scenarios > max_scenarios)
    return;
  smps.note = (aggregate ? "single-" : "multi-") + smps.note;
  


  auto const & master_range = core.stage_submatrix.at(stage_1st->get_timestage());
  int no_master_vars = master_range.col_idx_end - master_range.col_idx_begin;
  int no_master_rows = master_range.row_idx_end - master_range.row_idx_begin;

  auto const & sub_range = core.stage_submatrix.at(stage_2nd->get_timestage());
  int no_sub_vars = sub_range.col_idx_end - sub_range.col_idx_begin;
  int no_sub_rows = sub_range.row_idx_end - sub_range.row_idx_begin;

  CsvLogger log ("/tmp/dataset.csv");
  log << smps.note << no_scenarios << no_master_vars << no_master_rows << no_sub_vars << no_sub_rows;
  log.newline();

  // Highs highs;
  // REQUIRE(build_stochastic_problem(highs, smps.corefile, smps.timefile, smps.stochfile));
  // REQUIRE(highs.getNumCol() == no_master_vars + no_sub_vars * no_scenarios);
  // REQUIRE(highs.getNumRow() == no_master_rows + no_sub_rows * no_scenarios);
  // highs.run();
  // auto expected = highs.getObjectiveValue();
  
  std::vector<double> starting_point;
  Highs highs2;
  switch (start) {
    case obj_start:
      REQUIRE(build_stochastic_problem(highs2, smps.corefile, smps.timefile, smps.stochfile));
      zero_costs(highs2); smps.note += "-obj_start";
    break;
    case feas_start:
      smps.note += "-feas_start";
    break;
    case core_obj_start:
      highs2.passModel(core); smps.note += "-core_obj_start";
    break;
    case core_feas_start:
      highs2.passModel(core); zero_costs(highs2); smps.note += "-core_feas_start";
    break;
  }
  highs2.run();
  starting_point = highs2.getSolution().col_value;
  if (start == feas_start) REQUIRE(starting_point.empty());
  std::ofstream("/tmp/linking.csv", std::ios::app) << smps.note << ",";
  std::ofstream("/tmp/iteration.csv", std::ios::app) << smps.note  << "," << std::endl;
  std::ofstream("/tmp/cut_distances.csv", std::ios::app) << smps.note  << "," << std::endl;
  std::ofstream("/tmp/ubd_lbd.csv", std::ios::app) << smps.note  << "," << std::endl;
  std::ofstream("/tmp/ipm_stats.csv", std::ios::app) << smps.note  << "," << std::endl;

  auto opts =   getMasterOpts(use_simplex);
  auto res = benders_l_shaped(core_and_tree.first, core_and_tree.second, starting_point, smps.sub_lb, 1e-3, max_iters, aggregate, opts.first, opts.second);
  res.note(smps.expected, smps.note);
  if (smps.expected != kHighsInf) {
    REQUIRE(smps.note == smps.note);
    REQUIRE(no_scenarios == no_scenarios);
    REQUIRE(std::fabs(res.result - smps.expected) < 1e-3);
  }
  
}
//   // {dir + "assets/assets.cor", dir + "assets/assets.tim", dir + "assets/assets.sto.small", "assets-small", -1e4, false, -723.839},
//   // {dir + "assets/assets.cor", dir + "assets/assets.tim", dir + "assets/assets.sto.large", "assets-large", -1e4, false, -695.963},

// auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/assets/assets";
// TEST_CASE("test-2", "[highs-benders]") {
//   run(instance + ".cor", instance + ".tim", instance + ".sto.small", "chem-old", -1e6);
// }

// TEST_CASE("test1", "[highs-benders]") {
//   multi_run({instance + ".cor", instance + ".tim", instance + ".sto.small", "chem", -1e6, false, kHighsInf}, true, 1e6, feas_start);
// }

// TEST_CASE("run-instances", "[highs-benders]") {
//   for (auto test : tests)
//     multi_run(test, true, 1e6, feas_start);
// }


// TEST_CASE("test-quick", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/sg/sgpf5y6";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + ".tim";
//   auto stochfile = instance + ".sce";
//   multi_run({corefile, timefile, stochfile, "4node", 0, true, 0}, 1e6, feas_start);

// }


// TEST_CASE("test-quick", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/environ/env";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + ".tim";
//   auto stochfile = instance + ".sto.3780";
  
//   auto core_and_tree = build_stochastic_tree(corefile, timefile, stochfile);
//   auto & tree = core_and_tree.second;
//   auto core = core_and_tree.first;
//   auto & stage_1st = tree.root->get_child(0);
//   auto & stage_2nd = stage_1st->get_child(0);
//   int no_scenarios = stage_1st->get_no_children();


//   auto const & master_range = core.stage_submatrix.at(stage_1st->get_timestage());
//   int no_master_vars = master_range.col_idx_end - master_range.col_idx_begin;
//   int no_master_rows = master_range.row_idx_end - master_range.row_idx_begin;

//   auto const & sub_range = core.stage_submatrix.at(stage_2nd->get_timestage());
//   int no_sub_vars = sub_range.col_idx_end - sub_range.col_idx_begin;
//   int no_sub_rows = sub_range.row_idx_end - sub_range.row_idx_begin;

//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   REQUIRE(highs.getNumCol() == no_master_vars + no_sub_vars * no_scenarios);
//   REQUIRE(highs.getNumRow() == no_master_rows + no_sub_rows * no_scenarios);
//   highs.run();
//   REQUIRE(no_scenarios == no_scenarios);
//   REQUIRE(highs.getObjectiveValue() == 0.);
// }

// TEST_CASE("run-slp-instances", "[highs-benders]") {
//   // for (auto test : tests)
//   //   multi_run(test, 250, obj_start);
//   for (auto test : slptests)
//     multi_run(test, 1e6, feas_start);
//   // for (auto test : tests)
//   //   multi_run(test, 250, core_obj_start);
//   // for (auto test : tests)
//   //   multi_run(test, 250, core_feas_start);
// }
// TEST_CASE("run-istances", "[highs-benders]") {
//   multi_run({dir + "storm/stormG2.cor", dir + "storm/stormG2.tim", dir + "storm/stormG2_8.sto", "storm-8", 0, true, 15535235.73}, 1e4, obj_start);
//   //  multi_run({dir + "sg/sgpf5y3.cor", dir + "sg/sgpf5y3.tim", dir + "sg/sgpf5y3.sce", "scen-3", -1e4, true, -3084.234}, 1e4, feas_start);
// }

// TEST_CASE("run-instances", "[highs-benders]") {
  // for (auto test : tests)
  //   multi_run(test, 250, obj_start);
  // for (auto test : tests)
  //   multi_run(test, true, 1e6, feas_start);
  // for (auto test : tests)
  //   multi_run(test, 250, core_obj_start);
  // for (auto test : tests)
  //   multi_run(test, 250, core_feas_start);
// }

// HighsLp get_simple_test_problem() {
//   /*
//     Problem
//     min 5 + [1 2 -1]^T x
//     s.t.
//       [1 1 0         = 1
//       0 0 1  [x0     <= 1
//       0 0 0   x1     = 0
//       1 0 1   x2]    <= 2
//       1 1 1]         >= 1.5
//       x >= 0
//   */
//   std::vector<HighsInt> csr_index {0,1,2,0,2,0,1,2};
//   std::vector<double> csr_values(8, 1);
//   std::vector<HighsInt> csr_starts {0,2,3,3,5,8};
//   HighsLp lp;
//   lp.offset_ = 5;
//   lp.num_col_ = 3;
//   lp.num_row_ = 5;
//   lp.col_lower_ = {0, 0, 0};
//   lp.col_upper_ = {inf, inf, inf};
//   lp.col_cost_ = {1, 2, -1};
//   lp.row_lower_ = {1, -inf, 0, -inf, 1.5};
//   lp.row_upper_ = {1, 1, 0, 2, inf};
//   lp.a_matrix_.format_ = MatrixFormat::kRowwise;
//   lp.a_matrix_.start_ = csr_starts;
//   lp.a_matrix_.index_ = csr_index;
//   lp.a_matrix_.value_ = csr_values;
//   lp.a_matrix_.num_row_ = 5;
//   lp.a_matrix_.num_col_ = 3;
//   return lp;
// }

// HighsLp get_simple_multi_test_problem() {
//   /*
//     Problem
//     min 5 + [1 2 -1 -3]^T x
//     s.t.
//       [1 1 0 0         = 1
//       0 0 1 0  [x0     <= 1
//       0 0 0 0  x1      = 0
//       1 0 1 0  x2      <= 2
//       1 1 1 0  x3]     >= 1.5
//       0 0 0 1          <= 1
//       1 0 0 1]         = 1.5
//       ]          
//       x >= 0
//   */
//   std::vector<HighsInt> csr_index {0,1,2,0,2,0,1,2,3,0,3};
//   std::vector<double> csr_values(11, 1);
//   std::vector<HighsInt> csr_starts {0,2,3,3,5,8,9,11};
//   HighsLp lp;
//   lp.offset_ = 5;
//   lp.num_col_ = 4;
//   lp.num_row_ = 7;
//   lp.col_lower_ = {0, 0, 0, 0};
//   lp.col_upper_ = {inf, inf, inf, inf};
//   lp.col_cost_ = {1, 2, -1, -3};
//   lp.row_lower_ = {1, -inf, 0, -inf, 1.5, -inf, 1.5};
//   lp.row_upper_ = {1, 1, 0, 2, inf, 1, 1.5};
//   lp.a_matrix_.format_ = MatrixFormat::kRowwise;
//   lp.a_matrix_.start_ = csr_starts;
//   lp.a_matrix_.index_ = csr_index;
//   lp.a_matrix_.value_ = csr_values;
//   lp.a_matrix_.num_row_ = 7;
//   lp.a_matrix_.num_col_ = 4;
//   return lp;
// }

// HighsLp get_second_test_problem() {
//   /*
//   min  [1 5 3 4 5 -6]^T x
//   [
//   	1 1  0  0 0 0      <=  4  
//   	1 0 -1 0 0 0       <=  0
//   	0 2  1 0 1 0   x   >=  3
//   	1 1  0 1 0 0       ==  5 
//   	0 0  0 0 1 0       <=  2
//   	1 0  0 0 0 1       ==  1
//   ]
//       x >= 0
//   */
//   std::vector<HighsInt> csr_index {0,1, 0,2, 1,2,4, 0,1,3, 4, 0,5};
//   std::vector<double> csr_values {1,1, 1,-1, 2,1,1, 1,1,1, 1, 1,1 };
//   std::vector<HighsInt> csr_starts {0,2,4,7,10,11,13};
//   HighsLp lp;
//   lp.offset_ = 0;
//   lp.num_col_ = 6;
//   lp.num_row_ = 6;
//   lp.col_lower_ = {0, 0, 0, 0, 0, 0};
//   lp.col_upper_ = {inf, inf, inf, inf, inf, inf};
//   lp.col_cost_ = {1, 5, 3, 4, 5, -6};
//   lp.row_lower_ = {-inf, -inf, 3, 5, -inf, 1};
//   lp.row_upper_ = {4, 0, inf, 5, 2, 1};
//   lp.a_matrix_.format_ = MatrixFormat::kRowwise;
//   lp.a_matrix_.start_ = csr_starts;
//   lp.a_matrix_.index_ = csr_index;
//   lp.a_matrix_.value_ = csr_values;
//   lp.a_matrix_.num_row_ = 6;
//   lp.a_matrix_.num_col_ = 6;
//   return lp;
// }
// TEST_CASE("test-find-column-index", "[highs_benders]") {
//     /*
//       Matrix [
//         1 1 0
//         0 0 1
//         0 0 0
//         1 0 1
//         1 1 1
//       ]
//     */
//     std::vector<HighsInt> csr_starts {0,2,3,3,5,8};
//     std::vector<HighsInt> column_indices {0, 0, 1, 3, 3, 4, 4, 4};
//     std::vector<HighsInt> results;
//     for (int i = 0; i < 8; ++i)
//       results.push_back(find_row_index(csr_starts, i));
//     REQUIRE(results == column_indices);
// }
  
// TEST_CASE("test-division", "[highs_benders]") {
//   /*
//     Matrix [
//       1 0
//       0 1
//       1 1
//     ], with first column being a complicating variable
//   */
//   std::vector<HighsInt> csr_index {0, 1, 0, 1};
//   std::vector<HighsInt> csr_starts {0, 1, 2, 4};
//   std::set<HighsInt> master_indices {0};
//   auto result = divide_rows(csr_index, csr_starts, master_indices);
//   REQUIRE(result.inset_only_rows == std::set<HighsInt>{0, 2});
//   REQUIRE(result.other_rows == std::set<HighsInt>{1, 2});
//   REQUIRE(result.mixed_rows == std::set<HighsInt>{2});
// }
  
// TEST_CASE("test-division-v2", "[highs_benders]") {
//     /*
//       Matrix [
//         1 1 0
//         0 0 1
//         0 0 0
//         1 0 1
//         1 1 1
//       ], with complicating variables 0, 1
//     */
//     std::vector<HighsInt> csr_index {0,1,2,0,2,0,1,2};
//     std::vector<HighsInt> csr_starts {0,2,3,3,5,8};
//     std::set<HighsInt> master_indices {0, 1};
//     auto result = divide_rows(csr_index, csr_starts, master_indices);
//     REQUIRE(result.inset_only_rows == std::set<HighsInt>{0, 3, 4});
//     REQUIRE(result.other_rows == std::set<HighsInt>{1, 3, 4});
//     REQUIRE(result.mixed_rows == std::set<HighsInt>{3, 4});
// }

// TEST_CASE("test-division-v3", "[highs_benders]") {
//     /*
//       Matrix [
//         ​101001​
//          010100​
//          100010​
//          010101​
//          001010​
//          100101​
//       ], with complicating variables 1, 2
//     */
//     std::vector<HighsInt> csr_index {0,2,5,1,3,0,4,1,3,5,2,4,0,3,5};
//     std::vector<HighsInt> csr_starts {0,3,5,7,10,12,15};
//     std::set<HighsInt> master_indices {1, 2};
//     auto result = divide_rows(csr_index, csr_starts, master_indices);
//     REQUIRE(result.inset_only_rows == std::set<HighsInt>{0, 1, 3, 4});
//     REQUIRE(result.other_rows == std::set<HighsInt>{0, 1, 2, 3, 4, 5});
//     REQUIRE(result.mixed_rows == std::set<HighsInt>{0, 1, 3, 4});
// }

// TEST_CASE("test-division-empty-master", "[highs_benders]") {
//     /*
//       Matrix [
//         ​101001​
//          010100​
//          100010​
//          010101​
//          001010​
//          100101​
//       ], with no complicating variables 
//     */
//     std::vector<HighsInt> csr_index {0,2,5,1,3,0,4,1,3,5,2,4,0,3,5};
//     std::vector<HighsInt> csr_starts {0,3,5,7,10,12,15};
//     std::set<HighsInt> master_indices {};
//     auto result = divide_rows(csr_index, csr_starts, master_indices);
//     REQUIRE(result.inset_only_rows== std::set<HighsInt>{});
//     REQUIRE(result.other_rows == std::set<HighsInt>{0, 1, 2, 3, 4, 5});
//     REQUIRE(result.mixed_rows == std::set<HighsInt>{});
// }

// TEST_CASE("test-division-full-master", "[highs_benders]") {
//     /*
//       Matrix [
//         ​101001​
//          010100​
//          100010​
//          010101​
//          001010​
//          100101​
//       ], with every variable being complicating
//     */
//     std::vector<HighsInt> csr_index {0,2,5,1,3,0,4,1,3,5,2,4,0,3,5};
//     std::vector<HighsInt> csr_starts {0,3,5,7,10,12,15};
//     std::set<HighsInt> master_indices {0, 1, 2, 3, 4, 5};
//     auto result = divide_rows(csr_index, csr_starts, master_indices);
//     REQUIRE(result.inset_only_rows == std::set<HighsInt>{0, 1, 2, 3, 4, 5});
//     REQUIRE(result.other_rows == std::set<HighsInt>{});
//     REQUIRE(result.mixed_rows == std::set<HighsInt>{});
// }

// TEST_CASE("test-division-disjoint", "[highs_benders]") {
//     /*
//       Matrix [
//          1 0
//          0 1
//       ], with 0 as a complicating variable
//     */
//     std::vector<HighsInt> csr_index {0, 1};
//     std::vector<HighsInt> csr_starts {0, 1, 2};
//     std::set<HighsInt> master_indices {0};
//     auto result = divide_rows(csr_index, csr_starts, master_indices);
//     REQUIRE(result.inset_only_rows == std::set<HighsInt>{0});
//     REQUIRE(result.other_rows == std::set<HighsInt>{1});
// }

// TEST_CASE("test-set-master-values-highs-instance", "[highs-benders]") {
//   /*
//     Matrix [
//       1 0
//       0 1
//       1 1
//     ], with first column being a complicating variable
//   */
//   std::vector<HighsInt> csr_index {0, 1, 0, 1};
//   std::vector<double> csr_values = {1, 1, 1, 1};
//   std::vector<HighsInt> csr_starts {0, 1, 2, 4};
//   std::set<HighsInt> master_indices {0};
//   std::vector<double> master_values {1};
//   HighsLp lp;
//   lp.num_col_ = 2;
//   lp.num_row_ = 3;
//   lp.col_lower_.assign(lp.num_col_, 0);
//   lp.col_upper_.assign(lp.num_col_, inf);
//   lp.col_cost_.assign(lp.num_col_, 0);
//   lp.row_lower_.assign(lp.num_row_, 0);
//   lp.row_upper_.assign(lp.num_row_, 0);
//   lp.a_matrix_.format_ = MatrixFormat::kRowwise;
//   lp.a_matrix_.start_ = csr_starts;
//   lp.a_matrix_.index_ = csr_index;
//   lp.a_matrix_.value_ = csr_values;
//   Highs instance;
//   instance.passModel(lp);
//   auto result = fix_master_variables(instance, master_indices, master_values);
//   auto res_lp  = instance.getLp();
//   REQUIRE(result);
//   REQUIRE(res_lp.col_lower_ == std::vector<double> {1, 0});
//   REQUIRE(res_lp.col_upper_ == std::vector<double> {1, inf});

// }

// TEST_CASE("test-set-complement", "[highs-benders]") {
//   REQUIRE(sequence_complement({0, 1, 2}, 5) == std::set<HighsInt>{3, 4});
//   REQUIRE(sequence_complement({0}, 5) == std::set<HighsInt>{1, 2, 3, 4});
//   REQUIRE(sequence_complement({}, 5) == std::set<HighsInt>{0, 1, 2, 3, 4});
//   REQUIRE(sequence_complement({0, 3}, 2) == std::set<HighsInt>{1});
//   REQUIRE(sequence_complement({0, 3}, 0) == std::set<HighsInt>{});
// }

// TEST_CASE("test-create-master", "[highs-benders]") {
//   /*
//     Matrix [
//       1 1 0
//       0 0 1
//       0 0 0
//       1 0 1
//       1 1 1
//     ], with complicating variables 0, 1
//   */
//   std::vector<HighsInt> csr_index {0,1,2,0,2,0,1,2};
//   std::vector<double> csr_values(8, 1);
//   std::vector<HighsInt> csr_starts {0,2,3,3,5,8};
//   HighsLp lp;
//   lp.offset_ = 5;
//   lp.num_col_ = 3;
//   lp.num_row_ = 5;
//   lp.col_lower_ = {1, 2, 3};
//   lp.col_upper_ = {1, 2, 3};
//   lp.col_cost_ = {1, 2, 3};
//   lp.row_lower_ = {1, 2, 3, 4, 5};
//   lp.row_upper_ = {1, 2, 3, 4, 5};
//   lp.a_matrix_.format_ = MatrixFormat::kRowwise;
//   lp.a_matrix_.start_ = csr_starts;
//   lp.a_matrix_.index_ = csr_index;
//   lp.a_matrix_.value_ = csr_values;
//   lp.a_matrix_.num_row_ = 5;
//   lp.a_matrix_.num_col_ = 3;
//   std::set<HighsInt> master_variables {0, 1};

//   /*
//     We expect:
//     min 5 + x0 + 2x1 + z
//     s.t.
//     [ 1 1 0    [ x0  = 1
//       0 0 0 ]    x1  = 3
//                  z ] 
//       1 <= x0 <=1, 2 <= x1 <= 2, z = 0
//   */
//   HighsLp expected;
//   expected.offset_ = 5;
//   expected.num_col_ = 3;
//   expected.num_row_ = 2;
//   expected.col_lower_ = {1, 2, 0};
//   expected.col_upper_= {1, 2, 0};
//   expected.col_cost_ = {1, 2, 1};
//   expected.row_upper_ = {1, 3};
//   expected.row_lower_= {1, 3};
//   expected.a_matrix_.format_ = MatrixFormat::kRowwise; //TODO: can we leave it this way?
//   expected.a_matrix_.start_ = {0, 2, 2};
//   expected.a_matrix_.index_ = {0, 1};
//   expected.a_matrix_.value_ = {1, 1};
//   expected.a_matrix_.num_row_ = 2; //TODO: delete the empty row?
//   expected.a_matrix_.num_col_ = 3;


//   RowDivision rd;
//   rd.inset_only_rows = {0};
//   rd.other_rows = {1, 3, 4};
//   rd.mixed_rows = {3, 4};
//   Highs master;
//   create_master_problem(master, lp, master_variables, rd.other_rows);
//   auto result = master.getLp();
//   result.ensureRowwise();
//   REQUIRE(result.a_matrix_ == expected.a_matrix_);
//   REQUIRE(result.col_cost_  == expected.col_cost_);
//   REQUIRE(result.col_upper_  == expected.col_upper_);
//   REQUIRE(result.col_lower_ == expected.col_lower_);
//   REQUIRE(result == expected);
// }

// // TEST_CASE("test-create-subproblem", "[highs-benders]") {
// //   auto lp = get_simple_test_problem();
// //   std::set<HighsInt> master_variables {0, 1};

// //   HighsLp expected = lp;
// //   expected.offset_ = 0;
// //   expected.col_cost_ = {0, 0, -1};
// //   Highs subproblem;
// //   create_subproblem(subproblem, lp, master_variables);
// //   auto result = subproblem.getLp();
// //   result.ensureRowwise();
// //   REQUIRE(result  == expected);
// // }

// TEST_CASE("test-create-feas-subproblem", "[highs-benders]") {
//   /*
//     We expect:
//     min alpha1_- + alpha2_+
//     s.t.
//       [1 1 0  0   0            = 1
//        0 0 1  0   0  [x0       <= 1
//        0 0 0  0   0   x1       = 0
//        1 0 1 -1   0   x2       <= 2
//        1 1 1  0  1]   alpha_+  >= 1.5
//                    alpha_-]
//       x >= 0, alpha >= 0
//   */
//   auto lp = get_simple_test_problem();
//   std::set<HighsInt> master_variables {0, 1};

//   std::vector<HighsInt> csr_index {0, 1, 2, 0, 2, 3, 0, 1, 2, 4};
//   std::vector<double> csr_values {1, 1, 1, 1, 1, -1, 1, 1, 1, 1};
//   std::vector<HighsInt> csr_starts {0, 2, 3, 3, 6, 10};
//   HighsLp expected;
//   expected.offset_ = 0;
//   expected.num_col_ = 5;
//   expected.num_row_ = 5;
//   expected.col_lower_ = {0, 0, 0, 0, 0};
//   expected.col_upper_ = {inf, inf, inf, inf, inf};
//   expected.col_cost_ = {0, 0, 0, 1, 1};
//   expected.row_lower_ = {1, -inf, 0, -inf, 1.5};
//   expected.row_upper_ = {1, 1, 0, 2, inf};
//   expected.a_matrix_.format_ = MatrixFormat::kRowwise;
//   expected.a_matrix_.start_ = csr_starts;
//   expected.a_matrix_.index_ = csr_index;
//   expected.a_matrix_.value_ = csr_values;
//   expected.a_matrix_.num_row_ = 5;
//   expected.a_matrix_.num_col_ = 5;

//   Highs subproblem;
//   subproblem.passModel(lp);
//   auto division = divide_rows(lp.a_matrix_, master_variables);
//   REQUIRE(division.mixed_rows == std::set<HighsInt> {3, 4});
//   create_feasibility_subproblem(subproblem, lp, master_variables, division.mixed_rows);
//   auto result = subproblem.getLp();
//   result.ensureRowwise();
//   REQUIRE(result.a_matrix_.index_  == expected.a_matrix_.index_);
//   REQUIRE(result.a_matrix_.start_ == expected.a_matrix_.start_);
//   REQUIRE(result.a_matrix_.value_ == expected.a_matrix_.value_);
//   REQUIRE(result.a_matrix_  == expected.a_matrix_);
//   REQUIRE(result.col_cost_  == expected.col_cost_);
//   REQUIRE(result.col_lower_  == expected.col_lower_);
//   REQUIRE(result.row_lower_  == expected.row_lower_);
//   REQUIRE(result.col_upper_  == expected.col_upper_);
//   REQUIRE(result.row_upper_  == expected.row_upper_);
//   REQUIRE(result  == expected);
// }

// TEST_CASE("test-create-feas-subproblem-2", "[highs-benders]") {
//   /*
//     We expect:
//     min 0 + 0^T x + alpha1_+ + alpha1_- +  alpha2_- +  alpha3_+ 
//     s.t.
//       [1 1 0 1 -1  0 0               =  1
//        0 0 1 0  0  0 0     [x        <= 1
//        0 0 0 0  0  0 0      alpha]   =  0
//        1 0 1 0  0 -1 0               <= 2
//        1 1 1 0  0  0 1  ]            >= 1.5

//       ]
//       x >= 0, alpha >= 0
//   */
//   auto lp = get_simple_test_problem();
//   std::set<HighsInt> master_variables {0};

//   std::vector<HighsInt> csr_index {0, 1, 3, 4, 2, 0, 2, 5, 0, 1, 2, 6};
//   std::vector<double> csr_values {1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1};
//   std::vector<HighsInt> csr_starts {0, 4, 5, 5, 8, 12};
//   HighsLp expected;
//   expected.offset_ = 0;
//   expected.num_col_ = 7;
//   expected.num_row_ = 5;
//   expected.col_lower_ = {2, 0, 0, 0, 0, 0, 0};
//   expected.col_upper_ = {2, inf, inf, inf, inf, inf, inf};
//   expected.col_cost_ = {0, 0, 0, 1, 1, 1, 1};
//   expected.row_lower_ = {1, -inf, 0, -inf, 1.5};
//   expected.row_upper_ = {1, 1, 0, 2, inf};
//   expected.a_matrix_.format_ = MatrixFormat::kRowwise;
//   expected.a_matrix_.start_ = csr_starts;
//   expected.a_matrix_.index_ = csr_index;
//   expected.a_matrix_.value_ = csr_values;
//   expected.a_matrix_.num_row_ = 5;
//   expected.a_matrix_.num_col_ = 7;

//   Highs subproblem;
//   subproblem.passModel(lp);
//   auto division = divide_rows(lp.a_matrix_, master_variables);
//   create_feasibility_subproblem(subproblem, lp, master_variables, division.mixed_rows);
//   fix_master_variables(subproblem, master_variables, {2});
//   auto result = subproblem.getLp();
//   result.ensureRowwise();
//   REQUIRE(result.a_matrix_.start_  == expected.a_matrix_.start_);
//   REQUIRE(result.a_matrix_.index_  == expected.a_matrix_.index_);
//   REQUIRE(result.a_matrix_.value_ == expected.a_matrix_.value_);
//   REQUIRE(result.col_lower_ == expected.col_lower_);
//   REQUIRE(result.col_upper_ == expected.col_upper_);
//   REQUIRE(result  == expected);
//   subproblem.run();
//   REQUIRE(subproblem.getModelStatus() == HighsModelStatus::kOptimal);
// }

// TEST_CASE("test-get-multipliers", "[highs-benders]") {
//   auto lp = get_simple_test_problem();
//   std::set<HighsInt> master_variables {0}; //  with complicating variables 0

//   Highs subproblem;
//   create_subproblem(subproblem, lp, master_variables);

//   fix_master_variables(subproblem, master_variables, {0});
//   auto status = subproblem.run();
//   REQUIRE(status == HighsStatus::kOk);
//   REQUIRE(subproblem.getModelStatus() == HighsModelStatus::kOptimal);
//   // auto multipliers = get_all_multipliers(subproblem);
//   // REQUIRE(multipliers == std::vector<double> {2, 2, -1});
//   // REQUIRE(multipliers == std::vector<double> {2, 0, 0});
//   auto master_multipliers = get_master_multipliers(subproblem, master_variables);
//   REQUIRE(master_multipliers == std::vector<double> {2});

//   fix_master_variables(subproblem, master_variables, {2});
//   status = subproblem.run();
//   REQUIRE(status == HighsStatus::kOk);
//   REQUIRE(subproblem.getModelStatus() == HighsModelStatus::kInfeasible);
//   bool has_dual_ray;
//   double dual_ray[5];
//   subproblem.getDualRay(has_dual_ray, dual_ray);
//   REQUIRE(has_dual_ray);
//   REQUIRE(std::vector<double>(dual_ray, dual_ray + 5) == std::vector<double> {-1, 0, 0, 0, 0});
// }

// TEST_CASE("test-add-objective-cut", "[highs-benders]") {
//   auto lp = get_simple_test_problem();
//   std::set<HighsInt> master_variables {0}; //  with complicating variables 0
//   BendersProblems problems;
//   decompose_problem(problems, lp, master_variables);
//   auto & master = problems.master;
//   auto & subproblem = problems.subproblem;
  
//   auto num_col = master.getLp().num_col_;
//   auto num_row = master.getLp().num_row_;

//   std::vector<double> master_values {0};
//   fix_master_variables(subproblem, master_variables, master_values);
//   auto status = subproblem.run();
//   REQUIRE(status == HighsStatus::kOk);
//   REQUIRE(subproblem.getModelStatus() == HighsModelStatus::kOptimal);
  
//   add_cut(master, subproblem, master_variables, master_values, CutType::Objective);
//   unfreeze_mu(master, master_variables);
//   auto new_master = master.getLp();
//   /*
//     We want the new master to be in form:
//       min 5 + x_0 + z
//       0 x_0 + 0 z = 0
//       2 x_0 + z >= 1

//       x_0 >= 0, z free
//     */
//   HighsLp expected;
//   expected.offset_ = 5;
//   expected.num_col_ = 2;
//   expected.num_row_ = 2;
//   expected.col_lower_ = {0, -inf};
//   expected.col_upper_ = {inf, inf};
//   expected.col_cost_ = {1, 1};
//   expected.row_lower_ = {0, 1};
//   expected.row_upper_ = {0, inf};
//   expected.a_matrix_.format_ = MatrixFormat::kRowwise;
//   expected.a_matrix_.start_ = {0,0,2};
//   expected.a_matrix_.index_ = {0,1};
//   expected.a_matrix_.value_ = {2,1};
//   expected.a_matrix_.num_row_ = 2;
//   expected.a_matrix_.num_col_ = 2;
//   REQUIRE(new_master.num_row_ == num_row + 1);
//   REQUIRE(new_master == expected);
// }

// TEST_CASE("test-create_nonzero-vector", "[highs-benders]") {
//   REQUIRE(create_nonzero_vector({1, 2}) == NonZeroVector {2, {0, 1}, {1, 2}});
//   REQUIRE(create_nonzero_vector({1, 1, 1}) == NonZeroVector {3, {0, 1, 2}, {1, 1, 1}});
//   REQUIRE(create_nonzero_vector({1, 0, 2, 0, 0, 3, 0}) == NonZeroVector {3, {0, 2, 5}, {1, 2, 3}});
//   REQUIRE(create_nonzero_vector({0, 0}) == NonZeroVector {0, {}, {}});
//   REQUIRE(create_nonzero_vector({}) == NonZeroVector {0, {}, {}});
//  }

// TEST_CASE("test-discover-master-variables", "[highs-benders]") {
//   REQUIRE(discover_master_variables({"EC1", "EC2", "EG1", "EH2", "ECC", "EC4"}, "EC\\d") == std::set<HighsInt> {0, 1, 5});
//   REQUIRE(discover_master_variables({"EC1", "EC2", "EG1", "EH2", "ECC", "EC4"}, "EX") == std::set<HighsInt> {});
//   REQUIRE(discover_master_variables({}, "EC\\d") == std::set<HighsInt> {});
// }

// // TEST_CASE("test-add-feasibility-cut", "[highs-benders]") {
// //   auto lp = get_simple_test_problem();
// //   std::set<HighsInt> master_variables {0}; //  with complicating variables 0
// //   BendersProblems problems;
// //   decompose_problem(problems, lp, master_variables);
// //   auto & master = problems.master;
// //   auto & subproblem = problems.subproblem;
// //   auto & feas_subproblem = problems.feas_subproblem;

// //   auto num_col = master.getLp().num_col_;
// //   auto num_row = master.getLp().num_row_;

// //   std::vector<double> master_values {2};
// //   fix_master_variables(subproblem, master_variables, master_values);
// //   auto status = HighsStatus::kOk;
// //   // auto status = subproblem.run();
// //   REQUIRE(status == HighsStatus::kOk);
// //   // REQUIRE(subproblem.getModelStatus() == HighsModelStatus::kInfeasible);
// //   fix_master_variables(feas_subproblem, master_variables, master_values);
// //   status = feas_subproblem.run();
// //   REQUIRE(status == HighsStatus::kOk);
// //   REQUIRE(feas_subproblem.getModelStatus() == HighsModelStatus::kOptimal);

// //   double dual_objective;
// //   feas_subproblem.getDualObjectiveValue(dual_objective);
// //   auto multipliers = get_master_multipliers(feas_subproblem, master_variables);
// //   auto old_value_multiple = std::inner_product(multipliers.begin(), multipliers.end(), master_values.begin(), 0.0);
// //   auto nonzero_multipliers = create_nonzero_vector(multipliers);
// //   add_cut(master, feas_subproblem, master_variables, master_values, CutType::Feasibility);
// //   auto new_master = master.getLp();
// //   /*
// //     We want the new master to be in form:
// //       min 5 + x_0 + z
// //       0 x_0 + 0 z = 0
// //       -1 x_0 + 0z >= -1

// //       x_0 >= 0
// //     */
// //   HighsLp expected;
// //   expected.offset_ = 5;
// //   expected.num_col_ = 2;
// //   expected.num_row_ = 2;
// //   expected.col_lower_ = {0, mu_lb};
// //   expected.col_upper_ = {inf, inf};
// //   expected.col_cost_ = {1, 1};
// //   expected.row_lower_ = {0, -1};
// //   expected.row_upper_ = {0, inf};
// //   expected.a_matrix_.format_ = MatrixFormat::kRowwise;
// //   expected.a_matrix_.start_ = {0,0,1};
// //   expected.a_matrix_.index_ = {0};
// //   expected.a_matrix_.value_ = {-1}; //?
// //   expected.a_matrix_.num_row_ = 2;
// //   expected.a_matrix_.num_col_ = 2;
// //   REQUIRE(new_master.num_row_ == num_row + 1);
// //   REQUIRE(expected.a_matrix_.start_  ==  new_master.a_matrix_.start_);
// //   REQUIRE(expected.a_matrix_.index_ ==  new_master.a_matrix_.index_);
// //   REQUIRE(expected.a_matrix_.value_ ==  new_master.a_matrix_.value_);
// //   REQUIRE(expected.a_matrix_ ==  new_master.a_matrix_);
// //   REQUIRE(expected.col_lower_ ==  new_master.col_lower_);
// //   REQUIRE(expected.col_upper_ ==  new_master.col_upper_);
// //   REQUIRE(expected.col_cost_ == new_master.col_cost_ );
// //   REQUIRE(expected.row_lower_ ==  new_master.row_lower_);
// //   REQUIRE(expected.row_upper_ ==  new_master.row_upper_);
// //   REQUIRE(new_master == expected);
// // }

// TEST_CASE("test-solve-simple-system", "[highs-benders]") {
//   auto lp = get_simple_test_problem();
//   lp.col_names_ = {"m1", "s1", "s2"};
//   Highs nodecomp;
//   nodecomp.passModel(lp);
//   nodecomp.run();
//   auto expected = nodecomp.getObjectiveValue();
//   auto res = benders(lp, "m\\d", {2});
//   REQUIRE(std::abs(res.result- expected) < 1e-3);
//   REQUIRE(res.iter == 3);

// }

// TEST_CASE("test-solve-simple-system-2", "[highs-benders]") {
//   auto lp = get_simple_test_problem();
//   lp.col_names_ = {"m1", "s1", "s2"};
//   Highs nodecomp;
//   nodecomp.passModel(lp);
//   nodecomp.run();
//   auto expected = nodecomp.getObjectiveValue();
//   // auto res = benders(lp, "m\\d", {2});
//   auto res = benders(lp, std::set<HighsInt>{0, 2}, {2, 0});
//   REQUIRE(std::abs(res.result- expected) < 1e-3);
//   REQUIRE(res.iter == 3);
// }

// TEST_CASE("test-solve-second-system", "[highs-benders]") {
//   auto lp = get_second_test_problem();
//   lp.col_names_ = {"m1", "m2", "m3", "s1", "s2", "s3"};
//   Highs nodecomp;
//   nodecomp.passModel(lp);
//   nodecomp.run();
//   auto expected = nodecomp.getObjectiveValue();
//   // auto res = benders(lp, "m\\d", {0, 1.5, 0});
//   auto res = benders(lp, "m\\d", {0, 0, 0});
//   REQUIRE(std::abs(res.result- expected) < 1e-3);
//   REQUIRE(res.iter == 4);
// }

// TEST_CASE("test-solve-blending", "[highs-benders]") {
//   auto path = std::string(HIGHS_DIR) + "/check/instances/blending.mps";
//   Highs nodecomp;
//   nodecomp.readModel(path);
//   auto lp = nodecomp.getLp();
//   nodecomp.run();
//   auto expected = nodecomp.getObjectiveValue();
//   lp.ensureRowwise();
//   auto res = benders(lp, "P0", std::vector<double> (40, 0));
//   REQUIRE(std::abs(res.result- expected) < 1e-3);
//   REQUIRE(res.iter == 1); //TODO suprisingly low
// }

// TEST_CASE("test-solve-afiro", "[highs-benders]") {
//   auto path = std::string(HIGHS_DIR) + "/check/instances/afiro.mps";
//   Highs nodecomp;
//   nodecomp.readModel(path);
//   auto lp = nodecomp.getLp();
//   nodecomp.run();
//   auto expected = nodecomp.getObjectiveValue();
//   lp.ensureRowwise();
//   auto res = benders(lp, "X0\\d", std::vector<double> (40, 0));
//   REQUIRE(std::abs(res.result- expected) < 1e-3);
//   REQUIRE(res.iter == 7);
// }

// TEST_CASE("2test-solve-simple-system", "[highs-benders]") {
//   auto lp = get_simple_test_problem();
//   lp.col_names_ = {"m1", "s1", "s2"};
//   Highs nodecomp;
//   nodecomp.passModel(lp);
//   nodecomp.run();
//   auto expected = nodecomp.getObjectiveValue();
//   auto res = benders2(lp, "m\\d", {2});
//   REQUIRE(std::abs(res.result- expected) < 1e-3);
//   REQUIRE(res.iter == 3);
// }

// TEST_CASE("2test-solve-simple-system-2", "[highs-benders]") {
//   auto lp = get_simple_test_problem();
//   lp.col_names_ = {"m1", "s1", "s2"};
//   Highs nodecomp;
//   nodecomp.passModel(lp);
//   nodecomp.run();
//   auto expected = nodecomp.getObjectiveValue();
//   // auto res = benders(lp, "m\\d", {2});
//   auto res = benders2(lp, std::set<HighsInt>{0, 2}, {2, 0});
//   REQUIRE(std::abs(res.result- expected) < 1e-3);
//   REQUIRE(res.iter == 3);
// }

// TEST_CASE("2test-solve-second-system", "[highs-benders]") {
//   auto lp = get_second_test_problem();
//   lp.col_names_ = {"m1", "m2", "m3", "s1", "s2", "s3"};
//   Highs nodecomp;
//   nodecomp.passModel(lp);
//   nodecomp.run();
//   auto expected = nodecomp.getObjectiveValue();
//   // auto res = benders(lp, "m\\d", {0, 1.5, 0});
//   auto res = benders2(lp, "m\\d", {0, 0, 0});
//   REQUIRE(std::abs(res.result- expected) < 1e-3);
//   REQUIRE(res.iter == 4);
// }

// TEST_CASE("2test-solve-blending", "[highs-benders]") {
//   auto path = std::string(HIGHS_DIR) + "/check/instances/blending.mps";
//   Highs nodecomp;
//   nodecomp.readModel(path);
//   auto lp = nodecomp.getLp();
//   nodecomp.run();
//   auto expected = nodecomp.getObjectiveValue();
//   lp.ensureRowwise();
//   auto res = benders2(lp, "P0", std::vector<double> (40, 0));
//   REQUIRE(std::abs(res.result- expected) < 1e-3);
//   REQUIRE(res.iter == 1);
// }

// TEST_CASE("2test-solve-afiro", "[highs-benders]") {
//   auto path = std::string(HIGHS_DIR) + "/check/instances/afiro.mps";
//   Highs nodecomp;
//   nodecomp.readModel(path);
//   auto lp = nodecomp.getLp();
//   nodecomp.run();
//   auto expected = nodecomp.getObjectiveValue();
//   lp.ensureRowwise();
//   auto res = benders2(lp, "X0\\d", std::vector<double> (40, 0));
//   REQUIRE(std::abs(res.result- expected) < 1e-3);
//   REQUIRE(res.iter == 7);
// }

// TEST_CASE("test-solve-multi-simple-system-", "[highs-benders]") {
//   auto lp = get_simple_multi_test_problem();
//   Highs nodecomp;
//   nodecomp.passModel(lp);
//   nodecomp.run();
//   auto expected = nodecomp.getObjectiveValue();
//   // auto res = benders(lp, "m\\d", {2});
//   auto res = multi_benders(lp, {0}, {{1,2}, {3}}, {2, 0});
//   REQUIRE(std::abs(res.result- expected) < 1e-3);
//   REQUIRE(res.iter == 4);
// }

// TEST_CASE("test-benders-solve-indep-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxm/fxm";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + "2.tim";
//   auto stochfile = instance + "2_6.sto";
//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   highs.run();
//   auto expected = highs.getObjectiveValue();
//   auto c1 = highs.getSolution().col_value;
//   auto lp = highs.getModel().lp_;
//   lp.ensureRowwise();
//   std::set<int> master_vars;
//   for (int i = 0; i < 114 + 0  * (457 - 114); ++i) master_vars.emplace(i);
  
  
//   std::vector<double> starting_point (master_vars.size(), 0);
  
//   Highs highs2;
//   REQUIRE(build_stochastic_problem(highs2, corefile, timefile, stochfile));
//   for (int i = 0; i < highs2.getNumCol(); ++i) highs2.changeColCost(i, 0);
//   modify(highs2);
//   highs2.run();
//   starting_point = highs2.getSolution().col_value;
//   std::ofstream("/tmp/iteration.csv", std::ios::app) << "indep" << "," << std::endl;
//   std::ofstream("/tmp/cut_distances.csv", std::ios::app) << "indep" << "," << std::endl;
//   std::ofstream("/tmp/ubd_lbd.csv", std::ios::app) << "indep" << "," << std::endl;
//   auto res = benders(lp, master_vars, starting_point, 1e-3, max_iters);
//   res.note(expected, "indep");
//   REQUIRE(expected == expected);
//   REQUIRE(res.result == res.result);
//   REQUIRE(res.feas_iters == res.feas_iters);
//   REQUIRE(res.UBD_iters == res.UBD_iters);
  
//   REQUIRE(res == BendersRet {expected, -1}); 
//   // res = benders2(lp, master_vars, starting_point);
//   // REQUIRE(res == BendersRet {expected, 41}); 
// }

// TEST_CASE("test-multi-benders-solve-indep-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxm/fxm";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + "2.tim";
//   auto stochfile = instance + "2_6.sto";
//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   highs.run();

  
//   auto expected = highs.getObjectiveValue();
//   auto lp = highs.getModel().lp_;

//   auto optim = highs.getSolution().col_value;

//   auto c = lp.col_cost_;
//   double sum  = lp.offset_;
//   for (int i = 0; i < 114; ++i) sum += optim.at(i) * c.at(i);
//   for (int j = 0; j < 457-114; ++j)
//     for (int i = 0; i < 6; ++i)
//       sum += optim.at(114 + j) * c.at(114 + i * (457-114) + j);
//   REQUIRE(std::fabs(sum - expected) < 1e-8);
//   lp.ensureRowwise();
//   std::set<int> master_vars;
//   for (int i = 0; i < 114; ++i) master_vars.emplace(i);
//   std::vector<double> starting_point (master_vars.size(), 0);
//   std::vector<std::set<HighsInt>> subproblem_variables(6);
//   for (int i = 0; i < 6; ++i)
//     for (int j = 0; j < 457 - 114; ++j)
//       subproblem_variables.at(i).emplace(114 + i * (457 - 114) + j);
  
//   Highs highs2;
//   REQUIRE(build_stochastic_problem(highs2, corefile, timefile, stochfile));
//   for (int i = 0; i < highs2.getNumCol(); ++i) highs2.changeColCost(i, 0);
//   // for (int i = 0; i < 6; ++i) highs2.changeColCost(167 + (457-114)*i, -1);
//   modify(highs2);
//   highs2.run();
//   starting_point = highs2.getSolution().col_value;
//   // starting_point = highs.getSolution().col_value;

//   std::ofstream("/tmp/iteration.csv", std::ios::app) << "indep_mult" << "," << std::endl;
//   std::ofstream("/tmp/cut_distances.csv", std::ios::app) << "indep_mult" << "," << std::endl;
//   std::ofstream("/tmp/ubd_lbd.csv", std::ios::app) << "indep_mult" << "," << std::endl;
//   auto res = multi_benders(lp, master_vars, subproblem_variables, starting_point, 1e-3, max_iters);
//   res.note(expected, "indep_mult");
//   REQUIRE(res == BendersRet {expected, -1}); 
// }

// TEST_CASE("test-benders-solve-indep-bigger-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxm/fxm";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + "2.tim";
//   auto stochfile = instance + "2_16.sto";
//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   highs.run();
//   auto expected = highs.getObjectiveValue();
//   auto lp = highs.getModel().lp_;
//   lp.ensureRowwise();
//   std::set<int> master_vars;
//   for (int i = 0; i < 114 + 0  * (457 - 144); ++i) master_vars.emplace(i);
//   std::vector<double> starting_point (master_vars.size(), 0);

//   Highs highs2;
//   REQUIRE(build_stochastic_problem(highs2, corefile, timefile, stochfile));
//   for (int i = 0; i < highs2.getNumCol(); ++i) highs2.changeColCost(i, 0);
//   modify(highs2);
//   highs2.run();
//   starting_point = highs2.getSolution().col_value;
//   std::ofstream("/tmp/iteration.csv", std::ios::app) << "indep_bigger" << "," << std::endl;
//   std::ofstream("/tmp/cut_distances.csv", std::ios::app) << "indep_bigger" << "," << std::endl;
//   std::ofstream("/tmp/ubd_lbd.csv", std::ios::app) << "indep_bigger" << "," << std::endl;
//   auto res = benders(lp, master_vars, starting_point, 1e-3, 1e2);
//   res.note(expected, "indep_bigger");
//   REQUIRE(res == BendersRet {expected, 41}); 
//   // res = benders2(lp, master_vars, starting_point);
//   // REQUIRE(res == BendersRet {expected, 41}); 
// }

// TEST_CASE("test-multi-benders-solve-indep-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxm/fxm";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + "2.tim";
//   auto stochfile = instance + "3_6.sto";
//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   highs.run();

  
//   auto expected = highs.getObjectiveValue();
//   auto lp = highs.getModel().lp_;

//   auto optim = highs.getSolution().col_value;

//   auto c = lp.col_cost_;
//   double sum  = lp.offset_;
//   for (int i = 0; i < 114; ++i) sum += optim.at(i) * c.at(i);
//   for (int j = 0; j < 457-114; ++j)
//     for (int i = 0; i < 6; ++i)
//       sum += optim.at(114 + j) * c.at(114 + i * (457-114) + j);
//   REQUIRE(std::fabs(sum - expected) < 1e-8);
//   lp.ensureRowwise();
//   std::set<int> master_vars;
//   for (int i = 0; i < 114; ++i) master_vars.emplace(i);
//   std::vector<double> starting_point (master_vars.size(), 0);
//   std::vector<std::set<HighsInt>> subproblem_variables(6);
//   for (int i = 0; i < 6; ++i)
//     for (int j = 0; j < 457 - 114; ++j)
//       subproblem_variables.at(i).emplace(114 + i * (457 - 114) + j);
  
//   Highs highs2;
//   REQUIRE(build_stochastic_problem(highs2, corefile, timefile, stochfile));
//   for (int i = 0; i < highs2.getNumCol(); ++i) highs2.changeColCost(i, 0);
//   for (int i = 0; i < 6; ++i) highs2.changeColCost(167 + (457-114)*i, -1);
//   modify(highs2);
//   highs2.run();
//   starting_point = highs2.getSolution().col_value;
//   // starting_point = highs.getSolution().col_value;

//   std::ofstream("/tmp/iteration.csv", std::ios::app) << "indep_mult" << "," << std::endl;
//   std::ofstream("/tmp/cut_distances.csv", std::ios::app) << "indep_mult" << "," << std::endl;
//   std::ofstream("/tmp/ubd_lbd.csv", std::ios::app) << "indep_mult" << "," << std::endl;
//   auto res = multi_benders(lp, master_vars, subproblem_variables, starting_point, 1e-3, max_iters);
//   res.note(expected, "indep_mult");
//   REQUIRE(res == BendersRet {expected, 41}); 
// }

// TEST_CASE("test-benders-solve-indep-bigger-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxm/fxm";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + "2.tim";
//   auto stochfile = instance + "2_16.sto";
//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   highs.run();
//   auto expected = highs.getObjectiveValue();
//   auto lp = highs.getModel().lp_;
//   lp.ensureRowwise();
//   std::set<int> master_vars;
//   for (int i = 0; i < 114 + 0  * (457 - 144); ++i) master_vars.emplace(i);
//   std::vector<double> starting_point (master_vars.size(), 0);

//   Highs highs2;
//   REQUIRE(build_stochastic_problem(highs2, corefile, timefile, stochfile));
//   for (int i = 0; i < highs2.getNumCol(); ++i) highs2.changeColCost(i, 0);
//   modify(highs2);
//   highs2.run();
//   starting_point = highs2.getSolution().col_value;
//   std::ofstream("/tmp/iteration.csv", std::ios::app) << "indep_bigger" << "," << std::endl;
//   std::ofstream("/tmp/cut_distances.csv", std::ios::app) << "indep_bigger" << "," << std::endl;
//   std::ofstream("/tmp/ubd_lbd.csv", std::ios::app) << "indep_bigger" << "," << std::endl;
//   auto res = benders(lp, master_vars, starting_point, 1e-3, 1e2);
//   res.note(expected, "indep_bigger");
//   REQUIRE(res == BendersRet {expected, 41}); 
//   // res = benders2(lp, master_vars, starting_point);
//   // REQUIRE(res == BendersRet {expected, 41}); 
// }


// TEST_CASE("test-multi-benders-solve-indep-bigger-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxm/fxm";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + "2.tim";
//   auto stochfile = instance + "2_16.sto";
//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   highs.run();
//   auto expected = highs.getObjectiveValue();
//   auto lp = highs.getModel().lp_;
//   lp.ensureRowwise();
//   std::set<int> master_vars;
//   for (int i = 0; i < 114; ++i) master_vars.emplace(i);
//   std::vector<double> starting_point (master_vars.size(), 0);
//   std::vector<std::set<HighsInt>> subproblem_variables(16);
//   for (int i = 0; i < 16; ++i)
//     for (int j = 0; j < 457 - 114; ++j)
//       subproblem_variables.at(i).emplace(114 + i * (457 - 114) + j);
//   std::ofstream("/tmp/iteration.csv", std::ios::app) << "indep_mult_bigger" << "," << std::endl;
//   std::ofstream("/tmp/cut_distances.csv", std::ios::app) << "indep_mult_bigger" << "," << std::endl;
//   std::ofstream("/tmp/ubd_lbd.csv", std::ios::app) << "indep_mult_bigger" << "," << std::endl;
//   Highs highs2;
//   REQUIRE(build_stochastic_problem(highs2, corefile, timefile, stochfile));
//   for (int i = 0; i < highs2.getNumCol(); ++i) highs2.changeColCost(i, 0);
//   modify(highs2);
//   highs2.run();
//   starting_point = highs2.getSolution().col_value;

//   auto res = multi_benders(lp, master_vars, subproblem_variables, starting_point, 1e-3, 1e2);
//   res.note(expected, "indep_mult_bigger");
//   REQUIRE(res == BendersRet {expected, 41}); 
// }

// TEST_CASE("test-benders-solve-block-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/pltexp/pltexpA2";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + ".tim";
//   auto stochfile = instance + "_6.sto";
//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   highs.run();
//   auto expected = highs.getObjectiveValue();
//   auto lp = highs.getModel().lp_;
//   lp.ensureRowwise();
//   std::set<int> master_vars;
//   for (int i = 0; i < 188; ++i) master_vars.emplace(i);
//   std::vector<double> starting_point (master_vars.size(), 0);

//   Highs highs2;
//   REQUIRE(build_stochastic_problem(highs2, corefile, timefile, stochfile));
//   for (int i = 0; i < highs2.getNumCol(); ++i) highs2.changeColCost(i, 0);
//   modify(highs2);
//   highs2.run();
//   starting_point = highs2.getSolution().col_value;

//   REQUIRE(2 == 2);
//   std::ofstream("/tmp/iteration.csv", std::ios::app) << "block" << "," << std::endl;
//   std::ofstream("/tmp/cut_distances.csv", std::ios::app) << "block" << "," << std::endl;
//   std::ofstream("/tmp/ubd_lbd.csv", std::ios::app) << "block" << "," << std::endl;
//   auto res = benders(lp, master_vars, starting_point, 1e-3, max_iters);
//   res.note(expected, "block");
//   REQUIRE(res == BendersRet {expected, 2}); // TODO too low?
//   // res = benders2(lp, master_vars, starting_point, 1e-3);
//   // REQUIRE(res == BendersRet {expected, 2}); // TODO too low?
// }

// TEST_CASE("test-multi-benders-solve-block-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/pltexp/pltexpA2";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + ".tim";
//   auto stochfile = instance + "_6.sto";
//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   highs.run();
//   auto expected = highs.getObjectiveValue();
//   auto lp = highs.getModel().lp_;
//   lp.ensureRowwise();
//   std::set<int> master_vars;
//   for (int i = 0; i < 188; ++i) master_vars.emplace(i);
//   std::vector<double> starting_point (master_vars.size(), 0);
//   std::vector<std::set<HighsInt>> subproblem_variables(6);
//   for (int i = 0; i < 6; ++i)
//     for (int j = 0; j < 460 - 188; ++j)
//       subproblem_variables.at(i).emplace(188 + i * (460 - 188) + j);

//   Highs highs2;
//   REQUIRE(build_stochastic_problem(highs2, corefile, timefile, stochfile));
//   auto c = highs2.getModel().lp_.col_cost_;
//   for (int i = 0; i < highs2.getNumCol(); ++i) highs2.changeColCost(i, -c.at(i));
//   for (int i = 0; i < highs2.getNumCol(); ++i) highs2.changeColCost(i, 0);
//   modify(highs2);
//   highs2.run();
//   starting_point = highs2.getSolution().col_value;

//   std::ofstream("/tmp/iteration.csv", std::ios::app) << "block_mult" << "," << std::endl;
//   std::ofstream("/tmp/cut_distances.csv", std::ios::app) << "block_mult" << "," << std::endl;
//   std::ofstream("/tmp/ubd_lbd.csv", std::ios::app) << "block_mult" << "," << std::endl;
//   auto res = multi_benders(lp, master_vars, subproblem_variables, starting_point, 1e-3, max_iters);
//   res.note(expected, "block_mult");
//   REQUIRE(res == BendersRet {expected, 2});
// }

// TEST_CASE("test-benders-solve-bigger-block-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/pltexp/pltexpA2";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + ".tim";
//   auto stochfile = instance + "_16.sto";
//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   highs.run();
//   auto expected = highs.getObjectiveValue();
//   auto lp = highs.getModel().lp_;
//   lp.ensureRowwise();
//   std::set<int> master_vars;
//   for (int i = 0; i < 188; ++i) master_vars.emplace(i);
//   std::vector<double> starting_point (master_vars.size(), 0);

//   Highs highs2;
//   REQUIRE(build_stochastic_problem(highs2, corefile, timefile, stochfile));
//   for (int i = 0; i < highs2.getNumCol(); ++i) highs2.changeColCost(i, 0);
//   modify(highs2);
//   highs2.run();
//   starting_point = highs2.getSolution().col_value;

//   std::ofstream("/tmp/iteration.csv", std::ios::app) << "block_bigger" << "," << std::endl;
//   std::ofstream("/tmp/cut_distances.csv", std::ios::app) << "block_bigger" << "," << std::endl;
//   std::ofstream("/tmp/ubd_lbd.csv", std::ios::app) << "block_bigger" << "," << std::endl;
//   auto res = benders(lp, master_vars, starting_point);
//   res.note(expected, "block_bigger");
//   REQUIRE(res == BendersRet {expected, 2}); // TODO too low?
//   // res = benders2(lp, master_vars, starting_point, 1e-3);
//   // REQUIRE(res == BendersRet {expected, 2}); // TODO too low?
// }

// TEST_CASE("test-multi-benders-solve-bigger-block-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/pltexp/pltexpA2";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + ".tim";
//   auto stochfile = instance + "_16.sto";
//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   highs.run();
//   auto expected = highs.getObjectiveValue();
//   auto lp = highs.getModel().lp_;
//   lp.ensureRowwise();
//   std::set<int> master_vars;
//   for (int i = 0; i < 188; ++i) master_vars.emplace(i);
//   std::vector<double> starting_point (master_vars.size(), 0);
//   std::vector<std::set<HighsInt>> subproblem_variables(16);
//   for (int i = 0; i < 16; ++i)
//     for (int j = 0; j < 460 - 188; ++j)
//       subproblem_variables.at(i).emplace(188 + i * (460 - 188) + j);
//   std::ofstream("/tmp/iteration.csv", std::ios::app) << "block_mult_bigger" << "," << std::endl;
//   std::ofstream("/tmp/cut_distances.csv", std::ios::app) << "block_mult_bigger" << "," << std::endl;
//   std::ofstream("/tmp/ubd_lbd.csv", std::ios::app) << "block_mult_bigger" << "," << std::endl;
//   Highs highs2;
//   REQUIRE(build_stochastic_problem(highs2, corefile, timefile, stochfile));
//   for (int i = 0; i < highs2.getNumCol(); ++i) highs2.changeColCost(i, 0);
//   modify(highs2);
//   highs2.run();
//   // starting_point = highs2.getSolution().col_value;

//   auto res = multi_benders(lp, master_vars, subproblem_variables, starting_point);
//   res.note(expected, "block_mult_bigger");
//   REQUIRE(res == BendersRet {expected, 2});
// }

// TEST_CASE("test-benders-solve-scen-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/sg/sgpf5y3";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + ".tim";
//   auto stochfile = instance + ".sce";
//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   highs.run();
//   auto expected = highs.getObjectiveValue();
//   auto lp = highs.getModel().lp_;
//   lp.ensureRowwise();
//   std::set<int> master_vars;
//   for (int i = 0; i < 139; ++i) master_vars.emplace(i);
//   std::vector<double> starting_point (master_vars.size(), 0);
//   // starting_point = highs.getSolution().col_value;
  
//   Highs highs2;
//   REQUIRE(build_stochastic_problem(highs2, corefile, timefile, stochfile));
//   for (int i = 0; i < highs2.getNumCol(); ++i) highs2.changeColCost(i, 0);
//   modify(highs2);
//   highs2.run();
//   starting_point = highs2.getSolution().col_value;
//   std::ofstream("/tmp/iteration.csv", std::ios::app) << "scen" << "," << std::endl;
//   std::ofstream("/tmp/cut_distances.csv", std::ios::app) << "scen" << "," << std::endl;
//   std::ofstream("/tmp/ubd_lbd.csv", std::ios::app) << "scen" << "," << std::endl;
//   auto res = benders(lp, master_vars, starting_point, 1e-3, max_iters);
//   res.note(expected, "scen");
//   REQUIRE(res == BendersRet {expected, 9});
  
// // / res = benders2(lp, master_vars, starting_point, 1e-3);
// // / REQUIRE(res == BendersRet {expected, 9});
// }

// TEST_CASE("test-multi-benders-solve-scen-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/sg/sgpf5y3";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + ".tim";
//   auto stochfile = instance + ".sce";
//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   highs.run();
//   auto expected = highs.getObjectiveValue();
//   auto lp = highs.getModel().lp_;
//   auto num_per_sub = 79 + 5 * 79;
//   REQUIRE(lp.num_col_ == 139 + 5 * num_per_sub);
//   lp.ensureRowwise();
//   std::set<int> master_vars;
//   for (int i = 0; i < 139; ++i) master_vars.emplace(i);
//   std::vector<double> starting_point (master_vars.size(), 0);
//   std::vector<std::set<HighsInt>> subproblem_variables(5);
//   for (int i = 0; i < 5; ++i)
//     for (int j = 0; j < num_per_sub; ++j)
//       subproblem_variables.at(i).emplace(139 + i * num_per_sub + j);
//   Highs highs2;
//   REQUIRE(build_stochastic_problem(highs2, corefile, timefile, stochfile));
//   auto c = lp.col_cost_;
//   for (int i = 0; i < highs2.getNumCol(); ++i) highs2.changeColCost(i, -c.at(i));
//   modify(highs2);
//   highs2.run();
//   starting_point = highs2.getSolution().col_value;
//   std::ofstream("/tmp/iteration.csv", std::ios::app) << "scen_mult" << "," << std::endl;
//   std::ofstream("/tmp/cut_distances.csv", std::ios::app) << "scen_mult" << "," << std::endl;
//   std::ofstream("/tmp/ubd_lbd.csv", std::ios::app) << "scen_mult" << "," << std::endl;
//   auto res = multi_benders(lp, master_vars, subproblem_variables, starting_point, 1e-3, max_iters);
//   res.note(expected, "scen_mult");
//   REQUIRE(res == BendersRet {expected, 9});
// }

// TEST_CASE("test-multi-benders-solve-new-indep-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxm/fxm";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + "2.tim";
//   auto stochfile = instance + "3_6.sto";
//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   highs.run();

//   // REQUIRE(highs.getObjectiveValue() == 0);
//   // REQUIRE(highs.getNumCol() == 0);
  
//   auto expected = highs.getObjectiveValue();
//   auto lp = highs.getModel().lp_;

//   // auto optim = highs.getSolution().col_value;

//   // auto c = lp.col_cost_;
//   // double sum  = lp.offset_;
//   // for (int i = 0; i < 114; ++i) sum += optim.at(i) * c.at(i);
//   // for (int j = 0; j < 457-114; ++j)
//   //   for (int i = 0; i < 6; ++i)
//   //     sum += optim.at(114 + j) * c.at(114 + i * (457-114) + j);
//   // REQUIRE(std::fabs(sum - expected) < 1e-8);
//   lp.ensureRowwise();
//   std::set<int> master_vars;
//   for (int i = 0; i < 114; ++i) master_vars.emplace(i);
//   std::vector<double> starting_point (master_vars.size(), 0);
//   std::vector<std::set<HighsInt>> subproblem_variables(36);
//   for (int i = 0; i < 36; ++i)
//     for (int j = 0; j < 457 - 114; ++j)
//       subproblem_variables.at(i).emplace(114 + i * (457 - 114) + j);
  
//   Highs highs2;
//   REQUIRE(build_stochastic_problem(highs2, corefile, timefile, stochfile));
//   for (int i = 0; i < highs2.getNumCol(); ++i) highs2.changeColCost(i, 0);
//   modify(highs2);
//   highs2.run();
//   starting_point = highs2.getSolution().col_value;
//   // // starting_point = highs.getSolution().col_value;

//   std::ofstream("/tmp/iteration.csv", std::ios::app) << "new_indep_mult" << "," << std::endl;
//   std::ofstream("/tmp/cut_distances.csv", std::ios::app) << "new_indep_mult" << "," << std::endl;
//   std::ofstream("/tmp/ubd_lbd.csv", std::ios::app) << "new_indep_mult" << "," << std::endl;
//   auto res = multi_benders(lp, master_vars, subproblem_variables, starting_point, 1e-3, max_iters);
//   res.note(expected, "indep_mult");
//   REQUIRE(res == BendersRet {expected, 41}); 
// }

// TEST_CASE("test-fxm-1-stage", "[highs-benders]") {
//   // ROW idx: 165 is the problem
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxmtemp/fxm";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + "2.tim";
//   auto stochfile = instance + "2_6.sto";
//   Highs highs;
  
//   FilereaderMps mps;
//   HighsModel model;
//   REQUIRE(mps.readModelFromFile(HighsOptions(), corefile, model) == FilereaderRetcode::kOk);
//   highs.passModel(model);
//   ObjSense s;
//   REQUIRE(highs.getObjectiveSense(s) == HighsStatus::kOk);
//   REQUIRE(s == ObjSense::kMinimize);
//   REQUIRE(highs.getNumRow() == 92 * 1 + (174-92)*1 + (240-174)*1 + (330 - 240)*1);
//   highs.run();
  
//   auto expected = highs.getObjectiveValue();
//   std::vector<double> cs, xs;
//   auto x = highs.getSolution().col_value;
//   auto c = highs.getLp().col_cost_;
//   for (int i = 0; i < highs.getNumCol(); ++i)
//     if (c.at(i) != 0) {
//       cs.push_back(c.at(i));
//       xs.push_back(x.at(i));
//     }
//   // REQUIRE(cs == std::vector<double> {});
//   REQUIRE(xs == std::vector<double> {});
// }

// TEST_CASE("test-fxm-1-stage-expers", "[highs-benders]") {
//   // ROW idx: 165 is the problem
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxmtemp/fxm";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + "2.tim";
//   auto stochfile = instance + "2_6.sto";
//   Highs highs;
  
//   FilereaderMps mps;
//   HighsModel model;
//   REQUIRE(mps.readModelFromFile(HighsOptions(), corefile, model) == FilereaderRetcode::kOk);
//   highs.passModel(model);
//   ObjSense s;
//   REQUIRE(highs.getObjectiveSense(s) == HighsStatus::kOk);
//   REQUIRE(s == ObjSense::kMinimize);
//   REQUIRE(highs.getNumRow() == 92 * 1 + (174-92)*1 + (240-174)*1 + (330 - 240)*1);
//   highs.run();
  
//   auto expected = highs.getObjectiveValue();
//   std::vector<double> cs, xs;
//   auto x = highs.getSolution().col_value;
//   auto c = highs.getLp().col_cost_;
//   for (int i = 0; i < highs.getNumCol(); ++i)
//     if (c.at(i) != 0) {
//       cs.push_back(c.at(i));
//       xs.push_back(x.at(i));
//     }
//   // REQUIRE(cs == std::vector<double> {});
//   // REQUIRE(xs == std::vector<double> {});

//   auto dual = highs.getSolution().row_dual;
//   std::vector<int> to_delete;
  
//   std::vector<int> indices;
//   std::vector<double> vals;
//   for (int i = 0; i < dual.size(); ++i)
//     if (dual.at(i) != 0) {
//       indices.push_back(i);
//       vals.push_back(dual.at(i));
//     }
//     else to_delete.push_back(i);
//   //  REQUIRE(vals == std::vector<double> {});
//   //  REQUIRE(indices == std::vector<int> {});
//   // std::vector<int> to_delete;
//   // for (int i = 0; i < indices.size(); ++i)
//   //   if (vals.at(i) > 0.5)
//   //     to_delete.push_back(indices.at(i));
//   // REQUIRE(to_delete == std::vector<int> {});
//   highs.deleteRows(to_delete.size(), to_delete.data());
//   // REQUIRE(to_delete.size() == 0);
//   // REQUIRE(highs.getNumRow() == 0);
//   to_delete = {4, 7, 10, 11};
//   highs.deleteRows(to_delete.size(), to_delete.data());
//   highs.run();
//   REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
//   // REQUIRE(highs.getNumRows() == 0);
  
//   REQUIRE(mps.writeModelToFile(HighsOptions(), instance + "_xd" +".mps",highs.getModel()) == HighsStatus::kOk);
// }


// TEST_CASE("test-fxm-2-stage", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxmtemp/fxm";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + "2.tim";
//   auto stochfile = instance + "2_6.sto";
//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   ObjSense s;
//   REQUIRE(highs.getObjectiveSense(s) == HighsStatus::kOk);
//   REQUIRE(s == ObjSense::kMinimize);
//   REQUIRE(highs.getNumRow() == 92 * 1 + (174-92)*6 + (240-174)*6 + (330 - 240)*6);
//   highs.run();
//   auto expected = highs.getObjectiveValue();
//   std::vector<double> cs, xs;
//   auto x = highs.getSolution().col_value;
//   auto c = highs.getLp().col_cost_;
//   for (int i = 0; i < highs.getNumCol(); ++i)
//     if (c.at(i) != 0) {
//       cs.push_back(c.at(i));
//       xs.push_back(x.at(i));
//     }
//   // REQUIRE(cs == std::vector<double> {});
//   REQUIRE(xs == std::vector<double> {});
// }

// TEST_CASE("test-fxm-2v2-stage", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxmtemp/fxm";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + "2.tim";
//   auto stochfile = instance + "2_16.sto";
//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   ObjSense s;
//   REQUIRE(highs.getObjectiveSense(s) == HighsStatus::kOk);
//   REQUIRE(s == ObjSense::kMinimize);
//   REQUIRE(highs.getNumRow() == 92 * 1 + (174-92)*16 + (240-174)*16 + (330 - 240)*16);
//   highs.run();
//   auto expected = highs.getObjectiveValue();
//   std::vector<double> cs, xs;
//   auto x = highs.getSolution().col_value;
//   auto c = highs.getLp().col_cost_;
//   for (int i = 0; i < highs.getNumCol(); ++i)
//     if (c.at(i) != 0) {
//       cs.push_back(c.at(i));
//       xs.push_back(x.at(i));
//     }
//   // REQUIRE(cs == std::vector<double> {});
//   REQUIRE(xs == std::vector<double> {});
// }

// TEST_CASE("test-fxm-3-stage", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxmtemp/fxm";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + "3.tim";
//   auto stochfile = instance + "3_6.sto";
//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   ObjSense s;
//   REQUIRE(highs.getObjectiveSense(s) == HighsStatus::kOk);
//   REQUIRE(s == ObjSense::kMinimize);
//   REQUIRE(highs.getNumRow() == 92 * 1 + (174-92)*6 + (240-174)*6*6 + (330 - 240)*6*6);
//   highs.run();
//   auto expected = highs.getObjectiveValue();
//   std::vector<double> cs, xs;
//   auto x = highs.getSolution().col_value;
//   auto c = highs.getLp().col_cost_;
//   for (int i = 0; i < highs.getNumCol(); ++i)
//     if (c.at(i) != 0) {
//       cs.push_back(c.at(i));
//       xs.push_back(x.at(i));
//     }
//   // REQUIRE(cs == std::vector<double> {});
//   REQUIRE(xs == std::vector<double> {});
// }

// TEST_CASE("test-fxm-3v2-stage", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxmtemp/fxm";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + "3.tim";
//   auto stochfile = instance + "3_16.sto";
//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   ObjSense s;
//   REQUIRE(highs.getObjectiveSense(s) == HighsStatus::kOk);
//   REQUIRE(s == ObjSense::kMinimize);
//   REQUIRE(highs.getNumRow() == 92 * 1 + (174-92)*16 + (240-174)*16*16 + (330 - 240)*16*16);
//   highs.run();
//   auto expected = highs.getObjectiveValue();
//   std::vector<double> cs, xs;
//   auto x = highs.getSolution().col_value;
//   auto c = highs.getLp().col_cost_;
//   for (int i = 0; i < highs.getNumCol(); ++i)
//     if (c.at(i) != 0) {
//       cs.push_back(c.at(i));
//       xs.push_back(x.at(i));
//     }
//   // REQUIRE(cs == std::vector<double> {});
//   REQUIRE(xs == std::vector<double> {});
// }

// TEST_CASE("test-fxm-4-stage", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxmtemp/fxm";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + "4.tim";
//   auto stochfile = instance + "4_6.sto";
//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   ObjSense s;
//   REQUIRE(highs.getObjectiveSense(s) == HighsStatus::kOk);
//   REQUIRE(s == ObjSense::kMinimize);
//   REQUIRE(highs.getNumRow() == 92 * 1 + (174-92)*6 + (240-174)*6*6 + (330 - 240)*6*6*6);
//   highs.run();
//   auto expected = highs.getObjectiveValue();
//   std::vector<double> cs, xs;
//   auto x = highs.getSolution().col_value;
//   auto c = highs.getLp().col_cost_;
//   for (int i = 0; i < highs.getNumCol(); ++i)
//     if (c.at(i) != 0) {
//       cs.push_back(c.at(i));
//       xs.push_back(x.at(i));
//     }
//   // REQUIRE(cs == std::vector<double> {});
//   REQUIRE(xs == std::vector<double> {});
// }

// TEST_CASE("test-fxm-4v2-stage", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxmtemp/fxm";
//   auto corefile = instance + ".cor";
//   auto timefile = instance + "4.tim";
//   auto stochfile = instance + "4_16.sto";
//   Highs highs;
//   REQUIRE(build_stochastic_problem(highs, corefile, timefile, stochfile));
//   ObjSense s;
//   REQUIRE(highs.getObjectiveSense(s) == HighsStatus::kOk);
//   REQUIRE(s == ObjSense::kMinimize);
//   REQUIRE(highs.getNumRow() == 92 * 1 + (174-92)*16 + (240-174)*16*16 + (330 - 240)*16*16*16);
//   highs.run();
//   auto expected = highs.getObjectiveValue();
//   std::vector<double> cs, xs;
//   auto x = highs.getSolution().col_value;
//   auto c = highs.getLp().col_cost_;
//   for (int i = 0; i < highs.getNumCol(); ++i)
//     if (c.at(i) != 0) {
//       cs.push_back(c.at(i));
//       xs.push_back(x.at(i));
//     }
//   // REQUIRE(cs == std::vector<double> {});
//   REQUIRE(xs == std::vector<double> {});
// }



// TEST_CASE("test-benders-solve-indep-2-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxm/fxm";
//   run(instance + ".cor", instance + "2.tim", instance + "2_6.sto", "indep-2", 114, 457-114, 92, 330-92, 6, 0);
// }

// TEST_CASE("test-benders-solve-indep-2-bigger-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxm/fxm";
//   run(instance + ".cor", instance + "2.tim", instance + "2_16.sto", "indep-2-bigger", 114, 457-114, 92, 330-92, 16, 0);
// }

// TEST_CASE("test-benders-solve-indep-3-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxm/fxm";
//   run(instance + ".cor", instance + "3.tim", instance + "3_6.sto", "indep-3", 114, 457-114, 92, 330-92, 6, 0);
// }

// TEST_CASE("test-benders-solve-indep-3-bigger-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxm/fxm";
//   run(instance + ".cor", instance + "3.tim", instance + "3_16.sto", "indep-3-bigger", 114, 457-114, 92, 330-92, 16, 0);
// }

// TEST_CASE("test-benders-solve-indep-4-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxm/fxm";
//   run(instance + ".cor", instance + "4.tim", instance + "4_6.sto", "indep-4", 114, 457-114, 92, 330-92, 6*6, 0);
// }

// TEST_CASE("test-benders-solve-indep-4-bigger-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxm/fxm";
//   run(instance + ".cor", instance + "4.tim", instance + "4_16.sto", "indep-4-bigger", 114, 457-114, 92, 330-92, 16*16, 0);
// }

// TEST_CASE("test-benders-solve-block-2-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/pltexp/pltexpA";
//   run(instance + "2.cor", instance + "2.tim", instance + "2_6.sto", "block-2", 188, 272, 62, 104, 6, -100);
// }

// TEST_CASE("test-benders-solve-block-2-bigger-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/pltexp/pltexpA";
//   run(instance + "2.cor", instance + "2.tim", instance + "2_16.sto", "block-2-bigger", 188, 272, 62, 104, 16, -100);
// }

// TEST_CASE("test-benders-solve-block-3-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/pltexp/pltexpA";
//   auto coresize = core_size(instance + "3.cor");
//   run(instance + "3.cor", instance + "3.tim", instance + "3_6.sto", "block-3", 188, 
//     coresize.first - 188, 62, coresize.second - 62, 6*6, -100);
// }

// TEST_CASE("test-benders-solve-block-3-bigger-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/pltexp/pltexpA";
//   auto coresize = core_size(instance + "3.cor");
//   run(instance + "3.cor", instance + "3.tim", instance + "3_16.sto", "block-3-bigger", 188, 
//     coresize.first - 188, 62, coresize.second - 62, 16*16, -100);
// }

// TEST_CASE("test-benders-solve-block-4-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/pltexp/pltexpA";
//   auto coresize = core_size(instance + "4.cor");
//   run(instance + "4.cor", instance + "4.tim", instance + "4_6.sto", "block-4", 188, 
//     coresize.first - 188, 62, coresize.second - 62, 6*6*6, -100);
// }

// TEST_CASE("test-benders-solve-block-4-bigger-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/pltexp/pltexpA";
//   auto coresize = core_size(instance + "4.cor");
//   run(instance + "4.cor", instance + "4.tim", instance + "4_16.sto", "block-4-bigger", 188, 
//     coresize.first - 272, 62, coresize.second - 62, 16*16*16, -100);
// }

// TEST_CASE("test-benders-solve-storm-8", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/storm/stormG2";
//   auto coresize = core_size(instance + ".cor");
//   run(instance + ".cor", instance + ".tim", instance + "_8.sto", "storm-8", 121, 
//     coresize.first - 121, 185, coresize.second - 185, 8, 0);
// }

// TEST_CASE("test-benders-solve-storm-27", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/storm/stormG2";
//   auto coresize = core_size(instance + ".cor");
//   run(instance + ".cor", instance + ".tim", instance + "_27.sto", "storm-27", 121, 
//     coresize.first - 121, 185, coresize.second - 185, 27, 0);
// }

// TEST_CASE("test-benders-solve-storm-125", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/storm/stormG2";
//   auto coresize = core_size(instance + ".cor");
//   run(instance + ".cor", instance + ".tim", instance + "_125.sto", "storm-125", 121, 
//     coresize.first - 121, 185, coresize.second - 185, 125, 0);
// }

// TEST_CASE("test-benders-solve-storm-1000", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/storm/stormG2";
//   auto coresize = core_size(instance + ".cor");
//   run(instance + ".cor", instance + ".tim", instance + "_1000.sto", "storm-1000", 121, 
//     coresize.first - 121, 185, coresize.second - 185, 1000, 0);
// }

// TEST_CASE("test-benders-solve-multi-indep-2-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxm/fxm";
//   multi_run({instance + ".cor", instance + "2.tim", instance + "2_6.sto", "multi-indep-2", 0, false});
// }

// TEST_CASE("test-benders-solve-multi-indep-2-bigger-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxm/fxm";
//   multi_run({instance + ".cor", instance + "2.tim", instance + "2_16.sto", "multi-indep-2-bigger", 0, false});
// }

// TEST_CASE("test-benders-solve-multi-indep-3-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxm/fxm";
//   multi_run({instance + ".cor", instance + "3.tim", instance + "3_6.sto", "multi-indep-3", 0, false});
// }

// TEST_CASE("test-benders-solve-multi-indep-3-bigger-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxm/fxm";
//   multi_run({instance + ".cor", instance + "3.tim", instance + "3_16.sto", "multi-indep-3-bigger", 0, false});
// }

// TEST_CASE("test-benders-solve-multi-indep-4-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxm/fxm";
//   multi_run(instance + ".cor", instance + "4.tim", instance + "4_6.sto", "multi-indep-4", 114, 457-114, 92, 330-92, 6*6, 0);
// }

// TEST_CASE("test-benders-solve-multi-indep-4-bigger-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/fxm/fxm";
//   multi_run(instance + ".cor", instance + "4.tim", instance + "4_16.sto", "multi-indep-4-bigger", 114, 457-114, 92, 330-92, 16*16, 0);
// }

// TEST_CASE("test-benders-solve-multi-block-2-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/pltexp/pltexpA";
//   multi_run(instance + "2.cor", instance + "2.tim", instance + "2_6.sto", "multi-block-2", 188, 272, 62, 104, 6, -100);
// }

// TEST_CASE("test-benders-solve-multi-block-2-bigger-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/pltexp/pltexpA";
//   multi_run(instance + "2.cor", instance + "2.tim", instance + "2_16.sto", "multi-block-2-bigger", 188, 272, 62, 104, 16, -100);
// }

// TEST_CASE("test-benders-solve-multi-block-3-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/pltexp/pltexpA";
//   auto coresize = core_size(instance + "3.cor");
//   multi_run(instance + "3.cor", instance + "3.tim", instance + "3_6.sto", "multi-block-3", 188, 
//     coresize.first - 188, 62, coresize.second - 62, 6*6, -100);
// }

// TEST_CASE("test-benders-solve-multi-block-3-bigger-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/pltexp/pltexpA";
//   auto coresize = core_size(instance + "3.cor");
//   multi_run(instance + "3.cor", instance + "3.tim", instance + "3_16.sto", "multi-block-3-bigger", 188, 
//     coresize.first - 188, 62, coresize.second - 62, 16*16, -100);
// }

// TEST_CASE("test-benders-solve-multi-block-4-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/pltexp/pltexpA";
//   auto coresize = core_size(instance + "4.cor");
//   multi_run(instance + "4.cor", instance + "4.tim", instance + "4_6.sto", "multi-block-4", 188, 
//     coresize.first - 188, 62, coresize.second - 62, 6*6*6, -100);
// }

// TEST_CASE("test-benders-solve-multi-block-4-bigger-smps", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/pltexp/pltexpA";
//   auto coresize = core_size(instance + "4.cor");
//   multi_run(instance + "4.cor", instance + "4.tim", instance + "4_16.sto", "multi-block-4-bigger", 188, 
//     coresize.first - 188, 62, coresize.second - 62, 16*16*16, -100);
// }

// TEST_CASE("test-benders-solve-multi-storm-8", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/storm/stormG2";
//   auto coresize = core_size(instance + ".cor");
//   multi_run(instance + ".cor", instance + ".tim", instance + "_8.sto", "multi-storm-8", 121, 
//     coresize.first - 121, 185, coresize.second - 185, 8, 0);
// }

// TEST_CASE("test-benders-solve-multi-storm-27", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/storm/stormG2";
//   auto coresize = core_size(instance + ".cor");
//   multi_run(instance + ".cor", instance + ".tim", instance + "_27.sto", "multi-storm-27", 121, 
//     coresize.first - 121, 185, coresize.second - 185, 27, 0);
// }

// TEST_CASE("test-benders-solve-multi-storm-125", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/storm/stormG2";
//   auto coresize = core_size(instance + ".cor");
//   multi_run(instance + ".cor", instance + ".tim", instance + "_125.sto", "multi-storm-125", 121, 
//     coresize.first - 121, 185, coresize.second - 185, 125, 0);
// }

// TEST_CASE("test-benders-solve-multi-storm-1000", "[highs-benders]") {
//   auto instance = std::string(HIGHS_DIR) + "/check/instances/stoch/storm/stormG2";
//   auto coresize = core_size(instance + ".cor");
//   multi_run(instance + ".cor", instance + ".tim", instance + "_1000.sto", "multi-storm-1000", 121, 
//     coresize.first - 121, 185, coresize.second - 185, 1000, 0);
// }


std::vector<SmpsTestCase> read_tests {
  {dir + "fxm/fxm.cor", dir + "fxm/fxm2.tim", dir + "fxm/fxm2_6.sto", "indep-2", 0, false, 18417.066},
  {dir + "fxm/fxm.cor", dir + "fxm/fxm2.tim", dir + "fxm/fxm2_16.sto", "indep-2-bigger", 0, false, 18416.759},
  {dir + "fxm/fxm.cor", dir + "fxm/fxm3.tim", dir + "fxm/fxm3_6.sto", "indep-3", 0, false, 18615.74},
  {dir + "fxm/fxm.cor", dir + "fxm/fxm3.tim", dir + "fxm/fxm3_16.sto", "indep-3-bigger", 0, false, 18438.995},
  {dir + "fxm/fxm.cor", dir + "fxm/fxm4.tim", dir + "fxm/fxm4_6.sto", "indep-4", 0, false, 18616.05},
  {dir + "fxm/fxm.cor", dir + "fxm/fxm4.tim", dir + "fxm/fxm4_16.sto", "indep-4-bigger", 0, false, 18438.995},

  // {dir + "pltexp/pltexpA2.cor", dir + "pltexp/pltexpA2.tim", dir + "pltexp/pltexpA2_6.sto", "block-2", -100, false, -9.47935},
  // {dir + "pltexp/pltexpA2.cor", dir + "pltexp/pltexpA2.tim", dir + "pltexp/pltexpA2_16.sto", "block-2-bigger", -100, false, -9.66331},
  // {dir + "pltexp/pltexpA3.cor", dir + "pltexp/pltexpA3.tim", dir + "pltexp/pltexpA3_6.sto", "block-3", -100, false, -13.9694},
  // {dir + "pltexp/pltexpA3.cor", dir + "pltexp/pltexpA3.tim", dir + "pltexp/pltexpA3_16.sto", "block-3-bigger", -100, false, -14.2675},
  // {dir + "pltexp/pltexpA4.cor", dir + "pltexp/pltexpA4.tim", dir + "pltexp/pltexpA4_6.sto", "block-4", -100, false, -19.5994},
  
  // {dir + "pltexp/pltexpA4.cor", dir + "pltexp/pltexpA4.tim", dir + "pltexp/pltexpA4_16.sto", "block-4-bigger", -100, false, kHighsInf},
  // {dir + "pltexp/pltexpA5.cor", dir + "pltexp/pltexpA5.tim", dir + "pltexp/pltexpA5_6.sto", "block-5", -100, false, kHighsInf},
  // {dir + "pltexp/pltexpA5.cor", dir + "pltexp/pltexpA5.tim", dir + "pltexp/pltexpA5_16.sto", "block-5-bigger", -100, false, kHighsInf},
  // {dir + "pltexp/pltexpA6.cor", dir + "pltexp/pltexpA6.tim", dir + "pltexp/pltexpA6_6.sto", "block-6", -100, false, kHighsInf},
  // {dir + "pltexp/pltexpA6.cor", dir + "pltexp/pltexpA6.tim", dir + "pltexp/pltexpA6_16.sto", "block-6-bigger", -100, false, kHighsInf},

  // {dir + "pltexp/pltexpA3.cor", dir + "pltexp/pltexpA3.tim", dir + "pltexp/pltexpB3_6.sto", "blockB-3", -100, false, -13.6432},
  // {dir + "pltexp/pltexpA4.cor", dir + "pltexp/pltexpA4.tim", dir + "pltexp/pltexpB4_6.sto", "blockB-4", -100, false, -17.9282},
  // {dir + "pltexp/pltexpA5.cor", dir + "pltexp/pltexpA5.tim", dir + "pltexp/pltexpB5_6.sto", "blockB-5", -100, false, -23.8434},

  {dir + "storm/stormG2.cor", dir + "storm/stormG2.tim", dir + "storm/stormG2_8.sto", "storm-8", 0, false, 15535235.73},
  {dir + "storm/stormG2.cor", dir + "storm/stormG2.tim", dir + "storm/stormG2_27.sto", "storm-27", 0, false, 15508982.306},
  {dir + "storm/stormG2.cor", dir + "storm/stormG2.tim", dir + "storm/stormG2_125.sto", "storm-125", 0, false, 15512091.185},
  {dir + "storm/stormG2.cor", dir + "storm/stormG2.tim", dir + "storm/stormG2_1000.sto", "storm-1000", 0, false, 15802590.244},

  // {dir + "sg/sgpf5y3.cor", dir + "sg/sgpf5y3.tim", dir + "sg/sgpf5y3.sce", "scen-3", -1e4, false, -3084.234},
  // {dir + "sg/sgpf5y4.cor", dir + "sg/sgpf5y4.tim", dir + "sg/sgpf5y4.sce", "scen-4", -1e4, false, -4248.131},
  // {dir + "sg/sgpf5y5.cor", dir + "sg/sgpf5y5.tim", dir + "sg/sgpf5y5.sce", "scen-5", -1e4, false, -5593.008},
  // {dir + "sg/sgpf5y6.cor", dir + "sg/sgpf5y6.tim", dir + "sg/sgpf5y6.sce", "scen-6", -1e4, false, kHighsInf},

  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.1", "4node-1", 0, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.2", "4node-2", 0, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.4", "4node-4", 0, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.8", "4node-8", 0, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.16", "4node-16", 0, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.32", "4node-32", 0, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.64", "4node-64", 0, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.128", "4node-128", 0, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.256", "4node-256", 0, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.512", "4node-512", 0, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.1024", "4node-1024", 0, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.2048", "4node-2048", 0, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.4096", "4node-4096", 0, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.8192", "4node-8192", 0, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.16384", "4node-16384", 0, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.32768", "4node-32768", 0, false, kHighsInf},

  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.1", "4node-base-1", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.2", "4node-base-2", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.4", "4node-base-4", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.8", "4node-base-8", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.16", "4node-base-16", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.32", "4node-base-32", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.64", "4node-base-64", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.128", "4node-base-128", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.256", "4node-base-256", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.512", "4node-base-512", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.1024", "4node-base-1024", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.2048", "4node-base-2048", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.4096", "4node-base-4096", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.8192", "4node-base-8192", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.16384", "4node-base-16384", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.32768", "4node-base-32768", 0, false, kHighsInf},

  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.16old", "4node-old-16", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.16old", "4node-base-old-16", 0, false, kHighsInf},

  // {dir + "chem/chem.cor", dir + "chem/chem.tim", dir + "chem/chem.sto", "chem", -1e6, false, -13009.167},
  // {dir + "chem/chem.cor.base", dir + "chem/chem.tim", dir + "chem/chem.sto.base", "chem-base", -1e6, false, -13009.167},

  // {dir + "airlift/AIRL.cor", dir + "airlift/AIRL.tim", dir + "airlift/AIRL.sto.first", "airlift1", -1e6, false, 249101.672}, 
  // {dir + "airlift/AIRL.cor", dir + "airlift/AIRL.tim", dir + "airlift/AIRL.sto.second", "airlift2", -1e6, false, 269665.498},
  // {dir + "airlift/AIRL.cor", dir + "airlift/AIRL.tim", dir + "airlift/randgen.sto", "airlift-randgen", -1e6, false, kHighsInf},

  // {dir + "phone/phone.cor", dir + "phone/phone.tim", dir + "phone/phone.sto", "phone", 0, false, 36.9}, 
  // {dir + "phone/phone.cor", dir + "phone/phone.tim", dir + "phone/phone.sto.1", "phone1", 0, false, 36.9}, 

  // {dir + "stocfor2/stocfor2.cor", dir + "stocfor2/stocfor2.tim", dir + "stocfor2/stocfor2.sto", "stocfor", -1e6, false, -39772.448},

  // {dir + "environ/env.cor", dir + "environ/env.tim", dir + "environ/env.sto.first", "env-first", -1e6, false, kHighsInf},
  // {dir + "environ/env.cor", dir + "environ/env.tim", dir + "environ/env.sto.loose", "env-loose", -1e6, false, kHighsInf},
  // {dir + "environ/env.cor", dir + "environ/env.tim", dir + "environ/env.sto.aggr", "env-aggr", -1e6, false, kHighsInf},
  // {dir + "environ/env.cor2", dir + "environ/env.tim", dir + "environ/env.sto.imp", "env-imp", -1e6, false, kHighsInf},
  // {dir + "environ/env.cor2", dir + "environ/env.tim", dir + "environ/env.sto.1200", "env-1200", -1e6, false, kHighsInf},
  // {dir + "environ/env.cor2", dir + "environ/env.tim", dir + "environ/env.sto.1875", "env-1875", -1e6, false, kHighsInf},
  // {dir + "environ/env.cor2", dir + "environ/env.tim", dir + "environ/env.sto.3780", "env-3780", -1e6, false, kHighsInf},
  // {dir + "environ/env.cor2", dir + "environ/env.tim", dir + "environ/env.sto.5292", "env-5292", -1e6, false, kHighsInf},
  // {dir + "environ/env.cor2", dir + "environ/env.tim", dir + "environ/env.sto.lrge", "env-lrge", -1e6, false, kHighsInf},
  // {dir + "environ/env.cor2", dir + "environ/env.tim", dir + "environ/env.sto.xlrge", "env-xlrge", -1e6, false, kHighsInf},

  // {dir + "environ/env.cor.diss", dir + "environ/env.tim", dir + "environ/env.sto.first", "env-diss-first", -1e6, false, kHighsInf},
  // {dir + "environ/env.cor.diss", dir + "environ/env.tim", dir + "environ/env.sto.loose", "env-diss-loose", -1e6, false, kHighsInf},
  // {dir + "environ/env.cor.diss2", dir + "environ/env.tim", dir + "environ/env.sto.aggr", "env-diss-aggr", -1e6, false, kHighsInf},
  // {dir + "environ/env.cor.diss2", dir + "environ/env.tim", dir + "environ/env.sto.imp", "env-diss-imp", -1e6, false, kHighsInf},
  // {dir + "environ/env.cor.diss2", dir + "environ/env.tim", dir + "environ/env.sto.1200", "env-diss-1200", -1e6, false, kHighsInf},
  // {dir + "environ/env.cor.diss2", dir + "environ/env.tim", dir + "environ/env.sto.1875", "env-diss-1875", -1e6, false, kHighsInf},
  // {dir + "environ/env.cor.diss2", dir + "environ/env.tim", dir + "environ/env.sto.3780", "env-diss-3780", -1e6, false, kHighsInf},
  // {dir + "environ/env.cor.diss2", dir + "environ/env.tim", dir + "environ/env.sto.5292", "env-diss-5292", -1e6, false, kHighsInf},
  // {dir + "environ/env.cor.diss2", dir + "environ/env.tim", dir + "environ/env.sto.lrge", "env-diss-lrge", -1e6, false, kHighsInf},
  // {dir + "environ/env.cor.diss2", dir + "environ/env.tim", dir + "environ/env.sto.xlrge", "env-diss-xlrge", -1e6, false, kHighsInf},

  // {dir + "electric/LandS.cor", dir + "electric/LandS.tim", dir + "electric/LandS.sto", "electric", -1e6, false, kHighsInf},
  // {dir + "electric/LandS.cor", dir + "electric/LandS.tim", dir + "electric/LandS_blocks.sto", "electric_blocks", -1e6, false, kHighsInf},

  // {dir + "electric_3stage/LandS.cor", dir + "electric_3stage/LandS.tim", dir + "electric_3stage/LandS.sto.dep", "electric_3_dep", -1e6, false, kHighsInf},
  // {dir + "electric_3stage/LandS.cor", dir + "electric_3stage/LandS.tim", dir + "electric_3stage/LandS.sto.indep", "electric_3_indep", -1e6, false, kHighsInf},
  // {dir + "electric_3stage/LandS.cor", dir + "electric_3stage/LandS.tim", dir + "electric_3stage/LandS_blocks.sto", "electric_3_blocks", -1e6, false, kHighsInf},

  // {dir + "assets/assets.cor", dir + "assets/assets.tim", dir + "assets/assets.sto.small", "assets-small", -1e4, false, -723.839},
  // {dir + "assets/assets.cor", dir + "assets/assets.tim", dir + "assets/assets.sto.large", "assets-large", -1e4, false, -695.963},
};


// TEST_CASE("test-read", "[highs-benders]") {
//   for (auto r : read_tests) {
//     REQUIRE(r.note == r.note);
//     multi_run(r, true, 500, core_obj_start, false);
//   }
//   for (auto r : read_tests) {
//     REQUIRE(r.note == r.note);
//     multi_run(r, true, 500, core_obj_start, true);
//   }
//     for (auto r : read_tests) {
//     REQUIRE(r.note == r.note);
//     multi_run(r, true, 500, feas_start, false);
//   }
//   for (auto r : read_tests) {
//     REQUIRE(r.note == r.note);
//     multi_run(r, true, 500, feas_start, true);
//   }
// }

TEST_CASE("test-read", "[highs-benders]") {
  SmpsTestCase test {dir + "fxm/fxm.cor", dir + "fxm/fxm2.tim", dir + "fxm/fxm2_6.sto", "indep-2", 0, false, 18417.066};
  multi_run(test, true, 500, feas_start, false);
  multi_run(test, true, 500, feas_start, true);
  // multi_run(test, true, 500, core_obj_start, false);
  // multi_run(test, true, 500, core_obj_start, true);
}
