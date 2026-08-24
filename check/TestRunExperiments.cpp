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

// const double inf = kHighsInf;
const int max_iters = 500;

void zero_costs(Highs & highs) {
  std::vector<double> zeros (highs.getNumCol(), 0);
  highs.changeColsCost(0, highs.getNumCol()-1, zeros.data());
}

std::pair<std::vector<OptionValue>, MasterAdaptationParams> getMasterOpts(
    bool use_simplex=true, int start=1, int delta=2, int max_steps=15, bool flip=false, 
    bool on_improve=false, int every_n=2, double gamma=0.0, double rounding=1e-4) {
  double ipm_acc = 1e-6;
  double ipm_feas = 1e-8;
  int num_optim_steps = start;
  std::string solver = use_simplex ? kSimplexString : kHipoString;
  ipm_feas = solver == kHipoString ? ipm_feas : 1e-8;
  return {
    {
      OptionValue("ipm_iteration_limit", num_optim_steps),
      // OptionValue("dual_feasibility_tolerance", ipm_feas),
      // OptionValue("primal_feasibility_tolerance", ipm_feas),
      OptionValue("solver", solver),
      OptionValue("optimality_tolerance", ipm_acc),
      OptionValue("ipm_optimality_tolerance", ipm_acc),
      OptionValue("run_crossover", kHighsOffString),
      OptionValue("max_centring_steps", 20),
      OptionValue("presolve", kHighsOnString),
      OptionValue("centring_gamma", 1-1e-5),

      OptionValue("recentring_step", 1.0),
    },
    MasterAdaptationParams {ipm_acc, ipm_feas, num_optim_steps, delta, max_steps, flip, on_improve, every_n, gamma, rounding}
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

std::vector<SmpsTestCase> big_ones {  
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.1024", "4node-1024", 0, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.2048", "4node-2048", 0, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.4096", "4node-4096", 0, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.8192", "4node-8192", 0, false, kHighsInf},

  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.1024", "4node-base-1024", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.2048", "4node-base-2048", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.4096", "4node-base-4096", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.8192", "4node-base-8192", 0, false, kHighsInf},

  {dir + "environ/env.cor2", dir + "environ/env.tim", dir + "environ/env.sto.1200", "env-1200", -1e6, false, kHighsInf},
  {dir + "environ/env.cor2", dir + "environ/env.tim", dir + "environ/env.sto.1875", "env-1875", -1e6, false, kHighsInf},
  {dir + "environ/env.cor2", dir + "environ/env.tim", dir + "environ/env.sto.3780", "env-3780", -1e6, false, kHighsInf},
  {dir + "environ/env.cor2", dir + "environ/env.tim", dir + "environ/env.sto.5292", "env-5292", -1e6, false, kHighsInf},
  {dir + "environ/env.cor2", dir + "environ/env.tim", dir + "environ/env.sto.lrge", "env-lrge", -1e6, false, kHighsInf},

  {dir + "environ/env.cor.diss2", dir + "environ/env.tim", dir + "environ/env.sto.1200", "env-diss-1200", -1e6, false, kHighsInf},
  {dir + "environ/env.cor.diss2", dir + "environ/env.tim", dir + "environ/env.sto.1875", "env-diss-1875", -1e6, false, kHighsInf},
  {dir + "environ/env.cor.diss2", dir + "environ/env.tim", dir + "environ/env.sto.3780", "env-diss-3780", -1e6, false, kHighsInf},
  {dir + "environ/env.cor.diss2", dir + "environ/env.tim", dir + "environ/env.sto.5292", "env-diss-5292", -1e6, false, kHighsInf},
  {dir + "environ/env.cor.diss2", dir + "environ/env.tim", dir + "environ/env.sto.lrge", "env-diss-lrge", -1e6, false, kHighsInf},

  {dir + "pltexp/pltexpA4.cor", dir + "pltexp/pltexpA4.tim", dir + "pltexp/pltexpA4_16.sto", "block-4-bigger", -100, false, kHighsInf},
  {dir + "pltexp/pltexpA5.cor", dir + "pltexp/pltexpA5.tim", dir + "pltexp/pltexpA5_6.sto", "block-5", -100, false, kHighsInf},
  
};


std::vector<SmpsTestCase> bigger_ones {
  {dir + "phone/phone.cor", dir + "phone/phone.tim", dir + "phone/phone.sto", "phone", 0, false, 36.9}, 
  {dir + "assets/assets.cor", dir + "assets/assets.tim", dir + "assets/assets.sto.large", "assets-large", -1e4, false, -695.963},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.16384", "4node-16384", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.16384", "4node-base-16384", 0, false, kHighsInf},
};





std::vector<SmpsTestCase> training {
  {dir + "fxm/fxm.cor", dir + "fxm/fxm2.tim", dir + "fxm/fxm2_6.sto", "indep-2", 0, false, 18417.066},
  {dir + "fxm/fxm.cor", dir + "fxm/fxm2.tim", dir + "fxm/fxm2_16.sto", "indep-2-bigger", 0, false, 18416.759},
  {dir + "fxm/fxm.cor", dir + "fxm/fxm3.tim", dir + "fxm/fxm3_6.sto", "indep-3", 0, false, 18615.74},
  {dir + "fxm/fxm.cor", dir + "fxm/fxm3.tim", dir + "fxm/fxm3_16.sto", "indep-3-bigger", 0, false, 18438.995},
  {dir + "fxm/fxm.cor", dir + "fxm/fxm4.tim", dir + "fxm/fxm4_6.sto", "indep-4", 0, false, 18616.05},
  {dir + "fxm/fxm.cor", dir + "fxm/fxm4.tim", dir + "fxm/fxm4_16.sto", "indep-4-bigger", 0, false, 18438.995},

  {dir + "pltexp/pltexpA2.cor", dir + "pltexp/pltexpA2.tim", dir + "pltexp/pltexpA2_6.sto", "block-2", -100, false, -9.47935},
  {dir + "pltexp/pltexpA2.cor", dir + "pltexp/pltexpA2.tim", dir + "pltexp/pltexpA2_16.sto", "block-2-bigger", -100, false, -9.66331},
  {dir + "pltexp/pltexpA3.cor", dir + "pltexp/pltexpA3.tim", dir + "pltexp/pltexpA3_6.sto", "block-3", -100, false, -13.9694},
  {dir + "pltexp/pltexpA3.cor", dir + "pltexp/pltexpA3.tim", dir + "pltexp/pltexpA3_16.sto", "block-3-bigger", -100, false, -14.2675},
  {dir + "pltexp/pltexpA4.cor", dir + "pltexp/pltexpA4.tim", dir + "pltexp/pltexpA4_6.sto", "block-4", -100, false, -19.5994},

  {dir + "pltexp/pltexpA3.cor", dir + "pltexp/pltexpA3.tim", dir + "pltexp/pltexpB3_6.sto", "blockB-3", -100, false, -13.6432},
  {dir + "pltexp/pltexpA4.cor", dir + "pltexp/pltexpA4.tim", dir + "pltexp/pltexpB4_6.sto", "blockB-4", -100, false, -17.9282},
  {dir + "pltexp/pltexpA5.cor", dir + "pltexp/pltexpA5.tim", dir + "pltexp/pltexpB5_6.sto", "blockB-5", -100, false, -23.8434},

  {dir + "storm/stormG2.cor", dir + "storm/stormG2.tim", dir + "storm/stormG2_8.sto", "storm-8", 0, false, 15535235.73},
  {dir + "storm/stormG2.cor", dir + "storm/stormG2.tim", dir + "storm/stormG2_27.sto", "storm-27", 0, false, 15508982.306},
  {dir + "storm/stormG2.cor", dir + "storm/stormG2.tim", dir + "storm/stormG2_125.sto", "storm-125", 0, false, 15512091.185},
  {dir + "storm/stormG2.cor", dir + "storm/stormG2.tim", dir + "storm/stormG2_1000.sto", "storm-1000", 0, false, 15802590.244},

  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.1", "4node-1", 0, false, 413.388},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.2", "4node-2", 0, false, 414.013},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.4", "4node-4", 0, false, 416.512},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.8", "4node-8", 0, false, 418.512},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.16", "4node-16", 0, false, 423.013},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.32", "4node-32", 0, false, 423.013},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.64", "4node-64", 0, false, 423.013},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.128", "4node-128", 0, false, 423.013},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.256", "4node-256", 0, false, 425.375},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.512", "4node-512", 0, false, 429.962},

  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.1", "4node-base-1", 0, false, 413.388},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.2", "4node-base-2", 0, false, 414.013},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.4", "4node-base-4", 0, false, 414.387},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.8", "4node-base-8", 0, false, 414.687},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.16", "4node-base-16", 0, false, 414.688},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.32", "4node-base-32", 0, false, 416.600},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.64", "4node-base-64", 0, false, 416.600},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.128", "4node-base-128", 0, false, 416.600},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.256", "4node-base-256", 0, false, 417.162},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.512", "4node-base-512", 0, false, 420.292},

  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.16old", "4node-old-16", 0, false, 83094.076},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.16old", "4node-base-old-16", 0, false, 6482.567},

  {dir + "chem/chem.cor", dir + "chem/chem.tim", dir + "chem/chem.sto", "chem", -1e6, false, -13009.167},
  {dir + "chem/chem.cor.base", dir + "chem/chem.tim", dir + "chem/chem.sto.base", "chem-base", -1e6, false, -13009.167},

  {dir + "airlift/AIRL.cor", dir + "airlift/AIRL.tim", dir + "airlift/AIRL.sto.first", "airlift1", -1e6, false, 249101.672}, 
  {dir + "airlift/AIRL.cor", dir + "airlift/AIRL.tim", dir + "airlift/AIRL.sto.second", "airlift2", -1e6, false, 269665.498},
  {dir + "airlift/AIRL.cor", dir + "airlift/AIRL.tim", dir + "airlift/randgen.sto", "airlift-randgen", -1e6, false, 250262.050},

  {dir + "phone/phone.cor", dir + "phone/phone.tim", dir + "phone/phone.sto.1", "phone1", 0, false, 36.9}, 
  
  {dir + "stocfor1/stocfor1.cor", dir + "stocfor1/stocfor1.tim", dir + "stocfor1/stocfor1.sto", "stocfor1", -1e6, false, -41131.983},
  {dir + "stocfor2/stocfor2.cor", dir + "stocfor2/stocfor2.tim", dir + "stocfor2/stocfor2.sto", "stocfor2", -1e6, false, -39772.448},
  {dir + "stocfor3/stocfor3.cor", dir + "stocfor3/stocfor3.tim", dir + "stocfor3/stocfor3.sto", "stocfor3", -1e6, false, -40898.240},

  {dir + "environ/env.cor", dir + "environ/env.tim", dir + "environ/env.sto.first", "env-first", -1e6, false, 19777.445},
  {dir + "environ/env.cor", dir + "environ/env.tim", dir + "environ/env.sto.loose", "env-loose", -1e6, false, 19777.445},
  {dir + "environ/env.cor", dir + "environ/env.tim", dir + "environ/env.sto.aggr", "env-aggr", -1e6, false, 20478.699},
  {dir + "environ/env.cor2", dir + "environ/env.tim", dir + "environ/env.sto.imp", "env-imp", -1e6, false, 22265.255},

  {dir + "environ/env.cor.diss", dir + "environ/env.tim", dir + "environ/env.sto.first", "env-diss-first", -1e6, false, 14794.608},
  {dir + "environ/env.cor.diss", dir + "environ/env.tim", dir + "environ/env.sto.loose", "env-diss-loose", -1e6, false, 14794.608},
  {dir + "environ/env.cor.diss", dir + "environ/env.tim", dir + "environ/env.sto.aggr", "env-diss-aggr", -1e6, false, 15963.929},
  {dir + "environ/env.cor.diss2", dir + "environ/env.tim", dir + "environ/env.sto.imp", "env-diss-imp", -1e6, false, 20773.889},

  // {dir + "electric/LandS.cor", dir + "electric/LandS.tim", dir + "electric/LandS.sto", "electric", -1e6, false, kHighsInf},
  // {dir + "electric/LandS.cor", dir + "electric/LandS.tim", dir + "electric/LandS_blocks.sto", "electric_blocks", -1e6, false, kHighsInf},

  {dir + "assets/assets.cor", dir + "assets/assets.tim", dir + "assets/assets.sto.small", "assets-small", -1e4, false, -723.839},
};



std::vector<SmpsTestCase> biggest_ones {
  {dir + "environ/env.cor.diss2", dir + "environ/env.tim", dir + "environ/env.sto.xlrge", "env-diss-xlrge", -1e6, false, kHighsInf},
  {dir + "environ/env.cor2", dir + "environ/env.tim", dir + "environ/env.sto.xlrge", "env-xlrge", -1e6, false, kHighsInf},
  {dir + "pltexp/pltexpA6.cor", dir + "pltexp/pltexpA6.tim", dir + "pltexp/pltexpA6_6.sto", "block-6", -100, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.32768", "4node-32768", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.32768", "4node-base-32768", 0, false, kHighsInf},
  


};

std::vector<SmpsTestCase> xd_search {

  // {dir + "pltexp/pltexpA3.cor", dir + "pltexp/pltexpA3.tim", dir + "pltexp/pltexpB3_6.sto", "blockB-3", -100, false, -13.6432},
  // {dir + "pltexp/pltexpA4.cor", dir + "pltexp/pltexpA4.tim", dir + "pltexp/pltexpB4_6.sto", "blockB-4", -100, false, -17.9282},

  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.4", "4node-4", 0, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.8", "4node-8", 0, false, kHighsInf},
  
  // {dir + "airlift/AIRL.cor", dir + "airlift/AIRL.tim", dir + "airlift/randgen.sto", "airlift-randgen", -1e6, false, kHighsInf},
};

// TEST_CASE("test-read", "[highs-benders]") {
//     for (auto r : training)
//       multi_run(r, true, 1e5, feas_start, getMasterOpts(false, 5, 1, 30, false, false, 2));
//       // multi_run(r, true, 1e5, feas_start, getMasterOpts(true, 0, 0, 0, false, false, 1, 0.5, 1e-2));
          
// }


int multi_run(SmpsTestCase smps, bool aggregate, int max_scenarios, Start start, 
  std::pair<std::vector<OptionValue>, MasterAdaptationParams> params) {
  auto core_and_tree = build_stochastic_tree(smps.corefile, smps.timefile, smps.stochfile);
  auto & tree = core_and_tree.second;
  auto core = core_and_tree.first;
  auto & stage_1st = tree.root->get_child(0);
  auto & stage_2nd = stage_1st->get_child(0);
  int no_scenarios = stage_1st->get_no_children();
  if (no_scenarios > max_scenarios)
    return 0;
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
      highs2.passModel(core); apply_options(highs2, params.first); smps.note += "-core_obj_start";
    break;
    case core_feas_start:
      highs2.passModel(core); zero_costs(highs2); apply_options(highs2, params.first); smps.note += "-core_feas_start";
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
  std::ofstream("/tmp/errors.csv", std::ios::app) << smps.note  << "," << std::endl;
  std::ofstream ("/tmp/gammas.csv", std::ios::app) << smps.note  << "," << std::endl;

  // auto opts =   getMasterOpts(use_simplex);
  auto res = benders_l_shaped(core_and_tree.first, core_and_tree.second, starting_point, smps.sub_lb, 1e-3, max_iters, aggregate, params.first, params.second);
  res.note(smps.expected, smps.note);
  // if (smps.expected != kHighsInf) {
  //   REQUIRE(smps.note == smps.note);
  //   REQUIRE(no_scenarios == no_scenarios);
  //   REQUIRE(std::fabs(res.result - smps.expected) < 1e-3);
  // }
  auto diff = smps.expected != kHighsInf ? std::fabs(res.result - smps.expected) : 0.0;
  return diff < 1e-3 ? res.iter : max_iters;
  
}

int multi_run2(SmpsTestCase smps, int max_scenarios, Start start, MasterProblem & master_solver) {
  auto core_and_tree = build_stochastic_tree(smps.corefile, smps.timefile, smps.stochfile);
  auto & tree = core_and_tree.second;
  auto core = core_and_tree.first;
  auto & stage_1st = tree.root->get_child(0);
  auto & stage_2nd = stage_1st->get_child(0);
  int no_scenarios = stage_1st->get_no_children();
  if (no_scenarios > max_scenarios)
    return 0;
  


  auto const & master_range = core.stage_submatrix.at(stage_1st->get_timestage());
  int no_master_vars = master_range.col_idx_end - master_range.col_idx_begin;
  int no_master_rows = master_range.row_idx_end - master_range.row_idx_begin;

  auto const & sub_range = core.stage_submatrix.at(stage_2nd->get_timestage());
  int no_sub_vars = sub_range.col_idx_end - sub_range.col_idx_begin;
  int no_sub_rows = sub_range.row_idx_end - sub_range.row_idx_begin;

  CsvLogger log ("/tmp/dataset.csv");
  log << smps.note << no_scenarios << no_master_vars << no_master_rows << no_sub_vars << no_sub_rows;
  log.newline();
  
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
  }
  highs2.run();
  starting_point = highs2.getSolution().col_value;
  if (start == feas_start) REQUIRE(starting_point.empty());
  std::ofstream("/tmp/linking.csv", std::ios::app) << smps.note << ",";
  std::ofstream("/tmp/iteration.csv", std::ios::app) << smps.note  << "," << std::endl;
  std::ofstream("/tmp/cut_distances.csv", std::ios::app) << smps.note  << "," << std::endl;
  std::ofstream("/tmp/ubd_lbd.csv", std::ios::app) << smps.note  << "," << std::endl;
  std::ofstream("/tmp/ipm_stats.csv", std::ios::app) << smps.note  << "," << std::endl;
  std::ofstream("/tmp/errors.csv", std::ios::app) << smps.note  << "," << std::endl;
  std::ofstream ("/tmp/gammas.csv", std::ios::app) << smps.note  << "," << std::endl;

  auto res = benders_l_shaped2(core_and_tree.first, core_and_tree.second, starting_point, smps.sub_lb, master_solver, 1e-3, max_iters);
  res.note(smps.expected, smps.note);
  auto diff = smps.expected != kHighsInf ? std::fabs(res.result - smps.expected) : 0.0;
  return diff < 1e-3 ? res.iter : max_iters;
  
}


// TEST_CASE("test-read2", "[highs-benders]") {
//     for (auto r : training) {
//     //   ProximalIPMMasterProblem master_solver {5, 30, 2};
//       // LevelSetIpmMasterProblem master_solver {0.49, 0.51};
//       // LevelSetQpMasterProblem master_solver {0.5, 0.9};
//       StandardMasterProblem master_solver {}; 
//       multi_run2(r, 1e5, feas_start, master_solver);
//     }
     
          
// }

// TEST_CASE("test-read", "[highs-benders]") {
//    for (auto r : training)
//      multi_run(r, true, 1e5, feas_start, getMasterOpts(true));
// }

// TEST_CASE("test-read", "[highs-benders]") {
//   // for (auto dataset : {training})
//   //for (auto dataset : {training, big_ones, bigger_ones})
//   //  for (auto r : dataset)
//   //    multi_run(r, true, 1e5, feas_start, getMasterOpts(true));
//  // for (auto r : training)
//  //   multi_run(r, true, 1e5, feas_start, getMasterOpts(false, 5, 1, 30, false, false, 2));
//  // for (auto r : big_ones)
//  //   multi_run(r, true, 1e5, feas_start, getMasterOpts(false, 5, 1, 30, false, false, 4));
//  // for (auto r : bigger_ones)
//  //   multi_run(r, true, 1e5, feas_start, getMasterOpts(false, 5, 1, 30, false, false, 16));  
//   for (auto dataset : {training, big_ones, bigger_ones})
//     for (auto r : dataset)
//       multi_run(r, false, 1e5, feas_start, getMasterOpts(true, 0, 0, 0, false, false, 1, 0.5, 1e-8));
          
      
// }

// TEST_CASE("test-read", "[highs-benders]") {
//     for (auto gamma : {0.5})
//       for (auto mod : {0.0})
//         for (auto dataset : {biggest_ones})
//           for (auto r : dataset)
//           // if (gamma - mod >= 0)
//             multi_run(r, false, 1e5, feas_start, getMasterOpts(true, 0, 0, 0, false, false, 1, gamma, mod));
// }

// TEST_CASE("test-read-2", "[highs-benders]") {
//   bool skip = true;
//   for (auto g : {0.5, 0.9})  for (auto r : training) {
//     multi_run(r, false, 1e5, feas_start, getMasterOpts(true, 0, 0, 0, false, false, 0, g));
//   }
//   for (auto g : {0.5, 0.9})  for (auto r : big_ones) {
//     multi_run(r, false, 1e5, feas_start, getMasterOpts(true, 0, 0, 0, false, false, 0, g));
//   }
// }
// TEST_CASE("test-read", "[highs-benders]") {
//   // for (auto r : missing) multi_run(r, true, 1e5, feas_start, getMasterOpts(true));
//   for (auto every_n : {6, 8, 10, 12}) for (auto r : big_ones) 
//     multi_run(r, true, 1e5, feas_start, getMasterOpts(false, 5, 1, 30, false, false, every_n));
//   // for (auto r : missing) multi_run(r, false, 1e5, feas_start, getMasterOpts());
// }

// TEST_CASE("test-read", "[highs-benders]") {
//   for (auto smps: read_tests) {
//     auto core_and_tree = build_stochastic_tree(smps.corefile, smps.timefile, smps.stochfile);
//     auto & tree = core_and_tree.second;
//     auto core = core_and_tree.first;
//     auto & stage_1st = tree.root->get_child(0);
//     auto & stage_2nd = stage_1st->get_child(0);
//     int no_scenarios = stage_1st->get_no_children();

//     auto const & master_range = core.stage_submatrix.at(stage_1st->get_timestage());
//     int no_master_vars = master_range.col_idx_end - master_range.col_idx_begin;
//     int no_master_rows = master_range.row_idx_end - master_range.row_idx_begin;

//     auto const & sub_range = core.stage_submatrix.at(stage_2nd->get_timestage());
//     int no_sub_vars = sub_range.col_idx_end - sub_range.col_idx_begin;
//     int no_sub_rows = sub_range.row_idx_end - sub_range.row_idx_begin;


//     std::set<HighsInt> master_variables;
//     auto const & node_ranges = core.stage_submatrix.at(stage_1st->get_timestage());
//     for (int i = node_ranges.col_idx_begin; i < node_ranges.col_idx_end; ++i) master_variables.emplace(i);

//     MultiBendersProblems problems;
//     auto row_division = divide_rows(core.a_matrix_, master_variables);
//     int no_linking = row_division.mixed_rows.size();

//     CsvLogger log ("/tmp/datasize.csv");
//     log << smps.note << no_scenarios << no_master_vars << no_master_rows << no_sub_vars << no_sub_rows 
//       << no_master_vars + no_sub_vars * no_scenarios << no_master_rows + no_sub_rows * no_scenarios
//       << no_linking;
//     log.newline();
//   }
   
// }
