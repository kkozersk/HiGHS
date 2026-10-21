#include "benders/Benders.h"
#include "HCheckConfig.h"
#include "io/SMPS.h"
#include "catch.hpp"
#include <cmath>
#include <vector>

const int max_iters = 500;


auto dir = std::string(HIGHS_DIR) + "/check/instances/stoch/";
struct SmpsTestCase {
  std::string corefile, timefile, stochfile, note;
  double sub_lb;
  bool aggregate;
  double expected;
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
  
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.16384", "4node-16384", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.16384", "4node-base-16384", 0, false, kHighsInf},
  {dir + "phone/phone.cor", dir + "phone/phone.tim", dir + "phone/phone.sto", "phone", 0, false, 36.9}, 
  {dir + "assets/assets.cor", dir + "assets/assets.tim", dir + "assets/assets.sto.large", "assets-large", -1e4, false, -695.963},
};

std::vector<SmpsTestCase> big2 {  
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.16384", "4node-base-16384", 0, false, kHighsInf},
  {dir + "phone/phone.cor", dir + "phone/phone.tim", dir + "phone/phone.sto", "phone", 0, false, 36.9}, 
  {dir + "assets/assets.cor", dir + "assets/assets.tim", dir + "assets/assets.sto.large", "assets-large", -1e4, false, -695.963},
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

  {dir + "electric/LandS.cor", dir + "electric/LandS.tim", dir + "electric/LandS.sto", "electric", -1e6, false, kHighsInf},
  {dir + "electric/LandS.cor", dir + "electric/LandS.tim", dir + "electric/LandS_blocks.sto", "electric-blocks", -1e6, false, kHighsInf},

  {dir + "electric_3stage/LandS.cor", dir + "electric_3stage/LandS.tim", dir + "electric_3stage/LandS.sto.dep", "electric-3-dep", -1e6, false, kHighsInf},
  {dir + "electric_3stage/LandS.cor", dir + "electric_3stage/LandS.tim", dir + "electric_3stage/LandS.sto.indep", "electric-3-indep", -1e6, false, kHighsInf},
  {dir + "electric_3stage/LandS.cor", dir + "electric_3stage/LandS.tim", dir + "electric_3stage/LandS_blocks.sto", "electric-3-blocks", -1e6, false, kHighsInf},

  {dir + "assets/assets.cor", dir + "assets/assets.tim", dir + "assets/assets.sto.small", "assets-small", -1e4, false, -723.839},
};

std::vector<SmpsTestCase> biggest_ones {
  {dir + "environ/env.cor.diss2", dir + "environ/env.tim", dir + "environ/env.sto.xlrge", "env-diss-xlrge", -1e6, false, kHighsInf},
  {dir + "environ/env.cor2", dir + "environ/env.tim", dir + "environ/env.sto.xlrge", "env-xlrge", -1e6, false, kHighsInf},
  {dir + "pltexp/pltexpA6.cor", dir + "pltexp/pltexpA6.tim", dir + "pltexp/pltexpA6_6.sto", "block-6", -100, false, kHighsInf},
  {dir + "cargo/4node.cor", dir + "cargo/4node.tim", dir + "cargo/4node.sto.32768", "4node-32768", 0, false, kHighsInf},
  {dir + "cargo/4node.cor.base", dir + "cargo/4node.tim", dir + "cargo/4node.sto.32768", "4node-base-32768", 0, false, kHighsInf},
  


};

int multi_run(SmpsTestCase smps, int max_scenarios, MasterProblem & master_solver, std::string const & logfile="/tmp/results.csv") {
  auto core_and_tree = build_stochastic_tree(smps.corefile, smps.timefile, smps.stochfile);
  assert(core_and_tree.second.root != nullptr);
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

  auto res = benders_l_shaped(core_and_tree.first, core_and_tree.second, {}, smps.sub_lb, master_solver, 1e-6, max_iters, BendersAlgorithm::GAP::RELGAP);
  res.note(smps.expected, smps.note, logfile);
  auto diff = smps.expected != kHighsInf ? std::fabs(res.result - smps.expected) : 0.0;
  return diff < 1e-3 ? res.iter : max_iters;
  
}

std::vector<SmpsTestCase> new_ones {
  // {dir + "Cornell/Weekly-Small/core_nospace.mps",  dir + "Cornell/Weekly-Small/time_nospace.mps",  dir + "Cornell/Weekly-Small/stoch_nospace.mps",  "weekly-small", -1e6, false, kHighsInf}, 
  // {dir + "Cornell/Weekly-Medium/core_nospace.mps", dir + "Cornell/Weekly-Medium/time_nospace.mps", dir + "Cornell/Weekly-Medium/stoch_nospace.mps", "weekly-medium", -1e6, false, kHighsInf}, 
  // {dir + "Cornell/Weekly-Large/core_nospace.mps",  dir + "Cornell/Weekly-Large/time_nospace.mps",  dir + "Cornell/Weekly-Large/stoch_nospace.mps",  "weekly-large", -1e6, false, kHighsInf}, 

  // {dir + "Cornell/Bi-Weekly-Small/core_nospace.mps",  dir + "Cornell/Bi-Weekly-Small/time_nospace.mps",  dir + "Cornell/Bi-Weekly-Small/stoch_nospace.mps",  "bi-weekly-small", -1e6, false, kHighsInf}, 
  // {dir + "Cornell/Bi-Weekly-Medium/core_nospace.mps", dir + "Cornell/Bi-Weekly-Medium/time_nospace.mps", dir + "Cornell/Bi-Weekly-Medium/stoch_nospace.mps", "bi-weekly-medium", -1e6, false, kHighsInf}, 
  // {dir + "Cornell/Bi-Weekly-Large/core_nospace.mps",  dir + "Cornell/Bi-Weekly-Large/time_nospace.mps",  dir + "Cornell/Bi-Weekly-Large/stoch_nospace.mps",  "bi-weekly-large", -1e6, false, kHighsInf}, 

  // {dir + "Cornell/Product-Small/core_nospace.mps",  dir + "Cornell/Product-Small/time_nospace.mps",  dir + "Cornell/Product-Small/stoch_nospace.mps",  "product-small", -1e6, false, kHighsInf}, 
  // {dir + "Cornell/Product-Medium/core_nospace.mps", dir + "Cornell/Product-Medium/time_nospace.mps", dir + "Cornell/Product-Medium/stoch_nospace.mps", "product-medium", -1e6, false, kHighsInf}, 
  // {dir + "Cornell/Product-Large/core_nospace.mps",  dir + "Cornell/Product-Large/time_nospace.mps",  dir + "Cornell/Product-Large/stoch_nospace.mps",  "product-large", -1e6, false, kHighsInf}, 

  // {dir + "Cornell/Electricity-Small/core_nospace.mps",  dir + "Cornell/Electricity-Small/time_nospace.mps",  dir + "Cornell/Electricity-Small/stoch_nospace.mps",  "electricity-small", -1e6, false, kHighsInf}, 
  // {dir + "Cornell/Electricity-Medium/core_nospace.mps", dir + "Cornell/Electricity-Medium/time_nospace.mps", dir + "Cornell/Electricity-Medium/stoch_nospace.mps", "electricity-medium", -1e6, false, kHighsInf}, 
  // {dir + "Cornell/Electricity-Large/core_nospace.mps",  dir + "Cornell/Electricity-Large/time_nospace.mps",  dir + "Cornell/Electricity-Large/stoch_nospace.mps",  "electricity-large", -1e6, false, kHighsInf}, 
  
  // {dir + "saphir/saphir-50.cor", dir + "saphir/saphir-50.tim", dir + "saphir/saphir-50.sto", "saphir-50", -1e6, false, kHighsInf},
  // {dir + "saphir/saphir-100.cor", dir + "saphir/saphir-100.tim", dir + "saphir/saphir-100.sto", "saphir-100", -1e6, false, kHighsInf},
  // {dir + "saphir/saphir-200.cor", dir + "saphir/saphir-200.tim", dir + "saphir/saphir-200.sto", "saphir-200", -1e6, false, kHighsInf},
  // {dir + "saphir/saphir-500.cor", dir + "saphir/saphir-500.tim", dir + "saphir/saphir-500.sto", "saphir-500", -1e6, false, kHighsInf},
  // {dir + "saphir/saphir-1000.cor", dir + "saphir/saphir-1000.tim", dir + "saphir/saphir-1000.sto", "saphir-1000", -1e6, false, kHighsInf},

  // {dir + "rand/rand0.cor", dir + "rand/rand0.tim", dir + "rand/rand0-2000.sto", "rand0-2000", -1e6, false, kHighsInf},
  // {dir + "rand/rand0.cor", dir + "rand/rand0.tim", dir + "rand/rand0-4000.sto", "rand0-4000", -1e6, false, kHighsInf},
  // {dir + "rand/rand0.cor", dir + "rand/rand0.tim", dir + "rand/rand0-6000.sto", "rand0-6000", -1e6, false, kHighsInf},
  // {dir + "rand/rand0.cor", dir + "rand/rand0.tim", dir + "rand/rand0-8000.sto", "rand0-8000", -1e6, false, kHighsInf},
  // {dir + "rand/rand0.cor", dir + "rand/rand0.tim", dir + "rand/rand0-10000.sto", "rand0-10000", -1e6, false, kHighsInf},

  // {dir + "rand/rand1.cor", dir + "rand/rand1.tim", dir + "rand/rand1-2000.sto", "rand1-2000", -1e6, false, kHighsInf},
  // {dir + "rand/rand1.cor", dir + "rand/rand1.tim", dir + "rand/rand1-4000.sto", "rand1-4000", -1e6, false, kHighsInf},
  // {dir + "rand/rand1.cor", dir + "rand/rand1.tim", dir + "rand/rand1-6000.sto", "rand1-6000", -1e6, false, kHighsInf},
  // {dir + "rand/rand1.cor", dir + "rand/rand1.tim", dir + "rand/rand1-8000.sto", "rand1-8000", -1e6, false, kHighsInf},
  // {dir + "rand/rand1.cor", dir + "rand/rand1.tim", dir + "rand/rand1-10000.sto", "rand1-10000", -1e6, false, kHighsInf},

  // {dir + "rand/rand2.cor", dir + "rand/rand2.tim", dir + "rand/rand2-2000.sto", "rand2-2000", -1e6, false, kHighsInf},
  // {dir + "rand/rand2.cor", dir + "rand/rand2.tim", dir + "rand/rand2-4000.sto", "rand2-4000", -1e6, false, kHighsInf},
  // {dir + "rand/rand2.cor", dir + "rand/rand2.tim", dir + "rand/rand2-6000.sto", "rand2-6000", -1e6, false, kHighsInf},
  // {dir + "rand/rand2.cor", dir + "rand/rand2.tim", dir + "rand/rand2-8000.sto", "rand2-8000", -1e6, false, kHighsInf},
  // {dir + "rand/rand2.cor", dir + "rand/rand2.tim", dir + "rand/rand2-10000.sto", "rand2-10000", -1e6, false, kHighsInf},
  
  // {dir + "SIPLIB/dcap/dcap233_200.cor", dir + "SIPLIB/dcap/dcap233_200.tim", dir + "SIPLIB/dcap/dcap233_200.sto", "dcap233_200", -1e6, false, kHighsInf},
  // {dir + "SIPLIB/dcap/dcap233_300.cor", dir + "SIPLIB/dcap/dcap233_300.tim", dir + "SIPLIB/dcap/dcap233_300.sto", "dcap233_300", -1e6, false, kHighsInf},
  // {dir + "SIPLIB/dcap/dcap233_500.cor", dir + "SIPLIB/dcap/dcap233_500.tim", dir + "SIPLIB/dcap/dcap233_500.sto", "dcap233_500", -1e6, false, kHighsInf},

  // {dir + "SIPLIB/dcap/dcap243_200.cor", dir + "SIPLIB/dcap/dcap243_200.tim", dir + "SIPLIB/dcap/dcap243_200.sto", "dcap243_200", -1e6, false, kHighsInf},
  // {dir + "SIPLIB/dcap/dcap243_300.cor", dir + "SIPLIB/dcap/dcap243_300.tim", dir + "SIPLIB/dcap/dcap243_300.sto", "dcap243_300", -1e6, false, kHighsInf},
  // {dir + "SIPLIB/dcap/dcap243_500.cor", dir + "SIPLIB/dcap/dcap243_500.tim", dir + "SIPLIB/dcap/dcap243_500.sto", "dcap243_500", -1e6, false, kHighsInf},

  // {dir + "SIPLIB/dcap/dcap332_200.cor", dir + "SIPLIB/dcap/dcap332_200.tim", dir + "SIPLIB/dcap/dcap332_200.sto", "dcap332_200", -1e6, false, kHighsInf},
  // {dir + "SIPLIB/dcap/dcap332_300.cor", dir + "SIPLIB/dcap/dcap332_300.tim", dir + "SIPLIB/dcap/dcap332_300.sto", "dcap332_300", -1e6, false, kHighsInf},
  // {dir + "SIPLIB/dcap/dcap332_500.cor", dir + "SIPLIB/dcap/dcap332_500.tim", dir + "SIPLIB/dcap/dcap332_500.sto", "dcap332_500", -1e6, false, kHighsInf},

  // {dir + "SIPLIB/dcap/dcap342_200.cor", dir + "SIPLIB/dcap/dcap342_200.tim", dir + "SIPLIB/dcap/dcap342_200.sto", "dcap342_200", -1e6, false, kHighsInf},
  // {dir + "SIPLIB/dcap/dcap342_300.cor", dir + "SIPLIB/dcap/dcap342_300.tim", dir + "SIPLIB/dcap/dcap342_300.sto", "dcap342_300", -1e6, false, kHighsInf},
  // {dir + "SIPLIB/dcap/dcap342_500.cor", dir + "SIPLIB/dcap/dcap342_500.tim", dir + "SIPLIB/dcap/dcap342_500.sto", "dcap342_500", -1e6, false, kHighsInf},

  // {dir + "SIPLIB/semi/semi.core", dir + "SIPLIB/semi/semi.time", dir + "SIPLIB/semi/semi2.stoch", "semi2", -1e6, false, kHighsInf},
  // {dir + "SIPLIB/semi/semi.core", dir + "SIPLIB/semi/semi.time", dir + "SIPLIB/semi/semi3.stoch", "semi3", -1e6, false, kHighsInf},
  // {dir + "SIPLIB/semi/semi.core", dir + "SIPLIB/semi/semi.time", dir + "SIPLIB/semi/semi4.stoch", "semi4", -1e6, false, kHighsInf},
  
  // {dir + "SIPLIB/sizes/sizes.cor", dir + "SIPLIB/sizes/sizes.tim", dir + "SIPLIB/sizes/sizes3.sto", "sizes3", -1e6, false, kHighsInf},
  // {dir + "SIPLIB/sizes/sizes.cor", dir + "SIPLIB/sizes/sizes.tim", dir + "SIPLIB/sizes/sizes5.sto", "sizes5", -1e6, false, kHighsInf},
  // {dir + "SIPLIB/sizes/sizes.cor", dir + "SIPLIB/sizes/sizes.tim", dir + "SIPLIB/sizes/sizes10.sto", "sizes10", -1e6, false, kHighsInf},

  // {dir + "wright/20term/20.cor", dir + "wright/20term/20.tim", dir + "wright/20term/20.sto", "20term", -1e6, false, kHighsInf},
  // {dir + "wright/gbd/gbd.cor", dir + "wright/gbd/gbd.tim", dir + "wright/gbd/gbd.sto", "gbd", -1e6, false, kHighsInf},
  // {dir + "wright/ssn/ssn.cor", dir + "wright/ssn/ssn.tim", dir + "wright/ssn/ssn.sto", "ssn", -1e6, false, kHighsInf},
  // {dir + "wright/storm/storm.cor", dir + "wright/storm/storm.tim", dir + "wright/storm/storm.sto", "storm", -1e6, false, kHighsInf},

  // { dir + "SIPLIB/smkp/smkp_1.cor", dir + "SIPLIB/smkp/smkp_1.tim", dir + "SIPLIB/smkp/smkp_1.sto", "smkp_1", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_2.cor", dir + "SIPLIB/smkp/smkp_2.tim", dir + "SIPLIB/smkp/smkp_2.sto", "smkp_2", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_3.cor", dir + "SIPLIB/smkp/smkp_3.tim", dir + "SIPLIB/smkp/smkp_3.sto", "smkp_3", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_4.cor", dir + "SIPLIB/smkp/smkp_4.tim", dir + "SIPLIB/smkp/smkp_4.sto", "smkp_4", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_5.cor", dir + "SIPLIB/smkp/smkp_5.tim", dir + "SIPLIB/smkp/smkp_5.sto", "smkp_5", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_6.cor", dir + "SIPLIB/smkp/smkp_6.tim", dir + "SIPLIB/smkp/smkp_6.sto", "smkp_6", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_7.cor", dir + "SIPLIB/smkp/smkp_7.tim", dir + "SIPLIB/smkp/smkp_7.sto", "smkp_7", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_8.cor", dir + "SIPLIB/smkp/smkp_8.tim", dir + "SIPLIB/smkp/smkp_8.sto", "smkp_8", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_9.cor", dir + "SIPLIB/smkp/smkp_9.tim", dir + "SIPLIB/smkp/smkp_9.sto", "smkp_9", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_10.cor", dir + "SIPLIB/smkp/smkp_10.tim", dir + "SIPLIB/smkp/smkp_10.sto", "smkp_10", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_11.cor", dir + "SIPLIB/smkp/smkp_11.tim", dir + "SIPLIB/smkp/smkp_11.sto", "smkp_11", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_12.cor", dir + "SIPLIB/smkp/smkp_12.tim", dir + "SIPLIB/smkp/smkp_12.sto", "smkp_12", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_13.cor", dir + "SIPLIB/smkp/smkp_13.tim", dir + "SIPLIB/smkp/smkp_13.sto", "smkp_13", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_14.cor", dir + "SIPLIB/smkp/smkp_14.tim", dir + "SIPLIB/smkp/smkp_14.sto", "smkp_14", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_15.cor", dir + "SIPLIB/smkp/smkp_15.tim", dir + "SIPLIB/smkp/smkp_15.sto", "smkp_15", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_16.cor", dir + "SIPLIB/smkp/smkp_16.tim", dir + "SIPLIB/smkp/smkp_16.sto", "smkp_16", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_17.cor", dir + "SIPLIB/smkp/smkp_17.tim", dir + "SIPLIB/smkp/smkp_17.sto", "smkp_17", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_18.cor", dir + "SIPLIB/smkp/smkp_18.tim", dir + "SIPLIB/smkp/smkp_18.sto", "smkp_18", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_19.cor", dir + "SIPLIB/smkp/smkp_19.tim", dir + "SIPLIB/smkp/smkp_19.sto", "smkp_19", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_20.cor", dir + "SIPLIB/smkp/smkp_20.tim", dir + "SIPLIB/smkp/smkp_20.sto", "smkp_20", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_21.cor", dir + "SIPLIB/smkp/smkp_21.tim", dir + "SIPLIB/smkp/smkp_21.sto", "smkp_21", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_22.cor", dir + "SIPLIB/smkp/smkp_22.tim", dir + "SIPLIB/smkp/smkp_22.sto", "smkp_22", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_23.cor", dir + "SIPLIB/smkp/smkp_23.tim", dir + "SIPLIB/smkp/smkp_23.sto", "smkp_23", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_24.cor", dir + "SIPLIB/smkp/smkp_24.tim", dir + "SIPLIB/smkp/smkp_24.sto", "smkp_24", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_25.cor", dir + "SIPLIB/smkp/smkp_25.tim", dir + "SIPLIB/smkp/smkp_25.sto", "smkp_25", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_26.cor", dir + "SIPLIB/smkp/smkp_26.tim", dir + "SIPLIB/smkp/smkp_26.sto", "smkp_26", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_27.cor", dir + "SIPLIB/smkp/smkp_27.tim", dir + "SIPLIB/smkp/smkp_27.sto", "smkp_27", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_28.cor", dir + "SIPLIB/smkp/smkp_28.tim", dir + "SIPLIB/smkp/smkp_28.sto", "smkp_28", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_29.cor", dir + "SIPLIB/smkp/smkp_29.tim", dir + "SIPLIB/smkp/smkp_29.sto", "smkp_29", -1e6, false, kHighsInf},
  // { dir + "SIPLIB/smkp/smkp_30.cor", dir + "SIPLIB/smkp/smkp_30.tim", dir + "SIPLIB/smkp/smkp_30.sto", "smkp_30", -1e6, false, kHighsInf},
   
  // { dir + "/SIPLIB/sslp/sslp_10_50_1000.cor" , dir + "/SIPLIB/sslp/sslp_10_50_1000.tim" , dir + "SIPLIB/sslp/sslp_10_50_1000.sto", "sslp_10_50_1000", -1e5, false, kHighsInf},
  // { dir + "/SIPLIB/sslp/sslp_10_50_100.cor" , dir + "/SIPLIB/sslp/sslp_10_50_100.tim" , dir + "SIPLIB/sslp/sslp_10_50_100.sto", "sslp_10_50_100", -1e5, false, kHighsInf},
  // { dir + "/SIPLIB/sslp/sslp_10_50_2000.cor" , dir + "/SIPLIB/sslp/sslp_10_50_2000.tim" , dir + "SIPLIB/sslp/sslp_10_50_2000.sto", "sslp_10_50_2000", -1e5, false, kHighsInf},
  // { dir + "/SIPLIB/sslp/sslp_10_50_500.cor" , dir + "/SIPLIB/sslp/sslp_10_50_500.tim" , dir + "SIPLIB/sslp/sslp_10_50_500.sto", "sslp_10_50_500", -1e5, false, kHighsInf},
  // { dir + "/SIPLIB/sslp/sslp_10_50_50.cor" , dir + "/SIPLIB/sslp/sslp_10_50_50.tim" , dir + "SIPLIB/sslp/sslp_10_50_50.sto", "sslp_10_50_50", -1e5, false, kHighsInf},
  // { dir + "/SIPLIB/sslp/sslp_15_45_10.cor" , dir + "/SIPLIB/sslp/sslp_15_45_10.tim" , dir + "SIPLIB/sslp/sslp_15_45_10.sto", "sslp_15_45_10", -1e5, false, kHighsInf},
  // { dir + "/SIPLIB/sslp/sslp_15_45_15.cor" , dir + "/SIPLIB/sslp/sslp_15_45_15.tim" , dir + "SIPLIB/sslp/sslp_15_45_15.sto", "sslp_15_45_15", -1e5, false, kHighsInf},
  // { dir + "/SIPLIB/sslp/sslp_15_45_5.cor" , dir + "/SIPLIB/sslp/sslp_15_45_5.tim" , dir + "SIPLIB/sslp/sslp_15_45_5.sto", "sslp_15_45_5", -1e5, false, kHighsInf},
  // { dir + "/SIPLIB/sslp/sslp_5_25_100.cor" , dir + "/SIPLIB/sslp/sslp_5_25_100.tim" , dir + "SIPLIB/sslp/sslp_5_25_100.sto", "sslp_5_25_100", -1e5, false, kHighsInf},
  // { dir + "/SIPLIB/sslp/sslp_5_25_50.cor" , dir + "/SIPLIB/sslp/sslp_5_25_50.tim" , dir + "SIPLIB/sslp/sslp_5_25_50.sto", "sslp_5_25_50", -1e5, false, kHighsInf},

  // {dir + "SIPLIB/Vaccine/vac100a.cor", dir +  "SIPLIB/Vaccine/vac100a.tim", dir +  "SIPLIB/Vaccine/vac100a.sto", "vac100a", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac100b.cor", dir +  "SIPLIB/Vaccine/vac100b.tim", dir +  "SIPLIB/Vaccine/vac100b.sto", "vac100b", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac100c.cor", dir +  "SIPLIB/Vaccine/vac100c.tim", dir +  "SIPLIB/Vaccine/vac100c.sto", "vac100c", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac100d.cor", dir +  "SIPLIB/Vaccine/vac100d.tim", dir +  "SIPLIB/Vaccine/vac100d.sto", "vac100d", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac100e.cor", dir +  "SIPLIB/Vaccine/vac100e.tim", dir +  "SIPLIB/Vaccine/vac100e.sto", "vac100e", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac250a.cor", dir +  "SIPLIB/Vaccine/vac250a.tim", dir +  "SIPLIB/Vaccine/vac250a.sto", "vac250a", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac250b.cor", dir +  "SIPLIB/Vaccine/vac250b.tim", dir +  "SIPLIB/Vaccine/vac250b.sto", "vac250b", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac250c.cor", dir +  "SIPLIB/Vaccine/vac250c.tim", dir +  "SIPLIB/Vaccine/vac250c.sto", "vac250c", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac250d.cor", dir +  "SIPLIB/Vaccine/vac250d.tim", dir +  "SIPLIB/Vaccine/vac250d.sto", "vac250d", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac250e.cor", dir +  "SIPLIB/Vaccine/vac250e.tim", dir +  "SIPLIB/Vaccine/vac250e.sto", "vac250e", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac500a.cor", dir +  "SIPLIB/Vaccine/vac500a.tim", dir +  "SIPLIB/Vaccine/vac500a.sto", "vac500a", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac500b.cor", dir +  "SIPLIB/Vaccine/vac500b.tim", dir +  "SIPLIB/Vaccine/vac500b.sto", "vac500b", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac500c.cor", dir +  "SIPLIB/Vaccine/vac500c.tim", dir +  "SIPLIB/Vaccine/vac500c.sto", "vac500c", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac500d.cor", dir +  "SIPLIB/Vaccine/vac500d.tim", dir +  "SIPLIB/Vaccine/vac500d.sto", "vac500d", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac500e.cor", dir +  "SIPLIB/Vaccine/vac500e.tim", dir +  "SIPLIB/Vaccine/vac500e.sto", "vac500e", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac750a.cor", dir +  "SIPLIB/Vaccine/vac750a.tim", dir +  "SIPLIB/Vaccine/vac750a.sto", "vac750a", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac750b.cor", dir +  "SIPLIB/Vaccine/vac750b.tim", dir +  "SIPLIB/Vaccine/vac750b.sto", "vac750b", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac750c.cor", dir +  "SIPLIB/Vaccine/vac750c.tim", dir +  "SIPLIB/Vaccine/vac750c.sto", "vac750c", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac750d.cor", dir +  "SIPLIB/Vaccine/vac750d.tim", dir +  "SIPLIB/Vaccine/vac750d.sto", "vac750d", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac750e.cor", dir +  "SIPLIB/Vaccine/vac750e.tim", dir +  "SIPLIB/Vaccine/vac750e.sto", "vac750e", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac1000a.cor", dir +  "SIPLIB/Vaccine/vac1000a.tim", dir +  "SIPLIB/Vaccine/vac1000a.sto", "vac1000a", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac1000b.cor", dir +  "SIPLIB/Vaccine/vac1000b.tim", dir +  "SIPLIB/Vaccine/vac1000b.sto", "vac1000b", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac1000c.cor", dir +  "SIPLIB/Vaccine/vac1000c.tim", dir +  "SIPLIB/Vaccine/vac1000c.sto", "vac1000c", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac1000d.cor", dir +  "SIPLIB/Vaccine/vac1000d.tim", dir +  "SIPLIB/Vaccine/vac1000d.sto", "vac1000d", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac1000e.cor", dir +  "SIPLIB/Vaccine/vac1000e.tim", dir +  "SIPLIB/Vaccine/vac1000e.sto", "vac1000e", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac2000a.cor", dir +  "SIPLIB/Vaccine/vac2000a.tim", dir +  "SIPLIB/Vaccine/vac2000a.sto", "vac2000a", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac2000b.cor", dir +  "SIPLIB/Vaccine/vac2000b.tim", dir +  "SIPLIB/Vaccine/vac2000b.sto", "vac2000b", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac2000c.cor", dir +  "SIPLIB/Vaccine/vac2000c.tim", dir +  "SIPLIB/Vaccine/vac2000c.sto", "vac2000c", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac2000d.cor", dir +  "SIPLIB/Vaccine/vac2000d.tim", dir +  "SIPLIB/Vaccine/vac2000d.sto", "vac2000d", -1e5, false, kHighsInf},
  // {dir + "SIPLIB/Vaccine/vac2000e.cor", dir +  "SIPLIB/Vaccine/vac2000e.tim", dir +  "SIPLIB/Vaccine/vac2000e.sto", "vac2000e", -1e5, false, kHighsInf},
  
  {dir + "watson/watc-10.cor", dir + "watson/watc-10.tim", dir + "watson/watc-10-16.sto", "watc-10-16", -1e5, false, kHighsInf},
  {dir + "watson/watc-10.cor", dir + "watson/watc-10.tim", dir + "watson/watc-10-32.sto", "watc-10-32", -1e5, false, kHighsInf},
  {dir + "watson/watc-10.cor", dir + "watson/watc-10.tim", dir + "watson/watc-10-64.sto", "watc-10-64", -1e5, false, kHighsInf},
  {dir + "watson/watc-10.cor", dir + "watson/watc-10.tim", dir + "watson/watc-10-128.sto", "watc-10-128", -1e5, false, kHighsInf},
  {dir + "watson/watc-10.cor", dir + "watson/watc-10.tim", dir + "watson/watc-10-256.sto", "watc-10-256", -1e5, false, kHighsInf},
  {dir + "watson/watc-10.cor", dir + "watson/watc-10.tim", dir + "watson/watc-10-512.sto", "watc-10-512", -1e5, false, kHighsInf},
  {dir + "watson/watc-10.cor", dir + "watson/watc-10.tim", dir + "watson/watc-10-1024.sto", "watc-10-1024", -1e5, false, kHighsInf},
  {dir + "watson/watc-10.cor", dir + "watson/watc-10.tim", dir + "watson/watc-10-1152.sto", "watc-10-1152", -1e5, false, kHighsInf},
  {dir + "watson/watc-10.cor", dir + "watson/watc-10.tim", dir + "watson/watc-10-1536.sto", "watc-10-1536", -1e5, false, kHighsInf},
  {dir + "watson/watc-10.cor", dir + "watson/watc-10.tim", dir + "watson/watc-10-1920.sto", "watc-10-1920", -1e5, false, kHighsInf},
  {dir + "watson/watc-10.cor", dir + "watson/watc-10.tim", dir + "watson/watc-10-2304.sto", "watc-10-2304", -1e5, false, kHighsInf},
  {dir + "watson/watc-10.cor", dir + "watson/watc-10.tim", dir + "watson/watc-10-2688.sto", "watc-10-2688", -1e5, false, kHighsInf},

  {dir + "watson/wati-10.cor", dir + "watson/wati-10.tim", dir + "watson/wati-10-16.sto", "wati-10-16", -1e5, false, kHighsInf},
  {dir + "watson/wati-10.cor", dir + "watson/wati-10.tim", dir + "watson/wati-10-32.sto", "wati-10-32", -1e5, false, kHighsInf},
  {dir + "watson/wati-10.cor", dir + "watson/wati-10.tim", dir + "watson/wati-10-64.sto", "wati-10-64", -1e5, false, kHighsInf},
  {dir + "watson/wati-10.cor", dir + "watson/wati-10.tim", dir + "watson/wati-10-128.sto", "wati-10-128", -1e5, false, kHighsInf},
  {dir + "watson/wati-10.cor", dir + "watson/wati-10.tim", dir + "watson/wati-10-256.sto", "wati-10-256", -1e5, false, kHighsInf},
  {dir + "watson/wati-10.cor", dir + "watson/wati-10.tim", dir + "watson/wati-10-512.sto", "wati-10-512", -1e5, false, kHighsInf},
  {dir + "watson/wati-10.cor", dir + "watson/wati-10.tim", dir + "watson/wati-10-1024.sto", "wati-10-1024", -1e5, false, kHighsInf},
  {dir + "watson/wati-10.cor", dir + "watson/wati-10.tim", dir + "watson/wati-10-1152.sto", "wati-10-1152", -1e5, false, kHighsInf},
  {dir + "watson/wati-10.cor", dir + "watson/wati-10.tim", dir + "watson/wati-10-1536.sto", "wati-10-1536", -1e5, false, kHighsInf},
  {dir + "watson/wati-10.cor", dir + "watson/wati-10.tim", dir + "watson/wati-10-1920.sto", "wati-10-1920", -1e5, false, kHighsInf},
  
  // {dir + "RANDOMRHS/IC-K1.cor", dir + "RANDOMRHS/IC-K1.tim", dir + "RANDOMRHS/IC-K1.sto", "IC-K1", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-K2.cor", dir + "RANDOMRHS/IC-K2.tim", dir + "RANDOMRHS/IC-K2.sto", "IC-K2", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-K3.cor", dir + "RANDOMRHS/IC-K3.tim", dir + "RANDOMRHS/IC-K3.sto", "IC-K3", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-K4.cor", dir + "RANDOMRHS/IC-K4.tim", dir + "RANDOMRHS/IC-K4.sto", "IC-K4", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-K5.cor", dir + "RANDOMRHS/IC-K5.tim", dir + "RANDOMRHS/IC-K5.sto", "IC-K5", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-K6.cor", dir + "RANDOMRHS/IC-K6.tim", dir + "RANDOMRHS/IC-K6.sto", "IC-K6", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-K7.cor", dir + "RANDOMRHS/IC-K7.tim", dir + "RANDOMRHS/IC-K7.sto", "IC-K7", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-K8.cor", dir + "RANDOMRHS/IC-K8.tim", dir + "RANDOMRHS/IC-K8.sto", "IC-K8", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-K9.cor", dir + "RANDOMRHS/IC-K9.tim", dir + "RANDOMRHS/IC-K9.sto", "IC-K9", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-K10.cor", dir + "RANDOMRHS/IC-K10.tim", dir + "RANDOMRHS/IC-K10.sto", "IC-K10", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T1-1.cor", dir + "RANDOMRHS/IC-T1-1.tim", dir + "RANDOMRHS/IC-T1-1.sto", "IC-T1-1", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T1-2.cor", dir + "RANDOMRHS/IC-T1-2.tim", dir + "RANDOMRHS/IC-T1-2.sto", "IC-T1-2", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T1-3.cor", dir + "RANDOMRHS/IC-T1-3.tim", dir + "RANDOMRHS/IC-T1-3.sto", "IC-T1-3", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T2-1.cor", dir + "RANDOMRHS/IC-T2-1.tim", dir + "RANDOMRHS/IC-T2-1.sto", "IC-T2-1", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T2-2.cor", dir + "RANDOMRHS/IC-T2-2.tim", dir + "RANDOMRHS/IC-T2-2.sto", "IC-T2-2", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T2-3.cor", dir + "RANDOMRHS/IC-T2-3.tim", dir + "RANDOMRHS/IC-T2-3.sto", "IC-T2-3", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T3-1.cor", dir + "RANDOMRHS/IC-T3-1.tim", dir + "RANDOMRHS/IC-T3-1.sto", "IC-T3-1", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T3-2.cor", dir + "RANDOMRHS/IC-T3-2.tim", dir + "RANDOMRHS/IC-T3-2.sto", "IC-T3-2", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T3-3.cor", dir + "RANDOMRHS/IC-T3-3.tim", dir + "RANDOMRHS/IC-T3-3.sto", "IC-T3-3", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T4-1.cor", dir + "RANDOMRHS/IC-T4-1.tim", dir + "RANDOMRHS/IC-T4-1.sto", "IC-T4-1", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T4-2.cor", dir + "RANDOMRHS/IC-T4-2.tim", dir + "RANDOMRHS/IC-T4-2.sto", "IC-T4-2", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T4-3.cor", dir + "RANDOMRHS/IC-T4-3.tim", dir + "RANDOMRHS/IC-T4-3.sto", "IC-T4-3", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T5-1.cor", dir + "RANDOMRHS/IC-T5-1.tim", dir + "RANDOMRHS/IC-T5-1.sto", "IC-T5-1", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T5-2.cor", dir + "RANDOMRHS/IC-T5-2.tim", dir + "RANDOMRHS/IC-T5-2.sto", "IC-T5-2", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T5-3.cor", dir + "RANDOMRHS/IC-T5-3.tim", dir + "RANDOMRHS/IC-T5-3.sto", "IC-T5-3", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T6-1.cor", dir + "RANDOMRHS/IC-T6-1.tim", dir + "RANDOMRHS/IC-T6-1.sto", "IC-T6-1", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T6-2.cor", dir + "RANDOMRHS/IC-T6-2.tim", dir + "RANDOMRHS/IC-T6-2.sto", "IC-T6-2", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T6-3.cor", dir + "RANDOMRHS/IC-T6-3.tim", dir + "RANDOMRHS/IC-T6-3.sto", "IC-T6-3", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T7-1.cor", dir + "RANDOMRHS/IC-T7-1.tim", dir + "RANDOMRHS/IC-T7-1.sto", "IC-T7-1", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T7-2.cor", dir + "RANDOMRHS/IC-T7-2.tim", dir + "RANDOMRHS/IC-T7-2.sto", "IC-T7-2", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T7-3.cor", dir + "RANDOMRHS/IC-T7-3.tim", dir + "RANDOMRHS/IC-T7-3.sto", "IC-T7-3", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T8-1.cor", dir + "RANDOMRHS/IC-T8-1.tim", dir + "RANDOMRHS/IC-T8-1.sto", "IC-T8-1", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T8-2.cor", dir + "RANDOMRHS/IC-T8-2.tim", dir + "RANDOMRHS/IC-T8-2.sto", "IC-T8-2", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T8-3.cor", dir + "RANDOMRHS/IC-T8-3.tim", dir + "RANDOMRHS/IC-T8-3.sto", "IC-T8-3", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T9-1.cor", dir + "RANDOMRHS/IC-T9-1.tim", dir + "RANDOMRHS/IC-T9-1.sto", "IC-T9-1", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T9-2.cor", dir + "RANDOMRHS/IC-T9-2.tim", dir + "RANDOMRHS/IC-T9-2.sto", "IC-T9-2", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T9-3.cor", dir + "RANDOMRHS/IC-T9-3.tim", dir + "RANDOMRHS/IC-T9-3.sto", "IC-T9-3", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T10-1.cor", dir + "RANDOMRHS/IC-T10-1.tim", dir + "RANDOMRHS/IC-T10-1.sto", "IC-T10-1", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T10-2.cor", dir + "RANDOMRHS/IC-T10-2.tim", dir + "RANDOMRHS/IC-T10-2.sto", "IC-T10-2", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T10-3.cor", dir + "RANDOMRHS/IC-T10-3.tim", dir + "RANDOMRHS/IC-T10-3.sto", "IC-T10-3", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T11-1.cor", dir + "RANDOMRHS/IC-T11-1.tim", dir + "RANDOMRHS/IC-T11-1.sto", "IC-T11-1", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T11-2.cor", dir + "RANDOMRHS/IC-T11-2.tim", dir + "RANDOMRHS/IC-T11-2.sto", "IC-T11-2", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T11-3.cor", dir + "RANDOMRHS/IC-T11-3.tim", dir + "RANDOMRHS/IC-T11-3.sto", "IC-T11-3", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T12-1.cor", dir + "RANDOMRHS/IC-T12-1.tim", dir + "RANDOMRHS/IC-T12-1.sto", "IC-T12-1", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T12-2.cor", dir + "RANDOMRHS/IC-T12-2.tim", dir + "RANDOMRHS/IC-T12-2.sto", "IC-T12-2", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T12-3.cor", dir + "RANDOMRHS/IC-T12-3.tim", dir + "RANDOMRHS/IC-T12-3.sto", "IC-T12-3", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T13-1.cor", dir + "RANDOMRHS/IC-T13-1.tim", dir + "RANDOMRHS/IC-T13-1.sto", "IC-T13-1", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T13-2.cor", dir + "RANDOMRHS/IC-T13-2.tim", dir + "RANDOMRHS/IC-T13-2.sto", "IC-T13-2", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T13-3.cor", dir + "RANDOMRHS/IC-T13-3.tim", dir + "RANDOMRHS/IC-T13-3.sto", "IC-T13-3", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T14-1.cor", dir + "RANDOMRHS/IC-T14-1.tim", dir + "RANDOMRHS/IC-T14-1.sto", "IC-T14-1", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T14-2.cor", dir + "RANDOMRHS/IC-T14-2.tim", dir + "RANDOMRHS/IC-T14-2.sto", "IC-T14-2", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T14-3.cor", dir + "RANDOMRHS/IC-T14-3.tim", dir + "RANDOMRHS/IC-T14-3.sto", "IC-T14-3", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T15-1.cor", dir + "RANDOMRHS/IC-T15-1.tim", dir + "RANDOMRHS/IC-T15-1.sto", "IC-T15-1", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T15-2.cor", dir + "RANDOMRHS/IC-T15-2.tim", dir + "RANDOMRHS/IC-T15-2.sto", "IC-T15-2", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T15-3.cor", dir + "RANDOMRHS/IC-T15-3.tim", dir + "RANDOMRHS/IC-T15-3.sto", "IC-T15-3", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T16-1.cor", dir + "RANDOMRHS/IC-T16-1.tim", dir + "RANDOMRHS/IC-T16-1.sto", "IC-T16-1", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T16-2.cor", dir + "RANDOMRHS/IC-T16-2.tim", dir + "RANDOMRHS/IC-T16-2.sto", "IC-T16-2", -1e5, false, kHighsInf},
  // {dir + "RANDOMRHS/IC-T16-3.cor", dir + "RANDOMRHS/IC-T16-3.tim", dir + "RANDOMRHS/IC-T16-3.sto", "IC-T16-3", -1e5, false, kHighsInf},
};
// //XD

TEST_CASE("test-read2", "[highs-benders]") {
  for (auto r : training) {
   StandardMasterProblem master_solver {};
   multi_run(r, 1e5, master_solver);
  }
  // for (auto r : training) {
  //  // LevelSetIpmMasterProblem master_solver {0.5, 0.5};
  //    LevelSetIpmMasterProblem master_solver {0.49, 0.51};
  //    multi_run(r, 1e5, master_solver);
  // }
  // for (auto r : training) {
  //   LevelSetQpMasterProblem master_solver {0.5, 0.9};
  //   multi_run(r, 1e5, master_solver);
  // }
  // for (auto D : {1.}) for (auto r : training) {
  //  PrimalDualMasterProblem master_solver {D};
  //  multi_run(r, 1e5, master_solver);
  // }
}


// TEST_CASE("test-read2", "[highs-benders]") {
//   std::string algorithm, testset;
//   std::ifstream ("/tmp/algo.txt") >> algorithm;
//   std::ifstream ("/tmp/testset.txt") >> testset;
//   SmpsTestCase test;

//   if (testset == "env-diss-xlrge") test = biggest_ones[0];
//   if (testset == "env-xlrge") test = biggest_ones[1];
//   if (testset == "block-6") test = biggest_ones[2];
//   if (testset == "4node-32768") test = biggest_ones[3];
// //   if (testset == "4node-base-32768") test = biggest_ones[4];

// //   if (algorithm == "no_decomp") {
// //     StandardMasterProblem master_solver {}; 
// //     multi_run(test, 1e5, master_solver, "/home/s2860325/Desktop/results_no_decomp.csv");
// //   if (testset == "4node-base-32768") test = biggest_ones[4];


//     multi_run(test, 1e5, master_solver, "/home/s2860325/Desktop/results_no_decomp.csv");
  // }
//   if (algorithm == "proximal") {
//     ProximalIPMMasterProblem master_solver {5, 16};
//     multi_run(test, 1e5, master_solver, "/home/s2860325/Desktop/results_proximal.csv");
//   }
//   if (algorithm == "ipm_level_set") {
//     LevelSetIpmMasterProblem master_solver {0.49, 0.51};
//     multi_run(test, 1e5, master_solver, "/home/s2860325/Desktop/results_ipm_level_set.csv");
//   }
//   if (algorithm == "qp_level_set") {
//     LevelSetQpMasterProblem master_solver {0.5, 0.9};
//     multi_run(test, 1e5, master_solver, "/home/s2860325/Desktop/results_qp_level_set.csv");
//   }
//   if (algorithm == "small") {
//     for (auto test : training) {
//       ProximalIPMMasterProblem master_solver {5, 2};
//       multi_run(test, 1e5, master_solver, "/home/s2860325/Desktop/results_proximal.csv");
//     }
//     for (auto test : training) {
//       LevelSetIpmMasterProblem master_solver {0.49, 0.51};
//       multi_run(test, 1e5, master_solver, "/home/s2860325/Desktop/results_ipm_level_set.csv");
//     }
//     for (auto test : training) {
//       LevelSetQpMasterProblem master_solver {0.5, 0.9};
//       multi_run(test, 1e5, master_solver, "/home/s2860325/Desktop/results_qp_level_set.csv");
//     }
//     for (auto test : big_ones) {
//       ProximalIPMMasterProblem master_solver {5, 4};
//       multi_run(test, 1e5, master_solver, "/home/s2860325/Desktop/results_proximal.csv");
//     }
//     for (auto test : big_ones) {
//       LevelSetIpmMasterProblem master_solver {0.49, 0.51};
//       multi_run(test, 1e5, master_solver, "/home/s2860325/Desktop/results_ipm_level_set.csv");
//     }
//     for (auto test : big_ones) {
//       LevelSetQpMasterProblem master_solver {0.5, 0.9};
//       multi_run(test, 1e5, master_solver, "/home/s2860325/Desktop/results_qp_level_set.csv");
//     }
    
//   }
// }
