#include <cmath>
#include <iostream>
#include <vector>

#include "HCheckConfig.h"
#include "Highs.h"
#include "catch.hpp"
#include "io/Filereader.h"
#include "ipm/hipo/ipm/Model.h"
#include "ipm/hipo/ipm/Solver.h"
#include "ipm/hipo/ipm/Status.h"
#include "lp_data/HighsCallback.h"
#include "parallel/HighsParallel.h"
#include "simplex/SimplexConst.h"

const bool dev_run = false;

void runHipoTest(
    Highs& highs, const std::string& model, const double expected_obj,
    const HighsModelStatus& expected_model_status = HighsModelStatus::kOptimal,
    const std::string& presolve = kHighsOnString) {
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("solver", kHipoString);
  highs.setOptionValue("timeless_log", kHighsOnString);
  highs.setOptionValue("presolve", presolve);

  std::string filename = std::string(HIGHS_DIR) + "/check/instances/" + model;
  highs.readModel(filename);

  HighsStatus status = highs.run();
  REQUIRE(status == HighsStatus::kOk);
  REQUIRE(highs.getModelStatus() == expected_model_status);

  if (expected_model_status == HighsModelStatus::kOptimal) {
    const double actual_obj = highs.getObjectiveValue();
    REQUIRE(std::abs(actual_obj - expected_obj) / std::abs(expected_obj) <
            1e-4);
  }
}

TEST_CASE("test-hipo-afiro", "[highs_hipo]") {
  Highs highs;
  runHipoTest(highs, "afiro.mps", -464.753);
  highs.resetGlobalScheduler(true);
}

TEST_CASE("test-hipo-deterministic", "[highs_hipo]") {
  // Test that hipo finds the exact same solution if run twice

  std::string model = "80bau3b.mps";
  const double expected_obj = 9.8722e5;

  HighsInt iter_1, iter_2;
  HighsSolution solution_1, solution_2;

  Highs highs;
  highs.setOptionValue(kRunCrossoverString, kHighsOffString);

  runHipoTest(highs, model, expected_obj);
  solution_1 = highs.getSolution();
  iter_1 = highs.getInfo().ipm_iteration_count;

  runHipoTest(highs, model, expected_obj);
  solution_2 = highs.getSolution();
  iter_2 = highs.getInfo().ipm_iteration_count;

  REQUIRE(iter_1 == iter_2);
  REQUIRE(solution_1.value_valid == solution_2.value_valid);
  REQUIRE(solution_1.dual_valid == solution_2.dual_valid);
  REQUIRE(solution_1.col_value == solution_2.col_value);
  REQUIRE(solution_1.row_value == solution_2.row_value);
  REQUIRE(solution_1.col_dual == solution_2.col_dual);
  REQUIRE(solution_1.row_dual == solution_2.row_dual);
}

HighsLp get_dummy_lp_with_lb_ub(std::vector<double> const & lb, std::vector<double> const & ub) {
  REQUIRE(lb.size() == ub.size());
  HighsLp lp;
  lp.num_col_ = lb.size();
  lp.num_row_ = 1;
  lp.col_lower_ = lb;
  lp.col_upper_ = ub;
  lp.col_cost_ = std::vector<double>(lb.size(), 0);
  lp.row_lower_ = {1};
  lp.row_upper_ = {1};
  lp.a_matrix_.num_col_ = lb.size();
  lp.a_matrix_.num_row_ = 1;
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.index_ = {0};
  lp.a_matrix_.value_ = {1};
  lp.a_matrix_.start_ = {0,1};
  lp.ensureColwise();
  return lp;
}

TEST_CASE("test-is-centred-solutions", "[highs_hipo]") {
  hipo::Model model;
  auto lp = get_dummy_lp_with_lb_ub({0, 0, -kHighsInf, -kHighsInf}, {2, kHighsInf, 3.5, kHighsInf});
  REQUIRE(model.init(lp, {}) == 0);
  std::vector<double>
      xl {1, 1.5, 0, 0},
      zl {1, 1, 0, 0},
      xu {1, 0, 1, 0},
      zu {1, 0, 1, 0};
  double mu = 1.125;
  double gamma = 0.01;
  REQUIRE(hipo::isWellCentered(mu, gamma, model, xl, zl, xu, zu));

  gamma = 0.8;
  REQUIRE(!hipo::isWellCentered(mu, gamma, model, xl, zl, xu, zu));

  mu = 49./40.;
  REQUIRE(hipo::isWellCentered(mu, gamma, model, xl, zl, xu, zu));
}

TEST_CASE("test-eps-centring", "[highs_hipo]") {
  hipo::Model model;
  std::vector<char> dummy_cons {'<'};
  auto lp = get_dummy_lp_with_lb_ub({0, 0, 0, 0}, {INFINITY, INFINITY, INFINITY, INFINITY});
  REQUIRE(model.init(lp, {}) == 0);
  auto eps = 1e-8;
  std::vector<double>
      xl (4,eps),
      zl (4,eps),
      xu (4,0),
      zu (4,0);
  double gamma = 1;
  double mu = eps * eps;
  REQUIRE(hipo::isWellCentered(mu, gamma, model, xl, zl, xu, zu));
}

// TODO centredness check might fail for free vars, because of the initial bounds
TEST_CASE("test-no-bound-centring", "[highs_hipo]") {
  hipo::Model model;
  auto lp = get_dummy_lp_with_lb_ub({-INFINITY, -INFINITY, -INFINITY, -INFINITY}, {INFINITY, INFINITY, INFINITY, INFINITY});
  REQUIRE(model.init(lp, {}) == 0);
  std::vector<double>
      xl (4,0),
      zl (4,0),
      xu (4,0),
      zu (4,0);
  double gamma = 0.1;
  double mu = 0;
  REQUIRE(hipo::isWellCentered(mu, gamma, model, xl, zl, xu, zu));
}

TEST_CASE("test-is-centred-fixed-vars", "[highs_hipo]") {
  hipo::Model model;
  auto lp = get_dummy_lp_with_lb_ub({0, 0, -INFINITY, 3}, {2, INFINITY, 3.5, 3});
  REQUIRE(model.init(lp, {}) == 0);
  std::vector<double>
      xl {1, 1.5, 0, 0},
      zl {1, 1, 0, 0},
      xu {1, 0, 1, 0},
      zu {1, 0, 1, 0};
  double mu = 1.125;
  double gamma = 0.01;
  REQUIRE(hipo::isWellCentered(mu, gamma, model, xl, zl, xu, zu));

  gamma = 0.8;
  REQUIRE(!hipo::isWellCentered(mu, gamma, model, xl, zl, xu, zu));

  mu = 49./40.;
  REQUIRE(hipo::isWellCentered(mu, gamma, model, xl, zl, xu, zu));
}

bool inexact_vector_comparison(std::vector<double> const & a, std::vector<double> const & b, double eps=1e-3) {
  if (a.size() != b.size())
    return false;
  for (int i = 0; i < a.size(); ++i)
    if (abs(a[i] - b[i]) > eps)
      return false;
  return true;
}

TEST_CASE("test-centring-procedure", "[highs_hipo]") {
  std::vector<HighsInt> csr_index {0, 1};
  std::vector<double> csr_values {1, 1};
  std::vector<HighsInt> csr_starts {0, 2};
  HighsLp lp;
  lp.offset_ = 0;
  lp.num_col_ = 2;
  lp.num_row_ = 1;
  lp.col_lower_ = {0, 0};
  lp.col_upper_ = {INFINITY, INFINITY};
  lp.col_cost_ = {1, 1};
  lp.row_lower_ = {-INFINITY};
  lp.row_upper_ = {1};
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.start_ = csr_starts;
  lp.a_matrix_.index_ = csr_index;
  lp.a_matrix_.value_ = csr_values;
  lp.a_matrix_.num_row_ = 1;
  lp.a_matrix_.num_col_ = 2;
  
  Highs highs;
  highs.passModel(lp);
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("solver", kHipoString);
  highs.setOptionValue("max_centring_steps_hipo", 100);
  highs.setOptionValue("timeless_log", kHighsOnString);
  highs.setOptionValue("ipm_iteration_limit", 0);
  highs.setOptionValue("presolve", kHighsOffString);
  highs.setOptionValue("run_crossover", kHighsOffString);
  highs.setOptionValue("refine_with_ipx", false);
  highs.setOptionValue("fixed_mu", 0.5);
  highs.setOptionValue("centring_gamma", 1e-8);
  highs.run();
  auto solution = highs.getSolution();
  REQUIRE(inexact_vector_comparison(solution.col_value, {0.25, 0.25}));
  REQUIRE(inexact_vector_comparison(solution.col_dual, {2, 2}));
  REQUIRE(inexact_vector_comparison(solution.row_dual, {-1}));
}

TEST_CASE("test-centring-procedure-with-slack", "[highs_hipo]") {
  std::vector<HighsInt> csr_index {0, 0, 0};
  std::vector<double> csr_values {1, 1, 1};
  std::vector<HighsInt> csr_starts {0, 1, 2, 3};
  HighsLp lp;
  lp.offset_ = 0;
  lp.num_col_ = 3;
  lp.num_row_ = 1;
  lp.col_lower_ = {0, 0, 0};
  lp.col_upper_ = {INFINITY, INFINITY, INFINITY};
  lp.col_cost_ = {1, 1, 0};
  lp.row_lower_ = {1};
  lp.row_upper_ = {1};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = csr_starts;
  lp.a_matrix_.index_ = csr_index;
  lp.a_matrix_.value_ = csr_values;
  lp.a_matrix_.num_row_ = 1;
  lp.a_matrix_.num_col_ = 3;
  
  Highs highs;
  highs.passModel(lp);
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("solver", kHipoString);
  highs.setOptionValue("max_centring_steps_hipo", 100);
  highs.setOptionValue("timeless_log", kHighsOnString);
  highs.setOptionValue("ipm_iteration_limit", 0);
  highs.setOptionValue("presolve", kHighsOffString);
  highs.setOptionValue("run_crossover", kHighsOffString);
  highs.setOptionValue("refine_with_ipx", false);
  highs.setOptionValue("fixed_mu", 0.5);
  highs.setOptionValue("centring_gamma", 1e-8);

  highs.run();
  // REQUIRE( == HighsStatus::kOk);
  auto solution = highs.getSolution();
  REQUIRE(inexact_vector_comparison(solution.col_value, {0.25, 0.25, 0.5}));
  REQUIRE(inexact_vector_comparison(solution.col_dual, {2, 2, 1}));
  REQUIRE(inexact_vector_comparison(solution.row_dual, {-1}));
}

TEST_CASE("test-centring-procedure-fixed-var", "[highs_hipo]") {
  std::vector<HighsInt> csr_index {0, 0};
  std::vector<double> csr_values {1, 1};
  std::vector<HighsInt> csr_starts {0, 1, 2, 2};
  HighsLp lp;
  lp.offset_ = 0;
  lp.num_col_ = 3;
  lp.num_row_ = 1;
  lp.col_lower_ = {0, 0, 3};
  lp.col_upper_ = {INFINITY, INFINITY, 3};
  lp.col_cost_ = {1, 1, 0};
  lp.row_lower_ = {-INFINITY};
  lp.row_upper_ = {1};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = csr_starts;
  lp.a_matrix_.index_ = csr_index;
  lp.a_matrix_.value_ = csr_values;
  lp.a_matrix_.num_row_ = 1;
  lp.a_matrix_.num_col_ = 3;
  
  Highs highs;
  highs.passModel(lp);
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("solver", kHipoString);
  highs.setOptionValue("max_centring_steps_hipo", 100);
  highs.setOptionValue("timeless_log", kHighsOnString);
  highs.setOptionValue("ipm_iteration_limit", 0);
  highs.setOptionValue("presolve", kHighsOffString);
  highs.setOptionValue("run_crossover", kHighsOffString);
  highs.setOptionValue("refine_with_ipx", false);
  highs.setOptionValue("fixed_mu", 0.5);
  highs.setOptionValue("centring_gamma", 1e-8);
  highs.run();
  // REQUIRE( == HighsStatus::kOk);
  auto solution = highs.getSolution();
  // REQUIRE(solution.col_dual == std::vector<double> {2, 2, 0});
  REQUIRE(inexact_vector_comparison(solution.col_value, {0.25, 0.25, 3}));
  REQUIRE(inexact_vector_comparison(solution.col_dual, {2, 2, 0}));
  REQUIRE(inexact_vector_comparison(solution.row_dual, {-1}));
}

TEST_CASE("test-recentring-afiro", "[highs_hipo]") {
  std::string model = "afiro.mps";
  const double expected_obj = -464.753;

  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("solver", kHipoString);
  highs.setOptionValue("timeless_log", kHighsOnString);
  highs.setOptionValue("ipm_iteration_limit", 5);
  highs.setOptionValue("max_centring_steps_hipo", 10);
  highs.setOptionValue("presolve", kHighsOffString);
  double gamma = 0.5;
  highs.setOptionValue("centring_gamma", gamma);

  std::string filename = std::string(HIGHS_DIR) + "/check/instances/" + model;
  highs.readModel(filename);

  highs.run();

  highs.resetGlobalScheduler(true);
  auto solution = highs.getSolution();
  auto x = solution.col_value;
  auto s = solution.col_dual;
  double mu = std::inner_product(x.begin(), x.end(), s.begin(), 0.) / x.size();
  for(int i = 0; i < x.size(); ++i) {
    REQUIRE(gamma * mu <= x[i] * s[i]);
    REQUIRE(x[i] * s[i] <= mu / gamma);
  }
  
}

TEST_CASE("test-hipo-options", "[highs_hipo]") {
  // test all combinations of options for hipo

  std::string model = "adlittle.mps";
  const double expected_obj = 2.2549e5;
  Highs highs;

  std::vector<std::string> orders = {kHighsChooseString, kHipoMetisString,
                                     kHipoAmdString, kHipoRcmString};
  std::vector<std::string> systems = {kHighsChooseString, kHipoNormalEqString,
                                      kHipoAugmentedString};
  std::vector<std::string> parallels = {kHighsOnString, kHighsOffString,
                                        kHighsChooseString};
  std::vector<std::string> partypes = {kHipoTreeString, kHipoNodeString,
                                       kHipoBothString};

  for (auto& order : orders) {
    highs.setOptionValue(kHipoOrderingString, order);
    for (auto& system : systems) {
      highs.setOptionValue(kHipoSystemString, system);
      for (auto& parallel : parallels) {
        highs.setOptionValue(kParallelString, parallel);
        for (auto& partype : partypes) {
          highs.setOptionValue(kHipoParallelString, partype);
          runHipoTest(highs, model, expected_obj);
        }
      }
    }
  }

  highs.resetGlobalScheduler(true);
}

TEST_CASE("test-hipo-qp", "[highs_hipo]") {
  Highs highs;
  runHipoTest(highs, "qptestnw.lp", -6.4500);
  runHipoTest(highs, "qjh.lp", -5.2500);
  runHipoTest(highs, "primal1.mps", -3.501296e-2);
  highs.resetGlobalScheduler(true);
}

TEST_CASE("test-hipo-infeas", "[highs_hipo]") {
  const HighsModelStatus expected_status = HighsModelStatus::kInfeasible;
  Highs highs;
  runHipoTest(highs, "bgetam.mps", 0, expected_status, "off");
  runHipoTest(highs, "forest6.mps", 0, expected_status, "off");
  runHipoTest(highs, "klein1.mps", 0, expected_status, "off");
  highs.resetGlobalScheduler(true);
}

TEST_CASE("test-hipo-freevar", "[highs_hipo]") {
  const HighsModelStatus expected_status = HighsModelStatus::kOptimal;
  Highs highs;
  runHipoTest(highs, "perold.mps", -9.381e3, expected_status, "on");
  runHipoTest(highs, "perold.mps", -9.381e3, expected_status, "off");
  highs.resetGlobalScheduler(true);
}
