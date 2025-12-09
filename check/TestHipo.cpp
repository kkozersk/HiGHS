#include "HCheckConfig.h"
#include "Highs.h"
#include "catch.hpp"
#include "io/Filereader.h"
#include "ipm/hipo/ipm/Model.h"
#include "ipm/hipo/ipm/Solver.h"
#include "ipm/hipo/ipm/Status.h"
#include "lp_data/HighsCallback.h"
#include "parallel/HighsParallel.h"

// Example for using HiPO from its C++ interface. The program solves the Netlib
// problem afiro.

// #include <unistd.h>

#include <cmath>
#include <iostream>
#include <vector>

const bool dev_run = false;

TEST_CASE("test-hipo-afiro", "[highs_hipo]") {
  // Test that hipo runs and finds correct solution for afiro

  std::string model = "afiro.mps";
  const double expected_obj = -464.753;

  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("solver", kHipoString);
  highs.setOptionValue("timeless_log", kHighsOnString);

  std::string filename = std::string(HIGHS_DIR) + "/check/instances/" + model;
  highs.readModel(filename);

  HighsStatus status = highs.run();
  REQUIRE(status == HighsStatus::kOk);

  const double actual_obj = highs.getObjectiveValue();
  REQUIRE(std::abs(actual_obj - expected_obj) < 0.001);

  highs.resetGlobalScheduler(true);
}

TEST_CASE("test-hipo-deterministic", "[highs_hipo]") {
  // Test that hipo finds the exact same solution if run twice

  std::string model = "80bau3b.mps";
  std::string filename = std::string(HIGHS_DIR) + "/check/instances/" + model;

  HighsInt iter_1, iter_2;
  HighsSolution solution_1, solution_2;

  {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue(kSolverString, kHipoString);
    highs.setOptionValue(kParallelString, kHighsOnString);
    highs.setOptionValue(kRunCrossoverString, kHighsOffString);
    highs.readModel(filename);
    HighsStatus status = highs.run();
    REQUIRE(status == HighsStatus::kOk);
    solution_1 = highs.getSolution();
    iter_1 = highs.getInfo().ipm_iteration_count;
    highs.resetGlobalScheduler(true);
  }
  {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue(kSolverString, kHipoString);
    highs.setOptionValue(kParallelString, kHighsOnString);
    highs.setOptionValue(kRunCrossoverString, kHighsOffString);
    highs.readModel(filename);
    HighsStatus status = highs.run();
    REQUIRE(status == HighsStatus::kOk);
    solution_2 = highs.getSolution();
    iter_2 = highs.getInfo().ipm_iteration_count;
    highs.resetGlobalScheduler(true);
  }

  REQUIRE(iter_1 == iter_2);
  REQUIRE(solution_1.value_valid == solution_2.value_valid);
  REQUIRE(solution_1.dual_valid == solution_2.dual_valid);
  REQUIRE(solution_1.col_value == solution_2.col_value);
  REQUIRE(solution_1.row_value == solution_2.row_value);
  REQUIRE(solution_1.col_dual == solution_2.col_dual);
  REQUIRE(solution_1.row_dual == solution_2.row_dual);
}

TEST_CASE("test-is-centred-solutions", "[highs_hipo]") {
  hipo::Model model;
  std::vector<double> zeros(4, 0);
  std::vector<hipo::Int> dummy_matrix (5,0);
  std::vector<double> lb {0, 0, -INFINITY, -INFINITY};
  std::vector<double> ub {2, INFINITY, 3.5, INFINITY};
  std::vector<char> dummy_cons {'<'};
  auto res = model.init(4, 0, zeros.data(), zeros.data(), lb.data(), ub.data(),
     dummy_matrix.data(), dummy_matrix.data(), zeros.data(), dummy_cons.data(), 0);
  REQUIRE(res == 0);
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
  std::vector<double> zeros(4, 0);
  std::vector<hipo::Int> dummy_matrix (5,0);
  std::vector<double> lb {0, 0, 0, 0};
  std::vector<double> ub {INFINITY, INFINITY, INFINITY, INFINITY};
  std::vector<char> dummy_cons {'<'};
  auto res = model.init(4, 0, zeros.data(), zeros.data(), lb.data(), ub.data(),
     dummy_matrix.data(), dummy_matrix.data(), zeros.data(), dummy_cons.data(), 0);
  REQUIRE(res == 0);
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

TEST_CASE("test-no-bound-centring", "[highs_hipo]") {
  hipo::Model model;
  std::vector<double> zeros(4, 0);
  std::vector<hipo::Int> dummy_matrix (5,0);
  std::vector<double> lb {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  std::vector<double> ub {INFINITY, INFINITY, INFINITY, INFINITY};
  std::vector<char> dummy_cons {'<'};
  auto res = model.init(4, 0, zeros.data(), zeros.data(), lb.data(), ub.data(),
     dummy_matrix.data(), dummy_matrix.data(), zeros.data(), dummy_cons.data(), 0);
  REQUIRE(res == 0);
  std::vector<double>
      xl (4,0),
      zl (4,0),
      xu (4,0),
      zu (4,0);
  double gamma = 0.1;
  double mu = 0;
  REQUIRE(hipo::isWellCentered(mu, gamma, model, xl, zl, xu, zu));
}
