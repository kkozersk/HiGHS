#include "HCheckConfig.h"
#include "Highs.h"
#include "SpecialLps.h"
#include "catch.hpp"

const bool dev_run = true;
const double double_equal_tolerance = 1e-5;

bool objectiveOk(const double optimal_objective,
                 const double require_optimal_objective,
                 const bool dev_run = false);

void solve(Highs& highs, std::string presolve,
           const HighsModelStatus require_model_status,
           const double require_optimal_objective = 0,
           const double require_iteration_count = -1);

bool objectiveOk(const double optimal_objective,
                 const double require_optimal_objective, const bool dev_run) {
  double error = std::fabs(optimal_objective - require_optimal_objective) /
                 std::max(1.0, std::fabs(require_optimal_objective));
  bool error_ok = error < 1e-10;
  if (!error_ok && dev_run)
    printf("Objective is %g but require %g (error %g)\n", optimal_objective,
           require_optimal_objective, error);
  return error_ok;
}

void solve(Highs& highs, std::string presolve,
           const HighsModelStatus require_model_status,
           const double require_optimal_objective,
           const double require_iteration_count) {
  if (!dev_run) highs.setOptionValue("output_flag", false);
//   highs.resetGlobalScheduler(true);
  const HighsInfo& info = highs.getInfo();
  REQUIRE(highs.setOptionValue("presolve", presolve) == HighsStatus::kOk);

  REQUIRE(highs.setBasis() == HighsStatus::kOk);

  REQUIRE(highs.run() == HighsStatus::kOk);

  REQUIRE(highs.getModelStatus() == require_model_status);

  if (require_model_status == HighsModelStatus::kOptimal) {
    REQUIRE(objectiveOk(info.objective_function_value,
                        require_optimal_objective, dev_run));
  }
  REQUIRE(highs.resetOptions() == HighsStatus::kOk);

  highs.resetGlobalScheduler(true);
}

TEST_CASE("issue-2173", "[highs_test_mip_solver]") {
  std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/issue-2173.mps";
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("mip_rel_gap", 0);
  highs.setOptionValue("mip_abs_gap", 0);
  highs.readModel(filename);
  const HighsModelStatus require_model_status = HighsModelStatus::kOptimal;
  const double optimal_objective = -26770.8075489;
  solve(highs, kHighsOnString, require_model_status, optimal_objective);
}

TEST_CASE("parallel-mip-determinism", "[highs_test_mip_solver]") {
  std::string filename = std::string(HIGHS_DIR) + "/check/instances/bell5.mps";
  HighsInt num_runs = 6;
  std::vector<HighsInt> lp_iters(num_runs);
  for (HighsInt i = 0; i < num_runs; i++) {
    Highs highs;
    highs.setOptionValue("output_flag", dev_run);
    highs.setOptionValue("mip_rel_gap", 0);
    highs.setOptionValue("threads", 2);
    highs.setOptionValue("parallel", kHighsOnString);
    if (i % 2 == 0) highs.setOptionValue("mip_search_simulate_concurrency", 1);
    highs.readModel(filename);
    const HighsModelStatus require_model_status = HighsModelStatus::kOptimal;
    const double optimal_objective = 8966406.491519;
    solve(highs, kHighsOffString, require_model_status, optimal_objective);
    lp_iters[i] = highs.getInfo().simplex_iteration_count;
    if (i > 0) {
      REQUIRE(lp_iters[i] == lp_iters[0]);
    }
  }
}
