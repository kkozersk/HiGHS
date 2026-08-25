#ifndef HIPO_OPTION_H
#define HIPO_OPTION_H

#include "Parameters.h"
#include "io/HighsIO.h"
#include "lp_data/HighsOptions.h"

namespace hipo {

struct Options {
  // Solver options
  std::string nla = kHighsChooseString;
  std::string crossover = kHighsOffString;
  std::string parallel = kHighsChooseString;
  std::string parallel_type = kHipoBothString;
  std::string ordering = kHighsChooseString;

  // Ipm parameters
  Int max_iter = kMaxIterDefault;
  double feasibility_tol = kIpmTolDefault;
  double optimality_tol = kIpmTolDefault;
  double crossover_tol = kIpmTolDefault;
  bool refine_with_ipx = true;
  double time_limit = -1.0;
  Int block_size = 0;
  double frozen_mu = 0.5;
  Int max_recentring_iter = 100;
  double centring_gamma = 0.1;
  double recentring_step = 0.3;

  // Logging
  bool display = true;
  bool timeless_log = false;
  const HighsLogOptions* log_options = nullptr;
};

}  // namespace hipo

#endif
