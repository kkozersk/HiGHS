// #include <Highs.h>
#include <highs/benders/Benders.h>
#include <iostream>
#include <string>
#include <cstdlib>
const int max_iters = 5000;
const double eps = 1e-6;
const std::string baselogfile = "/tmp/results";
const std::string dir = "/home/s2860325/Desktop/HiGHS_IPM_Benders/";

int main() {
  Highs highs;
  std::string corefile =  getenv("COREFILE");
  std::string stochfile =  getenv("STOCHFILE");
  std::string timefile =  getenv("TIMEFILE");
  std::string note = getenv("NOTE");
  double expected = std::stod(getenv("EXPECTED"));
  double sub_lb = std::stod(getenv("SUBLB"));
  std::string algorithm = getenv("ALGORITHM");
  std::cout << getenv("---------------------") << std::endl;
  std::cout << note << " " << algorithm << std::endl;
  std::cout << getenv("---------------------") << std::endl;
    
  auto core_and_tree = build_stochastic_tree(dir + corefile, dir + timefile, dir + stochfile);
  MasterProblem * master;
  if (algorithm == "standard") { master = new StandardMasterProblem(); }
  else if (algorithm == "qp")  { master = new LevelSetQpMasterProblem(0.5, 0.49); }
  else if (algorithm == "pd")  { master = new PrimalDualMasterProblem(1.); }
  else if (algorithm == "ipm") { master = new LevelSetIpmMasterProblem(0.49, 0.51); }
  else { assert (1==0); }
  auto res = benders_l_shaped(core_and_tree.first, core_and_tree.second, {}, sub_lb, *master, eps, max_iters, BendersAlgorithm::GAP::RELGAP);
  auto logfile = baselogfile + "_" + algorithm + ".csv";
  res.note(expected, note, logfile);

  delete master;
  return 0;
}

