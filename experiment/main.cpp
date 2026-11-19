// #include <Highs.h>
#include <highs/benders/Benders.h>
#include <iostream>
#include <string>
#include <cstdlib>
const int max_iters = 5000;
const std::string dir = "../";

int main() {
  Highs highs;
  double eps = std::stod(getenv("EPS"));
  std::string baselogfile = getenv("BASELOGFILE");
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
  std::string recentring_accuracy = getenv("RECENTRING"); // 1e-5
  std::string gamma = getenv("GAMMA"); //0.5
  std::string omega = getenv("OMEGA"); //0.9
  std::string Tl = getenv("TL"); // 0.49
  std::string Tu = getenv("TU"); // 0.51
  std::string D = getenv("D"); // 0.51
    
  auto core_and_tree = build_stochastic_tree(dir + corefile, dir + timefile, dir + stochfile);
  MasterProblem * master;
  if (algorithm == "standard") { master = new StandardMasterProblem(); }
  else if (algorithm == "qp")  { master = new LevelSetQpMasterProblem(std::stod(gamma), std::stod(omega)); }
  else if (algorithm == "pd")  { master = new PrimalDualMasterProblem(std::stod(D), std::stod(recentring_accuracy)); }
  else if (algorithm == "ipm") { master = new LevelSetIpmMasterProblem(std::stod(Tl), std::stod(Tu), std::stod(recentring_accuracy)); }
  else { assert (1==0); }
  auto res = benders_l_shaped(core_and_tree.first, core_and_tree.second, {}, sub_lb, *master, eps, max_iters, BendersAlgorithm::GAP::RELGAP);
  auto logfile = baselogfile + "_" + algorithm + ".csv";
  res.note(expected, note, logfile);

  delete master;
  return 0;
}

