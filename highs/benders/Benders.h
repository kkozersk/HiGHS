#pragma once
#include <fstream>
#include <ios>
#include <set>
#include <regex>
#include <string>
#include <iostream>
#include "util/HighsInt.h"
#include "Highs.h"
#include "lp_data/HighsLp.h"
#include "io/SMPS.h"
#include "util/HighsSparseMatrix.h"

struct RowDivision {
  std::set<HighsInt> inset_only_rows;
  std::set<HighsInt> other_rows;
  std::set<HighsInt> mixed_rows;
  std::set<HighsInt> diff_other;
};

struct BendersProblems {
  Highs master;
  Highs subproblem;
  Highs feas_subproblem;
};

struct MultiBendersProblems {
  Highs master;
  std::vector<Highs> subproblems;
  std::vector<Highs> feas_subproblems;
};

struct NonZeroVector {
  HighsInt number_of_nonzeros;
  std::vector<HighsInt> nonzero_indices;
  std::vector<double> nonzero_values;
  bool operator==(NonZeroVector const & other) const {
    return number_of_nonzeros == other.number_of_nonzeros &&
      nonzero_indices == other.nonzero_indices &&
      nonzero_values == other.nonzero_values;
  }
};

struct BendersIterationInfo {
  bool was_subproblem_feasible=true;
  bool was_error=false;
};

enum CutType { Objective, Feasibility };

struct CutData {
  std::vector<double> master_multipliers;
  double dual_objective;
};
double ipm_acc = 1e-2;
double ipm_feas = 1e-7;
std::string used_solver = kHipoString;
struct  BendersRet {
  double result;
  int iter;
  std::vector<int> UBD_iters;
  std::vector<int> feas_iters;
  std::vector<int> recentring_steps;
  std::vector<int> optim_steps;
  std::vector<double> pinf, dinf;
  BendersRet(double res=kHighsInf, int iter=-1, std::vector<int> UBD_iters= std::vector<int>{}, std::vector<int> feas_iters= std::vector<int>{},
     std::vector<int> recentring_steps = {}, std::vector<int> optim_steps = {},
    std::vector<double> pinf={}, std::vector<double> dinf = {})
    : result(res), iter(iter), UBD_iters(UBD_iters), feas_iters(feas_iters), recentring_steps(recentring_steps), optim_steps(optim_steps), pinf(pinf), dinf(dinf) {}
  void note(double expected, std::string problem) {
    std::ofstream of("/tmp/results.csv", std::ios_base::app);
    auto last_imp = UBD_iters.empty() ? -1 : UBD_iters.back();
    
    double rec_sum = std::accumulate(recentring_steps.begin(), recentring_steps.end(), 0.);
    double rec_mean = recentring_steps.empty() ? 0 : rec_sum / recentring_steps.size();
    int max_rec = recentring_steps.empty() ? 0 : *std::max_element(recentring_steps.begin(), recentring_steps.end());
    
    double opt_sum = std::accumulate(optim_steps.begin(), optim_steps.end(), 0.);
    double opt_mean = optim_steps.empty() ? 0 : opt_sum / optim_steps.size();
    int max_opt = optim_steps.empty() ? 0 : *std::max_element(optim_steps.begin(), optim_steps.end());

    double pinf_sum = std::accumulate(pinf.begin(), pinf.end(), 0.);
    double pinf_mean = pinf.empty() ? 0 : pinf_sum / pinf.size();

    double dinf_sum = std::accumulate(dinf.begin(), dinf.end(), 0.);
    double dinf_mean = dinf.empty() ? 0 : dinf_sum / dinf.size();
    
    of << problem << "," << ipm_acc << "," << result - expected << ","  << iter << "," << last_imp << "," << feas_iters.size() 
      << "," << rec_mean << "," << max_rec << "," << opt_mean << "," << max_opt << "," << pinf_mean << "," << dinf_mean << std::endl;
  }
  bool operator==(BendersRet const & ret) const {
    std::cout << result << " " << ret.result << " " << std::abs(result - ret.result) << " "  << iter << " " << ret.iter << std::endl;
    return std::abs(result - ret.result) < 1e-3 && (iter == ret.iter || ret.iter == -1);
     }
  std::string operator()() const { return std::to_string(result) + " " + std::to_string(iter);}
};

class CsvLogger : public std::ofstream {
  public:
  CsvLogger(std::string const & filename) : std::ofstream(filename, std::ios::app) {};
  virtual ~CsvLogger()=default;
  CsvLogger & operator<<(double val) { std::ofstream::operator<<(val) << ","; return *this;}
  CsvLogger & operator<<(int val) { std::ofstream::operator<<(val) << ","; return *this;}
  CsvLogger & operator<<(bool val) { std::ofstream::operator<<(val) << ","; return *this;}
  CsvLogger & operator<<(std::string const & val) { std::operator<<(*this, val) << ","; return *this; }
  CsvLogger & newline() { std::endl(*this); return *this; }
  CsvLogger & log_predicate(std::vector<bool> const & preds, std::string const & on_true, std::string const & on_false) {
    for (auto pred : preds) operator<<(pred ? on_true : on_false);
    return newline();
  }
  template <typename T>
  CsvLogger & log_with_note(T const & val, std::string const & note) {
    std::ofstream::operator<<(val) << note << ","; return *this;
  }
  template <typename T>
  CsvLogger & operator<<(std::vector<T> const & vals) {
    for (auto const & val: vals) operator<<(val);
    return newline();
  }
};

HighsInt find_row_index(std::vector<HighsInt> const & csr_starts, HighsInt index);
RowDivision divide_rows(std::vector<HighsInt> const & csr_index, std::vector<HighsInt> const & csr_starts, std::set<HighsInt> const & master_variables); 
RowDivision divide_rows(HighsSparseMatrix & constraint_matrix, std::set<HighsInt> const & master_variables);
void fix_variable(Highs & problem, HighsInt variable_index, double value);
// void unfreeze_mu(Highs & master, std::set<HighsInt> const & master_variables, HighsInt subproblem_no=0);
bool fix_master_variables(Highs & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values);
std::set<HighsInt> sequence_complement(std::set<HighsInt> const & set, HighsInt max_number);
void create_master_problem(Highs & master, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & subproblem_rows, double subproblem_lb, int no_subproblems=1);
void create_subproblem(Highs & subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & master_only_rows);
void create_feasibility_subproblem(Highs & feas_subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & mixed_rows,  std::set<HighsInt> const & master_only_rows);
void decompose_problem(BendersProblems & problems, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, RowDivision const & row_division, double subproblem_lb);
void decompose_problem(BendersProblems & problems, HighsLp & base_problem, std::set<HighsInt> const & master_variables, double subproblem_lb);
BendersProblems decompose_problem(HighsLp const & problem, std::set<HighsInt> const & master_variables, RowDivision const & row_division, double subproblem_lb); 
BendersProblems decompose_problem(HighsLp & problem, std::set<HighsInt> const & master_variables, double subproblem_lb);
// std::vector<double> get_all_multipliers(Highs const & subproblem);
std::vector<double> get_master_multipliers(Highs const & subproblem, std::set<HighsInt> const & master_variables);
NonZeroVector create_nonzero_vector(std::vector<double> const & base_vector);
NonZeroVector add_mu_entry(NonZeroVector vector, HighsInt mu_index);
void add_nonzero_row(Highs & problem, double lower, double upper, NonZeroVector const & row_vector);
void add_nonzero_col(Highs & problem, double col_cost, double col_lower, double col_upper, NonZeroVector const & col_vector);
// void add_cut(BendersProblems & problems, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values, CutType cut_type);
std::pair<std::vector<double>, double> add_cut(Highs & master, CutData const & cut, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values, CutType cut_type, int subproblem_no=0);
std::pair<std::vector<double>, double> add_cut(Highs & master, Highs const & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values, CutType cut_type, int subproblem_no=0);
// void solve_feasibility_subproblem(Highs & subproblem);
// BendersIterationInfo solve_feasibility_subproblem(Highs & feas_subproblem); 
std::set<HighsInt> discover_master_variables(std::vector<std::string> const & variable_names, std::regex const & master_name_pattern);
std::set<HighsInt> discover_master_variables(std::vector<std::string> const & variable_names, std::string const & master_name_pattern);
BendersIterationInfo solve_feasibility_subproblem(Highs & subproblem, BendersIterationInfo info, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values);
BendersIterationInfo solve_subproblem(Highs & subproblem, BendersIterationInfo info, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values);
BendersIterationInfo solve_master(Highs & master, BendersIterationInfo info);
double calculate_solution_cost(Highs const & master, double subproblem_cost, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values);
double calculate_solution_cost(Highs const & master, Highs const & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values);
BendersRet benders(HighsLp & base_problem, std::set<HighsInt> const & master_variables, std::vector<double> const & starting_point, double subproblem_lb, double eps=1e-3, int max_iter=1e2);
BendersRet benders(HighsLp & base_problem, std::string const & master_name_pattern, std::vector<double> const & starting_point, double subproblem_lb, double eps=1e-3);
// BendersRet multi_benders(HighsLp & base_problem, std::set<HighsInt> const & master_variables, std::vector<std::set<HighsInt>> const & subproblem_variables, std::vector<double> const & starting_point, double subproblem_lb, double eps=1e-3, int max_iter=1e2);  
// BendersRet benders2(HighsLp & base_problem, std::set<HighsInt> const & master_variables, std::vector<double> const & starting_point, double subproblem_lb, double eps=1e-3);
// BendersRet benders2(HighsLp & base_problem, std::string const & master_name_pattern, std::vector<double> const & starting_point, double subproblem_lb, double eps=1e-3);
BendersRet multi_benders_loop(MultiBendersProblems & problems, std::set<HighsInt> const & master_variables,
                     std::vector<double> const & starting_point, double eps, int max_iter);
BendersRet benders_loop(BendersProblems & problems, std::set<HighsInt> const & master_variables, std::vector<double> const & starting_point, double eps, int max_iter);
std::vector<double> get_dual_costs(HighsLp const & lp);
std::vector<double> calculate_negated_reduced_costs(HighsSparseMatrix const & A, std::vector<double> const & dual);
CutData solve_feasibility_subproblem(Highs & subproblem, std::set<HighsInt> const & master_variables);
std::set<HighsInt> index_set_union(std::set<HighsInt> const & a, std::set<HighsInt> const & b);
std::set<HighsInt> index_set_intersection(std::set<HighsInt> const & a, std::set<HighsInt> const & b);
// void decompose_problem(MultiBendersProblems & problems, HighsLp & base_problem, std::set<HighsInt> const & master_variables, std::vector<std::set<HighsInt>> const & subproblems_variables);
// void create_subproblem(Highs & subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & other_rows, std::set<HighsInt> const & subproblem_variables, std::set<HighsInt> const & master_only_variables); 

BendersRet benders_l_shaped(SmpsCoreStructure & core, StochasticTree & tree, 
  std::set<HighsInt> const & master_variables, std::vector<double> const & starting_point, double subproblem_lb, double eps=1e-3, int max_iter=1e2);