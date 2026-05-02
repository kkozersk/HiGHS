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
  std::set<HighsInt> master_rows;
  std::set<HighsInt> subproblem_rows;
  std::set<HighsInt> mixed_rows;
  std::set<HighsInt> master_only_rows;
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
  NonZeroVector(HighsInt num_nz, std::vector<HighsInt> const & nz_ind, std::vector<double> const & nz_val)
  : number_of_nonzeros(num_nz), nonzero_indices(nz_ind), nonzero_values(nz_val) {}
  NonZeroVector(std::vector<double> const & vec) {
    for (int i = 0; i < vec.size(); ++i)
      if (vec.at(i) != 0) {
        nonzero_indices.push_back(i);
        nonzero_values.push_back(vec.at(i));
      }
    number_of_nonzeros = nonzero_indices.size();
  }
};

struct BendersIterationInfo {
  bool was_subproblem_feasible=true;
  bool was_error=false;
  double master_time=0;
  double sub_time=0;
};

enum CutType { Objective, Feasibility };

struct CutData {
  std::vector<double> master_multipliers;
  double dual_objective;
};

struct Cut {
  NonZeroVector coefficients;
  double rhs;
};

double ipm_acc = 1e-2;
double ipm_feas = 1e-7;
std::string used_solver = kHipoString;

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
    for (T const & val: vals) std::ofstream::operator<<(val) << ",";
    return newline();
  }
};

struct  BendersRet {
  double result;
  int iter;
  std::vector<int> UBD_iters;
  std::vector<int> feas_iters;
  std::vector<int> recentring_steps;
  std::vector<int> optim_steps;
  std::vector<double> pinf, dinf;
  double gap;
  double master_time;
  double sub_time;
  bool first_optim;
  BendersRet(double gap=0, double res=kHighsInf, BendersIterationInfo info={}, int iter=-1, bool first_optim = false, std::vector<int> UBD_iters= std::vector<int>{}, std::vector<int> feas_iters= std::vector<int>{},
     std::vector<int> recentring_steps = {}, std::vector<int> optim_steps = {},
    std::vector<double> pinf={}, std::vector<double> dinf = {})
    : master_time(info.master_time), sub_time(info.sub_time), result(res), iter(iter), first_optim(first_optim),
     UBD_iters(UBD_iters), feas_iters(feas_iters), recentring_steps(recentring_steps), optim_steps(optim_steps), pinf(pinf), dinf(dinf), gap(gap) {}
  void note(double expected, std::string problem) {
    CsvLogger of("/tmp/results.csv");
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
    of << problem << ipm_acc << result << result - expected  << gap << iter << last_imp << (int) feas_iters.size() 
       << rec_mean << max_rec << opt_mean << max_opt << pinf_mean << dinf_mean << master_time << sub_time << first_optim;
    of.newline();

    auto fp = std::fopen("/tmp/exac_results.csv", "a");
    fprintf(fp, "%s,%.3f\n", problem.c_str(), result);
    fclose(fp);
  }
  bool operator==(BendersRet const & ret) const {
    std::cout << result << " " << ret.result << " " << std::abs(result - ret.result) << " "  << iter << " " << ret.iter << std::endl;
    return std::abs(result - ret.result) < 1e-3 && (iter == ret.iter || ret.iter == -1);
     }
  std::string operator()() const { return std::to_string(result) + " " + std::to_string(iter);}
};

struct OptionValue {
  std::string optName;
  int intVal;
  double doubleVal;
  std::string strVal;
  enum {isInt, isDouble, isStr} type;

  OptionValue(std::string const & optName, int val) : optName(optName), intVal {val}, type{isInt} {}
  OptionValue(std::string const & optName, double val) : optName(optName), doubleVal {val}, type{isDouble} {}
  OptionValue(std::string const & optName, std::string const & val) : optName(optName), strVal {val}, type{isStr} {}
  void apply_to(Highs & problem) const {
    switch (type) {
      case isInt: problem.setOptionValue(optName, intVal); break;
      case isDouble: problem.setOptionValue(optName, doubleVal); break;
      case isStr: problem.setOptionValue(optName, strVal); break;
    }
  }
};

inline void apply_options(Highs & problem, std::vector<OptionValue> const & optionValues) {
  for (auto & option : optionValues) option.apply_to(problem);
}

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
void create_feasibility_subproblem(Highs & feas_subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & mixed_rows,  std::set<HighsInt> const & master_only_rows, 
  HighsSparseMatrix const & extension_matrix);
HighsSparseMatrix construct_extension_matrix(HighsLp const & base_problem, std::set<HighsInt> const & mixed_rows);
void decompose_problem(BendersProblems & problems, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, RowDivision const & row_division, double subproblem_lb);
void decompose_problem(BendersProblems & problems, HighsLp & base_problem, std::set<HighsInt> const & master_variables, double subproblem_lb);
BendersProblems decompose_problem(HighsLp const & problem, std::set<HighsInt> const & master_variables, RowDivision const & row_division, double subproblem_lb); 
BendersProblems decompose_problem(HighsLp & problem, std::set<HighsInt> const & master_variables, double subproblem_lb);
// std::vector<double> get_all_multipliers(Highs const & subproblem);
std::vector<double> get_master_multipliers(Highs const & subproblem, std::set<HighsInt> const & master_variables);
NonZeroVector create_nonzero_vector(std::vector<double> const & base_vector);
NonZeroVector add_mu_entry(NonZeroVector vector, HighsInt mu_index);
void add_nonzero_row(Highs & problem, double lower, double upper, NonZeroVector const & row_vector);
// void add_nonzero_col(Highs & problem, double col_cost, double col_lower, double col_upper, NonZeroVector const & col_vector);
// void add_cut(BendersProblems & problems, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values, CutType cut_type);
// std::pair<std::vector<double>, double> add_cut(Highs & master, CutData const & cut, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values, CutType cut_type, int subproblem_no=0);
// std::pair<std::vector<double>, double> add_cut(Highs & master, Highs const & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values, CutType cut_type, int subproblem_no=0);
Cut form_cut(Highs & master, Highs const & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values, CutType cut_type, int subproblem_no=0);
// void solve_feasibility_subproblem(Highs & subproblem);
// BendersIterationInfo solve_feasibility_subproblem(Highs & feas_subproblem); 
std::set<HighsInt> discover_master_variables(std::vector<std::string> const & variable_names, std::regex const & master_name_pattern);
std::set<HighsInt> discover_master_variables(std::vector<std::string> const & variable_names, std::string const & master_name_pattern);
BendersIterationInfo solve_feasibility_subproblem(Highs & subproblem, BendersIterationInfo info, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values);
BendersIterationInfo solve_subproblem(Highs & subproblem, BendersIterationInfo info, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values);
BendersIterationInfo solve_master(Highs & master, BendersIterationInfo info);
double calculate_solution_cost(Highs const & master, double subproblem_cost, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values);
double calculate_solution_cost(Highs const & master, Highs const & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values);
BendersRet benders(HighsLp & base_problem, std::set<HighsInt> const & master_variables, std::vector<double> const & starting_point, double subproblem_lb, double eps=1e-3, int max_iter=1e2, std::vector<OptionValue> const & masterOptions = {});
BendersRet benders(HighsLp & base_problem, std::string const & master_name_pattern, std::vector<double> const & starting_point, double subproblem_lb, double eps=1e-3, std::vector<OptionValue> const & masterOptions = {});
// BendersRet multi_benders(HighsLp & base_problem, std::set<HighsInt> const & master_variables, std::vector<std::set<HighsInt>> const & subproblem_variables, std::vector<double> const & starting_point, double subproblem_lb, double eps=1e-3, int max_iter=1e2);  
// BendersRet benders2(HighsLp & base_problem, std::set<HighsInt> const & master_variables, std::vector<double> const & starting_point, double subproblem_lb, double eps=1e-3);
// BendersRet benders2(HighsLp & base_problem, std::string const & master_name_pattern, std::vector<double> const & starting_point, double subproblem_lb, double eps=1e-3);
BendersRet multi_benders_loop(MultiBendersProblems & problems, std::set<HighsInt> const & master_variables,
                     std::vector<double> const & starting_point, double eps, int max_iter);
BendersRet multi_benders_loop_aggregated(MultiBendersProblems & problems, std::set<HighsInt> const & master_variables,
                     std::vector<double> const & starting_point, double eps, int max_iter);


BendersRet benders_loop(BendersProblems & problems, std::set<HighsInt> const & master_variables, std::vector<double> const & starting_point, double eps, int max_iter);
std::vector<double> get_dual_costs(HighsLp const & lp);
std::set<HighsInt> index_set_union(std::set<HighsInt> const & a, std::set<HighsInt> const & b);
std::set<HighsInt> index_set_intersection(std::set<HighsInt> const & a, std::set<HighsInt> const & b);
// void decompose_problem(MultiBendersProblems & problems, HighsLp & base_problem, std::set<HighsInt> const & master_variables, std::vector<std::set<HighsInt>> const & subproblems_variables);
// void create_subproblem(Highs & subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & other_rows, std::set<HighsInt> const & subproblem_variables, std::set<HighsInt> const & master_only_variables); 

BendersRet benders_l_shaped(SmpsCoreStructure & core, StochasticTree & tree, 
  std::vector<double> const & starting_point, double subproblem_lb,
  double eps=1e-3, int max_iter=1e2, bool aggregate_cuts=false,
  std::vector<OptionValue> const & masterOptions={}
);

inline double vecsum(std::vector<double> const & vec) {
  return std::accumulate(vec.begin(), vec.end(), 0.);
}

void add_vec_to_vec(std::vector<double> & base, NonZeroVector const & addition);

inline void zerovec(std::vector<double> & vec) {
  std::fill(vec.begin(), vec.end(), 0.);
}

inline std::vector<HighsInt> set_to_vector(std::set<HighsInt> const & set) {
  return {set.begin(), set.end()};
}

