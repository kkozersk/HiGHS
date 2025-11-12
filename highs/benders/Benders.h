#pragma once
#include <set>
#include <regex>
#include "HConst.h"
#include "HighsInt.h"
#include "Highs.h"
#include "HighsLp.h"
#include "HighsSparseMatrix.h"

//TODO: REMOVE
const double mu_lb = -1e4;

struct RowDivision {
  std::set<HighsInt> master_rows;
  std::set<HighsInt> subproblem_rows;
  std::set<HighsInt> mixed_rows;
};

struct BendersProblems {
  Highs master;
  Highs subproblem;
  Highs feas_subproblem;
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
  double UBD=kHighsInf;
  double LBD=-kHighsInf;
  bool was_subproblem_feasible=true;
  bool was_error=false;
};

enum CutType { Objective, Feasibility };

HighsInt find_row_index(std::vector<HighsInt> const & csr_starts, HighsInt index);
RowDivision divide_rows(std::vector<HighsInt> const & csr_index, std::vector<HighsInt> const & csr_starts, std::set<HighsInt> const & master_variables); 
RowDivision divide_rows(HighsSparseMatrix & constraint_matrix, std::set<HighsInt> const & master_variables);
void fix_variable(Highs & problem, HighsInt variable_index, double value);
bool fix_master_variables(Highs & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values);
std::set<HighsInt> sequence_complement(std::set<HighsInt> const & set, HighsInt max_number);
void create_master_problem(Highs & master, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & subproblem_rows);
void create_subproblem(Highs & subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables);
void create_feasibility_subproblem(Highs & feas_subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & mixed_rows);
void decompose_problem(BendersProblems & problems, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, RowDivision const & row_division);
void decompose_problem(BendersProblems & problems, HighsLp & base_problem, std::set<HighsInt> const & master_variables);
BendersProblems decompose_problem(HighsLp const & problem, std::set<HighsInt> const & master_variables, RowDivision const & row_division); 
BendersProblems decompose_problem(HighsLp & problem, std::set<HighsInt> const & master_variables);
std::vector<double> get_all_multipliers(Highs const & subproblem);
std::vector<double> get_master_multipliers(Highs const & subproblem, std::set<HighsInt> const & master_variables);
NonZeroVector create_nonzero_vector(std::vector<double> const & base_vector);
NonZeroVector add_mu_entry(NonZeroVector vector, HighsInt mu_index);
void add_nonzero_row(Highs & problem, double lower, double upper, NonZeroVector const & row_vector);
void add_nonzero_col(Highs & problem, double col_cost, double col_lower, double col_upper, NonZeroVector const & col_vector);
// void add_cut(BendersProblems & problems, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values, CutType cut_type);
void add_cut(Highs & master, Highs const & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values, CutType cut_type);
// void solve_feasibility_subproblem(Highs & subproblem);
BendersIterationInfo solve_feasibility_subproblem(Highs & feas_subproblem, BendersIterationInfo info); 
std::set<HighsInt> discover_master_variables(std::vector<std::string> const & variable_names, std::regex const & master_name_pattern);
std::set<HighsInt> discover_master_variables(std::vector<std::string> const & variable_names, std::string const & master_name_pattern);
BendersIterationInfo solve_subproblem(Highs & subproblem, BendersIterationInfo info);
BendersIterationInfo solve_master(Highs & master, BendersIterationInfo info);
double benders(HighsLp & base_problem, std::set<HighsInt> const & master_variables, std::vector<double> starting_point, double eps=1e-3);
double benders(HighsLp & base_problem, std::string const & master_name_pattern, std::vector<double> starting_point, double eps=1e-3);
