#include "Benders.h"
#include "HConst.h"
#include "Highs.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <regex>
#include "HighsInt.h"
#include "HighsStatus.h"
#include "HighsUtils.h"
#include "HighsLpUtils.h"
#include "highs_c_api.h"


HighsInt find_row_index(std::vector<HighsInt> const & csr_starts, HighsInt index) {
  auto pointer = std::upper_bound(csr_starts.begin(), csr_starts.end(), index);
  return std::distance(csr_starts.begin(), pointer) - 1;
}

RowDivision divide_rows(std::vector<HighsInt> const & csr_index, std::vector<HighsInt> const & csr_starts, std::set<HighsInt> const & master_variables) {
  auto is_master = [&master_variables](HighsInt index){ return master_variables.count(index) > 0; };
  std::set<HighsInt> master_rows, subproblem_rows, mixed_rows;
  for (int i = 0; i < csr_index.size(); ++i)
    (is_master(csr_index[i]) ? master_rows : subproblem_rows).insert(find_row_index(csr_starts, i));
  std::set_intersection(master_rows.begin(), master_rows.end(), subproblem_rows.begin(), subproblem_rows.end(),
                         std::inserter(mixed_rows, mixed_rows.begin()));
  return {master_rows, subproblem_rows, mixed_rows};
}

RowDivision divide_rows(HighsSparseMatrix & constraint_matrix, std::set<HighsInt> const & master_variables) {
  constraint_matrix.ensureRowwise();
  return divide_rows(constraint_matrix.index_, constraint_matrix.start_, master_variables);
}

void fix_variable(Highs & problem, HighsInt variable_index, double value) {
  problem.changeColBounds(variable_index, value, value);
}

bool fix_master_variables(Highs & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  int master_index = 0;
  for (auto subproblem_index : master_variables)
    fix_variable(subproblem, subproblem_index, master_values.at(master_index++));
  return true;
}

std::set<HighsInt> sequence_complement(std::set<HighsInt> const & set, HighsInt max_number) {
  std::set<HighsInt> complement;
  for (int i = 0; i < max_number; i++)
    if (set.count(i) == 0)
      complement.insert(i);
  return complement;
}

//TODO: remove
inline std::vector<HighsInt> set_to_vector(std::set<HighsInt> const & set) { return {set.begin(), set.end()}; }

void create_master_problem(Highs & master, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & subproblem_rows) {
  master.passModel(base_problem);
  auto nonmaster_rows = set_to_vector(subproblem_rows);
  master.deleteRows(nonmaster_rows.size(), nonmaster_rows.data());
  auto subproblem_variables = set_to_vector(sequence_complement(master_variables, base_problem.num_col_)); // TODO: should it be here?
  master.deleteCols(subproblem_variables.size(), subproblem_variables.data());
  master.addCol(1, mu_lb, kHighsInf, 0, nullptr, nullptr);
}

void create_subproblem(Highs & subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables) {
  subproblem.passModel(base_problem);
  subproblem.changeObjectiveOffset(0);
  auto nonsub_variables = set_to_vector(master_variables);
  std::vector<double> zeros (master_variables.size(), 0);
  subproblem.changeColsCost(master_variables.size(), nonsub_variables.data(), zeros.data());
}

void add_nonzero_col(Highs & problem, double col_cost, double col_lower, double col_upper, NonZeroVector const & col_vector) {
  problem.addCol(col_cost, 0, kHighsInf, col_vector.number_of_nonzeros, col_vector.nonzero_indices.data(), col_vector.nonzero_values.data());
}

void create_feasibility_subproblem(Highs & feas_subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & mixed_rows) {
  feas_subproblem.passModel(base_problem);
  feas_subproblem.changeObjectiveOffset(0);
  std::vector<double> zeros(base_problem.num_col_, 0);
  feas_subproblem.changeColsCost(0, base_problem.num_col_ - 1, zeros.data());
  for (auto i : mixed_rows) {
    if (base_problem.row_lower_.at(i) > -kHighsInf)
      add_nonzero_col(feas_subproblem, 1, 0, kHighsInf, NonZeroVector{1, {i}, {1}});
    if (base_problem.row_upper_.at(i) < kHighsInf) 
      add_nonzero_col(feas_subproblem, 1, 0, kHighsInf, NonZeroVector{1, {i}, {-1}});
  }
}

void decompose_problem(BendersProblems & problems, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, RowDivision const & row_division) {
  create_master_problem(problems.master, base_problem, master_variables, row_division.subproblem_rows);
  create_subproblem(problems.subproblem, base_problem, master_variables);
  create_feasibility_subproblem(problems.feas_subproblem, base_problem, master_variables, row_division.mixed_rows);
}

void decompose_problem(BendersProblems & problems, HighsLp & base_problem, std::set<HighsInt> const & master_variables) {
  auto row_division = divide_rows(base_problem.a_matrix_, master_variables);
  decompose_problem(problems, base_problem, master_variables, row_division);
}

// this can be done with reduced costs!
std::vector<double> get_all_multipliers(Highs const & subproblem) {
  std::vector<double> all_multipliers;
  auto const & dual = subproblem.getSolution().row_dual;
  auto const & A = subproblem.getLp().a_matrix_;
  A.productTranspose(all_multipliers, dual);
  return all_multipliers;
}

std::vector<double> get_master_multipliers(Highs const & subproblem, std::set<HighsInt> const & master_variables) {
  auto all_multipliers = get_all_multipliers(subproblem);
  std::vector<double> master_multipliers;
  for (auto i : master_variables)
    master_multipliers.push_back(all_multipliers.at(i));
  return master_multipliers;
}

NonZeroVector create_nonzero_vector(std::vector<double> const & base_vector) {
  HighsInt number_of_nonzeros = 0;
  std::vector<HighsInt> nonzero_indices;
  std::vector<double> nonzero_values;
  double vector_entry;
  for (int i = 0; i < base_vector.size(); ++i)
    if ((vector_entry = base_vector.at(i)) != 0) {
      nonzero_indices.push_back(i);
      nonzero_values.push_back(vector_entry);
      ++number_of_nonzeros;
    }
  return {number_of_nonzeros, nonzero_indices, nonzero_values};
}

NonZeroVector add_mu_entry(NonZeroVector vector, HighsInt mu_index) {
  vector.number_of_nonzeros += 1;
  vector.nonzero_indices.push_back(mu_index);
  vector.nonzero_values.push_back(1);
  return vector;
}

void add_nonzero_row(Highs & problem, double lower, double upper, NonZeroVector const & row_vector) {
  problem.addRow(lower, upper, row_vector.number_of_nonzeros, row_vector.nonzero_indices.data(), row_vector.nonzero_values.data());
}

void add_cut(Highs & master, Highs const & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values, CutType cut_type) {
  double dual_objective;
  subproblem.getDualObjectiveValue(dual_objective);
  auto multipliers = get_master_multipliers(subproblem, master_variables);
  auto old_value_multiple = std::inner_product(multipliers.begin(), multipliers.end(), master_values.begin(), 0.0);
  auto nonzero_multipliers = create_nonzero_vector(multipliers);
  if (cut_type == CutType::Objective) nonzero_multipliers = add_mu_entry(nonzero_multipliers, master_variables.size());
  add_nonzero_row(master, dual_objective + old_value_multiple, kHighsInf, nonzero_multipliers);
}

std::set<HighsInt> discover_master_variables(std::vector<std::string> const & variable_names, std::regex const & master_name_pattern) {
  std::set<HighsInt> master_variables;
  for (int i = 0; i < variable_names.size(); ++i)
    if (std::regex_search(variable_names.at(i), master_name_pattern))
      master_variables.insert(i);
  return master_variables;
}

std::set<HighsInt> discover_master_variables(std::vector<std::string> const & variable_names, std::string const & master_name_pattern) {
  return discover_master_variables(variable_names, std::regex(master_name_pattern));
}

// void solve_feasibility_subproblem(Highs & subproblem) {
    // bool has_dual_ray;
    // std::vector<double> dual_ray (subproblem.getLp().num_row_);
    // subproblem.getDualRay(has_dual_ray, dual_ray.data());
    // auto solution = subproblem.getSolution();
    // solution.row_dual = dual_ray;
    // subproblem.setSolution(solution);
// }

BendersIterationInfo solve_subproblem(Highs & subproblem, BendersIterationInfo info, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  fix_master_variables(subproblem, master_variables, master_values);
  info.was_error = info.was_error || subproblem.run() == HighsStatus::kError;
  info.was_subproblem_feasible = subproblem.getModelStatus() == HighsModelStatus::kOptimal;
  return info;
}


// BendersIterationInfo solve_feasibility_subproblem(Highs & feas_subproblem) {
//   BendersIterationInfo info;
//   info.was_error = feas_subproblem.run() == HighsStatus::kError;
//   return info;
// }

BendersIterationInfo solve_master(Highs & master, BendersIterationInfo info) {
  info.was_error = info.was_error || master.run() == HighsStatus::kError;
  return info;
}

double calculate_solution_cost(Highs const & master, Highs const & subproblem, std::set<HighsInt> const & master_variables) {
  double subproblem_cost = subproblem.getObjectiveValue();
  double master_cost = master.getObjectiveValue();
  int no_master_vars = master_variables.size();
  auto const & master_solution = master.getSolution().col_value;
  auto const & master_costs = master.getLp().col_cost_;
  double mu_cost = std::inner_product(master_solution.begin() + no_master_vars, master_solution.end(), master_costs.begin() + no_master_vars, 0.);
  return subproblem_cost + master_cost - mu_cost;
}

double benders(HighsLp & base_problem, std::string const & master_name_pattern, std::vector<double> const & starting_point, double eps) { 
  auto master_variables = discover_master_variables(base_problem.col_names_, master_name_pattern);
  return benders(base_problem, master_variables, starting_point, eps);
}

double benders(HighsLp & base_problem, std::set<HighsInt> const & master_variables, std::vector<double> const & starting_point, double eps) { 
  BendersProblems problems; decompose_problem(problems, base_problem, master_variables);
  BendersIterationInfo info;
  auto master_values = starting_point;
  int iter = 0;
  double UBD = kHighsInf, LBD = -kHighsInf;
  while (UBD - LBD > eps && !info.was_error && ++iter < 1e2) {
    info = solve_subproblem(problems.subproblem, info, master_variables, master_values);
    if (info.was_subproblem_feasible) {
      double solution_cost = calculate_solution_cost(problems.master, problems.subproblem, master_variables);
      // TODO: maximization?
      UBD = std::min(UBD, solution_cost);
      add_cut(problems.master, problems.subproblem, master_variables, master_values, CutType::Objective);
    }
    else {
      info = solve_subproblem(problems.feas_subproblem, info,  master_variables, master_values);
      add_cut(problems.master, problems.feas_subproblem, master_variables, master_values, CutType::Feasibility);
    }
    info = solve_master(problems.master, info);
    master_values = problems.master.getSolution().col_value;
    LBD = problems.master.getObjectiveValue();
  }
  return UBD;
}

