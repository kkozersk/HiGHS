#include "Benders.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>
#include <vector>
#include <regex>
#include "lp_data/HighsStatus.h"
#include "lp_data/HighsLpUtils.h"

HighsInt find_row_index(std::vector<HighsInt> const & csr_starts, HighsInt index) {
  auto pointer = std::upper_bound(csr_starts.begin(), csr_starts.end(), index);
  return std::distance(csr_starts.begin(), pointer) - 1;
}

RowDivision divide_rows(std::vector<HighsInt> const & csr_index, std::vector<HighsInt> const & csr_starts, std::set<HighsInt> const & master_variables) {
  auto is_master = [&master_variables](HighsInt index){ return master_variables.find(index) != master_variables.end(); };
  std::set<HighsInt> master_rows, subproblem_rows, mixed_rows;
  for (std::vector<HighsInt>::size_type i = 0; i < csr_index.size(); ++i)
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

void unfreeze_mu(Highs & master, std::set<HighsInt> const & master_variables, HighsInt subproblem_no) {
  master.changeColBounds(master_variables.size() + subproblem_no, -kHighsInf, kHighsInf);
}

bool fix_master_variables(Highs & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  int master_index = 0;
  for (auto subproblem_index : master_variables)
    fix_variable(subproblem, subproblem_index, master_values.at(master_index++));
  return true;
}

std::set<HighsInt> sequence_complement(std::set<HighsInt> const & set, HighsInt max_number) {
  std::set<HighsInt> complement;
  for (HighsInt i = 0; i < max_number; i++)
    if (set.count(i) == 0)
      complement.insert(i);
  return complement;
}

//TODO: remove
inline std::vector<HighsInt> set_to_vector(std::set<HighsInt> const & set) { return {set.begin(), set.end()}; }

void create_master_problem(Highs & master, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & subproblem_rows, int no_subproblems) {
  master.passModel(base_problem);
  auto nonmaster_rows = set_to_vector(subproblem_rows);
  master.deleteRows(nonmaster_rows.size(), nonmaster_rows.data());
  auto subproblem_variables = set_to_vector(sequence_complement(master_variables, base_problem.num_col_)); // TODO: should it be here?
  master.deleteCols(subproblem_variables.size(), subproblem_variables.data());
  for (int i = 0; i < no_subproblems; ++i)
    master.addCol(1, 0, 0, 0, nullptr, nullptr);
}

void create_subproblem(Highs & subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables) {
  subproblem.passModel(base_problem);
  subproblem.changeObjectiveOffset(0);
  auto nonsub_variables = set_to_vector(master_variables);
  std::vector<double> zeros (master_variables.size(), 0);
  subproblem.changeColsCost(master_variables.size(), nonsub_variables.data(), zeros.data());
}

void create_subproblem(Highs & subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & other_rows, std::set<HighsInt> const & subproblem_variables) {
  subproblem.passModel(base_problem);
  subproblem.changeObjectiveOffset(0);
  auto nonsub_variables = set_to_vector(sequence_complement(subproblem_variables, base_problem.num_col_));
  std::vector<double> zeros (nonsub_variables.size(), 0);
  subproblem.changeColsCost(nonsub_variables.size(), nonsub_variables.data(), zeros.data());
  subproblem.changeColsBounds(nonsub_variables.size(), nonsub_variables.data(),  zeros.data(), zeros.data());
  auto nonsub_rows = set_to_vector(other_rows);
  subproblem.deleteRows(nonsub_rows.size(), nonsub_rows.data());
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
  create_master_problem(problems.master, base_problem, master_variables, row_division.other_rows);
  create_subproblem(problems.subproblem, base_problem, master_variables);
  // create_feasibility_subproblem(problems.feas_subproblem, base_problem, master_variables, row_division.mixed_rows);
}

void decompose_problem(BendersProblems & problems, HighsLp & base_problem, std::set<HighsInt> const & master_variables) {
  auto row_division = divide_rows(base_problem.a_matrix_, master_variables);
  decompose_problem(problems, base_problem, master_variables, row_division);
}

std::set<HighsInt> index_set_union(std::set<HighsInt> const & a, std::set<HighsInt> const & b) {
  auto avec = set_to_vector(a);
  auto bvec = set_to_vector(b);
  std::set<HighsInt> a_union_b;
  std::set_union(avec.begin(), avec.end(), bvec.begin(), bvec.end(), std::inserter(a_union_b, a_union_b.begin()));
  return a_union_b;
}

void decompose_problem(MultiBendersProblems & problems, HighsLp & base_problem, std::set<HighsInt> const & master_variables, std::vector<std::set<HighsInt>> const & subproblems_variables) {
  auto row_division = divide_rows(base_problem.a_matrix_, master_variables);
  create_master_problem(problems.master, base_problem, master_variables, row_division.other_rows, subproblems_variables.size());
  problems.master.setOptionValue("presolve", kHighsOffString);
  problems.subproblems = std::vector<Highs> (subproblems_variables.size());
  for (std::vector<HighsInt>::size_type i = 0; i < subproblems_variables.size(); ++i) {
    std::set<HighsInt> const & subproblem_variables = subproblems_variables.at(i);
    auto master_and_subproblem_vars = index_set_union(subproblem_variables, master_variables);
    row_division = divide_rows(base_problem.a_matrix_, master_and_subproblem_vars);
    create_subproblem(problems.subproblems.at(i), base_problem, master_variables, row_division.other_rows, subproblem_variables);
    problems.subproblems.at(i).setOptionValue("presolve", kHighsOffString);
  }
}

std::vector<double> get_master_multipliers(Highs const & subproblem, std::set<HighsInt> const & master_variables) {
  auto const & reduced_costs = subproblem.getSolution().col_dual;
  std::vector<double> master_multipliers;
  for (auto i : master_variables) master_multipliers.push_back(-reduced_costs.at(i));
  return master_multipliers;
}

NonZeroVector create_nonzero_vector(std::vector<double> const & base_vector) {
  HighsInt number_of_nonzeros = 0;
  std::vector<HighsInt> nonzero_indices;
  std::vector<double> nonzero_values;
  double vector_entry;
  for (std::vector<double>::size_type i = 0; i < base_vector.size(); ++i)
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

void add_cut(Highs & master, CutData const & cut, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values, CutType cut_type, int subproblem_no) {
  auto old_value_multiple = std::inner_product(cut.master_multipliers.begin(), cut.master_multipliers.end(), master_values.begin(), 0.0);
  auto nonzero_multipliers = create_nonzero_vector(cut.master_multipliers);
  if (cut_type == CutType::Objective) nonzero_multipliers = add_mu_entry(nonzero_multipliers, master_variables.size() + subproblem_no);
  add_nonzero_row(master, cut.dual_objective + old_value_multiple, kHighsInf, nonzero_multipliers);
}

void add_cut(Highs & master, Highs const & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values, CutType cut_type, int subproblem_no) {
  double dual_objective;
  subproblem.getDualObjectiveValue(dual_objective);
  auto multipliers = get_master_multipliers(subproblem, master_variables);
  add_cut(master, {multipliers, dual_objective}, master_variables, master_values, cut_type, subproblem_no);
  // auto old_value_multiple = std::inner_product(multipliers.begin(), multipliers.end(), master_values.begin(), 0.0);
  // auto nonzero_multipliers = create_nonzero_vector(multipliers);
  // if (cut_type == CutType::Objective) nonzero_multipliers = add_mu_entry(nonzero_multipliers, master_variables.size());
  // add_nonzero_row(master, dual_objective + old_value_multiple, kHighsInf, nonzero_multipliers);
}

// void add_cut(Highs & master, double dual_objective, std::vector<double> const & dual_ray, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values, CutType cut_type) {
//   auto multipliers = get_master_multipliers(subproblem, master_variables);
//   auto old_value_multiple = std::inner_product(multipliers.begin(), multipliers.end(), master_values.begin(), 0.0);
//   auto nonzero_multipliers = create_nonzero_vector(multipliers);
//   if (cut_type == CutType::Objective) nonzero_multipliers = add_mu_entry(nonzero_multipliers, master_variables.size());
//   add_nonzero_row(master, dual_objective + old_value_multiple, kHighsInf, nonzero_multipliers);
// }

std::set<HighsInt> discover_master_variables(std::vector<std::string> const & variable_names, std::regex const & master_name_pattern) {
  std::set<HighsInt> master_variables;
  for (std::vector<std::string>::size_type i = 0; i < variable_names.size(); ++i)
    if (std::regex_search(variable_names.at(i), master_name_pattern))
      master_variables.insert(i);
  return master_variables;
}

std::set<HighsInt> discover_master_variables(std::vector<std::string> const & variable_names, std::string const & master_name_pattern) {
  return discover_master_variables(variable_names, std::regex(master_name_pattern));
}

std::vector<double> get_dual_costs(HighsLp const & lp) {
  std::vector<double> dual_prices (lp.num_row_);
  for (int i = 0; i < lp.num_row_; ++i)
    dual_prices[i] = lp.row_upper_.at(i) < kHighsInf ? lp.row_upper_.at(i) : lp.row_lower_.at(i);
  return dual_prices;
}

std::vector<double> calculate_negated_reduced_costs(HighsSparseMatrix const & A, std::vector<double> const & dual) {
  std::vector<double> negated_prices;
  A.productTranspose(negated_prices, dual);
  return negated_prices;
}

CutData solve_feasibility_subproblem(Highs & subproblem, std::set<HighsInt> const & master_variables) {
  bool has_dual_ray;
  std::vector<double> dual_ray (subproblem.getLp().num_row_);
  subproblem.getDualRay(has_dual_ray, dual_ray.data());
  auto dual_costs = get_dual_costs(subproblem.getLp());
  // auto const & row_lower = subproblem.getLp().row_lower_;
  // auto const & row_upper = subproblem.getLp().row_upper_;
  // std::vector<double> dual_prices (subproblem.getLp().num_row_);
  // for (int i = 0; i < subproblem.getLp().num_row_; ++i)
  //   dual_prices.at(i) = (row_upper.at(i) < kHighsInf ? row_upper.at(i) : row_lower.at(i));
  double dual_objective = std::abs(std::inner_product(dual_ray.begin(), dual_ray.end(), dual_costs.begin(), 0.));
  auto all_multipliers = calculate_negated_reduced_costs(subproblem.getLp().a_matrix_, dual_ray);
  std::vector<double> master_multipliers(master_variables.size());
  std::transform(master_variables.begin(), master_variables.end(), master_multipliers.begin(),
                  [&all_multipliers](HighsInt idx){ return all_multipliers.at(idx); });
  return {master_multipliers, dual_objective};
}

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

double calculate_solution_cost(Highs const & master, double subproblem_cost, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  auto const & master_costs = master.getLp().col_cost_;
  int no_master_vars = master_variables.size();
  double master_cost = std::inner_product(master_values.begin(), master_values.begin() + no_master_vars, master_costs.begin(), 0.);
  double offset; master.getObjectiveOffset(offset);
  return subproblem_cost + master_cost + offset;
}

double calculate_solution_cost(Highs const & master, Highs const & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  return calculate_solution_cost(master, subproblem.getObjectiveValue(), master_variables, master_values);
}

double benders(HighsLp & base_problem, std::string const & master_name_pattern, std::vector<double> const & starting_point, double eps) { 
  auto master_variables = discover_master_variables(base_problem.col_names_, master_name_pattern);
  return benders(base_problem, master_variables, starting_point, eps);
}

double benders(HighsLp & base_problem, std::set<HighsInt> const & master_variables, std::vector<double> const & starting_point, double eps, int max_iter) { 
  BendersProblems problems; decompose_problem(problems, base_problem, master_variables);
  BendersIterationInfo info;
  auto master_values = starting_point;
  int iter = 0;
  double UBD = kHighsInf, LBD = -kHighsInf;
  bool any_objective_cuts = false;
  while (UBD - LBD > eps && !info.was_error && ++iter < max_iter) {
    info = solve_subproblem(problems.subproblem, info, master_variables, master_values);
    if (info.was_subproblem_feasible) {
      double solution_cost = calculate_solution_cost(problems.master, problems.subproblem, master_variables, master_values);
      UBD = std::min(UBD, solution_cost);
      if (UBD - LBD <= eps) break;
      add_cut(problems.master, problems.subproblem, master_variables, master_values, CutType::Objective);
      if (!any_objective_cuts) {
        any_objective_cuts = true;
        unfreeze_mu(problems.master, master_variables);
      }
    }
    else {
      auto cut = solve_feasibility_subproblem(problems.subproblem, master_variables);
      add_cut(problems.master, cut, master_variables, master_values, CutType::Feasibility);
    }
    info = solve_master(problems.master, info);
    master_values = problems.master.getSolution().col_value;
    if (any_objective_cuts)
      LBD = problems.master.getObjectiveValue();
  }
  return UBD;
}

inline bool all(std::vector<bool> const & v) { return std::all_of(v.begin(), v.end(), [](bool x) { return x; }); }

double multi_benders(HighsLp & base_problem, std::set<HighsInt> const & master_variables,
                     std::vector<std::set<HighsInt>> const & subproblems_variables,
                     std::vector<double> const & starting_point, double eps, int max_iter) { 
  MultiBendersProblems problems;
  decompose_problem(problems, base_problem, master_variables, subproblems_variables);
  BendersIterationInfo info;
  auto master_values = starting_point;
  int iter = 0;
  double UBD = kHighsInf, LBD = -kHighsInf;
  int no_subproblems = subproblems_variables.size();
  std::vector<bool> any_objective_cuts(no_subproblems, false);
  bool all_objective_cuts = false;
  while (UBD - LBD > eps && !info.was_error && ++iter < max_iter) {
    double subproblem_costs = 0;
    bool all_feasible = true;
    for (int i = 0; i < no_subproblems; ++i) {
      auto & subproblem = problems.subproblems.at(i);
      info = solve_subproblem(subproblem, info, master_variables, master_values);
      if (info.was_subproblem_feasible) {
        if (!any_objective_cuts[i]) {
          any_objective_cuts[i] = true;
          unfreeze_mu(problems.master, master_variables, i);
        }
        subproblem_costs += subproblem.getObjectiveValue();
        add_cut(problems.master, subproblem, master_variables, master_values, CutType::Objective, i);
      }
      else {
        all_feasible = false;
        auto cut = solve_feasibility_subproblem(subproblem, master_variables);
        add_cut(problems.master, cut, master_variables, master_values, CutType::Feasibility, i);
      }
    }
    if (all_feasible) {
        double solution_cost = calculate_solution_cost(problems.master, subproblem_costs, master_variables, master_values);
        UBD = std::min(UBD, solution_cost);
        if (UBD - LBD <= eps) break;
    }
    info = solve_master(problems.master, info);
    master_values = problems.master.getSolution().col_value;
    if((all_objective_cuts = all_objective_cuts || all(any_objective_cuts)))
      LBD = problems.master.getObjectiveValue();
  }
  return UBD;
}

double benders2(HighsLp & base_problem, std::set<HighsInt> const & master_variables, std::vector<double> const & starting_point, double eps) {
  auto subproblem_variables = sequence_complement(master_variables, base_problem.num_col_);
  return multi_benders(base_problem, master_variables, {subproblem_variables}, starting_point, eps);
} 

double benders2(HighsLp & base_problem, std::string const & master_name_pattern, std::vector<double> const & starting_point, double eps) { 
  auto master_variables = discover_master_variables(base_problem.col_names_, master_name_pattern);
  return benders2(base_problem, master_variables, starting_point, eps);
}
