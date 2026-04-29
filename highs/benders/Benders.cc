#include "Benders.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>
#include <vector>
#include <regex>
#include "HConst.h"
#include "HighsOptions.h"
#include "lp_data/HighsStatus.h"
#include "lp_data/HighsLpUtils.h"
#include "ipm/hipo/ipm/Solver.h"

HighsInt find_row_index(std::vector<HighsInt> const & csr_starts, HighsInt index) {
  auto pointer = std::upper_bound(csr_starts.begin(), csr_starts.end(), index);
  return std::distance(csr_starts.begin(), pointer) - 1;
}

RowDivision divide_rows(std::vector<HighsInt> const & csr_index, std::vector<HighsInt> const & csr_starts, std::set<HighsInt> const & master_variables) {
  auto is_master = [&master_variables](HighsInt index){ return master_variables.find(index) != master_variables.end(); };
  std::set<HighsInt> master_rows, subproblem_rows, mixed_rows, master_only_rows;
  for (std::vector<HighsInt>::size_type i = 0; i < csr_index.size(); ++i)
    (is_master(csr_index[i]) ? master_rows : subproblem_rows).insert(find_row_index(csr_starts, i));
  std::set_intersection(master_rows.begin(), master_rows.end(), subproblem_rows.begin(), subproblem_rows.end(),
                         std::inserter(mixed_rows, mixed_rows.begin()));
  std::set_difference(master_rows.begin(), master_rows.end(), mixed_rows.begin(), mixed_rows.end(),
      std::inserter(master_only_rows, master_only_rows.begin()));
  return {master_rows, subproblem_rows, mixed_rows, master_only_rows};
}

RowDivision divide_rows(HighsSparseMatrix & constraint_matrix, std::set<HighsInt> const & master_variables) {
  constraint_matrix.ensureRowwise();
  return divide_rows(constraint_matrix.index_, constraint_matrix.start_, master_variables);
}

void fix_variable(Highs & problem, HighsInt variable_index, double value) {
  // TODO verify extra low bounds
  if (std::abs(value) < 1e-8) value = 0.;
  problem.changeColBounds(variable_index, value, value);
}

// void unfreeze_mu(Highs & master, std::set<HighsInt> const & master_variables, HighsInt subproblem_no) {
//   master.changeColBounds(master_variables.size() + subproblem_no, -kHighsInf, kHighsInf);
// }

bool fix_master_variables(Highs & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  int master_index = 0;
  for (auto subproblem_index : master_variables)
    fix_variable(subproblem, subproblem_index, master_values.at(master_index++));
  return true;
}

void save_master_variables(Highs & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  std::vector<double> result;
  std::vector<double> x (subproblem.getNumCol(), 0);
  int master_index = 0;
  for (auto subproblem_index : master_variables) {
    x.at(subproblem_index) = master_values.at(master_index++);
  }
  subproblem.getLp().a_matrix_.product(result, x);
  std::ofstream vs("/tmp/vs.csv", std::ios::app);
  for (auto r : result) vs << r << ",";
  vs << "\n";
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

void create_master_problem(Highs & master, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & subproblem_rows, double subproblem_lb, int no_subproblems) {
  master.passModel(base_problem);
  auto nonmaster_rows = set_to_vector(subproblem_rows);
  master.deleteRows(nonmaster_rows.size(), nonmaster_rows.data());
  auto subproblem_variables = set_to_vector(sequence_complement(master_variables, base_problem.num_col_)); // TODO: should it be here?
  master.deleteCols(subproblem_variables.size(), subproblem_variables.data());
  for (int i = 0; i < no_subproblems; ++i)
    master.addCol(1, subproblem_lb, kHighsInf, 0, nullptr, nullptr);
}

void create_subproblem(Highs & subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & master_only_rows) {
  subproblem.passModel(base_problem);
  subproblem.changeObjectiveOffset(0);
  auto nonsub_variables = set_to_vector(master_variables);
  std::vector<double> zeros (master_variables.size(), 0);
  subproblem.changeColsCost(master_variables.size(), nonsub_variables.data(), zeros.data());
  std::vector<int> to_delete = set_to_vector(master_only_rows);
  subproblem.deleteRows(to_delete.size(), to_delete.data());
}

// //TODO requires master rows to be before subproblems
// void create_subproblem(Highs & subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & other_rows, std::set<HighsInt> const & subproblem_variables, std::set<HighsInt> const & master_only_rows) {
//   subproblem.passModel(base_problem);
//   subproblem.changeObjectiveOffset(0);
//   auto nonsub_variables = set_to_vector(sequence_complement(subproblem_variables, base_problem.num_col_));
//   std::vector<double> zeros (nonsub_variables.size(), 0);
//   subproblem.changeColsCost(nonsub_variables.size(), nonsub_variables.data(), zeros.data());
//   subproblem.changeColsBounds(nonsub_variables.size(), nonsub_variables.data(),  zeros.data(), zeros.data());
//   auto nonsub_rows = set_to_vector(other_rows);
//   subproblem.deleteRows(nonsub_rows.size(), nonsub_rows.data());
//   std::vector<int> to_delete = set_to_vector(master_only_rows);
//   subproblem.deleteRows(to_delete.size(), to_delete.data());
// }

void add_nonzero_col(Highs & problem, double col_cost, double col_lower, double col_upper, NonZeroVector const & col_vector) {
  problem.addCol(col_cost, 0, kHighsInf, col_vector.number_of_nonzeros, col_vector.nonzero_indices.data(), col_vector.nonzero_values.data());
}

// does not delete other rows?
void create_feasibility_subproblem(Highs & feas_subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & mixed_rows, std::set<HighsInt> const & master_only_rows) {
  feas_subproblem.passModel(base_problem);
  feas_subproblem.changeObjectiveOffset(0);
  std::vector<double> zeros(base_problem.num_col_, 0);
  feas_subproblem.changeColsCost(0, base_problem.num_col_ - 1, zeros.data());
  // for (auto i : mixed_rows) {
  for (int i = 0; i < base_problem.num_row_; ++i) {
    if (base_problem.row_lower_.at(i) > -kHighsInf)
      add_nonzero_col(feas_subproblem, 1, 0, kHighsInf, NonZeroVector{1, {i}, {1}});
    if (base_problem.row_upper_.at(i) < kHighsInf) 
      add_nonzero_col(feas_subproblem, 1, 0, kHighsInf, NonZeroVector{1, {i}, {-1}});
  }
  std::vector<int> to_delete = set_to_vector(master_only_rows);
  feas_subproblem.deleteRows(to_delete.size(), to_delete.data());
  //TODO verify for deletion
  // for (auto i : master_rows)
    // feas_subproblem.changeRowBounds(i, -kHighsInf, kHighsInf);
}

void master_modifier(Highs & master) {
  master.setOptionValue("solver", used_solver);
  // master.setOptionValue("solver", kSimplexString);
  master.setOptionValue("optimality_tolerance", ipm_acc);
  master.setOptionValue("ipm_optimality_tolerance", ipm_acc);
  master.setOptionValue("run_crossover", kHighsOffString);
  master.setOptionValue("max_centring_steps", 1000);
  master.setOptionValue("presolve", kHighsOnString);
  // master.setOptionValue("ipm_iteration_limit", 10);
  master.setOptionValue("centring_gamma", 1-1e-5);
  master.setOptionValue("dual_feasibility_tolerance", ipm_feas);
  master.setOptionValue("primal_feasibility_tolerance", ipm_feas);
}
//TODO sub and feas sub have too many rows and columns
void decompose_problem(BendersProblems & problems, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, RowDivision const & row_division, double subproblem_lb) {
  create_master_problem(problems.master, base_problem, master_variables, row_division.other_rows, subproblem_lb);
  create_subproblem(problems.subproblem, base_problem, master_variables, row_division.diff_other);
  // master_modifier(problems.master);
  // problems.master.setOptionValue("presolve", kHighsOffString);
  // problems.subproblem.setOptionValue("presolve", kHighsOffString);
  create_feasibility_subproblem(problems.feas_subproblem, base_problem, master_variables, row_division.mixed_rows, row_division.diff_other);
}

void decompose_problem(BendersProblems & problems, HighsLp & base_problem, std::set<HighsInt> const & master_variables, double subproblem_lb) {
  auto row_division = divide_rows(base_problem.a_matrix_, master_variables);
  decompose_problem(problems, base_problem, master_variables, row_division, subproblem_lb);
}

std::set<HighsInt> index_set_union(std::set<HighsInt> const & a, std::set<HighsInt> const & b) {
  auto avec = set_to_vector(a);
  auto bvec = set_to_vector(b);
  std::set<HighsInt> a_union_b;
  std::set_union(avec.begin(), avec.end(), bvec.begin(), bvec.end(), std::inserter(a_union_b, a_union_b.begin()));
  return a_union_b;
}

std::set<HighsInt> index_set_intersection(std::set<HighsInt> const & a, std::set<HighsInt> const & b) {
  auto avec = set_to_vector(a);
  auto bvec = set_to_vector(b);
  std::set<HighsInt> a_union_b;
  std::set_intersection(avec.begin(), avec.end(), bvec.begin(), bvec.end(), std::inserter(a_union_b, a_union_b.begin()));
  return a_union_b;
}

// void decompose_problem(MultiBendersProblems & problems, HighsLp & base_problem, std::set<HighsInt> const & master_variables, std::vector<std::set<HighsInt>> const & subproblems_variables, double subproblem_lb) {
//   auto row_division = divide_rows(base_problem.a_matrix_, master_variables);
//   auto master_mixed_rows = row_division.mixed_rows;
//   auto master_only_rows = row_division.diff_other;
//   create_master_problem(problems.master, base_problem, master_variables, row_division.other_rows, subproblem_lb, subproblems_variables.size());
//   master_modifier(problems.master);
//   // problems.master.setOptionValue("presolve", kHighsOffString);
//   problems.subproblems = std::vector<Highs> (subproblems_variables.size());
//   problems.feas_subproblems = std::vector<Highs> (subproblems_variables.size());
//   for (std::vector<HighsInt>::size_type i = 0; i < subproblems_variables.size(); ++i) {
//     std::set<HighsInt> const & subproblem_variables = subproblems_variables.at(i);
//     auto master_and_subproblem_vars = index_set_union(subproblem_variables, master_variables);
//     row_division = divide_rows(base_problem.a_matrix_, master_and_subproblem_vars);
//     create_subproblem(problems.subproblems.at(i), base_problem, master_variables, row_division.other_rows, subproblem_variables, master_only_rows);
//     // problems.subproblems.at(i).setOptionValue("presolve", kHighsOffString);
    
//     auto sub_and_master_rows = row_division.inset_only_rows;
//     auto sub_mixed_rows = index_set_intersection(master_mixed_rows, sub_and_master_rows);
//     create_feasibility_subproblem(problems.feas_subproblems.at(i), base_problem, master_variables, sub_mixed_rows, master_only_rows);
//     // problems.feas_subproblems.at(i).setOptionValue("presolve", kHighsOffString);
//   }
// }

std::vector<double> get_master_multipliers(Highs const & subproblem, std::set<HighsInt> const & master_variables) {
  std::ofstream vm("/tmp/v_mults.csv", std::ios::app);
  for (auto & v : subproblem.getSolution().row_dual) vm << v << ",";
  vm << "\n";
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

std::pair<std::vector<double>, double> add_cut(Highs & master, CutData const & cut, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values, CutType cut_type, int subproblem_no) {
  auto old_value_multiple = std::inner_product(cut.master_multipliers.begin(), cut.master_multipliers.end(), master_values.begin(), 0.0);
  auto nonzero_multipliers = create_nonzero_vector(cut.master_multipliers);
  if (cut_type == CutType::Objective) nonzero_multipliers = add_mu_entry(nonzero_multipliers, master_variables.size() + subproblem_no);
  auto rhs  = cut.dual_objective + old_value_multiple;
  add_nonzero_row(master, rhs, kHighsInf, nonzero_multipliers);
  std::ofstream log ("/tmp/cut_log.txt", std::ios_base::app);
  for (int i = 0; i < nonzero_multipliers.number_of_nonzeros; ++i) {
      log << " + " << nonzero_multipliers.nonzero_values.at(i) <<  " x[" << nonzero_multipliers.nonzero_indices.at(i) <<"]"; 
  }
  log << " >= " << rhs << std::endl;

  std::ofstream log2 ("/tmp/cut_log.csv", std::ios_base::app);
  log2 << rhs << " <=";
  for (auto & x : cut.master_multipliers) log2 << "," << x;
  log2 << "\n"; 

  return {cut.master_multipliers, rhs};
}

std::pair<std::vector<double>, double> add_cut(Highs & master, Highs const & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values, CutType cut_type, int subproblem_no) {
  double dual_objective;
  subproblem.getDualObjectiveValue(dual_objective);
  auto multipliers = get_master_multipliers(subproblem, master_variables);
  return add_cut(master, {multipliers, dual_objective}, master_variables, master_values, cut_type, subproblem_no);
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

BendersIterationInfo solve_feasibility_subproblem(Highs & subproblem, BendersIterationInfo info, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  fix_master_variables(subproblem, master_variables, master_values);
  info.was_error = info.was_error || subproblem.run() == HighsStatus::kError ;//|| subproblem.getModelStatus() != HighsModelStatus::kOptimal;
  return info;
}

BendersIterationInfo solve_subproblem(Highs & subproblem, BendersIterationInfo info, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  fix_master_variables(subproblem, master_variables, master_values);
  save_master_variables(subproblem, master_variables, master_values);
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

BendersRet benders(HighsLp & base_problem, std::string const & master_name_pattern, std::vector<double> const & starting_point, double eps) { 
  auto master_variables = discover_master_variables(base_problem.col_names_, master_name_pattern);
  return benders(base_problem, master_variables, starting_point, eps);
}
void save_ubd_lbd(double lbd, double ubd, double acc) {
  std::ofstream("/tmp/bounds.csv", std::ios::app) << lbd << "," << ubd << "," << ubd-lbd
   << "," << (ubd-lbd)/(1 + std::fabs(ubd) + std::fabs(lbd)) << "," << acc << std::endl;
}

void save_iteration_data(double lbd, double ubd, double acc, double feas, bool feasible, double mean_orto, int max_orto_idx, double max_orto) {
  auto status = feasible ? "optimal cut" : "feasibility cut";
  std::ofstream("/tmp/iteration.csv", std::ios::app) 
    << lbd << "," << ubd << "," << ubd-lbd << "," << acc << "," << feas << ","
    << status << "," << mean_orto << "," << max_orto << "," << max_orto_idx << std::endl;
}

void save_iteration_data(double lbd, double ubd, double acc, 
  std::vector<bool> feasible, std::vector<std::tuple<double, int, double>> ortho) {
  std::ofstream file("/tmp/iteration.csv", std::ios::app);
  file << lbd << "," << ubd << "," << ubd-lbd << "," << acc;
  for (int i = 0; i < feasible.size(); ++i) {
    auto status = feasible.at(i) ? "optimal cut" : "feasibility cut";
    auto mean_ortho = std::get<0>(ortho.at(i));
    auto max_ortho_idx = std::get<1>(ortho.at(i));
    auto max_ortho = std::get<2>(ortho.at(i));
    file << "," << status << "," << mean_ortho << "," << max_ortho << "," << max_ortho_idx;
  }    
  file << std::endl; 
}

inline double norm(std::vector<double> const & vec) { 
  return std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.)); 
}

inline double ortho(std::vector<double> const & vec1, std::vector<double> const & vec2) {
  auto delim = norm(vec1) * norm(vec2);
  return delim == 0 ? 
    0 : std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0.) / delim;
}
inline double mean(std::vector<double> const & vec) {
  return vec.size() == 0 ? 0 : std::accumulate(vec.begin(), vec.end(), 0.) / vec.size();
}
inline std::pair<int, double> max_abs(std::vector<double> const & vec) {
  // double max_elem = -INFINITY;
  // int max_elem_idx = 0;
  if (vec.empty()) return {-1, -INFINITY};
  auto max = std::max_element(vec.begin(), vec.end(), [](double x, double y) { return fabs(x) < fabs(y);});
  return {std::distance(vec.begin(), max), *max};
  // for (int i = 0; i < vec.size(); ++i)
  // for (auto const & x : vec) 
  //   max_elem = std::max(max_elem, std::fabs(x));
  // return max_elem;
}

std::tuple<double,int,double> count_ortho(std::vector<std::vector<double>> const & cuts, std::vector<double> const & new_cut) {
  std::vector<double> orthos;
  std::transform(cuts.begin(), cuts.end(), std::back_inserter(orthos), 
    [&new_cut](std::vector<double> const & cut){return ortho(new_cut, cut);});
  auto s1 = orthos.size();
  auto s2 = cuts.size();
  auto max_idx = max_abs(orthos);
  return {mean(orthos), max_idx.first, max_idx.second};
}

void decrease_gap(double & acc, Highs & master, double div) {
    acc = std::max(acc / div, 1e-7);
    master.setOptionValue("optimality_tolerance", acc);
    master.setOptionValue("ipm_optimality_tolerance", acc);
    if (acc <= 1e-7) {
      master.setOptionValue("solver", kSimplexString);
      master.setOptionValue("dual_feasibility_tolerance", 1e-8);
      master.setOptionValue("primal_feasibility_tolerance", 1e-8);
    }
      
}

void decrease_feas(double & feas, Highs & master, double div) {
    feas = std::max(feas / div, 1e-8);
    master.setOptionValue("dual_feasibility_tolerance", feas);
    master.setOptionValue("primal_feasibility_tolerance", feas);
}

BendersRet benders(HighsLp & base_problem, std::set<HighsInt> const & master_variables, std::vector<double> const & starting_point, double subproblem_lb, double eps, int max_iter) { 
  BendersProblems problems;
  decompose_problem(problems, base_problem, master_variables, subproblem_lb);
  // BendersIterationInfo info;
  // solve_master(problems.master, info);
  // if (problems.master.getModelStatus() != HighsModelStatus::kOptimal) 
  //   assert(1 == 0);
  return benders_loop(problems, master_variables, starting_point, eps, max_iter);
}

BendersRet benders_loop(BendersProblems & problems, std::set<HighsInt> const & master_variables, std::vector<double> const & starting_point, double eps, int max_iter) { 
  int no_master_rows = problems.master.getNumRow();
  BendersIterationInfo info;
  auto master_values = starting_point;
  int iter = 0;
  double UBD = kHighsInf, LBD = -kHighsInf;
  bool any_objective_cuts = false;
  //TODO 1st iter infeas
  // TODO verify starting from a master feasible, if no point is passed
  // if (std::all_of(starting_point.begin(), starting_point.end(),[](double v){return v == 0.;})) {
  //   info = solve_master(problems.master, info);
  //   master_values = problems.master.getSolution().col_value;
  // }
  std::vector<int> UBD_updates {};
  std::vector<int> feasible_iters {};
  std::vector<int> recentring_counts {};
  std::vector<int> optim_counts {};
  std::vector<double> pinf {};
  std::vector<double> dinf {};
  master_modifier(problems.master);
  double acc = ipm_acc;
  double feas = ipm_feas;
  double rel_gap;
  double last_gap = INFINITY;
  double mean_orto, max_orto;
  int max_orto_idx;
  std::vector<bool> far_away, far_far_away;
  std::vector<std::vector<double>> feas_cuts, obj_cuts;
  std::vector<double> rhss;
  std::vector<bool> was_feas;
  std::vector<int> type_shifts;
  CsvLogger dist("/tmp/cut_distances.csv");
  CsvLogger ubd_lbd("/tmp/ubd_lbd.csv");
  CsvLogger xs("/tmp/xs.csv");
  CsvLogger duals("/tmp/duals.csv");
  // std::fstream dist("/tmp/cut_distances.csv", std::ios::app);
  // std::fstream ubd_lbd("/tmp/ubd_lbd.csv", std::ios::app);
  // std::fstream xs("/tmp/xs.csv", std::ios::app);
  // std::fstream duals("/tmp/duals.csv", std::ios::app);
  int oscillation_count = 0;
  int d_osci_count = 0;
  int big_oscil_count = 0;
  int streak = 0;
  bool last_optim = false;
  int cut_type_shift = 0;
  while (UBD - LBD > eps && !info.was_error && iter++ < max_iter) {
    bool was_feasible = true;
    info = solve_subproblem(problems.subproblem, info, master_variables, master_values);
    if (info.was_subproblem_feasible) {
      streak++;
      was_feas.push_back(was_feasible);
      // if (any_objective_cuts) decrease_gap(acc, problems.master, 2);
      if (streak > 1) decrease_gap(acc, problems.master, 2.5);
      if (streak > 0) decrease_feas(feas, problems.master, 5);
      double solution_cost = calculate_solution_cost(problems.master, problems.subproblem, master_variables, master_values);
      feasible_iters.push_back(iter);
      if (solution_cost < UBD) UBD_updates.push_back(iter);
      UBD = std::min(UBD, solution_cost);
      if (UBD - LBD <= eps) break;
      auto cut = add_cut(problems.master, problems.subproblem, master_variables, master_values, CutType::Objective);
      auto dat = count_ortho(obj_cuts, cut.first);
      mean_orto = std::get<0>(dat);
      max_orto_idx = std::get<1>(dat);
      max_orto = std::get<2>(dat);
      obj_cuts.push_back(cut.first);
      rhss.push_back(cut.second);
      // if (!any_objective_cuts) {
      //   any_objective_cuts = true;
      //   unfreeze_mu(problems.master, master_variables);
      // }
    }
    else {
      streak = 0;
      was_feasible = false;
      was_feas.push_back(was_feasible);
      info = solve_feasibility_subproblem(problems.feas_subproblem, info, master_variables, master_values);
      auto cut = add_cut(problems.master, problems.feas_subproblem, master_variables, master_values, CutType::Feasibility);
      auto dat = count_ortho(feas_cuts, cut.first);
      mean_orto = std::get<0>(dat);
      max_orto_idx = std::get<1>(dat);
      max_orto = std::get<2>(dat);
      feas_cuts.push_back(cut.first);
      rhss.push_back(cut.second);
    }
    if (iter > 1 && !was_feasible && last_optim)
      cut_type_shift = 1;
    else if (iter > 1 && was_feasible && !last_optim)
      cut_type_shift = 2;
    else cut_type_shift = 0;
    type_shifts.push_back(cut_type_shift);
    last_optim = was_feasible;
    if (std::isfinite(max_orto) && std::fabs(max_orto) > 0.99) {
      decrease_gap(acc, problems.master,  streak > 1 ? 2.5 : 10);
      // decrease_gap(acc, problems.master,  was_feasible ? 5 : 10);
    }
    // if (info.was_error) break;
    info = solve_master(problems.master, info);
    recentring_counts.push_back(recentring_count);
    optim_counts.push_back(optim_count);
    pinf.push_back(primal_feasibility);
    dinf.push_back(dual_feasibility);
    if (problems.master.getModelStatus() != HighsModelStatus::kOptimal)
      assert(1 == 0);
    master_values = problems.master.getSolution().col_value;
    xs << cut_type_shift << master_values;
    // for (auto x : master_values)
    //   xs << "," << x;
    // xs << "\n";
    auto d = problems.master.getSolution().row_dual;
    // if (any_objective_cuts)
      problems.master.getDualObjectiveValue(LBD);
    save_iteration_data(LBD, UBD, acc, feas, was_feasible, mean_orto, max_orto_idx, max_orto);
    ubd_lbd << UBD << LBD << was_feasible; ubd_lbd.newline();
    // ubd_lbd << UBD << "," << LBD << "," << was_feasible << "\n"; 
    auto row_sol = problems.master.getSolution().row_value;
        std::vector<int> ind(problems.master.getNumCol());
    std::vector<double> val(problems.master.getNumCol(), 0);
    int num_nz;
    far_away.push_back(false);
    for (int i = 0; i < problems.master.getNumRow() - no_master_rows; ++i) {
        problems.master.getModel().lp_.a_matrix_.getRow(no_master_rows + i, num_nz, ind.data(), val.data());
        double norm_a = norm(val);
        // double norm = 0;
        // for (int i = 0; i < num_nz; ++i) norm += val.at(i) * val.at(i);
        // norm = std::sqrt(norm);
        auto distance = (row_sol.at(no_master_rows + i) - rhss.at(i)) / norm_a;
        bool oscillation = far_away.at(i) && distance < 1 && d.at(no_master_rows + i) * norm_a > 1e-4; 
        dist.log_with_note(std::round(100 * distance) / 100., oscillation ? "(OSC)" : "");
        duals.log_with_note(d.at(no_master_rows + i) * norm_a, (oscillation ? "(OSC)" : ""));
        far_away.at(i) = distance >= 1;  
        oscillation_count += oscillation;
      }
    dist.newline();
    duals.newline();
  }
  xs.newline();
  duals.newline();
  dist.log_predicate(was_feas, "obj", "feas");
  // for (int i = 0; i < problems.master.getNumRow() - no_master_rows; ++i) {
  //     dist << (was_feas.at(i) ? "obj" : "feas");
  // }
  dist << type_shifts << oscillation_count;
  dist.newline();
  ubd_lbd.newline();
  return {UBD, iter, UBD_updates, feasible_iters, recentring_counts, optim_counts, pinf, dinf};
}

inline bool all(std::vector<bool> const & v) { return std::all_of(v.begin(), v.end(), [](bool x) { return x; }); }

// BendersRet multi_benders(HighsLp & base_problem, std::set<HighsInt> const & master_variables,
//                      std::vector<std::set<HighsInt>> const & subproblems_variables,
//                      std::vector<double> const & starting_point, double subproblem_lb, double eps, int max_iter) { 
//   MultiBendersProblems problems;
//   decompose_problem(problems, base_problem, master_variables, subproblems_variables, subproblem_lb);
//   return multi_benders_loop(problems, master_variables, subproblems_variables, starting_point, eps, max_iter);
// }

BendersRet multi_benders_loop(MultiBendersProblems & problems, std::set<HighsInt> const & master_variables,
                     std::vector<double> const & starting_point, double eps, int max_iter) { 
  BendersIterationInfo info;
  auto master_values = starting_point;
  int iter = 0;
  double UBD = kHighsInf, LBD = -kHighsInf;
  int no_subproblems = problems.subproblems.size();
  // std::vector<bool> any_objective_cuts(no_subproblems, false);
  // bool all_objective_cuts = false;
  std::vector<int> UBD_updates {};
  std::vector<int> feasible_iters {};
  std::vector<int> recentring_counts {};
  std::vector<int> optim_counts {};
  std::vector<double> pinf {};
  std::vector<double> dinf {};

  // std::vector<std::vector<std::vector<double>>> obj_cuts (subproblems_variables.size());
  // std::vector<std::vector<std::vector<double>>> feas_cuts (subproblems_variables.size());
  // std::vector<std::tuple<double, int, double>> ortho (subproblems_variables.size());
  std::vector<bool> was_feasible (no_subproblems);
  master_modifier(problems.master);
  double acc = ipm_acc;
  int oscillation_count = 0;
  bool was_all_feas = false;
  auto no_master_rows = problems.master.getNumRow();
  std::vector<double> rhss;
  std::vector<bool> was_feas;
  std::vector<bool> far_away;
  std::fstream ubd_lbd("/tmp/ubd_lbd.csv", std::ios::app);
  std::fstream dist("/tmp/cut_distances.csv", std::ios::app);
  std::fstream xs("/tmp/xs.csv", std::ios::app);
  std::fstream duals("/tmp/duals.csv", std::ios::app);
  while (UBD - LBD > eps && !info.was_error && iter++ < max_iter) {
    int feas_count = 0;
    double subproblem_costs = 0;
    bool all_feasible = true;
    for (int i = 0; i < no_subproblems; ++i) { 
      auto & subproblem = problems.subproblems.at(i);
      info = solve_subproblem(subproblem, info, master_variables, master_values);
      if (info.was_subproblem_feasible) {
        feas_count++;
        // was_feasible.at(i) = true;
        // if (!any_objective_cuts[i]) {
        //   any_objective_cuts[i] = true;
        //   unfreeze_mu(problems.master, master_variables, i);
        // }
        subproblem_costs += subproblem.getObjectiveValue();
        auto cut = add_cut(problems.master, subproblem, master_variables, master_values, CutType::Objective, i);
        // rhss.push_back(cut.second);
        was_feas.push_back(true);
        // ortho.at(i) =  count_ortho(obj_cuts.at(i), cut.first);
        // obj_cuts.at(i).push_back(cut.first);
      }
      else {
        // was_feasible.at(i) = false;
        all_feasible = false;
        auto & feas_subproblem = problems.feas_subproblems.at(i);
        info = solve_feasibility_subproblem(feas_subproblem, info, master_variables, master_values);
        auto cut = add_cut(problems.master, feas_subproblem, master_variables, master_values, CutType::Feasibility, i);
        // rhss.push_back(cut.second);
        // was_feas.push_back(false);
        // ortho.at(i) =  count_ortho(feas_cuts.at(i), cut.first);
        // feas_cuts.at(i).push_back(cut.first);
        // auto cut = solve_feasibility_subproblem(subproblem, master_variables);
        // add_cut(problems.master, cut, master_variables, master_values, CutType::Feasibility, i);
      }
    }
    // if (info.was_error) break;
    if (all_feasible) {
        if (was_all_feas) decrease_gap(acc, problems.master, 2.5);
        was_all_feas = true;
        feasible_iters.push_back(iter);
        double solution_cost = calculate_solution_cost(problems.master, subproblem_costs, master_variables, master_values);
        if (solution_cost < UBD)
          UBD_updates.push_back(iter);
        UBD = std::min(UBD, solution_cost);
        if (UBD - LBD <= eps) break;
        // if (all_parallel) break;
    }
    
    // bool all_parallel = true;
    // for (auto const & par : ortho)
      // if (!std::isfinite(std::get<2>(par)) || std::fabs(std::get<2>(par)) <= 0.99)
        // all_parallel = false;
    // if (all_parallel) decrease_gap(acc, problems.master, all_feasible ? 4 : 10);

    info = solve_master(problems.master, info);
    recentring_counts.push_back(recentring_count);
    optim_counts.push_back(optim_count);
    pinf.push_back(primal_feasibility);
    dinf.push_back(dual_feasibility);
    if (problems.master.getModelStatus() != HighsModelStatus::kOptimal)
      assert(1 == 0);
    master_values = problems.master.getSolution().col_value;
    for (auto x : master_values)
      xs << "," << x;
    xs << "\n";
    // if((all_objective_cuts = all_objective_cuts || all(any_objective_cuts)))
      problems.master.getDualObjectiveValue(LBD);
    ubd_lbd << UBD << "," << LBD << "," << feas_count << "\n"; 
      // LBD = problems.master.getObjectiveValue();
    // save_iteration_data(LBD, UBD, acc, was_feasible, ortho);
    // auto row_sol = problems.master.getSolution().row_value;
    // std::vector<int> ind(problems.master.getNumCol());
    // std::vector<double> val(problems.master.getNumCol(), 0);
    // int num_nz;
    // auto d = problems.master.getSolution().row_dual;
    // for (int i = 0; i < no_subproblems; ++i) far_away.push_back(false);
    // for (int i = 0; i < problems.master.getNumRow() - no_master_rows; ++i) {
    //     problems.master.getModel().lp_.a_matrix_.getRow(no_master_rows + i, num_nz, ind.data(), val.data());
    //     double norm_a = norm(val);
    //     // for (int i = 0; i < num_nz; ++i) norm += val.at(i) * val.at(i);
    //     // norm = std::sqrt(norm);
    //     auto distance = (row_sol.at(no_master_rows + i) - rhss.at(i)) / norm_a;
    //     bool oscillation = far_away.at(i) && distance < 1 && d.at(no_master_rows + i) * norm_a > 1e-4; 
    //     oscillation_count += oscillation;
    //     dist << std::round(100 * distance) / 100. << (oscillation ? "(OSC)" : "") << ",";
    //     far_away.at(i) = distance >= 1;   
    //     duals << d.at(no_master_rows + i) * norm_a << (oscillation ? "(OSC)" : "") << ",";
    //   }
    // dist << "\n";
    // duals << "\n";
  }
  // for (int i = 0; i < problems.master.getNumRow() - no_master_rows; ++i) {
  //     dist << (was_feas.at(i) ? "obj" : "feas") << ",";
  // }
  dist << "\n" << oscillation_count << "\n";
  ubd_lbd << "\n";
  xs << "\n";
  return {UBD, iter, UBD_updates, feasible_iters, recentring_counts, optim_counts, pinf, dinf};
}

// BendersRet benders2(HighsLp & base_problem, std::set<HighsInt> const & master_variables, std::vector<double> const & starting_point, double subproblem_lb, double eps) {
//   auto subproblem_variables = sequence_complement(master_variables, base_problem.num_col_);
//   return multi_benders(base_problem, master_variables, {subproblem_variables}, starting_point, subproblem_lb, eps);
// } 

// BendersRet benders2(HighsLp & base_problem, std::string const & master_name_pattern, std::vector<double> const & starting_point, double subproblem_lb, double eps) { 
//   auto master_variables = discover_master_variables(base_problem.col_names_, master_name_pattern);
//   return benders2(base_problem, master_variables, starting_point, subproblem_lb, eps);
// }

BendersRet benders_l_shaped(SmpsCoreStructure & core, StochasticTree & tree, 
  std::set<HighsInt> const & master_variables, std::vector<double> const & starting_point, double subproblem_lb, double eps, int max_iter) {
  assert(core.is_valid() && tree.root != nullptr && core.stage_submatrix.size() == 2 && tree.root->get_no_children() == 1);
  auto & stage_1st = tree.root->get_child(0);
  int no_subproblems = stage_1st->get_no_children();

  MultiBendersProblems problems;
  auto row_division = divide_rows(core.a_matrix_, master_variables);
  auto master_mixed_rows = row_division.mixed_rows;
  auto master_only_rows = row_division.diff_other;
  auto & master = problems.master;
  auto master_lp = modify_problem(core, *stage_1st);
  create_master_problem(master, master_lp, master_variables, row_division.other_rows, subproblem_lb, no_subproblems);
  master_modifier(master);

  problems.subproblems = std::vector<Highs> (no_subproblems);
  problems.feas_subproblems = std::vector<Highs> (no_subproblems);
  for (int i = 0; i < no_subproblems; ++i) {
    auto & sub = problems.subproblems.at(i);
    auto & feas_sub = problems.feas_subproblems.at(i);
    auto & stage_2nd = stage_1st->get_child(i);
    auto base_sub_lp = modify_problem(core, *stage_2nd);
    create_subproblem(sub, base_sub_lp, master_variables, master_only_rows);
    create_feasibility_subproblem(feas_sub, base_sub_lp, master_variables, master_mixed_rows, master_only_rows);
  }
  
  return multi_benders_loop(problems, master_variables, starting_point, eps, max_iter);

  // decompose_problem(problems, base_problem, master_variables, subproblems_variables, subproblem_lb);
  // return multi_benders_loop(problems, master_variables, subproblems_variables, starting_point, eps, max_iter);
  }