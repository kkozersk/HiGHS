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
#include "io/HMPSIO.h"


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
  problem.changeColBounds(variable_index, value, value);
}

bool fix_master_variables(Highs & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  // int master_index = 0;
  subproblem.changeColsBounds(
    master_variables.size(), 
    set_to_vector(master_variables).data(),
    master_values.data(),
    master_values.data()
  );
  // for (auto subproblem_index : master_variables)
  //   fix_variable(subproblem, subproblem_index, master_values.at(master_index++));
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
}

std::set<HighsInt> sequence_complement(std::set<HighsInt> const & set, HighsInt max_number) {
  std::set<HighsInt> complement;
  for (HighsInt i = 0; i < max_number; i++)
    if (set.count(i) == 0)
      complement.insert(i);
  return complement;
}

void create_master_problem(Highs & master, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & subproblem_rows, double subproblem_lb, int no_subproblems) {
  master.passModel(base_problem);
  auto nonmaster_rows = set_to_vector(subproblem_rows);
  master.deleteRows(nonmaster_rows.size(), nonmaster_rows.data());
  auto subproblem_variables = set_to_vector(sequence_complement(master_variables, base_problem.num_col_)); // TODO: should it be here?
  master.deleteCols(subproblem_variables.size(), subproblem_variables.data());
  std::vector<double> ones(no_subproblems, 1);
  std::vector<double> lb(no_subproblems, subproblem_lb);
  std::vector<double> ub(no_subproblems, kHighsInf);
  int original_col_count = master.getNumCol();
  master.addCols(no_subproblems, ones.data(), lb.data(), ub.data(), 0, nullptr, nullptr, nullptr);
  for (int i = 0; i < no_subproblems; ++i) {
    master.passColName(original_col_count + i, std::string("BENDMU")+std::to_string(i));
  }
  // master.setOptionValue("output_flag", false);
  // master.setOptionValue("log_to_console", false);

  // for (int i = 0; i < no_subproblems; ++i)
  //   master.addCol(1, subproblem_lb, kHighsInf, 0, nullptr, nullptr);
}

void create_subproblem(Highs & subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & master_only_rows) {
  subproblem.passModel(base_problem);
  subproblem.changeObjectiveOffset(0);
  auto nonsub_variables = set_to_vector(master_variables);
  std::vector<double> zeros (master_variables.size(), 0);
  subproblem.changeColsCost(master_variables.size(), nonsub_variables.data(), zeros.data());
  std::vector<int> to_delete = set_to_vector(master_only_rows);
  subproblem.deleteRows(to_delete.size(), to_delete.data());
  subproblem.setOptionValue("output_flag", false);
  subproblem.setOptionValue("log_to_console", false);
}

// void add_nonzero_col(Highs & problem, double col_cost, double col_lower, double col_upper, NonZeroVector const & col_vector) {
//   problem.addCol(col_cost, 0, kHighsInf, col_vector.number_of_nonzeros, col_vector.nonzero_indices.data(), col_vector.nonzero_values.data());
// }

void create_feasibility_subproblem(Highs & feas_subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & mixed_rows, std::set<HighsInt> const & master_only_rows) {
  auto a = construct_extension_matrix(base_problem, mixed_rows);
  create_feasibility_subproblem(feas_subproblem, base_problem, master_variables, mixed_rows, master_only_rows, a);
  // feas_subproblem.passModel(base_problem);
  // feas_subproblem.changeObjectiveOffset(0);
  // std::vector<double> zeros(base_problem.num_col_, 0);
  // feas_subproblem.changeColsCost(0, base_problem.num_col_ - 1, zeros.data());
  // HighsSparseMatrix a; a.num_row_ = base_problem.num_row_;
  // int counter = 0;
  // int row;
  // double row_val;
  // for (auto i : mixed_rows) {
  //   if (base_problem.row_lower_.at(i) > -kHighsInf) {
  //     counter++;
  //     row = i;
  //     row_val = 1;
  //     a.addVec(1, &row, &row_val);
  //   }
  //   if (base_problem.row_upper_.at(i) < kHighsInf) {
  //     counter++;
  //     row = i;
  //     row_val = -1;
  //     a.addVec(1, &row, &row_val);   
  //   }
  // }
  // std::vector<double> costs(counter, 1);
  // std::vector<double> lb(counter, 0);
  // std::vector<double> ub(counter, kHighsInf);
  // feas_subproblem.addCols(counter, costs.data(), lb.data(), ub.data(),
  //   a.numNz(), a.start_.data(), a.index_.data(), a.value_.data()
  // );
  // std::vector<int> to_delete = set_to_vector(master_only_rows);
  // feas_subproblem.deleteRows(to_delete.size(), to_delete.data());
}

HighsSparseMatrix construct_extension_matrix(HighsLp const & base_problem, std::set<HighsInt> const & mixed_rows) {
  HighsSparseMatrix a; 
  a.num_row_ = base_problem.num_row_;
  double row_val;
  for (auto i : mixed_rows) {
    if (base_problem.row_lower_.at(i) > -kHighsInf) {
      row_val = 1;
      a.addVec(1, &i, &row_val);
    }
    if (base_problem.row_upper_.at(i) < kHighsInf) {
      row_val = -1;
      a.addVec(1, &i, &row_val);   
    }
  }
  return a;
}

void create_feasibility_subproblem(Highs & feas_subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & mixed_rows,  std::set<HighsInt> const & master_only_rows, 
  HighsSparseMatrix const & extension_matrix) {
  
    // return;
      feas_subproblem.passModel(base_problem);
  feas_subproblem.changeObjectiveOffset(0);
  std::vector<double> zeros(base_problem.num_col_, 0);
  feas_subproblem.changeColsCost(0, base_problem.num_col_ - 1, zeros.data());
  std::vector<double> costs(extension_matrix.num_col_, 1);
  std::vector<double> lb(extension_matrix.num_col_, 0);
  std::vector<double> ub(extension_matrix.num_col_, kHighsInf);
  feas_subproblem.addCols(extension_matrix.num_col_, costs.data(), lb.data(), ub.data(),
    extension_matrix.numNz(), extension_matrix.start_.data(), extension_matrix.index_.data(), extension_matrix.value_.data()
  );
  std::vector<int> to_delete = set_to_vector(master_only_rows);
  feas_subproblem.deleteRows(to_delete.size(), to_delete.data());
  feas_subproblem.setOptionValue("output_flag", false);
  feas_subproblem.setOptionValue("log_to_console", false);
}

//TODO sub and feas sub have too many rows and columns
void decompose_problem(BendersProblems & problems, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, RowDivision const & row_division, double subproblem_lb) {
  create_master_problem(problems.master, base_problem, master_variables, row_division.subproblem_rows, subproblem_lb);
  create_subproblem(problems.subproblem, base_problem, master_variables, row_division.master_only_rows);
  create_feasibility_subproblem(problems.feas_subproblem, base_problem, master_variables, row_division.mixed_rows, row_division.master_only_rows);
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

void add_nonzero_row(Highs & problem, double lower, double upper, NonZeroVector const & row_vector, std::string const & name) {
  problem.addRow(lower, upper, row_vector.number_of_nonzeros, row_vector.nonzero_indices.data(), row_vector.nonzero_values.data());
  if (name != "") problem.passRowName(problem.getNumRow() - 1, name);
}

Cut form_cut(Highs const & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values, CutType cut_type, int subproblem_no) {
  double dual_objective;
  subproblem.getDualObjectiveValue(dual_objective);
  auto multipliers = get_master_multipliers(subproblem, master_variables);
  auto old_value_multiple = std::inner_product(multipliers.begin(), multipliers.end(), master_values.begin(), 0.0);
  auto nonzero_multipliers = create_nonzero_vector(multipliers);
  if (cut_type == CutType::Objective) nonzero_multipliers = add_mu_entry(nonzero_multipliers, master_variables.size() + subproblem_no);
  auto rhs  = dual_objective + old_value_multiple;

  return {nonzero_multipliers, rhs};
}

// std::pair<std::vector<double>, double> add_cut(Highs & master, CutData const & cut, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values, CutType cut_type, int subproblem_no) {
//   auto old_value_multiple = std::inner_product(cut.master_multipliers.begin(), cut.master_multipliers.end(), master_values.begin(), 0.0);
//   auto nonzero_multipliers = create_nonzero_vector(cut.master_multipliers);
//   if (cut_type == CutType::Objective) nonzero_multipliers = add_mu_entry(nonzero_multipliers, master_variables.size() + subproblem_no);
//   auto rhs  = cut.dual_objective + old_value_multiple;
//   add_nonzero_row(master, rhs, kHighsInf, nonzero_multipliers);
//   CsvLogger log("/tmp/cut_log.csv");
//   log.log_with_note(rhs, " <=") << cut.master_multipliers;
//   return {cut.master_multipliers, rhs};
// }

// std::pair<std::vector<double>, double> add_cut(Highs & master, Highs const & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values, CutType cut_type, int subproblem_no) {
//   double dual_objective;
//   subproblem.getDualObjectiveValue(dual_objective);
//   auto multipliers = get_master_multipliers(subproblem, master_variables);
//   return add_cut(master, {multipliers, dual_objective}, master_variables, master_values, cut_type, subproblem_no);
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

BendersIterationInfo solve_feasibility_subproblem(Highs & subproblem, BendersIterationInfo info, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  // assert(1==0);
  fix_master_variables(subproblem, master_variables, master_values);
  auto start = subproblem.getRunTime();
  info.was_error = info.was_error || subproblem.run() == HighsStatus::kError ;//|| subproblem.getModelStatus() != HighsModelStatus::kOptimal;
  auto end =  subproblem.getRunTime();
  info.sub_time += end - start;
  return info;
}

BendersIterationInfo solve_subproblem(Highs & subproblem, BendersIterationInfo info, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  fix_master_variables(subproblem, master_variables, master_values);
  save_master_variables(subproblem, master_variables, master_values);
  auto start = subproblem.getRunTime();
  info.was_error = info.was_error || subproblem.run() == HighsStatus::kError;
  auto end =  subproblem.getRunTime();
  info.sub_time += end - start;
  info.was_subproblem_feasible = subproblem.getModelStatus() == HighsModelStatus::kOptimal;
  // CsvLogger("/tmp/subvars.csv") << subproblem.getSolution().col_value;
  return info;
}

BendersIterationInfo solve_master(Highs & master, BendersIterationInfo info) {
  auto start = master.getRunTime();
  info.was_error = info.was_error || master.run() == HighsStatus::kError;
  auto end =  master.getRunTime();
  info.master_time += end - start;
  return info;
}

double calculate_solution_cost(Highs const & master, double subproblem_cost, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  auto const & master_costs = master.getLp().col_cost_;
  int no_master_vars = master_variables.size();
  double master_cost = std::inner_product(master_values.begin(), master_values.begin() + no_master_vars, master_costs.begin(), 0.);
  double offset; master.getObjectiveOffset(offset);
  return subproblem_cost + master_cost + offset;
}

double calculate_solution_cost(std::vector<double> const & master_costs, double offset, double subproblem_cost, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  int no_master_vars = master_variables.size();
  double master_cost = std::inner_product(master_values.begin(), master_values.begin() + no_master_vars, master_costs.begin(), 0.);
  return subproblem_cost + master_cost + offset;
}

double calculate_solution_cost(Highs const & master, Highs const & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  return calculate_solution_cost(master, subproblem.getObjectiveValue(), master_variables, master_values);
}
//TODO: Fix!
// BendersRet benders(HighsLp & base_problem, std::string const & master_name_pattern, std::vector<double> const & starting_point, double eps, std::vector<OptionValue> const & masterOptions) { 
//   auto master_variables = discover_master_variables(base_problem.col_names_, master_name_pattern);
//   return benders(base_problem, master_variables, starting_point);
// }

void save_ubd_lbd(double lbd, double ubd, double acc) {
  CsvLogger log("/tmp/bounds.csv");
  log << lbd << ubd << ubd-lbd << (ubd-lbd)/(1 + std::fabs(ubd) + std::fabs(lbd)) << acc;
  log.newline();
}

void save_iteration_data(double lbd, double ubd, double acc, double feas, bool feasible, double mean_orto, int max_orto_idx, double max_orto) {
  auto status = feasible ? "optimal cut" : "feasibility cut";
  CsvLogger log("/tmp/iteration.csv");
  log << lbd << ubd << ubd-lbd << acc << feas << status << mean_orto << max_orto << max_orto_idx;
  log.newline();
}

void save_iteration_data(double lbd, double ubd, double acc, 
  std::vector<bool> feasible, std::vector<std::tuple<double, int, double>> ortho) {
  CsvLogger log("/tmp/iteration.csv");

  log << lbd << ubd << ubd-lbd << acc;
  for (int i = 0; i < feasible.size(); ++i) {
    std::string status = feasible.at(i) ? "optimal cut" : "feasibility cut";
    auto mean_ortho = std::get<0>(ortho.at(i));
    auto max_ortho_idx = std::get<1>(ortho.at(i));
    auto max_ortho = std::get<2>(ortho.at(i));
    log << status  << mean_ortho << max_ortho << max_ortho_idx;
  }  
  log.newline();  
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
  if (vec.empty()) return {-1, -INFINITY};
  auto max = std::max_element(vec.begin(), vec.end(), [](double x, double y) { return fabs(x) < fabs(y);});
  return {std::distance(vec.begin(), max), *max};
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

void tighter(MasterAdaptationParams & params, Highs & master) {
  params.no_optim_steps = std::min(params.no_optim_steps + params.delta_steps, params.max_steps);
  master.setOptionValue("ipm_iteration_limit", params.no_optim_steps);
  // if (params.no_optim_steps >= params.max_steps) {
  //     master.setOptionValue("solver", kSimplexString);
  //     master.setOptionValue("dual_feasibility_tolerance", 1e-8);
  //     master.setOptionValue("primal_feasibility_tolerance", 1e-8);
  //   }
      
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

std::vector<double> quick_master_solve(Highs & master) {
  master.run();
  auto st = master.getModelStatus();
  CsvLogger ("/tmp/xs.csv") << master.getSolution().col_value;
  // TOOD -- asssert on error?
  return master.getSolution().col_value;
}

// void modify_master(Highs & master) {
    
//     master.setOptionValue("solver", used_solver);
//     master.setOptionValue("optimality_tolerance", ipm_acc);
//     master.setOptionValue("ipm_optimality_tolerance", ipm_acc);
//     master.setOptionValue("run_crossover", kHighsOffString);
//     master.setOptionValue("max_centring_steps", 1000);
//     master.setOptionValue("presolve", kHighsOnString);
//     master.setOptionValue("centring_gamma", 1-1e-5);
//     master.setOptionValue("dual_feasibility_tolerance", ipm_feas);
//     master.setOptionValue("primal_feasibility_tolerance", ipm_feas);
// }

BendersRet benders(HighsLp & base_problem, std::set<HighsInt> const & master_variables, std::vector<double> const & starting_point, double subproblem_lb,
   double eps, int max_iter, std::vector<OptionValue> const & masterOptions, MasterAdaptationParams params) { 
  BendersProblems problems;
  decompose_problem(problems, base_problem, master_variables, subproblem_lb);
  // modify_master(problems.master);
  apply_options(problems.master, masterOptions);
  return benders_loop(problems, master_variables, 
    starting_point.empty() ? quick_master_solve(problems.master) : starting_point, eps, max_iter, params);
}

std::vector<double> unravel(NonZeroVector const & vec) {
  if (vec.number_of_nonzeros == 0) return {};
  auto max_idx = *std::max_element(vec.nonzero_indices.begin(), vec.nonzero_indices.end());
  std::vector<double> result(1+max_idx, 0);
  for (int i = 0; i < vec.number_of_nonzeros; ++i)
    result.at(vec.nonzero_indices.at(i)) = vec.nonzero_values.at(i);
  return result;
}

BendersRet benders_loop(BendersProblems & problems, std::set<HighsInt> const & master_variables,
   std::vector<double> const & starting_point, double eps, int max_iter, MasterAdaptationParams params) { 
  int no_master_rows = problems.master.getNumRow();
  BendersIterationInfo info;
  auto master_values = starting_point;
  int iter = 0;
  double UBD = kHighsInf, LBD = -kHighsInf;
  bool any_objective_cuts = false;
  std::vector<int> UBD_updates {};
  std::vector<int> feasible_iters {};
  std::vector<int> recentring_counts {};
  std::vector<int> optim_counts {};
  double acc = params.ipm_acc;
  double feas = params.ipm_feas;
  double rel_gap;
  double last_gap = INFINITY;
  double mean_orto, max_orto;
  int max_orto_idx;
  std::vector<bool> far_away, far_far_away;
  std::vector<std::vector<double>> feas_cuts, obj_cuts;
  std::vector<double> rhss;
  std::vector<bool> was_feas;
  // std::vector<int> type_shifts;
  CsvLogger dist("/tmp/cut_distances.csv");
  CsvLogger ubd_lbd("/tmp/ubd_lbd.csv");
  CsvLogger xs("/tmp/xs.csv");
  CsvLogger duals("/tmp/duals.csv");
  // int oscillation_count = 0;
  // int d_osci_count = 0;
  // int big_oscil_count = 0;
  int streak = 0;
  bool last_optim = false;
  bool first_optim = false;
  // int cut_type_shift = 0;
  while (UBD - LBD > eps && !info.was_error && iter++ < max_iter) {
    bool was_feasible = true;
    info = solve_subproblem(problems.subproblem, info, master_variables, master_values);
    if (info.was_subproblem_feasible) {
      if (iter == 1) first_optim = true;
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
      // CsvLogger("/tmp/v_mults.csv") << problems.subproblem.getSolution().row_dual;
      auto cut = form_cut(problems.subproblem, master_variables, master_values, CutType::Objective);
      add_nonzero_row(problems.master, cut.rhs, kHighsInf, cut.coefficients);
      // auto cut = add_cut(problems.master, problems.subproblem, master_variables, master_values, CutType::Objective);
      auto un = unravel(cut.coefficients);
      auto dat = count_ortho(obj_cuts, un);
      mean_orto = std::get<0>(dat);
      max_orto_idx = std::get<1>(dat);
      max_orto = std::get<2>(dat);
      obj_cuts.push_back(un);
      rhss.push_back(cut.rhs);
      CsvLogger("/tmp/aggregate2.csv") << cut.rhs << un;
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
      // CsvLogger("/tmp/v_mults.csv") << problems.subproblem.getSolution().row_dual;
      auto cut = form_cut(problems.feas_subproblem, master_variables, master_values, CutType::Feasibility);
      add_nonzero_row(problems.master, cut.rhs, kHighsInf, cut.coefficients);
      // auto cut = add_cut(problems.master, problems.feas_subproblem, master_variables, master_values, CutType::Feasibility);
      auto un = unravel(cut.coefficients);
      auto dat = count_ortho(feas_cuts, un);
      mean_orto = std::get<0>(dat);
      max_orto_idx = std::get<1>(dat);
      max_orto = std::get<2>(dat);
      feas_cuts.push_back(un);
      rhss.push_back(cut.rhs);
      CsvLogger("/tmp/aggregate2.csv") << cut.rhs << un;
      
    }
    // if (iter > 1 && !was_feasible && last_optim)
    //   cut_type_shift = 1;
    // else if (iter > 1 && was_feasible && !last_optim)
    //   cut_type_shift = 2;
    // else cut_type_shift = 0;
    // type_shifts.push_back(cut_type_shift);
    last_optim = was_feasible;
    if (std::isfinite(max_orto) && std::fabs(max_orto) > 0.99) {
      // decrease_gap(acc, problems.master,  streak > 1 ? 2.5 : 10);
      // decrease_gap(acc, problems.master,  was_feasible ? 5 : 10);
    }
    // if (info.was_error) break;
    info = solve_master(problems.master, info);
    recentring_counts.push_back(recentring_count);
    optim_counts.push_back(optim_count);
    if (problems.master.getModelStatus() != HighsModelStatus::kOptimal && problems.master.getModelStatus() != HighsModelStatus::kUnknown)
      assert(1 == 0);
    master_values = problems.master.getSolution().col_value;
    // xs << cut_type_shift << master_values;
    xs << master_values;
    auto d = problems.master.getSolution().row_dual;
    problems.master.getDualObjectiveValue(LBD);
    save_iteration_data(LBD, UBD, acc, feas, was_feasible, mean_orto, max_orto_idx, max_orto);
    ubd_lbd << UBD << LBD << was_feasible; ubd_lbd.newline();
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
        // oscillation_count += oscillation;
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
  // dist << type_shifts << oscillation_count;
  dist.newline();
  ubd_lbd.newline();
  return {UBD-LBD, UBD, info, iter, first_optim, UBD_updates, feasible_iters, recentring_counts, optim_counts, params};
}

inline bool all(std::vector<bool> const & v) { return std::all_of(v.begin(), v.end(), [](bool x) { return x; }); }

BendersRet multi_benders_loop(MultiBendersProblems & problems, std::set<HighsInt> const & master_variables,
                     std::vector<double> const & starting_point, double eps, int max_iter, MasterAdaptationParams params) { 
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

  // std::vector<std::vector<std::vector<double>>> obj_cuts (subproblems_variables.size());
  // std::vector<std::vector<std::vector<double>>> feas_cuts (subproblems_variables.size());
  // std::vector<std::tuple<double, int, double>> ortho (subproblems_variables.size());
  std::vector<bool> was_feasible (no_subproblems);
  double acc = params.ipm_acc;
  double feas = params.ipm_feas;
  int oscillation_count = 0;
  bool was_all_feas = false;
  auto no_master_rows = problems.master.getNumRow();
  std::vector<double> rhss;
  std::vector<bool> was_feas;
  std::vector<bool> far_away;
  CsvLogger ubd_lbd("/tmp/ubd_lbd.csv");
  CsvLogger dist("/tmp/cut_distances.csv");
  CsvLogger xs("/tmp/xs.csv");
  CsvLogger duals("/tmp/duals.csv");
  std::vector<double> new_rhs(no_subproblems);
  std::vector<double> cut_ubs(no_subproblems, kHighsInf);
  HighsSparseMatrix new_cuts;
  bool first_optim = false;
  while (UBD - LBD > eps && !info.was_error && iter++ < max_iter) {
    zerovec(new_rhs);
    new_cuts.clear();
    new_cuts.ensureRowwise();
    new_cuts.num_col_ = problems.master.getNumCol(); 
    int feas_count = 0;
    double subproblem_costs = 0;
    bool all_feasible = true;
    for (int i = 0; i < no_subproblems; ++i) { 
      auto & subproblem = problems.subproblems.at(i);
      info = solve_subproblem(subproblem, info, master_variables, master_values);
      if (info.was_subproblem_feasible) {
        feas_count++;
        subproblem_costs += subproblem.getObjectiveValue();
        // CsvLogger("/tmp/v_mults.csv") << subproblem.getSolution().row_dual;
        auto cut = form_cut(subproblem, master_variables, master_values, CutType::Objective, i);
        new_rhs.at(i) = cut.rhs;
        new_cuts.addVec(cut.coefficients.number_of_nonzeros, cut.coefficients.nonzero_indices.data(), cut.coefficients.nonzero_values.data());
        was_feas.push_back(true);
      }
      else {
        all_feasible = false;
        auto & feas_subproblem = problems.feas_subproblems.at(i);
        info = solve_feasibility_subproblem(feas_subproblem, info, master_variables, master_values);
        // CsvLogger("/tmp/v_mults.csv") << feas_subproblem.getSolution().row_dual;
        auto cut = form_cut(feas_subproblem, master_variables, master_values, CutType::Feasibility, i);
        new_rhs.at(i) = cut.rhs;
        new_cuts.addVec(cut.coefficients.number_of_nonzeros, cut.coefficients.nonzero_indices.data(), cut.coefficients.nonzero_values.data());
      }
    }
    if (all_feasible) {
        if (iter == 1) first_optim = true;
        if (was_all_feas) decrease_gap(acc, problems.master, 2.5);
        decrease_feas(feas, problems.master, 5);
        was_all_feas = true;
        feasible_iters.push_back(iter);
        double solution_cost = calculate_solution_cost(problems.master, subproblem_costs, master_variables, master_values);
        if (solution_cost < UBD)
          UBD_updates.push_back(iter);
        UBD = std::min(UBD, solution_cost);
        if (UBD - LBD <= eps) break;
    }
    problems.master.addRows(
      new_cuts.num_row_, new_rhs.data(), cut_ubs.data(), 
      new_cuts.numNz(), new_cuts.start_.data(), new_cuts.index_.data(), new_cuts.value_.data()
    );
    info = solve_master(problems.master, info);
    recentring_counts.push_back(recentring_count);
    optim_counts.push_back(optim_count);
    // if (problems.master.getModelStatus() != HighsModelStatus::kOptimal && problems.master.getModelStatus() != HighsModelStatus::kUnknown)
    //   assert(1 == 0);
    master_values = problems.master.getSolution().col_value;
    xs << master_values;
    problems.master.getDualObjectiveValue(LBD);
    ubd_lbd << UBD << LBD << feas_count; 
    ubd_lbd.newline();

  }
  dist.newline() << oscillation_count;
  dist.newline();
  ubd_lbd.newline();
  xs.newline();
  return {UBD-LBD, UBD, info, iter, first_optim, UBD_updates, feasible_iters, recentring_counts, optim_counts, params};
}

BendersRet benders_l_shaped2(SmpsCoreStructure & core, StochasticTree & tree, 
  std::vector<double> const & starting_point, double subproblem_lb, MasterProblem & master_solver, double eps, int max_iter) {
  assert(core.is_valid() && tree.root != nullptr && core.stage_submatrix.size() == 2 && tree.root->get_no_children() == 1);
  auto & stage_1st = tree.root->get_child(0);
  int no_subproblems = stage_1st->get_no_children();

  std::set<HighsInt> master_variables;
  auto const & node_ranges = core.stage_submatrix.at(stage_1st->get_timestage());
  for (int i = node_ranges.col_idx_begin; i < node_ranges.col_idx_end; ++i) master_variables.emplace(i);

  MultiBendersProblems problems;
  auto row_division = divide_rows(core.a_matrix_, master_variables);
  auto & master = problems.master;
  auto master_lp = modify_problem(core, *stage_1st);
  std::ofstream("/tmp/linking.csv", std::ios::app) << row_division.mixed_rows.size() << "\n";
  create_master_problem(master, master_lp, master_variables, row_division.subproblem_rows, subproblem_lb, 1);
  // create_master_problem(master, master_lp, master_variables, row_division.subproblem_rows, subproblem_lb, aggregate_cuts ? 1 : no_subproblems);
  // auto start = starting_point.empty() ? quick_master_solve(problems.master) : starting_point;
  problems.subproblems = std::vector<Highs> (no_subproblems);
  problems.feas_subproblems = std::vector<Highs> (no_subproblems);
  auto a = construct_extension_matrix(core, row_division.mixed_rows);
  for (int i = 0; i < no_subproblems; ++i) {
    auto & stage_2nd = stage_1st->get_child(i);
    auto base_sub_lp = modify_problem(core, *stage_2nd);
    create_subproblem(problems.subproblems.at(i), base_sub_lp, master_variables, row_division.master_only_rows);
    create_feasibility_subproblem(problems.feas_subproblems.at(i), base_sub_lp, master_variables, row_division.mixed_rows, row_division.master_only_rows, a);
  }
  BendersAlgorithm benders(eps, max_iter);
  return benders.benders_loop(problems, master_variables, {}, master_solver);
}

void add_vec_to_vec(std::vector<double> & base, NonZeroVector const & addition) {
  for (int i = 0; i < addition.number_of_nonzeros; ++i)
    base.at(addition.nonzero_indices.at(i)) += addition.nonzero_values.at(i);
}

BendersRet  BendersAlgorithm::benders_loop(
    MultiBendersProblems & problems, std::set<HighsInt> const & master_variables,
    std::vector<double> const & starting_point, MasterProblem & master_solver) {
      double sub_time = 0;
      //TODO delete problems.master
      master_solver.pass_model(problems.master.getModel());
      std::vector<double> master_values = starting_point.empty() ? master_solver.starting_point(): starting_point;
      while (!is_gap_closed() && !error && iter++ < max_iter) {
        auto sub_res = solve_subproblems(problems.subproblems, problems.feas_subproblems, master_variables, master_values);
        sub_time += sub_res.time;
        if (error) break;
        double solution_cost = !sub_res.all_feasible ? kHighsInf : 
          calculate_solution_cost(problems.master, sub_res.subproblem_costs, master_variables, master_values);
        UBD = std::min(UBD, solution_cost);
        if (is_gap_closed()) break;
        master_solver.add_cut(sub_res.cut, iter);
        error = !master_solver.solve(UBD, LBD, eps, solution_cost);
        if (error) break;
        LBD = master_solver.getLBD();
        master_values = master_solver.getMasterValues();
      }
      BendersIterationInfo info;
      info.master_time = master_solver.get_master_time();
      info.sub_time = sub_time;
      BendersRet ret {UBD-LBD, UBD, info, iter};
      ret.was_error = error;
      return ret;
    }

  BendersAlgorithm::SubproblemsResult BendersAlgorithm::solve_subproblems(std::vector<Highs> & subproblems, 
    std::vector<Highs> & feas_subproblems, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
    
      //TODO drop multi vectors
      // TODO pass basis?
      // TODO pass info?
      // TODO tidy between subtypes
      std::vector<double> obj_cut_multipliers(master_variables.size()+1, 0);
      std::vector<double> feas_cut_multipliers(master_variables.size()+1, 0);
      double obj_rhs = 0;
      double feas_rhs = 0;
      double subproblem_costs = 0;
      bool all_feasible = true;
      BendersIterationInfo info;
      for (int i = 0; i < subproblems.size(); ++i) { 
        auto & subproblem = subproblems.at(i); //TODO: Unnecessary solution if infeasible
        auto & feas_subproblem = feas_subproblems.at(i);
        info = solve_subproblem(subproblem, info, master_variables, master_values);
        if (info.was_subproblem_feasible) {
          if (!all_feasible) continue;
          subproblem_costs += subproblem.getObjectiveValue();
          auto cut = form_cut(subproblem, master_variables, master_values, CutType::Objective);
          obj_rhs += cut.rhs;
          add_vec_to_vec(obj_cut_multipliers, cut.coefficients);
          obj_cut_multipliers.back() = 1; //TODO only works for aggregated
        }
        else {
          all_feasible = false;
          info = solve_feasibility_subproblem(feas_subproblem, info, master_variables, master_values);
          auto cut = form_cut(feas_subproblem, master_variables, master_values, CutType::Feasibility);
          feas_rhs += cut.rhs;
          add_vec_to_vec(feas_cut_multipliers, cut.coefficients);
        }
      }
      error = info.was_error;
      //TODO ugly
      return {
        all_feasible,
        {all_feasible ? obj_cut_multipliers : feas_cut_multipliers, all_feasible ? obj_rhs : feas_rhs},
        all_feasible ? subproblem_costs : kHighsInf,
        info.sub_time
      };

    }
void MasterProblem::add_cut_to_problem(Highs & problem, CutData const & cut, std::string const & name) {
  add_nonzero_row(problem, cut.dual_objective, kHighsInf, {cut.master_multipliers}, name);
}

void MasterProblem::add_cut(CutData const & cut, int iter) {
  add_cut_to_problem(master, cut, std::string("CUT") + std::to_string(iter));
}

bool StandardMasterProblem::solve(double UBD, double LBD, double eps, double solution_cost) {
  BendersIterationInfo info;
  info = solve_master(master, info);
  master_time += info.master_time;
  return !info.was_error;
}

bool ProximalIPMMasterProblem::solve(double UBD, double LBD, double eps, double solution_cost) {
    if (is_close(UBD, LBD, eps)) {
      master.setOptionValue("solver", kSimplexString);
      master.setOptionValue("ipm_optimality_tolerance", 1e-8);
    }
    if (solution_cost < kHighsInf && feas_iter_counter++ % increment_every_n_iter == 0) {
      optim_steps = std::min(optim_steps + 1, max_optim_steps);
      master.setOptionValue("ipm_iteration_limit", optim_steps);
    }
    BendersIterationInfo info;
    info = solve_master(master, info);
    master_time += info.master_time;
    return true;
    // return !info.was_error; //TODO better status checking
}

HighsHessian LevelSetQpMasterProblem::create_level_set_hessian(int num_master_variables, int num_mu) const {
  HighsHessian hess;
  hess.dim_ = num_master_variables + num_mu; //1?
  hess.format_ = HessianFormat::kTriangular;
  hess.value_ = std::vector<double>(hess.dim_, 1);
  std::fill(hess.value_.begin() + num_master_variables, hess.value_.end(), 0);
  hess.index_ = std::vector<int>(hess.dim_, 0);
  hess.start_ = std::vector<int>(hess.dim_ + 1, 0);
  for (int i = 0; i < hess.dim_; ++i) {
    hess.index_.at(i) = i;
    hess.start_.at(i) = i;
  }
  hess.start_.at(hess.dim_) = hess.dim_;
  return hess;
}

void LevelSetMasterProblem::pass_model(HighsModel const & model, int no_mu) {
  MasterProblem::pass_model(model, no_mu);
  level_set_master.passModel(model);
  add_nonzero_row(level_set_master, -kHighsInf, kHighsInf, {model.lp_.col_cost_}, "LEVEL");
  level_set_constraint = level_set_master.getNumRow() - 1;
  std::vector<double> zeros (level_set_master.getNumCol(), 0);
  level_set_master.changeColsCost(0, level_set_master.getNumCol()-1, zeros.data());
}



void LevelSetQpMasterProblem::pass_model(HighsModel const & model, int no_mu) {
  LevelSetMasterProblem::pass_model(model, no_mu);
  level_set_master.passHessian(create_level_set_hessian(model.lp_.num_col_ - no_mu, no_mu));
}

void LevelSetMasterProblem::add_cut(CutData const & cut, int iter) {
  MasterProblem::add_cut(cut, iter);
  add_cut_to_problem(level_set_master, cut, std::string("CUT") + std::to_string(iter));
}

std::pair<NonZeroVector, double> LevelSetQpMasterProblem::create_distance_costs() const {
  auto prev_solution = getMasterValues();
  std::vector<double> linear_factor(prev_solution.size(), 0);
  double constant = 0; // TODO num_mu inconsistent
  for (int i = 0; i < master.getNumCol() - num_mu; ++i) {
      auto val = prev_solution.at(i);
      val = std::round(val / rounding) * rounding;
      linear_factor.at(i) = -val;
      constant += val * val / 2;
  }      
  return {{linear_factor}, constant};

}

bool LevelSetQpMasterProblem::solve(double UBD, double LBD, double eps, double solution_cost) {
  auto distance_costs = create_distance_costs(); //TODO ugly placement  
  BendersIterationInfo info;
    info = solve_master(master, info);
    LBD = master.getObjectiveValue();
    double real_improvement = prev_UBD-solution_cost; 
    prev_UBD = UBD;
    if (in_level && solution_cost < kHighsInf && real_improvement > 0) {
      double r = real_improvement / projected_improvement;
      if (r <= 0.1)
        gamma = 1 - omega * (1-gamma);
      else if (r > 0.9)
        gamma *= omega;
    } 
    if ((in_level = is_in_level(UBD, LBD, eps))) {
      double target = LBD + gamma * (UBD-LBD);
      projected_improvement = UBD - target;
      level_set_master.changeRowBounds(level_set_constraint,  -kHighsInf, target);
      level_set_master.changeColsCost(distance_costs.first.number_of_nonzeros, distance_costs.first.nonzero_indices.data(), distance_costs.first.nonzero_values.data());
      level_set_master.changeObjectiveOffset(distance_costs.second);
      
      // std::vector<int> indices(level_set_master.getNumCol());
      // for (int i = 0; i < level_set_master.getNumCol(); ++i) indices[i] = i;
      // level_set_master.setSolution(level_set_master.getNumCol(), indices.data(), master.getSolution().col_value.data());
      info = solve_master(level_set_master, info);
      if (info.was_error) {
          if (++error_counter > 5) omega = 1.0;
          gamma = orig_gamma;
          in_level = false;
          info.was_error = false;
      }
        
    }
    master_time += info.master_time;
    return !info.was_error;
}

void LevelSetIpmMasterProblem::pass_model(HighsModel const & model, int no_mu) {
  LevelSetMasterProblem::pass_model(model, no_mu);
  level_set_master.setOptionValue("solver", kHipoString);
  level_set_master.setOptionValue("max_centring_steps", 100);
  level_set_master.setOptionValue("run_crossover", kHighsOffString);
  level_set_master.setOptionValue("centring_gamma", 1-1e-5);
  level_set_master.setOptionValue("recentring_step", 1.0);
  level_set_master.setOptionValue("ipm_iteration_limit", 0);
}


bool LevelSetIpmMasterProblem::solve(double UBD, double LBD, double eps, double solution_cost) {
    BendersIterationInfo info;
    info = solve_master(master, info);
    LBD = master.getObjectiveValue();
    if ((in_level = is_in_level(UBD, LBD, eps))) {
      if (UBD < kHighsInf) {
        double target_l = LBD + gamma_l * (UBD-LBD);
        double target_u = LBD + gamma_u * (UBD-LBD);
        level_set_master.changeRowBounds(level_set_constraint, target_l, target_u);
      }
      info = solve_master(level_set_master, info);
    }
    master_time += info.master_time;
    return !info.was_error;
}
//TODO delete BendersInfo

void ProximalIPMMasterProblem::pass_model(HighsModel const & model, int no_mu) {
  MasterProblem::pass_model(model, no_mu);
  master.setOptionValue("optimality_tolerance", 1e-6);
  master.setOptionValue("ipm_optimality_tolerance", 1e-6);
  master.setOptionValue("ipm_iteration_limit", optim_steps);
  master.setOptionValue("solver", kHipoString);
  master.setOptionValue("run_crossover", kHighsOffString);
  master.setOptionValue("centring_gamma", 1-1e-5);
  master.setOptionValue("max_centring_steps", 20);
  master.setOptionValue("recentring_step", 1.0);
  // master.setOptionValue("primal_feasibility_tolerance", 1e-8);
  // master.setOptionValue("dual_feasibility_tolerance", 1e-8);
}
//TODO rename master_problem?

std::vector<double> MasterProblem::starting_point() {
  BendersIterationInfo info;
  info = solve_master(master, info);
  master_time = info.master_time;
  if (master.getModelStatus() != HighsModelStatus::kUnbounded) // TODO what if imprecise?
    return master.getSolution().col_value;
  auto copy_costs = master.getLp().col_cost_;
  std::vector<double> zeros (master.getNumCol(), 0);
  master.changeColsCost(0, master.getNumCol()-1, zeros.data());
  info = solve_master(master, info);
  master_time = info.master_time;
  master.changeColsCost(0, master.getNumCol()-1, copy_costs.data());
  return master.getSolution().col_value; 
  //TODO what if error?
}