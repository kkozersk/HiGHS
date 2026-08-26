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
  subproblem.changeColsBounds(
    master_variables.size(), 
    set_to_vector(master_variables).data(),
    master_values.data(),
    master_values.data()
  );
  return true;
}

// void save_master_variables(Highs & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
//   std::vector<double> result;
//   std::vector<double> x (subproblem.getNumCol(), 0);
//   int master_index = 0;
//   for (auto subproblem_index : master_variables) {
//     x.at(subproblem_index) = master_values.at(master_index++);
//   }
//   subproblem.getLp().a_matrix_.product(result, x);
// }

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

void create_feasibility_subproblem(Highs & feas_subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & mixed_rows, std::set<HighsInt> const & master_only_rows) {
  auto a = construct_extension_matrix(base_problem, mixed_rows);
  create_feasibility_subproblem(feas_subproblem, base_problem, master_variables, mixed_rows, master_only_rows, a);
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
void decompose_problem(MultiBendersProblems & problems, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, RowDivision const & row_division, double subproblem_lb) {
  create_master_problem(problems.master, base_problem, master_variables, row_division.subproblem_rows, subproblem_lb);
  create_subproblem(problems.subproblems.at(0), base_problem, master_variables, row_division.master_only_rows);
  create_feasibility_subproblem(problems.feas_subproblems.at(0), base_problem, master_variables, row_division.mixed_rows, row_division.master_only_rows);
}

void decompose_problem(MultiBendersProblems & problems, HighsLp & base_problem, std::set<HighsInt> const & master_variables, double subproblem_lb) {
  auto row_division = divide_rows(base_problem.a_matrix_, master_variables);
  decompose_problem(problems, base_problem, master_variables, row_division, subproblem_lb);
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
  // save_master_variables(subproblem, master_variables, master_values);
  auto start = subproblem.getRunTime();
  info.was_error = info.was_error || subproblem.run() == HighsStatus::kError;
  auto end =  subproblem.getRunTime();
  info.sub_time += end - start;
  info.was_subproblem_feasible = subproblem.getModelStatus() == HighsModelStatus::kOptimal;
  return info;
}

// BendersIterationInfo solve_master(Highs & master, BendersIterationInfo info) {
//   auto start = master.getRunTime();
//   info.was_error = info.was_error || master.run() == HighsStatus::kError;
//   auto end =  master.getRunTime();
//   info.master_time += end - start;
//   return info;
// }

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
//TODO: Fix!
// BendersRet benders(HighsLp & base_problem, std::string const & master_name_pattern, std::vector<double> const & starting_point, double eps, std::vector<OptionValue> const & masterOptions) { 
//   auto master_variables = discover_master_variables(base_problem.col_names_, master_name_pattern);
//   return benders(base_problem, master_variables, starting_point);
// }

BendersRet benders(HighsLp & base_problem, std::set<HighsInt> const & master_variables, std::vector<double> const & starting_point, double subproblem_lb,
   double eps, int max_iter) { 
  MultiBendersProblems problems {1};
  StandardMasterProblem master_solver {};
  decompose_problem(problems, base_problem, master_variables, subproblem_lb);
  BendersAlgorithm benders(eps, max_iter);
  return benders.benders_loop(problems, master_variables, starting_point, master_solver);
}

// BendersRet multi_benders_loop(MultiBendersProblems & problems, std::set<HighsInt> const & master_variables,
//                      std::vector<double> const & starting_point, double eps, int max_iter, MasterAdaptationParams params) { 
//   BendersIterationInfo info;
//   auto master_values = starting_point;
//   int iter = 0;
//   double UBD = kHighsInf, LBD = -kHighsInf;
//   int no_subproblems = problems.subproblems.size();
//   bool was_all_feas = false;
//   std::vector<double> new_rhs(no_subproblems);
//   std::vector<double> cut_ubs(no_subproblems, kHighsInf);
//   HighsSparseMatrix new_cuts;
//   while (UBD - LBD > eps && !info.was_error && iter++ < max_iter) {
//     zerovec(new_rhs);
//     new_cuts.clear();
//     new_cuts.ensureRowwise();
//     new_cuts.num_col_ = problems.master.getNumCol(); 
//     int feas_count = 0;
//     double subproblem_costs = 0;
//     bool all_feasible = true;
//     for (int i = 0; i < no_subproblems; ++i) { 
//       auto & subproblem = problems.subproblems.at(i);
//       info = solve_subproblem(subproblem, info, master_variables, master_values);
//       if (info.was_subproblem_feasible) {
//         feas_count++;
//         subproblem_costs += subproblem.getObjectiveValue();
//         auto cut = form_cut(subproblem, master_variables, master_values, CutType::Objective, i);
//         new_rhs.at(i) = cut.rhs;
//         new_cuts.addVec(cut.coefficients.number_of_nonzeros, cut.coefficients.nonzero_indices.data(), cut.coefficients.nonzero_values.data());
//       }
//       else {
//         all_feasible = false;
//         auto & feas_subproblem = problems.feas_subproblems.at(i);
//         info = solve_feasibility_subproblem(feas_subproblem, info, master_variables, master_values);
//         auto cut = form_cut(feas_subproblem, master_variables, master_values, CutType::Feasibility, i);
//         new_rhs.at(i) = cut.rhs;
//         new_cuts.addVec(cut.coefficients.number_of_nonzeros, cut.coefficients.nonzero_indices.data(), cut.coefficients.nonzero_values.data());
//       }
//     }
//     if (all_feasible) {
//         double solution_cost = calculate_solution_cost(problems.master, subproblem_costs, master_variables, master_values);
//         UBD = std::min(UBD, solution_cost);
//         if (UBD - LBD <= eps) break;
//     }
//     problems.master.addRows(
//       new_cuts.num_row_, new_rhs.data(), cut_ubs.data(), 
//       new_cuts.numNz(), new_cuts.start_.data(), new_cuts.index_.data(), new_cuts.value_.data()
//     );
//     info = solve_master(problems.master, info);
//     // if (problems.master.getModelStatus() != HighsModelStatus::kOptimal && problems.master.getModelStatus() != HighsModelStatus::kUnknown)
//     //   assert(1 == 0);
//     master_values = problems.master.getSolution().col_value;
//     problems.master.getDualObjectiveValue(LBD);

//   }
//   return {UBD-LBD, UBD, info, iter};
// }

BendersRet benders_l_shaped(SmpsCoreStructure & core, StochasticTree & tree, 
  std::vector<double> const & starting_point, double subproblem_lb, MasterProblem & master_solver, double eps, int max_iter) {
  assert(core.is_valid() && tree.root != nullptr && core.stage_submatrix.size() == 2 && tree.root->get_no_children() == 1);
  auto & stage_1st = tree.root->get_child(0);
  int no_subproblems = stage_1st->get_no_children();

  std::set<HighsInt> master_variables;
  auto const & node_ranges = core.stage_submatrix.at(stage_1st->get_timestage());
  for (int i = node_ranges.col_idx_begin; i < node_ranges.col_idx_end; ++i) master_variables.emplace(i);

  MultiBendersProblems problems {no_subproblems};
  auto row_division = divide_rows(core.a_matrix_, master_variables);
  auto & master = problems.master;
  auto master_lp = modify_problem(core, *stage_1st);
  create_master_problem(master, master_lp, master_variables, row_division.subproblem_rows, subproblem_lb, 1);
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
  solve_problem_with_logging(master);
  return !error;
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
    solve_problem_with_logging(master);
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
  level_set_master.setOptionValue("output_flag", false);
  level_set_master.setOptionValue("log_to_console", false);
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
    solve_problem_with_logging(master);
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
      
      solve_problem_with_logging(level_set_master);
      if (error) {
          if (++error_counter > 5) omega = 1.0;
          gamma = orig_gamma;
          in_level = false;
          error = false;
      }
        
    }
    return !error;
}

void LevelSetIpmMasterProblem::pass_model(HighsModel const & model, int no_mu) {
  LevelSetMasterProblem::pass_model(model, no_mu);
  level_set_master.setOptionValue("solver", kHipoString);
  level_set_master.setOptionValue("max_centring_steps_hipo", 100);
  level_set_master.setOptionValue("run_crossover", kHighsOffString);
  level_set_master.setOptionValue("centring_gamma", 1-1e-5);
  level_set_master.setOptionValue("recentring_step", 1.0);
  level_set_master.setOptionValue("ipm_iteration_limit", 0);
  level_set_master.setOptionValue("refine_with_ipx", false);
}


bool LevelSetIpmMasterProblem::solve(double UBD, double LBD, double eps, double solution_cost) {
    solve_problem_with_logging(master);
    LBD = master.getObjectiveValue();
    if ((in_level = is_in_level(UBD, LBD, eps))) {
      if (UBD < kHighsInf) {
        double target_l = LBD + gamma_l * (UBD-LBD);
        double target_u = LBD + gamma_u * (UBD-LBD);
        level_set_master.changeRowBounds(level_set_constraint, target_l, target_u);
      }
      solve_problem_with_logging(level_set_master);
    }
    return !error;
}

void ProximalIPMMasterProblem::pass_model(HighsModel const & model, int no_mu) {
  MasterProblem::pass_model(model, no_mu);
  master.setOptionValue("optimality_tolerance", 1e-6);
  master.setOptionValue("ipm_optimality_tolerance", 1e-6);
  master.setOptionValue("ipm_iteration_limit", optim_steps);
  master.setOptionValue("solver", kHipoString);
  master.setOptionValue("run_crossover", kHighsOffString);
  master.setOptionValue("centring_gamma", 1-1e-5);
  master.setOptionValue("max_centring_steps_hipo", 20);
  master.setOptionValue("recentring_step", 1.0);
  master.setOptionValue("refine_with_ipx", false);
  // master.setOptionValue("primal_feasibility_tolerance", 1e-8);
  // master.setOptionValue("dual_feasibility_tolerance", 1e-8);
}

std::vector<double> MasterProblem::starting_point() {
  solve_problem_with_logging(master);
  if (master.getModelStatus() != HighsModelStatus::kUnbounded) // TODO what if imprecise?
    return master.getSolution().col_value;
  auto copy_costs = master.getLp().col_cost_;
  std::vector<double> zeros (master.getNumCol(), 0);
  master.changeColsCost(0, master.getNumCol()-1, zeros.data());
  solve_problem_with_logging(master);
  master.changeColsCost(0, master.getNumCol()-1, copy_costs.data());
  return master.getSolution().col_value; 
  //TODO what if error?
}

void MasterProblem::solve_problem_with_logging(Highs & problem) {
  auto start = master.getRunTime();
  error = problem.run() == HighsStatus::kError;
  auto end =  master.getRunTime();
  master_time += end - start;
}