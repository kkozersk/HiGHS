#include "Benders.h"
#include "HConst.h"
#include "Highs.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>
#include <vector>
#include "HighsInt.h"
#include "HighsStatus.h"
#include "HighsUtils.h"
#include "HighsLpUtils.h"


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

bool fix_master_variables(Highs & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  for (auto i : master_variables)
    subproblem.changeColBounds(i, master_values.at(i) , master_values.at(i));
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
  master.addCol(1, -kHighsInf, kHighsInf, 0, nullptr, nullptr);
}

void create_subproblem(Highs & subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables) {
  subproblem.passModel(base_problem);
  subproblem.changeObjectiveOffset(0);
  auto nonsub_variables = set_to_vector(master_variables);
  std::vector<double> zeros (master_variables.size(), 0);
  subproblem.changeColsCost(master_variables.size(), nonsub_variables.data(), zeros.data());
}

void decompose_problem(BendersProblems & problems, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, RowDivision const & row_division) {
  create_master_problem(problems.master, base_problem, master_variables, row_division.subproblem_rows);
  create_subproblem(problems.subproblem, base_problem, master_variables);
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

void add_objective_cut(Highs & master, Highs const & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  double dual_objective;
  subproblem.getDualObjectiveValue(dual_objective);
  auto multipliers = get_master_multipliers(subproblem, master_variables);
  auto old_value_multiple = std::inner_product(multipliers.begin(), multipliers.end(), master_values.begin(), 0.0);
  auto nonzero_multipliers = add_mu_entry(create_nonzero_vector(multipliers), master_variables.size());
  master.addRow(dual_objective + old_value_multiple, kHighsInf,
                 nonzero_multipliers.number_of_nonzeros,
                 nonzero_multipliers.nonzero_indices.data(),
                 nonzero_multipliers.nonzero_values.data());
}

// void add_feasibility_cut(Highs & master, Highs const & subproblem, std::set<HighsInt> master_variables, std::vector<double> master_values) {
//   double dual_objective;
//   subproblem.getDualObjectiveValue(dual_objective);
//   auto multipliers = get_master_multipliers(subproblem, master_variables);
//   auto old_value_multiple = std::inner_product(multipliers.begin(), multipliers.end(), master_values.begin(), 0.0);
//   master.addRow(dual_objective + old_value_multiple, kHighsInf, 0, nullptr, multipliers.data());
// }

void benders(HighsLp & base_problem, std::set<HighsInt> & master_variables) {
  BendersProblems problems;
  decompose_problem(problems, base_problem, master_variables);
  auto & master = problems.master;
  auto & subproblem = problems.subproblem;
  std::vector<double> dual_ray (base_problem.num_col_);
  std::vector<double> master_values(master_variables.size(), 0);
  double UBD = kHighsInf, LBD = -kHighsInf;
  auto starting_values =  std::vector<double>(master_variables.size(), 0);
  fix_master_variables(subproblem, master_variables, starting_values);
  while (UBD - LBD > 1e-3) {
    if (subproblem.run() == HighsStatus::kError) break;
    if (subproblem.getModelStatus() == HighsModelStatus::kInfeasible) {
      bool has_dual_ray;
      subproblem.getDualRay(has_dual_ray, dual_ray.data());
      // add_feasibility_cut(master, subproblem, master_variables, master_values);
    }
    else {
      // TODO: maximizing subproblem
       UBD = std::min(UBD, subproblem.getObjectiveValue());
       add_objective_cut(master, subproblem, master_variables, master_values);
    }
    if (master.run() == HighsStatus::kError) break;
    LBD = master.getObjectiveValue();
    auto master_values = master.getSolution().col_value;
    fix_master_variables(subproblem, master_variables, master_values); // Won't work with mu added
  }
}

