#include "Benders.h"
#include "Highs.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <vector>
#include "HighsInt.h"
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

void benders(HighsLp & base_problem, std::set<HighsInt> & master_variables) {
  BendersProblems problems;
  decompose_problem(problems, base_problem, master_variables);
  for (int i = 0; i < 1; ++i) {
    auto master_status = problems.master.run();
    auto const & master_solution = problems.master.getSolution();
    fix_master_variables(problems.subproblem, master_variables, master_solution.col_value); // Won't work with mu added
    auto subproblem_status = problems.subproblem.run();
    auto const & subproblem_solution = problems.subproblem.getSolution();
     // highs.resetGlobalScheduler(true);
  }
}

