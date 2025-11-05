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

HighsIndexCollection index_collection_from_set(std::set<HighsInt> const & index_set, HighsInt max_idx) {
  return index_collection_from_set(std::vector<HighsInt>(index_set.begin(), index_set.end()), max_idx);
}

HighsIndexCollection index_collection_from_set(std::vector<HighsInt> const & index_set, HighsInt collection_dimension) {
  HighsIndexCollection index_collection;
  index_collection.is_set_ = true;
  index_collection.set_ = index_set;
  index_collection.set_num_entries_ = index_set.size();
  index_collection.dimension_ = collection_dimension;
  return index_collection;
}

bool fix_master_variables(Highs & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  for (auto i : master_variables)
    subproblem.changeColBounds(i, master_values.at(i) , master_values.at(i));
  return true;
}

bool fix_master_variables(HighsLp & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  // TODO: What max(master_values) > size?
  if (master_variables.size() != master_values.size())
    return false;
  auto index_collection = index_collection_from_set(master_variables, subproblem.num_col_);
  changeLpColBounds(subproblem, index_collection, master_values, master_values);
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

HighsLp create_master_problem(HighsLp problem, std::set<HighsInt> const & master_variables, RowDivision const & row_division) {
  Highs master;
  master.passModel(problem);
  auto nonmaster_rows= set_to_vector(row_division.subproblem_rows);
  master.deleteRows(nonmaster_rows.size(), nonmaster_rows.data());
  auto subproblem_variables = set_to_vector(sequence_complement(master_variables, problem.num_col_)); // TODO: should it be here?
  master.deleteCols(subproblem_variables.size(), subproblem_variables.data());
  return master.getLp();
}

HighsLp create_subproblem(HighsLp problem, std::set<HighsInt> const & master_variables) {
  problem.offset_ = 0;
  for (auto i: master_variables)
    problem.col_cost_.at(i) = 0;
  return problem;
}

BendersProblems decompose_problem(HighsLp const & problem, std::set<HighsInt> const & master_variables, RowDivision const & row_division) {
  // TODO: Not optimal!
  auto master = create_master_problem(problem, master_variables, row_division);
  auto subproblem = create_subproblem(problem, master_variables);
  return {master, subproblem};
}

BendersProblems decompose_problem(HighsLp & problem, std::set<HighsInt> const & master_variables) {
  auto row_division = divide_rows(problem.a_matrix_, master_variables);
  return decompose_problem(problem, master_variables, row_division);
}

void benders(HighsLp & problem, std::set<HighsInt> & master_variables) {
  auto problems = decompose_problem(problem, master_variables);
  Highs highs_master, highs_subproblem;
  highs_subproblem.passModel(problems.subproblem);
  highs_master.passModel(problems.master);
  for (int i = 0; i < 1; ++i) {
    auto master_status = highs_master.run();
    auto const & master_solution = highs_master.getSolution();
    fix_master_variables(highs_subproblem, master_variables, master_solution.col_value); // Won't work with mu added
    auto subproblem_status = highs_subproblem.run();
    auto const & subproblem_solution = highs_subproblem.getSolution();
     // highs.resetGlobalScheduler(true);
  }
}

