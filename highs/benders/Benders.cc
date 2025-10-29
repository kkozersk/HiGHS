#include "Benders.h"
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

OverlappingRowDivision divide_overlapping_rows(std::vector<HighsInt> const & csr_index, std::vector<HighsInt> const & csr_starts, std::set<HighsInt> const & master_variables) {
  auto is_master = [&master_variables](HighsInt index){ return master_variables.count(index) > 0; };
  std::set<HighsInt> master_rows, subproblem_rows;
  for (int i = 0; i < csr_index.size(); ++i)
    (is_master(csr_index[i]) ? master_rows : subproblem_rows).insert(find_row_index(csr_starts, i));
  return {master_rows, subproblem_rows};
}

DisjointRowDivision divide_disjoint_rows(std::vector<HighsInt> const & csr_index, std::vector<HighsInt> const & csr_starts, std::set<HighsInt> const & master_variables) {
  auto overlapping_rows = divide_overlapping_rows(csr_index, csr_starts, master_variables);
  auto const & master_rows = overlapping_rows.master_rows;
  auto const & subproblem_rows = overlapping_rows.subproblem_rows;
  std::vector<HighsInt> mixed_rows, master_only_rows, subproblem_only_rows;
  std::set_intersection(master_rows.begin(), master_rows.end(), subproblem_rows.begin(), subproblem_rows.end(), std::back_inserter(mixed_rows));
  std::set_difference(master_rows.begin(), master_rows.end(), mixed_rows.begin(), mixed_rows.end(), std::back_inserter(master_only_rows));
  std::set_difference(subproblem_rows.begin(), subproblem_rows.end(), mixed_rows.begin(), mixed_rows.end(), std::back_inserter(subproblem_only_rows));
  return {master_only_rows, subproblem_only_rows, mixed_rows};
}

DisjointRowDivision divide_disjoint_rows(HighsSparseMatrix & constraint_matrix, std::set<HighsInt> const & master_variables) {
  constraint_matrix.ensureRowwise();
  return divide_disjoint_rows(constraint_matrix.index_, constraint_matrix.start_, master_variables);
}

HighsIndexCollection index_collection_from_set(std::set<HighsInt> const & index_set) {
  HighsIndexCollection index_collection;
  index_collection.is_set_ = true;
  index_collection.set_ = std::vector<HighsInt>(index_set.begin(), index_set.end());
  index_collection.set_num_entries_ = index_set.size();
  index_collection.dimension_ = index_set.size();
  return index_collection;
}

bool fix_master_variables(HighsLp & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values) {
  // TODO: What max(master_values) > size?
  if (master_variables.size() != master_values.size())
    return false;
  auto index_collection = index_collection_from_set(master_variables);
  changeLpColBounds(subproblem, index_collection, master_values, master_values);
  return true;
}
