#include "Benders.h"
#include <algorithm>
#include <iterator>
#include <vector>
#include "HighsInt.h"


HighsInt find_row_index(std::vector<HighsInt> const & csr_starts, HighsInt index) {
  auto pointer = std::upper_bound(csr_starts.begin(), csr_starts.end(), index);
  return std::distance(csr_starts.begin(), pointer) - 1;
}

OverlappingRowDivision divide_overlapping_rows(std::vector<HighsInt> const & csr_index, std::vector<HighsInt> const & csr_starts, std::set<HighsInt> const & master_variables) {
  auto is_master = [&master_variables](HighsInt index){ return master_variables.count(index) > 0; };
  std::set<HighsInt> master_rows, subproblem_rows;
  for (int i = 0; i < csr_index.size(); ++i) {
    if (is_master(csr_index[i]))
      master_rows.insert(find_row_index(csr_starts, i));
    else
      subproblem_rows.insert(find_row_index(csr_starts, i));
  }
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

