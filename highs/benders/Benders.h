#pragma once
#include <set>
#include "HighsInt.h"
#include "HighsLp.h"

struct DisjointRowDivision {
  std::vector<HighsInt> master_only_rows;
  std::vector<HighsInt> subproblem_only_rows;
  std::vector<HighsInt> mixed_rows;
};

struct OverlappingRowDivision {
  std::set<HighsInt> master_rows;
  std::set<HighsInt> subproblem_rows;
};

HighsInt find_row_index(std::vector<HighsInt> const & csr_starts, HighsInt index);
OverlappingRowDivision divide_overlapping_rows(std::vector<HighsInt> const & csr_index, std::vector<HighsInt> const & csr_starts, std::set<HighsInt> const & master_variables); 
DisjointRowDivision divide_disjoint_rows(std::vector<HighsInt> const & csr_index, std::vector<HighsInt> const & csr_starts, std::set<HighsInt> const & master_variables);
void solve_subproblem(HighsLp & problem, const std::set<HighsInt> & master_variables, const std::vector<double> & master_values, const DisjointRowDivision& division);
