#pragma once
#include <set>
#include "HighsInt.h"
#include "HighsLp.h"
#include "HighsSparseMatrix.h"

struct DisjointRowDivision {
  std::vector<HighsInt> master_only_rows;
  std::vector<HighsInt> subproblem_only_rows;
  std::vector<HighsInt> mixed_rows;
};

struct OverlappingRowDivision {
  std::set<HighsInt> master_rows;
  std::set<HighsInt> subproblem_rows;
};

struct BendersProblems {
  HighsLp master;
  HighsLp subproblem;
};

HighsInt find_row_index(std::vector<HighsInt> const & csr_starts, HighsInt index);
OverlappingRowDivision divide_overlapping_rows(std::vector<HighsInt> const & csr_index, std::vector<HighsInt> const & csr_starts, std::set<HighsInt> const & master_variables); 
DisjointRowDivision divide_disjoint_rows(std::vector<HighsInt> const & csr_index, std::vector<HighsInt> const & csr_starts, std::set<HighsInt> const & master_variables);
DisjointRowDivision divide_disjoint_rows(HighsSparseMatrix & constraint_matrix, std::set<HighsInt> const & master_variables);
HighsIndexCollection index_collection_from_set(std::set<HighsInt> const & index_set);
bool fix_master_variables(HighsLp & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values);
