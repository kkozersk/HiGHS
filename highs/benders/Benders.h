#pragma once
#include <iterator>
#include <set>
#include "HighsInt.h"
#include "HighsLp.h"
#include "HighsSparseMatrix.h"

struct RowDivision {
  std::set<HighsInt> master_rows;
  std::set<HighsInt> subproblem_rows;
  std::set<HighsInt> mixed_rows;
};

struct BendersProblems {
  HighsLp master;
  HighsLp subproblem;
};

HighsInt find_row_index(std::vector<HighsInt> const & csr_starts, HighsInt index);
RowDivision divide_rows(std::vector<HighsInt> const & csr_index, std::vector<HighsInt> const & csr_starts, std::set<HighsInt> const & master_variables); 
RowDivision divide_rows(HighsSparseMatrix & constraint_matrix, std::set<HighsInt> const & master_variables);
HighsIndexCollection index_collection_from_set(std::vector<HighsInt> const & index_set, HighsInt max_idx);
HighsIndexCollection index_collection_from_set(std::set<HighsInt> const & index_set, HighsInt max_idx);
bool fix_master_variables(Highs & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values);
bool fix_master_variables(HighsLp & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values);
std::set<HighsInt> sequence_complement(std::set<HighsInt> const & set, HighsInt max_number);
HighsLp create_master_problem(HighsLp problem, std::set<HighsInt> const & master_variables, RowDivision const & row_division);
HighsLp create_master_problem2(HighsLp problem, std::set<HighsInt> const & master_variables, RowDivision const & row_division);
HighsLp create_subproblem(HighsLp problem, std::set<HighsInt> const & master_variables); 
BendersProblems decompose_problem(HighsLp const & problem, std::set<HighsInt> const & master_variables, RowDivision const & row_division); 
BendersProblems decompose_problem(HighsLp & problem, std::set<HighsInt> const & master_variables);
void benders(HighsLp & problem, std::set<HighsInt> & master_variables);
