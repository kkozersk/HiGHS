#pragma once
#include <set>
#include "HighsInt.h"
#include "Highs.h"
#include "HighsLp.h"
#include "HighsSparseMatrix.h"

struct RowDivision {
  std::set<HighsInt> master_rows;
  std::set<HighsInt> subproblem_rows;
  std::set<HighsInt> mixed_rows;
};

struct BendersProblems {
  Highs master;
  Highs subproblem;
};

struct NonZeroVector {
  HighsInt number_of_nonzeros;
  std::vector<HighsInt> nonzero_indices;
  std::vector<double> nonzero_values;
};

HighsInt find_row_index(std::vector<HighsInt> const & csr_starts, HighsInt index);
RowDivision divide_rows(std::vector<HighsInt> const & csr_index, std::vector<HighsInt> const & csr_starts, std::set<HighsInt> const & master_variables); 
RowDivision divide_rows(HighsSparseMatrix & constraint_matrix, std::set<HighsInt> const & master_variables);
bool fix_master_variables(Highs & subproblem, std::set<HighsInt> const & master_variables, std::vector<double> const & master_values);
std::set<HighsInt> sequence_complement(std::set<HighsInt> const & set, HighsInt max_number);
void create_master_problem(Highs & master, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, std::set<HighsInt> const & subproblem_rows);
void create_subproblem(Highs & subproblem, HighsLp const & base_problem, std::set<HighsInt> const & master_variables);
void decompose_problem(BendersProblems & problems, HighsLp const & base_problem, std::set<HighsInt> const & master_variables, RowDivision const & row_division);
void decompose_problem(BendersProblems & problems, HighsLp & base_problem, std::set<HighsInt> const & master_variables);
BendersProblems decompose_problem(HighsLp const & problem, std::set<HighsInt> const & master_variables, RowDivision const & row_division); 
BendersProblems decompose_problem(HighsLp & problem, std::set<HighsInt> const & master_variables);
std::vector<double> get_all_multipliers(Highs const & subproblem);
std::vector<double> get_master_multipliers(Highs const & subproblem, std::set<HighsInt> master_variables);
NonZeroVector create_nonzero_vector(std::vector<double> base_vector);
void add_objective_cut(Highs & master, Highs const & subproblem, std::set<HighsInt> master_variables, std::vector<double> master_values);
void benders(HighsLp & problem, std::set<HighsInt> & master_variables);
