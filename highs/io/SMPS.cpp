#include "SMPS.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "Filereader.h"
#include "FilereaderMps.h"
#include "HighsLp.h"
#include "lp_data/HStruct.h"
#include "model/HighsModel.h"
#include "lp_data/HighsOptions.h"

void SmpsTimeStructure::read_file(std::istream & input) {
  std::string line;
  if (!getline(input, line) || !process_header(line)) return;
  if (!getline(input, line) || !process_period_header(line)) return;
  while (getline(input, line) && !is_ending(line)) {
    if (!process_period(line)) return;
  }
  if (timestage_entries.empty()) return;
  std::transform(timestage_entries.begin(), timestage_entries.end(), std::back_inserter(stage_names), [](TimeStageEntry const & val) {return val.stage_name; });
  if (!process_ending(line)) return;
  is_valid_ = true;
}

bool SmpsTimeStructure::process_header(std::string const & line) {
  std::string header;
  std::stringstream ss(line);
  ss >> header >> problem_name;
  if (header != "TIME") return false;
  return true;
}

bool SmpsTimeStructure::process_period_header(std::string const & line) {
  return line.substr(0, 7) == "PERIODS";
}

bool SmpsTimeStructure::process_period(std::string const & line) {
  std::string starting_column, starting_row, stage_name;
  std::stringstream ss(line);
  ss >> starting_column >> starting_row >> stage_name;
  if (starting_column.empty() || starting_row.empty() || stage_name.empty()) return false;
  timestage_entries.emplace_back(starting_row, starting_column, stage_name);
  return true;
}

bool SmpsTimeStructure::is_ending(std::string const & line) const {
  return line.empty() || line == "ENDATA";
}

bool SmpsTimeStructure::process_ending(std::string const & line) {
  return line == "ENDATA";
}

int SmpsTimeStructure::get_stage_index(std::string const & stage) const {
  auto it = std::find(stage_names.begin(), stage_names.end(), stage);
  return std::distance(stage_names.begin(), it);
}


SmpsCoreStructure & SmpsCoreStructure::operator=(HighsLp const & lp) {
  HighsLp::operator=(lp);
  if (!col_hash_.name2index.size()) col_hash_.form(col_names_);
  if (!row_hash_.name2index.size()) row_hash_.form(row_names_);
  return *this;
}

SmpsCoreStructure::SmpsCoreStructure(HighsOptions const & options, std::string const & filepath) {
  FilereaderMps mps;
  HighsModel model;
  is_valid_ = mps.readModelFromFile(options, filepath, model) == FilereaderRetcode::kOk;
  if (is_valid_) *this = model.lp_;
}

SmpsCoreStructure::SmpsCoreStructure(HighsLp const & lp) {
  *this=lp;
}

bool SmpsCoreStructure::load_time_stages(SmpsTimeStructure const & time_stage_data) {
  if (!verify_stages(time_stage_data, row_hash_, col_hash_)) return false;
  stage_submatrix = load_stage_submatrices(time_stage_data.get_entries(), row_hash_, col_hash_);
  return true;
}

Timestage2Range SmpsCoreStructure::load_stage_submatrices(std::vector<TimeStageEntry> const & timestage_indices, HighsNameHash const & row_name_hash,
                  HighsNameHash const & col_name_hash) {
  auto const & row2index = row_name_hash.name2index;
  auto const & col2index = col_name_hash.name2index;
  Timestage2Range result;
  for (int i = 0; i < timestage_indices.size(); ++i) {
    auto const & current_stage = timestage_indices.at(i);
    
    auto row_begin_idx = current_stage.row_idx_name == objective_name_ ? 0 : row2index.at(current_stage.row_idx_name);
    auto row_end_idx = i < timestage_indices.size() - 1 ? row2index.at(timestage_indices.at(i+1).row_idx_name) : num_row_;

    auto col_begin_idx = col2index.at(current_stage.col_idx_name);
    auto col_end_idx = i < timestage_indices.size() - 1 ? col2index.at(timestage_indices.at(i+1).col_idx_name) : num_col_;

    result[current_stage.stage_name] = {row_begin_idx, row_end_idx, col_begin_idx, col_end_idx};
  }

  return result;
}

bool SmpsCoreStructure::verify_stages(SmpsTimeStructure const & timestage_data, HighsNameHash const & row_hash, HighsNameHash const & col_hash) const {
  if (!is_valid_ || !timestage_data.is_valid()) return false;
  auto const & row2index = row_hash.name2index;
  auto const & col2index = col_hash.name2index;
  for (auto const & time_stage : timestage_data.get_entries()) {
    if (time_stage.row_idx_name != objective_name_  && row2index.find(time_stage.row_idx_name) == row2index.end()) return false;
    if (col2index.find(time_stage.col_idx_name) == col2index.end()) return false;
  }
  return true;
}

double Node::sum_children_prob() const {
  return std::accumulate(children.begin(), children.end(), 0.,
                [](double sum, std::unique_ptr<Node> const & node){return sum + node->node_probability;});
} 

bool Node::verify_children_probabilities() const {
  return is_leaf() || std::abs(sum_children_prob() - 1.) < 1e-6;
}

void Node::add_child(std::unique_ptr<Node> && child) {
  child->parent = this;
  children.push_back(std::move(child));
}

std::unique_ptr<SmpsStochasticStructure> read_stochastic_file(std::string const & filepath, SmpsCoreStructure const & core, SmpsTimeStructure const & time) {
  std::string header, problem_name, structure_type, distribution;
  std::ifstream(filepath) >> header >> problem_name >> structure_type >> distribution;
  if (distribution != "DISCRETE") return nullptr;
  if (structure_type == "INDEP")
    return std::unique_ptr<SmpsStochasticStructure>(new IndepStructure(filepath));
  if (structure_type == "BLOCK")
    return std::unique_ptr<SmpsStochasticStructure>(new BlockStructure(filepath));
  if (structure_type == "SCENARIO")
    return std::unique_ptr<SmpsStochasticStructure>(new ScenarioStructure(filepath, time, core));
  return nullptr;
}

IndepStructure::IndepStructure(std::istream & input)  {
  is_valid_ = read_from_file(input);
}

IndepStructure::IndepStructure(std::string const & filepath) {
  std::ifstream input(filepath);
  is_valid_ = read_from_file(input);
}

bool SmpsStochasticStructure::process_header(Tokens const & tokens, std::string & header, std::string & problem_name) const {
  if (tokens.size() != 2) return false;
  header = tokens[0];
  problem_name = tokens[1];
  return true;
}

bool SmpsStochasticStructure::process_structure(Tokens const & tokens, std::string & structure_type, std::string & distribution) const {
  if (tokens.size() != 2) return false;
  structure_type = tokens[0];
  distribution = tokens[1];
  return true;
}

bool IndepStructure::read_from_file(std::istream & input) {
    std::string header, problem_name, structure_type, distribution;
    return process_header(read_tokens(input), header, problem_name) &&
           process_structure(read_tokens(input), structure_type, distribution) &&    
            header == "STOCH" && structure_type == "INDEP" && distribution == "DISCRETE"
            && process_data(input) && !timestage_random_entries.empty();
}

bool IndepStructure::process_tokens(Tokens const & tokens, std::string & column, std::string & row, double & value, std::string & timeperiod, double & probability) const {
  if (tokens.size() != 5) return false;
  column = tokens[0];
  row = tokens[1];
  if (!str_to_dbl(tokens[2], value)) return false;
  timeperiod = tokens[3];
  return str_to_dbl(tokens[4], probability) && probability >= 0 && probability <= 1;
}

bool SmpsStochasticStructure::is_ending(Tokens const & tokens) const {
  return tokens.size() == 0 || tokens[0] == "ENDATA";
}

bool SmpsStochasticStructure::is_proper_ending(Tokens const & tokens) const {
    return tokens.size() == 1 && tokens[0] == "ENDATA";
}

bool IndepStructure::process_data(std::istream & input) {
  std::string temp_column, temp_row, temp_timeperiod, column, row, timeperiod, line;
  double value, probability;
  std::vector<RandomVariable::RandomValue> values;
  std::vector<RandomVariable> rvs;
  auto tokens = skip_initial_comments(input);
  if (!process_tokens(tokens, column, row, value, timeperiod, probability)) return false;
  values.emplace_back(probability, value);
  while (!(tokens = read_tokens(input)).empty() && !is_ending(tokens)) {
    if (is_comment(tokens)) continue;
    if (!process_tokens(tokens, temp_column, temp_row, value, temp_timeperiod, probability)) return false;
    if (temp_column != column || temp_row != row || temp_timeperiod != timeperiod) {
      rvs.emplace_back(column, row, values);
      values.clear();
    }
    if (temp_timeperiod != timeperiod) {
      timestage_random_entries.emplace_back(rvs, timeperiod);
      rvs.clear();
    }
    column = temp_column;
    row = temp_row;
    timeperiod = temp_timeperiod;
    values.emplace_back(probability, value);
  }
  rvs.emplace_back(column, row, values);
  timestage_random_entries.emplace_back(rvs, timeperiod);
  return is_proper_ending(tokens);
};

StochasticTree IndepStructure::constructTree() {
  auto root = std::unique_ptr<Node>(new Node("root"));
  std::vector<Node*> current_level = {root.get()}, next_level;
  for (auto const & timestage_rvs : timestage_random_entries) {
    for (auto const & random_vec_value : timestage_rvs.combine_variables())
      for (auto & node : current_level) {
        auto child = new Node(timestage_rvs.timestage, random_vec_value.probability, random_vec_value.lp_modifications);
        node->add_child(std::unique_ptr<Node>(child));
        next_level.push_back(child);
      }
    current_level = next_level;
    next_level.clear();
  }
  return StochasticTree(std::move(root));
}

RandomVector append_to_random_vector(RandomVariable const & rv, RandomVector const & rvec) {
  RandomVector result;
  if (rvec.size() == 0) {
    for (auto random_value : rv.values)
      result.emplace_back(random_value.probability, BlockLpEntry{{rv.row, rv.col, random_value.value}});
    return result;
  }
  for (auto random_value : rv.values)   
    for (auto const & vector_value : rvec) {
      double probability = random_value.probability * vector_value.probability;
      result.emplace_back(probability, vector_value.lp_modifications);
      LpEntry new_entry {rv.row, rv.col, random_value.value};
      result.back().lp_modifications.push_back(new_entry);
    }
  return result;
}

RandomVector TimestageRandomVariables::combine_variables() const {
  RandomVector result;
  for (auto const & rv: rvs) 
    result = append_to_random_vector(rv, result);
  return result;
}

//TODO: other class could also use that
Tokens read_tokens(std::istream & input) {
  std::string line;
  if (!getline(input, line)) return {};
  std::istringstream buffer(line);
  return {std::istream_iterator<std::string>(buffer), std::istream_iterator<std::string>()};
}

bool BlockStructure::read_from_file(std::istream & input) {
    std::string header, problem_name, structure_type, distribution;
    return process_header(read_tokens(input), header, problem_name) &&
           process_structure(read_tokens(input), structure_type, distribution) &&
           header == "STOCH" && structure_type == "BLOCKS" && distribution == "DISCRETE"
           && process_data(input) && !timestage_random_vectors.empty();
}

bool BlockStructure::is_new_block(Tokens const & tokens) const {
  return tokens.size() == 4 && tokens[0] == "BL";
}

bool BlockStructure::process_new_block(Tokens const & tokens, std::string & block_name, std::string & timeperiod, double & probability) const {
  if (tokens.size() != 4 || tokens[0] != "BL") return false;
  block_name = tokens[1];
  timeperiod = tokens[2];
  return str_to_dbl(tokens[3], probability) && probability >= 0 && probability <= 1;
}

bool BlockStructure::process_block_entry(Tokens const & tokens, std::string & column, std::string & row, double & value) const {
  if (tokens.size() != 3) return false;
  column = tokens[0];
  row = tokens[1];
  return str_to_dbl(tokens[2], value);
}

bool BlockStructure::has_timestage_changed(Tokens const & tokens, std::string const & timestage) const {
  return tokens.at(2) != timestage;
}

bool BlockStructure::has_block_changed(Tokens const & tokens, std::string const & block_name) const {
  return tokens.at(1) != block_name;
}

bool BlockStructure::process_data(std::istream & input) {
  std::string block_name, timeperiod, temp_timeperiod, column, row, line;
  double value, probability;
  BlockLpEntry in_block_entries;
  RandomVector rv;
  std::vector<RandomVector> rvs;
  
  auto tokens = skip_initial_comments(input);
  if (!is_new_block(tokens) || !process_new_block(tokens, block_name, timeperiod, probability))
    return false;
  while (!(tokens = read_tokens(input)).empty() && !is_ending(tokens)) {
    if (is_comment(tokens)) continue;
    if (is_new_block(tokens)) {
      if(in_block_entries.empty()) return false;
      rv.emplace_back(probability, in_block_entries);
      in_block_entries.clear();
      if (has_block_changed(tokens, block_name) || has_timestage_changed(tokens, timeperiod)) {
        rvs.push_back(rv);
        rv.clear();
      }
      if (has_timestage_changed(tokens, timeperiod)) {
        timestage_random_vectors.emplace_back(rvs, timeperiod);
        rvs.clear();
      }
      if (!process_new_block(tokens, block_name, timeperiod, probability)) return false;
    }
    else {
      if (!process_block_entry(tokens, column, row, value)) return false;
      in_block_entries.emplace_back(row, column, value);
    }
  }
  if(in_block_entries.empty()) return false;
  rv.emplace_back(probability, in_block_entries);
  rvs.push_back(rv);
  timestage_random_vectors.emplace_back(rvs, timeperiod);
  for (auto & rvs : timestage_random_vectors)
    for (auto & rv : rvs.rvs)
      if (!rv.fill_missing_entries()) return false;
  return is_proper_ending(tokens);
}

StochasticTree BlockStructure::constructTree() {
  auto root = std::unique_ptr<Node>(new Node("root"));
  std::vector<Node*> current_level = {root.get()}, next_level;
  for (auto const & timestage_rvs : timestage_random_vectors) {
    for (auto const & random_vec_value : timestage_rvs.combine_vectors())
      for (auto node : current_level) {
        auto child = new Node(timestage_rvs.timestage, random_vec_value.probability, random_vec_value.lp_modifications);
        node->add_child(std::unique_ptr<Node>(child));
        next_level.push_back(child);
      }
    current_level = next_level;
    next_level.clear();
  }
  return StochasticTree(std::move(root));
}

bool str_to_dbl(std::string const & str, double & val) {
  int pos;
  return sscanf(str.c_str(), "%lf%n", &val, &pos) == 1 && pos == str.length();
}

bool ScenarioStructure::is_new_scenario(Tokens const & tokens) const {
  return tokens.size() == 5 && tokens[0] == "SC";
}

bool ScenarioStructure::process_new_scenario(Tokens const & tokens, std::string & scenario_name, std::string & parent_scenario,
                           std::string & timeperiod, double & probability) const {
  if (tokens.size() != 5 || tokens[0] != "SC") return false;
  scenario_name = tokens[1];
  parent_scenario = tokens[2];
  timeperiod = tokens[4];
  return str_to_dbl(tokens[3], probability) && probability >= 0 && probability <= 1;
}

bool ScenarioStructure::process_scenario_entry(Tokens const & tokens, std::string & column, std::string & row, double & value) const {
  if (tokens.size() != 3) return false;
  column = tokens[0];
  row = tokens[1];
  return str_to_dbl(tokens[2], value);
}

bool ScenarioStructure::process_data(std::istream & input) {
  std::string scenario_name, parent_scenario, timeperiod, temp_timeperiod, column, row, line;
  double value, probability;
  BlockLpEntry in_scenario_entries;  
  auto tokens = skip_initial_comments(input);
  if (!is_new_scenario(tokens) || !process_new_scenario(tokens, scenario_name, parent_scenario, timeperiod, probability))
    return false;
  while (!(tokens = read_tokens(input)).empty() && !is_ending(tokens)) {
    if (is_comment(tokens)) continue;
    if (is_new_scenario(tokens)) {
        if (in_scenario_entries.empty()) return false;
        scenarios.emplace_back(in_scenario_entries, probability, timeperiod, scenario_name, parent_scenario);
        in_scenario_entries.clear();
        if (!process_new_scenario(tokens, scenario_name, parent_scenario, timeperiod, probability))  return false;
    } else {
      if (!process_scenario_entry(tokens, column, row, value)) return false;
      in_scenario_entries.emplace_back(row, column, value);    
    }
  }
  if (in_scenario_entries.empty()) return false;
  scenarios.emplace_back(in_scenario_entries, probability, timeperiod, scenario_name, parent_scenario);
  return is_proper_ending(tokens);
}

bool ScenarioStructure::read_from_file(std::istream & input) {
    std::string structure_type, distribution;
    return  process_structure(read_tokens(input), structure_type, distribution) &&
            structure_type == "SCENARIOS" && distribution == "DISCRETE"
            && process_data(input) && !scenarios.empty();
}

bool ScenarioStructure::construct_parent_mapping() {
  for (int i = 0; i < scenarios.size(); ++i) {
    auto const & scen = scenarios[i];
    name2scen_idx.emplace(scen.scenario_name, i);
    if (scen.parent_scenario != "ROOT" && name2scen_idx.find(scen.parent_scenario) == name2scen_idx.end())
        return false;
  }
  return true;
}

StochasticTree ScenarioStructure::constructTree() {
  auto root = std::unique_ptr<Node>(new Node("root"));
  std::map<std::pair<std::string, std::string>, Node *> scen_time2node {{{"ROOT", "ROOT"}, root.get()}};
  for (auto & scen : scenarios) {
    auto parent_timestage = scen.parent_scenario == "ROOT" ? "ROOT" : get_scenario(scen.parent_scenario).timestage;
    if (scen.parent_scenario != "ROOT") scen += get_scenario(scen.parent_scenario);
    Node * parent = scen_time2node.at({scen.parent_scenario, parent_timestage});
    for (int t = time.get_stage_index(scen.timestage); t < time.get_no_timestages(); ++t) {
      auto timestage = time.get_timestage(t);
      auto path_prob = t == time.get_no_timestages() - 1 ? scen.probability : 0;
      auto modifications = scen.filter_by_proper_timestage(core, timestage);
      auto child = new Node(timestage, path_prob, modifications);
      scen_time2node[{scen.scenario_name, timestage}] = child;
      parent->add_child(std::unique_ptr<Node>(child));
      parent = child;
    }
  }
  return root->rescale_tree_to_leaf_probability() ? StochasticTree(std::move(root)) : nullptr;
}

bool Node::rescale_to_children_probability() {
  if (is_leaf() && node_probability > 0) return true;
  auto child_prob = sum_children_prob();
  if (child_prob == 0) return false;
  node_probability = child_prob;
  for (auto & child : children) child->node_probability /= node_probability;
  return true;
}

bool Node::rescale_tree_to_leaf_probability() {
  for (auto & child : children)
    if(!child->rescale_tree_to_leaf_probability())
      return false;
  return rescale_to_children_probability();
}

bool RandomVector::fill_missing_entries() {
  if (size() == 0) return false;
  RandomVectorValue const & block_basis = at(0);
  std::for_each(begin() + 1, end(), [&block_basis](RandomVectorValue & entry) { entry += block_basis;});
  return std::all_of(cbegin() + 1, cend(), [&block_basis](RandomVectorValue const & entry) {
                     return entry.lp_modifications.size() == block_basis.lp_modifications.size(); });
}

void RandomVectorValue::operator+=(RandomVectorValue const & basis) {
  for (BlockLpEntry::size_type i = 0; i < basis.lp_modifications.size(); ++i) 
    if (i >= lp_modifications.size() || at(i).row != basis.at(i).row || at(i).col != basis.at(i).col)
      lp_modifications.insert(lp_modifications.begin() + i, basis.at(i));
}

bool SmpsStochasticStructure::is_comment(Tokens const & tokens) const {
  return tokens.size() > 0 && tokens.front().at(0) == '*';
}

Tokens SmpsStochasticStructure::skip_initial_comments(std::istream & input) const {
  Tokens tokens;
  while (!(tokens = read_tokens(input)).empty() && !is_ending(tokens) && is_comment(tokens))
    continue;
  return tokens;
 }

RandomVector TimestageRandomVectors::combine_vectors() const {
  RandomVector result;
  for (auto const & rv: rvs) 
    result = append_to_random_vector(rv, result);
  return result;
}

RandomVector append_to_random_vector(RandomVector const & to_append, RandomVector const & rvec) {
  if (rvec.size() == 0) return to_append;
  RandomVector result;
  for (auto const & vector_value : rvec) 
    for (auto const & random_value : to_append) {
        auto entries = vector_value.lp_modifications;
        entries.insert(entries.end(), random_value.lp_modifications.begin(), random_value.lp_modifications.end());
        double probability = random_value.probability * vector_value.probability;
        result.emplace_back(probability, entries);
      }
  return result;
}

// TODO move semantics?
void Node::insert_intermediate_child(Node * intermediate_child, std::unique_ptr<Node> & current_child, bool swap_probability) {
  if (swap_probability) {
    auto current_child_prob = current_child->node_probability;
    current_child->node_probability = intermediate_child->node_probability;
    intermediate_child->node_probability = current_child_prob; 
  }
  auto current_child_ptr = current_child.release();
  intermediate_child->add_child(std::unique_ptr<Node> {current_child_ptr});
  current_child.reset(intermediate_child);
  current_child->parent = this;
}

//TODO BOUNDS!!!!!
void add_node_entry(SmpsCoreStructure const & core, Node & node, Highs & result) {
    // TODO: redundant looping
    auto const & node_ranges = core.stage_submatrix.at(node.get_timestage());
    auto node_modifications = core.annotate_lp_entries(node.get_lp_modifications());
    node.set_in_problem_range(node_ranges.create_in_problem_range(result));
    node_ranges.expand_problem_by_range_vars(result, core.col_lower_, core.col_upper_);
    IdxTranslator translator {core.stage_submatrix, create_stochastic_path_translation(node)};
    for (int row = node_ranges.row_idx_begin; row < node_ranges.col_idx_end; ++row) {
      auto LB = core.row_lower_.at(row);
      auto UB = core.row_upper_.at(row);
      auto row_data = SparseVector::get_matrix_row(core.a_matrix_, row);
      for (auto const & mod : node_modifications)
        if (mod.row_idx == row && mod.is_rhs) {
          LB = update_lb(LB, mod.value);
          UB = update_ub(UB, mod.value);
        } else if (mod.row_idx == row)
           row_data.set(mod.col_idx, mod.value);
      row_data.translate_to_in_problem(translator);
      result.addRow(LB, UB, row_data.num_nz(), row_data.nz_indices.data(), row_data.nz_values.data());
    }        
    // TODO prepare and a single call to changeColCosts
    auto prob = node.get_in_tree_probability();
    for (int col = node_ranges.col_idx_begin; col < node_ranges.col_idx_end; ++col)
      if (core.col_cost_.at(col) != 0)
        result.changeColCost(translator(col), core.col_cost_.at(col) * prob);
    for (auto const & mod : node_modifications)
      if (mod.is_objective) 
        result.changeColCost(translator(mod.col_idx), mod.value * prob);
}

void add_node_tree_entries(SmpsCoreStructure const & core, Node & node, Highs & result) {
  add_node_entry(core, node, result); 
  for (int i = 0; i < node.get_no_children(); ++i)  add_node_tree_entries(core, *node.get_child(i), result);
}

//TODO const tree?
// TODO bread first?
void add_tree_entries(const SmpsCoreStructure &core, const StochasticTree &tree, Highs &result) {
  for (int i = 0; i < tree.root->get_no_children(); ++i) add_node_tree_entries(core, *tree.root->get_child(i), result);
}

LpIdxEntry SmpsCoreStructure::annotate_lp_entry(LpEntry const & entry) const {
  bool is_objective = entry.row == objective_name_;
  bool is_rhs = entry.col == "RHS";
  return {entry, is_objective, is_rhs, is_objective ? -1 : row_hash_.name2index.at(entry.row), is_rhs ? -1 : col_hash_.name2index.at(entry.col)};
}

std::vector<LpIdxEntry> SmpsCoreStructure::annotate_lp_entries(std::vector<LpEntry> const & entries) const {
  std::vector<LpIdxEntry> result; result.reserve(entries.size());
  std::transform(entries.begin(), entries.end(), std::back_inserter(result), [this](LpEntry const & entry) { return annotate_lp_entry(entry); });
  return result;
}

void SparseVector::set(unsigned index, double value) {
  auto idx  = std::distance(nz_indices.begin(), std::find(nz_indices.begin(), nz_indices.end(), index));
  if (idx < nz_indices.size()) {
    if (value == 0.) {
      nz_indices.erase(nz_indices.begin() + idx);
      nz_values.erase(nz_values.begin() + idx);
    } 
    else 
      nz_values.at(idx) = value;
    return;
  }
  if (value == 0.) return;
  idx = std::distance(nz_indices.begin(), std::lower_bound(nz_indices.begin(), nz_indices.end(), index));
  nz_indices.insert(nz_indices.begin() + idx, index);
  nz_values.insert(nz_values.begin() + idx, value);
}

double SparseVector::operator[](unsigned index) const {
  auto idx  = std::distance(nz_indices.begin(), std::find(nz_indices.begin(), nz_indices.end(), index));
  return idx < nz_indices.size() ? nz_values.at(idx) : 0;
}

void SparseVector::truncate(int num_nz) {
  if (num_nz > nz_indices.size()) return;
  nz_indices.resize(num_nz);
  nz_values.resize(num_nz);
 }

bool SubMatrixRange::expand_problem_by_range_vars(
  Highs & problem, std::vector<double> const & col_lower, std::vector<double> const & col_upper) const {
  //TODO integrality!
  auto status = problem.addVars(num_cols(),  col_lower.data() + col_idx_begin, col_upper.data() + col_idx_begin );
  return status ==  HighsStatus::kOk;
}

SparseVector SparseVector::get_matrix_row(HighsSparseMatrix const & A, int row_idx) {
    int num_nz;
    // TODO: memory saving
    SparseVector row_data {std::vector<int>(A.num_col_), std::vector<double>(A.num_col_)};
    A.getRow(row_idx, num_nz, row_data.nz_indices.data(), row_data.nz_values.data());
    row_data.truncate(num_nz);
    return row_data;
}

Timestage2Range create_stochastic_path_translation(Node const & node) {
  Timestage2Range timeperiod2range;
  for (Node const * parent = &node; parent != nullptr; parent = parent->get_parent())
    timeperiod2range[parent->get_timestage()] = parent->get_in_problem_range();
  return timeperiod2range;
}

void SparseVector::translate_to_in_problem(IdxTranslator const & translator) {
  std::transform(nz_indices.begin(), nz_indices.end(), nz_indices.begin(), [&translator](int idx) {return translator(idx);});
}

//TODO -1 val
int IdxTranslator::operator()(int idx) const {
  // TODO this can be made much faster
  auto it = std::find_if(in_core.begin(), in_core.end(),
                         [idx](std::pair<std::string, SubMatrixRange> const & val) { return idx < val.second.col_idx_end;});
  if (it == in_core.end()) return -1;
  int in_range_shift = idx - it->second.col_idx_begin;
  return in_problem.at(it->first).col_idx_begin + in_range_shift;
}

double Node::get_in_tree_probability() const {
  // TODO sum algorithm?
  auto probability = 1.;
  for (Node const * parent = this; parent != nullptr; parent = parent->get_parent())
    probability *= parent->get_node_probability();
  return probability;
}

bool build_stochastic_problem(Highs & problem,
                              std::string const & core_filename,
                              std::string const & time_filename,
                              std::string const & stoch_filename,
                              HighsOptions const & highs_mps_options) {
  
    SmpsCoreStructure core(highs_mps_options, core_filename);
    if (!core.is_valid()) return false;

    SmpsTimeStructure time(time_filename);
    if (!time.is_valid()) return false;

    auto stoch = read_stochastic_file(stoch_filename, core, time);
    if (stoch == nullptr || !stoch->is_valid()) return false;
    auto tree = stoch->constructTree();

    if (!core.load_time_stages(time)) return false;
    //TODO some checking should be done here
    add_tree_entries(core, tree, problem);
    return true;
}

bool build_stochastic_problem(Highs & problem, std::string const & mutual_name, HighsOptions const & highs_mps_options) {
  return build_stochastic_problem(problem, mutual_name + ".cor", mutual_name + ".tim", mutual_name + ".sto", highs_mps_options);
}

std::string SmpsCoreStructure::entry2stage(LpEntry const & entry) const {
  if (entry.row == objective_name_) {
    int col_idx = col_hash_.name2index.at(entry.col);
    for (auto const & time_submatrix : stage_submatrix)
      if (col_idx < time_submatrix.second.col_idx_end) return time_submatrix.first;
  }
  else {
    int row_idx = row_hash_.name2index.at(entry.row);
    for (auto const & time_submatrix : stage_submatrix)
      if (row_idx < time_submatrix.second.row_idx_end) return time_submatrix.first;
  }
  //TODO better error return
  return "";
}

BlockLpEntry ScenarioModifications::filter_by_proper_timestage(SmpsCoreStructure const & core, std::string const & timestage) const {
  BlockLpEntry modifications;
  std::copy_if(lp_modifications.begin(), lp_modifications.end(), std::back_inserter(modifications),
                [&core, &timestage](LpEntry const & entry) { return core.entry2stage(entry) == timestage;});
  return modifications;
}

void ScenarioModifications::operator+=(ScenarioModifications const & base) {
  for (auto const & mod : base.lp_modifications)
    if (lp_modifications.end() == std::find_if(lp_modifications.begin(), lp_modifications.end(),
                      [&mod](LpEntry const & entry) {return mod.col == entry.col && mod.row == entry.row;}))
      lp_modifications.push_back(mod);
}
