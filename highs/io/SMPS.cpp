#include "SMPS.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "Filereader.h"
#include "FilereaderMps.h"
#include "lp_data/HStruct.h"
#include "model/HighsModel.h"
#include "lp_data/HighsOptions.h"

void SmpsTimeStructure::read_file(std::istream & input) {
  std::string line;
  // getline(input, line);
  if (!getline(input, line) || !process_header(line)) return;
  if (!getline(input, line) || !process_period_header(line)) return;
  while (getline(input, line) && !is_ending(line)) {
    if (!process_period(line)) return;
  }
  if (col_stages.empty()) return;
  std::transform(col_stages.begin(), col_stages.end(), std::back_inserter(stage_names), [](IndexStage const & val) {return val.stage_name; });
  if (!process_ending(line)) return;
  is_valid_ = true;
  // input.close();
}

bool SmpsTimeStructure::process_header(std::string const & line) {
  // if (line.empty()) return false;
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
  col_stages.emplace_back(starting_column, stage_name);
  row_stages.emplace_back(starting_row, stage_name);
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

SmpsCoreStructure::SmpsCoreStructure(HighsOptions const & options, std::string const & filepath) {
  FilereaderMps mps;
  HighsModel model;
  is_valid_ = mps.readModelFromFile(options, filepath, model) == FilereaderRetcode::kOk;
  if (is_valid_) {
    HighsLp::operator=(model.lp_);
    if (!col_hash_.name2index.size()) col_hash_.form(col_names_);
    if (!row_hash_.name2index.size()) row_hash_.form(row_names_);
  }
}

bool SmpsCoreStructure::load_time_stages(SmpsTimeStructure const & time_stage_data) {
  if (!verify_time_stages(time_stage_data)) return false;
  col_time_stage = load_stages(time_stage_data.get_col_stages(), col_hash_, num_col_);
  row_time_stage = load_stages(time_stage_data.get_row_stages(), row_hash_, num_row_);
  return true;
}

std::vector<string> SmpsCoreStructure::load_stages(std::vector<IndexStage> const & stage_idx_data, HighsNameHash const & name_hash, int num_entries) {
  auto const & name2index = name_hash.name2index;
  std::vector<string> result(num_entries, stage_idx_data.back().stage_name);
  for (int i = 0; i < stage_idx_data.size() - 1; ++i) {
    auto const & time_stage = stage_idx_data.at(i);
    auto const & next_time_stage = stage_idx_data.at(i+1);
    auto stage_begin_idx = time_stage.starting_idx_name == objective_name_ ? 0 : name2index.at(time_stage.starting_idx_name);
    auto stage_end_idx = name2index.at(next_time_stage.starting_idx_name);
    std::fill(result.begin() + stage_begin_idx, result.begin() + stage_end_idx, time_stage.stage_name);
  }
  return result;
}

bool SmpsCoreStructure::verify_time_stages(SmpsTimeStructure const & time_stage_data) const {
  return is_valid_ && time_stage_data.is_valid() &&
      verify_stages(time_stage_data.get_col_stages(), col_hash_) &&
      verify_stages(time_stage_data.get_row_stages(), row_hash_);
}

bool SmpsCoreStructure::verify_stages(std::vector<IndexStage> const & idx_time_stages, HighsNameHash const & name_hash) const {
  for (auto const & time_stage : idx_time_stages) {
    if (name_hash.name2index.find(time_stage.starting_idx_name) == name_hash.name2index.end() 
      && time_stage.starting_idx_name != objective_name_)
      return false;
  }
  return true;
}

bool Node::verify_children_probabilities() const {
  double total_prob =  std::accumulate(children.begin(), children.end(), 0.,
                [](double sum, std::unique_ptr<Node> const & node){return sum + node->node_probability;});
  return std::abs(total_prob - 1.) < 1e-6;
}

void Node::add_child(std::unique_ptr<Node> && child) {
  child->parent = this;
  children.push_back(std::move(child));
}

std::unique_ptr<SmpsStochasticStructure> read_stochastic_file(std::string const & filepath) {
  std::string header, problem_name, structure_type, distribution;
  std::ifstream(filepath) >> header >> problem_name >> structure_type >> distribution;
  if (distribution != "DISCRETE") return nullptr;
  if (structure_type == "INDEP")
    return std::unique_ptr<SmpsStochasticStructure>(new IndepStructure(filepath));
  if (structure_type == "BLOCK")
    return std::unique_ptr<SmpsStochasticStructure>(new BlockStructure(filepath));
  if (structure_type == "SCENARIO")
    return std::unique_ptr<SmpsStochasticStructure>(new ScenarioStructure(filepath));
  return nullptr;
}

IndepStructure::IndepStructure(std::istream & input)  {
  is_valid_ = read_from_file(input);
}

IndepStructure::IndepStructure(std::string const & filepath) {
  std::ifstream input(filepath);
  is_valid_ = read_from_file(input);
  input.close();
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
    std::string header, problem_name, structure_type, distribution, line;
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
  auto tokens = read_tokens(input);
  if (!process_tokens(tokens, column, row, value, timeperiod, probability)) return false;
  values.emplace_back(probability, value);
  while (!(tokens = read_tokens(input)).empty() && !is_ending(tokens)) {
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

StochasticTree IndepStructure::constructTree(SmpsTimeStructure const & timestructure) const {
  auto root = std::unique_ptr<Node>(new Node("root"));
  std::vector<Node*> current_level = {root.get()}, next_level;
  for (auto const & timestage_rvs : timestage_random_entries) {
    for (auto const & random_vec_value : timestage_rvs.generate_vector())
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

RandomVector TimestageRandomVariables::generate_vector() const {
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
    std::string header, problem_name, structure_type, distribution, line;
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
  return tokens[2] != timestage;
}

bool BlockStructure::process_data(std::istream & input) {
  std::string block_name, timeperiod, temp_timeperiod, column, row, line;
  double value, probability;
  BlockLpEntry in_block_entries;
  RandomVector rv;
  
  auto tokens = read_tokens(input);
  if (!is_new_block(tokens) || !process_new_block(tokens, block_name, timeperiod, probability))
    return false;
  while (!(tokens = read_tokens(input)).empty() && !is_ending(tokens)) {
    if (is_new_block(tokens)) {
      rv.emplace_back(probability, in_block_entries);
      in_block_entries.clear();
      if (has_timestage_changed(tokens, timeperiod)) {
        timestage_random_vectors.emplace_back(rv, timeperiod);
        rv.clear();
      }
      if (!process_new_block(tokens, block_name, timeperiod, probability)) return false;
    }
    else {
      if (!process_block_entry(tokens, column, row, value)) return false;
      in_block_entries.emplace_back(row, column, value);
    }
  }
  rv.emplace_back(probability, in_block_entries);
  timestage_random_vectors.emplace_back(rv, timeperiod);
  return is_proper_ending(tokens);
}

StochasticTree BlockStructure::constructTree(SmpsTimeStructure const & timestructure) const {
  auto root = std::unique_ptr<Node>(new Node("root"));
  std::vector<Node*> current_level = {root.get()}, next_level;
  for (auto const & timestage_rvs : timestage_random_vectors) {
    for (auto const & random_vec_value : timestage_rvs.rvs)
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
  auto tokens = read_tokens(input);
  if (!is_new_scenario(tokens) || !process_new_scenario(tokens, scenario_name, parent_scenario, timeperiod, probability))
    return false;
  while (!(tokens = read_tokens(input)).empty() && !is_ending(tokens)) {
    if (is_new_scenario(tokens)) {
        scenarios.emplace_back(in_scenario_entries, probability, timeperiod, scenario_name, parent_scenario);
        in_scenario_entries.clear();
        if (!process_new_scenario(tokens, scenario_name, parent_scenario, timeperiod, probability))  return false;
    } else {
      if (!process_scenario_entry(tokens, column, row, value)) return false;
      in_scenario_entries.emplace_back(row, column, value);    
    }
  }
  scenarios.emplace_back(in_scenario_entries, probability, timeperiod, scenario_name, parent_scenario);
  return is_proper_ending(tokens);
}

bool ScenarioStructure::read_from_file(std::istream & input) {
    std::string structure_type, distribution, line;
    return  process_structure(read_tokens(input), structure_type, distribution) &&
            structure_type == "SCENARIOS" && distribution == "DISCRETE"
            && process_data(input) && !scenarios.empty();
}

StochasticTree ScenarioStructure::constructTree(SmpsTimeStructure const &) const {
  return {nullptr};
}
