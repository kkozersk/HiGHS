#include "SMPS.h"
#include <algorithm>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "Filereader.h"
#include "FilereaderMps.h"
#include "HStruct.h"
#include "HighsModel.h"
#include "HighsOptions.h"

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
  col_stages.push_back({starting_column, stage_name});
  row_stages.push_back({starting_row, stage_name});
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
    return std::unique_ptr<SmpsStochasticStructure>(new IndepStructure(problem_name, filepath));
  // if (structure_type == "BLOCK")
  //   return std::unique_ptr<SmpsStochasticStructure>(new BlockStructure(problem_name, filepath));
  // if (structure_type == "SCENARIO")
  //   return std::unique_ptr<SmpsStochasticStructure>(new ScenarioStructure(problem_name, filepath));
  return nullptr;
}

void read_from_file(std::istream & input);
IndepStructure::IndepStructure(std::string const & problem_name, std::istream & input) : SmpsStochasticStructure(problem_name) {
  is_valid_ = read_from_file(input);
}

IndepStructure::IndepStructure(std::string const & problem_name, std::string const & filepath) : SmpsStochasticStructure(problem_name) {
  std::ifstream input(filepath);
  is_valid_ = read_from_file(input);
  input.close();
}

bool IndepStructure::read_from_file(std::istream & input) {
    std::string header, problem_name, structure_type, distribution;
    input >> header >> problem_name >> structure_type >> distribution;
    return header == "STOCH" && problem_name == get_problem_name() && structure_type == "INDEP" && distribution == "DISCRETE"
            && process_data(input) && !modifications.empty();
}

bool IndepStructure::process_data(std::istream & input) {
  std::string temp_column, temp_row, temp_timeperiod, column, row, timeperiod;
  double value, probability;
  std::vector<RandomVariable::RandomValue> values;
  std::vector<RandomVariable> rvs;
  std::vector<TimestageRandomVariables> timestage_rvs;
  input >> column >> row >> value >> timeperiod >> probability;
  if (column == "" || row == "" || timeperiod == "" || probability < 0 || probability > 1) return false;
  values.push_back({probability, value});
  while (temp_column != "ENDATA") {
    input >> temp_column >> temp_row >> value >> temp_timeperiod >> probability;
    if (temp_column == "ENDATA" || temp_column != column || temp_row != row || temp_timeperiod != timeperiod) {
      if (!values.empty()) rvs.push_back({column, row, values});
      values.clear();
      column = temp_column;
      row = temp_row;
    }
    values.push_back({probability, value});
    if (temp_column == "ENDATA" || temp_timeperiod != timeperiod) {
      modifications.push_back({rvs, temp_timeperiod});
      rvs.clear();
      timeperiod = temp_timeperiod;
    }
  }
  return true;
};

StochasticTree IndepStructure::constructTree(SmpsTimeStructure const & timestructure) const {
  auto root = std::unique_ptr<Node>(new Node("root"));
  std::vector<Node*> current_level = {root.get()}, next_level;
  for (auto const & timestage_rvs : modifications) {
    for (auto const & random_vec_value : timestage_rvs.generate_vector())
      for (auto node : current_level) {
        auto child = new Node(timestage_rvs.timestage, random_vec_value.probability, random_vec_value.modifications);
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
      result.push_back({random_value.probability, {{rv.row, rv.col, random_value.value}}});
    return result;
  }
  for (auto random_value : rv.values)   
    for (auto const & vector_value : rvec) {
      double probability = random_value.probability * vector_value.probability;
      result.push_back({probability, vector_value.modifications});
      LpEntry new_entry {rv.row, rv.col, random_value.value};
      result.back().modifications.push_back(new_entry);
    }
  return result;
}

RandomVector TimestageRandomVariables::generate_vector() const {
  RandomVector result;
  for (auto const & rv: rvs) 
    result = append_to_random_vector(rv, result);
  return result;
}
