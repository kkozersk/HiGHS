#pragma once
#include <fstream>
#include <istream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "lp_data/HighsLp.h"

using Tokens = std::vector<std::string>;

struct TimeStage {
  std::string starting_column;
  std::string starting_row;
  std::string stage_name;
};


struct IndexStage {
  std::string starting_idx_name;
  std::string stage_name;
  bool operator==(IndexStage const & other) const { return other.starting_idx_name == starting_idx_name && other.stage_name == stage_name; }
  IndexStage(std::string const & starting_idx_name, std::string const & stage_name) : starting_idx_name(starting_idx_name), stage_name(stage_name) {}
};

class SmpsTimeStructure {
  public:
    SmpsTimeStructure(std::string const & filepath) {std::ifstream input(filepath.c_str()); read_file(input);};
    SmpsTimeStructure(std::istream & input_stream) { read_file(input_stream); };
    bool is_valid() const { return is_valid_; }
    std::string get_problem_name() const { return problem_name; }
    std::vector<std::string> get_stage_names() const { return stage_names; }
    std::vector<IndexStage> const & get_row_stages() const { return row_stages; }
    std::vector<IndexStage> const & get_col_stages() const { return col_stages; }
    int get_stage_index(std::string const & stage) const;
  private:
    void read_file(std::istream & input);

    std::vector<IndexStage> row_stages, col_stages;
    std::vector<std::string> stage_names;
    bool is_valid_ = false;
    std::string problem_name;

    bool process_header(std::string const & line);
    bool process_period_header(std::string const & line);
    bool process_period(std::string const & line);
    bool is_ending(std::string const & line) const;
    bool process_ending(std::string const & line);
};

class SmpsCoreStructure : public HighsLp {
  public:
    SmpsCoreStructure(HighsOptions const & options, std::string const & filepath);
    bool is_valid() const { return is_valid_; }
    bool load_time_stages(SmpsTimeStructure const & time_stage_data);

    std::vector<std::string> row_time_stage;
    std::vector<std::string> col_time_stage;
    
  private:
    bool is_valid_ = false;
    bool verify_time_stages(SmpsTimeStructure const & time_stage_data) const;
    bool verify_stages(std::vector<IndexStage> const & idx_time_stages, HighsNameHash const & name_hash) const;
    std::vector<string> load_stages(std::vector<IndexStage> const & stage_idx_data, HighsNameHash const & name_hash, int num_entries);
};

struct LpEntry {
  std::string row;
  std::string col; // can be RHS
  double value;

  bool operator==(LpEntry const & other) const { return row == other.row && col == other.col && value == other.value; }
  LpEntry(std::string const & row, std::string const & col, double value) : row(row), col(col), value(value) {}
};

using BlockLpEntry = std::vector<LpEntry>;

class Node {
  std::vector<std::unique_ptr<Node>> children;
  Node * parent = nullptr;
  std::vector<LpEntry> lp_modifications;
  double node_probability; // TODO 0 <= p <= 1
  std::string timestage;

  double sum_children_prob() const; 
  public:
    Node(std::string timestage, double node_probability = 1., std::vector<LpEntry> const & lp_modifications={}):
        lp_modifications(lp_modifications), node_probability(node_probability), timestage(timestage) {}
    void add_child(std::unique_ptr<Node> && child);
    bool verify_children_probabilities() const;
    std::unique_ptr<Node> & get_child(int index) { return children.at(index); }
    int get_no_children() const { return children.size(); }
    // std::vector<std::unique_ptr<Node>> const & get_children() { return children; }
    Node const * get_parent() const { return parent; };
    std::vector<LpEntry> get_lp_modifications() const { return lp_modifications; }
    double get_node_probability() const { return node_probability; }
    void fill_missing_child();
    void fill_tree();
    bool is_leaf() const { return children.empty(); }
  // TimeStage timestage;
  // double get_in_tree_probability() const;
};

struct StochasticTree {
  std::unique_ptr<Node> root;
  StochasticTree(std::unique_ptr<Node> && root) : root(std::move(root)) {}
  void fill_tree() { root->fill_tree(); }
};

class SmpsStochasticStructure {
  private:
    virtual bool process_data(std::istream & input) = 0;
    virtual bool read_from_file(std::istream & input) = 0;
  protected:
    bool process_header(Tokens const & tokens, std::string & header, std::string & problem_name) const;
    bool process_structure(Tokens const & tokens, std::string & structure_type, std::string & distribution) const; 
    std::string problem_name = "";
    bool is_ending(Tokens const & tokens) const;
    bool is_proper_ending(Tokens const & tokens) const;
    bool is_comment(Tokens const & tokens) const;
    bool is_valid_ = false;
    Tokens skip_initial_comments(std::istream & input) const;
  public:
    SmpsStochasticStructure() = default;
    virtual ~SmpsStochasticStructure() = default;
    virtual StochasticTree constructTree() const = 0;
    SmpsStochasticStructure(std::string problem_name) : problem_name(problem_name) {}
    std::string get_problem_name() const { return problem_name; }
    bool is_valid() const { return is_valid_; }
};

struct RandomVariable {
  struct RandomValue {
    double probability, value;
    bool operator==(RandomValue const & other) const { return probability == other.probability && value == other.value; }
    RandomValue(double probability, double value) : probability(probability), value(value) {}
  };
  std::string col, row;
  std::vector<RandomValue> values;
  bool operator==(RandomVariable const & other) const { return col == other.col && row == other.row && values == other.values; }
  RandomVariable(std::string const & col, std::string const & row, std::vector<RandomValue> const & values) : col(col), row(row), values(values) {}
};

struct RandomVectorValue {
  double probability;
  BlockLpEntry lp_modifications;
  bool operator==(RandomVectorValue const & other) const { return probability == other.probability && lp_modifications == other.lp_modifications; }
  RandomVectorValue(double probability, BlockLpEntry const & lp_modifications) : probability(probability), lp_modifications(lp_modifications) {}
  void operator+=(RandomVectorValue const & basis);
  LpEntry const & at(int index) const { return lp_modifications.at(index); }
 };

class RandomVector : public std::vector<RandomVectorValue> {
  public:
    using std::vector<RandomVectorValue>::vector;
    bool fill_missing_entries();
};

struct TimestageRandomVariables {
  std::vector<RandomVariable> rvs;
  std::string timestage;
  RandomVector combine_variables() const;
  bool operator==(TimestageRandomVariables const & other) const { return rvs == other.rvs && timestage == other.timestage; }
  TimestageRandomVariables(std::vector<RandomVariable> const & rvs, std::string const & timestage) : rvs(rvs), timestage(timestage) {}
};

struct TimestageRandomVectors {
  std::vector<RandomVector> rvs;
  std::string timestage;
  RandomVector combine_vectors() const;
  bool operator==(TimestageRandomVectors const & other) const { return rvs == other.rvs && timestage == other.timestage; }
  TimestageRandomVectors(std::vector<RandomVector> const & rvs, std::string const & timestage) : rvs(rvs), timestage(timestage) {}
  // bool fill_missing_entries();
};


class IndepStructure : public SmpsStochasticStructure {
  std::vector<TimestageRandomVariables> timestage_random_entries;

  bool process_tokens(Tokens const & tokens, std::string & column, std::string & row, double & value, std::string & timeperiod, double & probability) const;
  bool process_data(std::istream & input);
  bool read_from_file(std::istream & input);
  public:
    IndepStructure(std::istream & input);
    IndepStructure(std::string const & filename);
    virtual StochasticTree constructTree() const;
    int get_no_timestage_random_entries() const { return timestage_random_entries.size(); }
    TimestageRandomVariables const & get_timestage_random_entry(int index) { return timestage_random_entries.at(index); }
    // std::vector<TimestageRandomVariables> const & get_modifications() { return modifications; };
};

class BlockStructure : public SmpsStochasticStructure {
  std::vector<TimestageRandomVectors> timestage_random_vectors;
  bool process_data(std::istream & input);
  bool read_from_file(std::istream & input);
  bool is_new_block(Tokens const & tokens) const;
  bool process_new_block(Tokens const & tokens, std::string & block_name, std::string & timeperiod, double & probability) const;
  bool process_block_entry(Tokens const & tokens, std::string & column, std::string & row, double & value) const;
  bool has_timestage_changed(Tokens const & tokens, std::string const & timestage) const;
  bool has_block_changed(Tokens const & tokens, std::string const & block_name) const;
  public:
    BlockStructure(std::string const & filepath) { std::ifstream input(filepath); is_valid_ = read_from_file(input); }
    BlockStructure(std::istream & input) { is_valid_ = read_from_file(input); };
    virtual StochasticTree constructTree() const;
    int get_no_timestage_random_vectors() const { return timestage_random_vectors.size(); }
    TimestageRandomVectors const & get_timestage_random_vector(int index) { return timestage_random_vectors.at(index); }
};

struct ScenarioModifications {
  BlockLpEntry lp_modifications;
  double probability;
  std::string timestage;
  std::string scenario_name;
  std::string parent_scenario;
  ScenarioModifications(BlockLpEntry const & lp_modifications, double probability, std::string const & timestage,
      std::string const & scenario_name, std::string const & parent_scenario) : lp_modifications(lp_modifications), probability(probability),
      timestage(timestage), scenario_name(scenario_name), parent_scenario(parent_scenario) {}
  bool operator==(ScenarioModifications const & other) const {
    return lp_modifications == other.lp_modifications && probability == other.probability && timestage == other.timestage &&
    scenario_name == other.scenario_name && parent_scenario == other.parent_scenario;
  }
};

class ScenarioStructure : public SmpsStochasticStructure {
  std::vector<ScenarioModifications> scenarios;
  bool process_data(std::istream & input);
  bool read_from_file(std::istream & input);
  bool is_new_scenario(Tokens const & tokens) const;
  bool process_new_scenario(Tokens const & tokens, std::string & scenario_name, std::string & parent_scenario,
                           std::string & timeperiod, double & probability) const;
  bool process_scenario_entry(Tokens const & tokens, std::string & column, std::string & row, double & value) const;
  public:
    ScenarioStructure(std::string const & filepath) { std::ifstream input(filepath); is_valid_ = read_from_file(input); }
    ScenarioStructure(std::istream & input) { is_valid_ = read_from_file(input); };
    virtual StochasticTree constructTree() const;
    int get_no_scenarios() const { return scenarios.size(); }
    ScenarioModifications const & get_scenario(int index) { return scenarios.at(index); }
};

std::unique_ptr<SmpsStochasticStructure> read_stochastic_file(std::string const & filepath);
Tokens read_tokens(std::istream & input);
RandomVector append_to_random_vector(RandomVariable const & rv, RandomVector const & rvec);
RandomVector append_to_random_vector(RandomVector const & to_append, RandomVector const & rvec);
bool str_to_dbl(std::string const & str, double & val);
