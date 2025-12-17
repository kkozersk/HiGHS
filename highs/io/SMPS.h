#pragma once
#include <fstream>
#include <istream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "lp_data/HighsLp.h"

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
};

using BlockLpEntry = std::vector<LpEntry>;

class Node {
  std::vector<std::unique_ptr<Node>> children;
  Node * parent = nullptr;
  std::vector<LpEntry> lp_modifications;
  double node_probability; // TODO 0 <= p <= 1
  std::string timestage;
  public:
    Node(std::string timestage, double node_probability = 1., std::vector<LpEntry> const & lp_modifications={}):
        lp_modifications(lp_modifications), node_probability(node_probability), timestage(timestage) {}
    void add_child(std::unique_ptr<Node> && child);
    bool verify_children_probabilities() const;
    std::unique_ptr<Node> & get_child(int index) { return children.at(index); }
    // std::vector<std::unique_ptr<Node>> const & get_children() { return children; }
    Node const * get_parent() const { return parent; };
  // TimeStage timestage;
  // double get_in_tree_probability() const;
};

struct StochasticTree {
  std::unique_ptr<Node> root;
  StochasticTree(std::unique_ptr<Node> && root) : root(std::move(root)) {}
};

class SmpsStochasticStructure {
  protected:
    std::string problem_name = "";
    bool is_ending(std::string const & line) const { return line == "ENDATA" || line.empty(); };
    bool process_ending(std::string const & line) const { return line == "ENDATA"; };
    bool is_valid_ = false;
  public:
    SmpsStochasticStructure() = default;
    virtual ~SmpsStochasticStructure() = default;
    virtual StochasticTree constructTree(SmpsTimeStructure const &) const = 0;
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
 };
using RandomVector = std::vector<RandomVectorValue>;

struct TimestageRandomVariables {
  std::vector<RandomVariable> rvs;
  std::string timestage;
  RandomVector generate_vector() const;
  bool operator==(TimestageRandomVariables const & other) const { return rvs == other.rvs && timestage == other.timestage; }
  TimestageRandomVariables(std::vector<RandomVariable> const & rvs, std::string const & timestage) : rvs(rvs), timestage(timestage) {}
};

class IndepStructure : public SmpsStochasticStructure {
  std::vector<TimestageRandomVariables> timestage_random_entries;

  bool process_tokens(std::vector<std::string> const & tokens, std::string & column, std::string & row, double & value, std::string & timeperiod, double & probability) const;
  bool is_ending(std::vector<std::string> const & tokens) const;
  bool is_proper_ending(std::vector<std::string> const & tokens) const;
  bool process_data(std::istream & input);
  bool process_header(std::vector<std::string> const & tokens, std::string & header, std::string & problem_name) const;
  bool process_structure(std::vector<std::string> const & tokens, std::string & structure_type, std::string & distribution) const; 
  bool read_from_file(std::istream & input);
  public:
    IndepStructure(std::istream & input);
    IndepStructure(std::string const & filename);
    virtual StochasticTree constructTree(SmpsTimeStructure const &) const;
    int get_no_timestage_random_entries() const { return timestage_random_entries.size(); }
    TimestageRandomVariables const & get_timestage_random_entry(int index) { return timestage_random_entries.at(index); }
    // std::vector<TimestageRandomVariables> const & get_modifications() { return modifications; };
};

// class BlockStructure : public SmpsStochasticStructure {
//   struct BlockModifications {
//     std::vector<LpEntry> modifications;
//     double probability;
//     std::string period;
//     std::string block_name;
//   };
//   std::vector<BlockModifications> modifications;
//   public:
//     BlockStructure(std::string const & problem_name, std::string const & filename);
//     virtual StochasticTree constructTree(SmpsTimeStructure const &) const;
// };

// class ScenarioStructure : public SmpsStochasticStructure {
//   struct ScenarioModifications {
//     std::vector<LpEntry> modifications;
//     double probability;
//     std::string period;
//     std::string scenario_name;
//     std::string parent_scenario;
//   };
//   std::vector<ScenarioModifications> modifications;
//   public:
//     ScenarioStructure(std::string const & problem_name, std::string const & filename);
//     virtual StochasticTree constructTree(SmpsTimeStructure const &) const;
// };

std::unique_ptr<SmpsStochasticStructure> read_stochastic_file(std::string const & filepath);
std::vector<std::string> read_tokens(std::istream & input);
RandomVector append_to_random_vector(RandomVariable const & rv, RandomVector const & rvec);
