#ifndef HIPO_PRE_POST_PROCESS
#define HIPO_PRE_POST_PROCESS

#include <cassert>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include "ipm/hipo/auxiliary/IntConfig.h"

namespace hipo {

class Model;
struct Iterate;

struct PreprocessorPoint {
  std::vector<double>& x;
  std::vector<double>& xl;
  std::vector<double>& xu;
  std::vector<double>& slack;
  std::vector<double>& y;
  std::vector<double>& zl;
  std::vector<double>& zu;

  void assertConsistency(Int n, Int m) const;
};

struct PreprocessAction {
  Int n_pre, m_pre, n_post, m_post;

  virtual ~PreprocessAction() = default;
  virtual void apply(Model& model) = 0;
  virtual void undo(PreprocessorPoint& point, const Model& model,
                    const Iterate& it) const = 0;
  virtual void print(std::stringstream& stream) const = 0;
  virtual std::unique_ptr<PreprocessAction> copy() const = 0;
};

struct PreprocessEmptyRows : public PreprocessAction {
  std::vector<Int> rows_shift;
  Int empty_rows{};

  void apply(Model& model) override;
  void undo(PreprocessorPoint& point, const Model& model,
            const Iterate& it) const override;
  void print(std::stringstream& stream) const override;
  std::unique_ptr<PreprocessAction> copy() const override { return std::unique_ptr<PreprocessAction>(new PreprocessEmptyRows(*this)); }
};

struct PreprocessFixedVars : public PreprocessAction {
  Int fixed_vars{};
  std::vector<double> fixed_at;

  // information about the columns that get removed
  struct FixedVarsData {
    double c;
    std::vector<Int> indA, indQ;
    std::vector<double> valA, valQ;
  };
  std::map<Int, FixedVarsData> data;

  void apply(Model& model) override;
  void undo(PreprocessorPoint& point, const Model& model,
            const Iterate& it) const override;
  void print(std::stringstream& stream) const override;
  std::unique_ptr<PreprocessAction> copy() const override { return std::unique_ptr<PreprocessAction>(new PreprocessFixedVars(*this)); }
};

struct PreprocessScaling : public PreprocessAction {
  bool scaled = false;

  void apply(Model& model) override;
  void undo(PreprocessorPoint& point, const Model& model,
            const Iterate& it) const override;
  void print(std::stringstream& stream) const override;
  std::unique_ptr<PreprocessAction> copy() const override { return std::unique_ptr<PreprocessAction>(new PreprocessScaling(*this)); }
};

struct PreprocessFormulation : public PreprocessAction {
  void apply(Model& model) override;
  void undo(PreprocessorPoint& point, const Model& model,
            const Iterate& it) const override;
  void print(std::stringstream& stream) const override;
  std::unique_ptr<PreprocessAction> copy() const override { return std::unique_ptr<PreprocessAction>(new PreprocessFormulation(*this)); }
};

struct PreprocessFreeVars : public PreprocessAction {
  Int free_vars_count{};

  void apply(Model& model) override;
  void undo(PreprocessorPoint& point, const Model& model,
            const Iterate& it) const override;
  void print(std::stringstream& stream) const override;
  std::unique_ptr<PreprocessAction> copy() const override { return std::unique_ptr<PreprocessAction>(new PreprocessFreeVars(*this)); }
};

struct Preprocessor {
  std::vector<std::unique_ptr<PreprocessAction>> stack;

  void apply(Model& model);
  void undo(PreprocessorPoint& point, const Model& model,
            const Iterate& it) const;
  void print(std::stringstream& log_stream) const;
  Preprocessor() {};
  Preprocessor(Preprocessor const & other) { for (auto const & ptr : other.stack) stack.push_back(ptr->copy()); };
  Preprocessor const & operator=(Preprocessor const & other) { for (auto const & ptr : other.stack) stack.push_back(ptr->copy()); return *this; };
};

}  // namespace hipo

#endif
