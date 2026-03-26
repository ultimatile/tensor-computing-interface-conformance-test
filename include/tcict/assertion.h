#pragma once

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tcict {

/// Exception thrown when a conformance test assertion fails.
struct assertion_error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

}  // namespace tcict

/// Assert that a condition is true.
#define TCICT_ASSERT(cond)                                                                         \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::ostringstream oss_;                                                                     \
      oss_ << __FILE__ << ":" << __LINE__ << ": assertion failed: " << #cond;                      \
      throw ::tcict::assertion_error(oss_.str());                                                  \
    }                                                                                              \
  } while (false)

/// Assert that a condition is true, with a custom message.
#define TCICT_ASSERT_MSG(cond, msg)                                                                \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::ostringstream oss_;                                                                     \
      oss_ << __FILE__ << ":" << __LINE__ << ": " << (msg);                                        \
      throw ::tcict::assertion_error(oss_.str());                                                  \
    }                                                                                              \
  } while (false)

/// Assert that |a - b| < epsilon (floating-point comparison).
#define TCICT_ASSERT_CLOSE(a, b, epsilon)                                                          \
  do {                                                                                             \
    auto tcict_a_ = (a);                                                                           \
    auto tcict_b_ = (b);                                                                           \
    auto tcict_eps_ = (epsilon);                                                                   \
    if (!(std::abs(tcict_a_ - tcict_b_) < tcict_eps_)) {                                           \
      std::ostringstream oss_;                                                                     \
      oss_ << __FILE__ << ":" << __LINE__ << ": close assertion failed: |" << tcict_a_ << " - "   \
           << tcict_b_ << "| = " << std::abs(tcict_a_ - tcict_b_) << " >= " << tcict_eps_;        \
      throw ::tcict::assertion_error(oss_.str());                                                  \
    }                                                                                              \
  } while (false)

/// Assert that an expression does not throw.
#define TCICT_ASSERT_NOTHROW(expr)                                                                 \
  do {                                                                                             \
    try {                                                                                          \
      expr;                                                                                        \
    } catch (const std::exception& e_) {                                                           \
      std::ostringstream oss_;                                                                     \
      oss_ << __FILE__ << ":" << __LINE__ << ": unexpected exception: " << e_.what();              \
      throw ::tcict::assertion_error(oss_.str());                                                  \
    }                                                                                              \
  } while (false)

namespace tcict {
namespace detail {

// Function template implementation for TCICT_ASSERT_THROWS.
// Using a lambda avoids ASAN false positives caused by brace-init temporaries
// constructed inside a try-catch block within a macro expansion.
template <typename ExType, typename F>
void assert_throws_impl(const char* file, int line, const char* ex_name, F&& f) {
  bool caught = false;
  bool wrong_type = false;
  std::string unexpected_info;
  try {
    f();
  } catch (const ExType&) {
    caught = true;
  } catch (const std::exception& e) {
    wrong_type = true;
    std::ostringstream oss;
    oss << "unexpected std::exception: " << e.what();
    unexpected_info = oss.str();
  } catch (...) {
    wrong_type = true;
    unexpected_info = "unexpected non-std exception";
  }
  if (!caught) {
    std::ostringstream oss;
    oss << file << ":" << line << ": ";
    if (wrong_type) {
      oss << "expected exception " << ex_name << " but a different exception was thrown";
      if (!unexpected_info.empty()) {
        oss << " (" << unexpected_info << ")";
      }
    } else {
      oss << "expected exception " << ex_name << " not thrown";
    }
    throw ::tcict::assertion_error(oss.str());
  }
}

}  // namespace detail
}  // namespace tcict

/// Assert that an expression throws a specific exception type.
#define TCICT_ASSERT_THROWS(ExType, expr)                                                          \
  ::tcict::detail::assert_throws_impl<ExType>(__FILE__, __LINE__, #ExType, [&]() { expr; })
