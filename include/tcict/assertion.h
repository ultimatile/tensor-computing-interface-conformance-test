#pragma once

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace tcict {

/// Exception thrown when a conformance test assertion fails.
struct assertion_error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

namespace detail {

/// Throws the assertion_error reporting `what` at `file`:`line`.
[[noreturn]] inline void fail_at(const char* file, int line, const std::string& what) {
  std::ostringstream oss;
  oss << file << ":" << line << ": " << what;
  throw assertion_error(oss.str());
}

/// Instantiated with `decltype((expr))`, which is a reference for every
/// expression that is not a prvalue (a name, a member access, an element
/// access, ...); rejects those at compile time, since TCICT_NOTHROW_VALUE would
/// return a reference that could refer to a temporary dying inside it.
template <typename T> constexpr bool require_prvalue() {
  static_assert(!std::is_reference_v<T>,
                "TCICT_NOTHROW_VALUE takes a prvalue, such as a call returning by value");
  return true;
}

}  // namespace detail

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

// Evaluates the expression in a lambda and yields its result, turning a
// std::exception from it into an assertion_error at the file and line where the
// macro is used. CHECK runs first and is discarded.
#define TCICT_DETAIL_NOTHROW_CALL(CHECK, ...)                                                      \
  ([&]() -> decltype(auto) {                                                                       \
    (void)(CHECK);                                                                                 \
    try {                                                                                          \
      return __VA_ARGS__;                                                                          \
    } catch (const std::exception& e_) {                                                           \
      ::tcict::detail::fail_at(__FILE__, __LINE__,                                                 \
                               std::string("unexpected exception: ") + e_.what());                 \
    }                                                                                              \
  }())

/// Evaluate a prvalue expression, such as a call returning by value, and yield
/// its value, turning a std::exception from it into an assertion_error that names
/// the file and line where the macro is used. The value reaches the caller
/// without a copy or move: `auto r = TCICT_NOTHROW_VALUE(f());` needs nothing of
/// the result type that `auto r = f();` does not. Any other expression does not
/// compile, since its result could refer to a temporary. Variadic so that the
/// expression may contain commas.
#define TCICT_NOTHROW_VALUE(...)                                                                   \
  TCICT_DETAIL_NOTHROW_CALL(::tcict::detail::require_prvalue<decltype((__VA_ARGS__))>(),           \
                            __VA_ARGS__)

/// Assert that an expression does not throw.
#define TCICT_ASSERT_NOTHROW(expr)                                                                 \
  do {                                                                                             \
    (void)TCICT_DETAIL_NOTHROW_CALL(true, expr);                                                   \
  } while (false)
