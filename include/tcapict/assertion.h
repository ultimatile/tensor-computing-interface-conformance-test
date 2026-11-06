#pragma once

#include <cmath>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace tcapict {

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

/// Calls `f` and returns its result, reporting a std::exception from `f` as an
/// assertion_error at `file`:`line`: an assertion_error is reported followed by
/// its own message, and any other std::exception as unexpected. An exception
/// of another type passes through unchanged, since a handler catching every
/// type would also catch the unwind of a cancelled thread, which must not be
/// stopped. `f` must not return a reference, which could refer to a temporary
/// that has died by the time the caller reads it. A prvalue result reaches the
/// caller without a copy or move.
template <typename F> decltype(auto) nothrow_call(const char* file, int line, F&& f) {
  static_assert(!std::is_reference_v<std::invoke_result_t<F&>>,
                "the expression must not yield a reference: TCAPICT_NOTHROW_VALUE takes "
                "a prvalue, such as a call returning by value");
  try {
    return f();
  } catch (const assertion_error& e) {
    fail_at(file, line, e.what());
  } catch (const std::exception& e) {
    fail_at(file, line, std::string("unexpected exception: ") + e.what());
  }
}

}  // namespace detail

}  // namespace tcapict

/// Assert that a condition is true.
#define TCAPICT_ASSERT(cond)                                                                       \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::ostringstream oss_;                                                                     \
      oss_ << __FILE__ << ":" << __LINE__ << ": assertion failed: " << #cond;                      \
      throw ::tcapict::assertion_error(oss_.str());                                                \
    }                                                                                              \
  } while (false)

/// Assert that a condition is true, with a custom message.
#define TCAPICT_ASSERT_MSG(cond, msg)                                                              \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::ostringstream oss_;                                                                     \
      oss_ << __FILE__ << ":" << __LINE__ << ": " << (msg);                                        \
      throw ::tcapict::assertion_error(oss_.str());                                                \
    }                                                                                              \
  } while (false)

/// Assert that |a - b| < epsilon (floating-point comparison).
#define TCAPICT_ASSERT_CLOSE(a, b, epsilon)                                                        \
  do {                                                                                             \
    auto tcapict_a_ = (a);                                                                         \
    auto tcapict_b_ = (b);                                                                         \
    auto tcapict_eps_ = (epsilon);                                                                 \
    if (!(std::abs(tcapict_a_ - tcapict_b_) < tcapict_eps_)) {                                     \
      std::ostringstream oss_;                                                                     \
      oss_ << __FILE__ << ":" << __LINE__ << ": close assertion failed: |" << tcapict_a_ << " - " \
           << tcapict_b_ << "| = " << std::abs(tcapict_a_ - tcapict_b_) << " >= "                  \
           << tcapict_eps_;                                                                        \
      throw ::tcapict::assertion_error(oss_.str());                                                \
    }                                                                                              \
  } while (false)

/// Evaluate a prvalue expression, such as a call returning by value, and yield
/// its value, reporting a std::exception from it at the macro's use as
/// detail::nothrow_call does. The value reaches the caller without a copy or
/// move:
/// `auto r = TCAPICT_NOTHROW_VALUE(f());` needs nothing of the result type that
/// `auto r = f();` does not. Any other expression, a variable's name included,
/// does not compile, since its result could refer to a temporary. The reported
/// line is one of the lines the invocation spans; which one, for an invocation
/// spanning several, depends on the compiler. Variadic so that the expression
/// may contain commas.
#define TCAPICT_NOTHROW_VALUE(...)                                                                 \
  ::tcapict::detail::nothrow_call(__FILE__, __LINE__,                                              \
                               [&]() -> decltype(auto) { return (__VA_ARGS__); })

/// Assert that an expression does not throw a std::exception, reporting one at
/// the macro's use as detail::nothrow_call does. Variadic so that the
/// expression may contain commas.
#define TCAPICT_ASSERT_NOTHROW(...)                                                                \
  do {                                                                                             \
    ::tcapict::detail::nothrow_call(__FILE__, __LINE__, [&] { (void)(__VA_ARGS__); });             \
  } while (false)
