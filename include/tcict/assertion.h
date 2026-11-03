#pragma once

#include <cmath>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#if defined(__GLIBCXX__)
#include <cxxabi.h>
#endif

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

/// Calls `f` and returns its result, reporting an exception from `f` as an
/// assertion_error at `file`:`line`: an assertion_error is reported followed by
/// its own message, and an exception of any other type as unexpected. The one
/// exception left as it is, under libstdc++, unwinds a cancelled thread, and is
/// rethrown unchanged since swallowing it terminates the program. `f` must not return a reference, which could refer to a temporary that has
/// died by the time the caller reads it. A prvalue result reaches the caller
/// without a copy or move.
template <typename F> decltype(auto) nothrow_call(const char* file, int line, F&& f) {
  static_assert(!std::is_reference_v<std::invoke_result_t<F&>>,
                "the expression must not yield a reference: TCICT_NOTHROW_VALUE takes "
                "a prvalue, such as a call returning by value");
  try {
    return f();
  } catch (const assertion_error& e) {
    fail_at(file, line, e.what());
  } catch (const std::exception& e) {
    fail_at(file, line, std::string("unexpected exception: ") + e.what());
#if defined(__GLIBCXX__)
  } catch (abi::__forced_unwind&) {
    throw;
#endif
  } catch (...) {
    fail_at(file, line, "unexpected exception not catchable as std::exception");
  }
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

/// Evaluate a prvalue expression, such as a call returning by value, and yield
/// its value, reporting an exception from it at the macro's use as
/// detail::nothrow_call does. The value reaches the caller without a copy or
/// move:
/// `auto r = TCICT_NOTHROW_VALUE(f());` needs nothing of the result type that
/// `auto r = f();` does not. Any other expression, a variable's name included,
/// does not compile, since its result could refer to a temporary. The reported
/// line is one of the lines the invocation spans; which one, for an invocation
/// spanning several, depends on the compiler. Variadic so that the expression
/// may contain commas.
#define TCICT_NOTHROW_VALUE(...)                                                                   \
  ::tcict::detail::nothrow_call(__FILE__, __LINE__,                                                \
                               [&]() -> decltype(auto) { return (__VA_ARGS__); })

/// Assert that an expression does not throw, reporting an exception from it at
/// the macro's use as detail::nothrow_call does. Variadic so that the
/// expression may contain commas.
#define TCICT_ASSERT_NOTHROW(...)                                                                  \
  do {                                                                                             \
    ::tcict::detail::nothrow_call(__FILE__, __LINE__, [&] { (void)(__VA_ARGS__); });               \
  } while (false)
