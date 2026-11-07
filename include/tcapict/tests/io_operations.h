#pragma once

#include <tcapict/assertion.h>
#include <tcapict/elem_helper.h>
#include <tcapict/elem_sweep.h>
#include <tcapict/fixture.h>
#include <tcapict/skip.h>
#include <tcapict/zeroth_order.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace tcapict {

namespace detail {

// Removes the file at `path`, if any, when it goes out of scope, an assertion
// failing before then included; a removal that fails is ignored.
class removes_on_exit {
public:
  explicit removes_on_exit(std::string path) : path_(std::move(path)) {}
  removes_on_exit(const removes_on_exit &) = delete;
  removes_on_exit &operator=(const removes_on_exit &) = delete;
  ~removes_on_exit() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

private:
  std::string path_;
};

// A file path under the system temporary directory, which the object removes
// when it goes out of scope, as removes_on_exit does. A per-process counter
// keeps it distinct from the paths of other objects in this process;
// std::random_device and the clock make a collision with another process's
// path unlikely, not impossible. Its `.ten` extension is
// the one TCAPI V1's own `save` example uses, so a test using it asserts the
// call forms it makes, not how a backend treats a path without an extension.
class scratch_ten_file {
public:
  explicit scratch_ten_file(const char *stem) {
    static std::atomic<unsigned long> counter{0};
    std::random_device rd;
    const auto tag =
        std::to_string(rd()) + "_" +
        std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) +
        "_" + std::to_string(counter++);
    path_ = (std::filesystem::temp_directory_path() /
             (std::string("tcapict_") + stem + "_" + tag + ".ten"))
                .string();
  }
  scratch_ten_file(const scratch_ten_file &) = delete;
  scratch_ten_file &operator=(const scratch_ten_file &) = delete;
  ~scratch_ten_file() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }
  const std::string &str() const { return path_; }

private:
  std::string path_;
};

} // namespace detail

namespace tests {

// --- save/load roundtrip ---

template <typename TenT>
void test_save_load_roundtrip(tcapi_test_fixture<TenT>& fix) {
#if !defined(TCAPICT_SKIP_SAVE) && !defined(TCAPICT_SKIP_LOAD)
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  auto tensor = tcapi::zeros<TenT>(ctx, {2, 2});
  tcapi::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(1.0, 0.0));
  tcapi::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(1.0, 0.0));

  std::string filepath = "/tmp/tcapict_test_roundtrip";
  // Remove stale file to ensure save actually writes
  std::remove(filepath.c_str());

  TCAPICT_ASSERT_NOTHROW(tcapi::save(ctx, tensor, filepath));

  TenT loaded_tensor;
  TCAPICT_ASSERT_NOTHROW(loaded_tensor = tcapi::load<TenT>(ctx, filepath));

  TCAPICT_ASSERT(tcapi::shape(ctx, loaded_tensor) == tcapi::shape(ctx, tensor));

  auto val00 = tcapi::get_elem(ctx, loaded_tensor, {0, 0});
  auto val11 = tcapi::get_elem(ctx, loaded_tensor, {1, 1});
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(val00), 1.0, eps);
  TCAPICT_ASSERT_CLOSE(real_part<TenT>(val11), 1.0, eps);

  std::remove(filepath.c_str());
#else
  (void)fix;
#endif
}

// --- load verifies data integrity against original ---

template <typename TenT>
void test_load_data_integrity(tcapi_test_fixture<TenT>& fix) {
#if !defined(TCAPICT_SKIP_SAVE) && !defined(TCAPICT_SKIP_LOAD)
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  auto original = tcapi::eye<TenT>(ctx, 2);
  std::string filepath = "/tmp/tcapict_test_load_integrity";
  // Remove stale file to ensure save actually writes
  std::remove(filepath.c_str());
  tcapi::save(ctx, original, filepath);

  TenT loaded;
  TCAPICT_ASSERT_NOTHROW(loaded = tcapi::load<TenT>(ctx, filepath));
  TCAPICT_ASSERT(tcapi::shape(ctx, original) == tcapi::shape(ctx, loaded));
  TCAPICT_ASSERT(tcapi::close(ctx, original, loaded, eps));

  std::remove(filepath.c_str());
#else
  (void)fix;
#endif
}

// --- save/load roundtrip: zeroth-order ---

template <typename TenT>
void test_save_load_zeroth_order(tcapi_test_fixture<TenT>& fix) {
#if !defined(TCAPICT_SKIP_SAVE) && !defined(TCAPICT_SKIP_LOAD) && \
    defined(TCAPICT_HAS_ZEROTH_ORDER_SWEEP)
  auto& ctx = fix.context();
  auto tensor = tcapi::zeros<TenT>(ctx, {});
  TCAPICT_SET_ZEROTH_ORDER(fix, tensor, -6.5);

  std::string filepath = "/tmp/tcapict_test_zeroth_order_roundtrip";
  // Remove stale file to ensure save actually writes
  std::remove(filepath.c_str());
  TCAPICT_ASSERT_NOTHROW(tcapi::save(ctx, tensor, filepath));

  TenT loaded;
  TCAPICT_ASSERT_NOTHROW(loaded = tcapi::load<TenT>(ctx, filepath));
  TCAPICT_EXPECT_ZEROTH_ORDER(fix, loaded, -6.5);
  std::remove(filepath.c_str());
#else
  (void)fix;
#endif
}

// --- save/load: string-like storage ---

// TCAPI V1 declares `save(ctx, a, Storage &&strg)` and `load<TenT>(ctx,
// Storage &&strg)`, and: "C++ implementations are required to accept
// string-like inputs, such as `std::string`, `std::string_view`, or
// `const char*`". The other io tests pass a `std::string` variable; these pass
// each other form, the `std::string_view` and `std::string` ones as
// temporaries and the `const char*` one as a variable. Each asserts that the
// file exists at the path the storage names and that every element loads
// back.

#if !defined(TCAPICT_SKIP_SAVE) && !defined(TCAPICT_SKIP_LOAD) && \
    defined(TCAPICT_HAS_ELEM_SWEEP)
namespace io_check {

// Saves a {2, 3} ramp through `save_to(ctx, tensor, path)`, asserts that the
// file at `path` exists, loads it through `load_from(ctx, path)` and checks
// every element exactly; `stem` names the scratch file.
template <typename TenT, typename SaveTo, typename LoadFrom>
void expect_round_trip(tcapi_test_fixture<TenT> &fix, const char *stem,
                       SaveTo save_to, LoadFrom load_from) {
  auto &ctx = fix.context();
  const tcapi::shape_t<TenT> shape = {2, 3};
  auto at = ramp_over<TenT>(shape, 1.0);
  auto tensor = tcapi::zeros<TenT>(ctx, shape);
  TCAPICT_SET_ELEMS(fix, tensor, at);

  const detail::scratch_ten_file file(stem);
  TCAPICT_ASSERT_NOTHROW(save_to(ctx, tensor, file.str()));
  TCAPICT_ASSERT(std::filesystem::exists(file.str()));
  TenT loaded;
  TCAPICT_ASSERT_NOTHROW(loaded = load_from(ctx, file.str()));
  TCAPICT_EXPECT_ELEMS(fix, loaded, shape, at, tol_category::exact);
}

} // namespace io_check
#endif

template <typename TenT>
void test_save_load_string_view(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_SAVE) && !defined(TCAPICT_SKIP_LOAD) && \
    defined(TCAPICT_HAS_ELEM_SWEEP)
  // Each view ends before its buffer does. A backend reading the save view as
  // a null-terminated string through `data()` writes another file, so the
  // existence check fails; one reading the load view so looks for a file
  // that does not exist, which the load's no-throw check reports if the load
  // throws a std::exception and the element check reports if it returns.
  io_check::expect_round_trip(
      fix, "string_view",
      [](auto &ctx, const TenT &t, const std::string &path) {
        const std::string buffer = path + ".not_this_one";
        // A backend misreading the view writes here; remove it on every exit.
        const detail::removes_on_exit decoy(buffer);
        tcapi::save(ctx, t, std::string_view(buffer).substr(0, path.size()));
      },
      [](auto &ctx, const std::string &path) {
        const std::string buffer = path + ".not_this_one";
        return tcapi::load<TenT>(ctx,
                                 std::string_view(buffer).substr(0, path.size()));
      });
#else
  (void)fix;
#endif
}

template <typename TenT>
void test_save_load_c_string(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_SAVE) && !defined(TCAPICT_SKIP_LOAD) && \
    defined(TCAPICT_HAS_ELEM_SWEEP)
  io_check::expect_round_trip(
      fix, "c_string",
      [](auto &ctx, const TenT &t, const std::string &path) {
        const char *c_path = path.c_str();
        tcapi::save(ctx, t, c_path);
      },
      [](auto &ctx, const std::string &path) {
        const char *c_path = path.c_str();
        return tcapi::load<TenT>(ctx, c_path);
      });
#else
  (void)fix;
#endif
}

template <typename TenT>
void test_save_load_string_rvalue(tcapi_test_fixture<TenT> &fix) {
#if !defined(TCAPICT_SKIP_SAVE) && !defined(TCAPICT_SKIP_LOAD) && \
    defined(TCAPICT_HAS_ELEM_SWEEP)
  io_check::expect_round_trip(
      fix, "string_rvalue",
      [](auto &ctx, const TenT &t, const std::string &path) {
        tcapi::save(ctx, t, std::string(path));
      },
      [](auto &ctx, const std::string &path) {
        return tcapi::load<TenT>(ctx, std::string(path));
      });
#else
  (void)fix;
#endif
}

}}  // namespace tcapict::tests

// Bulk registration helper: invokes X(..., "category", test_fn) once per test.
// See include/tcapict/adapters/doctest.h for usage.
#define TCAPICT_FOREACH_IO_OPERATIONS_TEST_ALL_TYPES(X, ...) \
  X(__VA_ARGS__, "io_operations", test_save_load_roundtrip) \
  X(__VA_ARGS__, "io_operations", test_load_data_integrity) \
  X(__VA_ARGS__, "io_operations", test_save_load_zeroth_order) \
  X(__VA_ARGS__, "io_operations", test_save_load_string_view) \
  X(__VA_ARGS__, "io_operations", test_save_load_c_string) \
  X(__VA_ARGS__, "io_operations", test_save_load_string_rvalue)
