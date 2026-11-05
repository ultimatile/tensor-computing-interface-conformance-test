#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/elem_sweep.h>
#include <tcict/fixture.h>
#include <tcict/skip.h>
#include <tcict/zeroth_order.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <string_view>
#include <system_error>

namespace tcict {

namespace detail {

// A file path under the system temporary directory, removed when the object
// goes out of scope, an assertion failing before then included. A
// per-process counter keeps it distinct from the paths of other objects in
// this process; std::random_device and the clock make a collision with
// another process's path unlikely, not impossible. Its `.ten` extension is
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
             (std::string("tcict_") + stem + "_" + tag + ".ten"))
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
void test_save_load_roundtrip(tci_test_fixture<TenT>& fix) {
#if !defined(TCICT_SKIP_SAVE) && !defined(TCICT_SKIP_LOAD)
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  auto tensor = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, tensor, {0, 0}, make_elem<TenT>(1.0, 0.0));
  tci::set_elem(ctx, tensor, {1, 1}, make_elem<TenT>(1.0, 0.0));

  std::string filepath = "/tmp/tcict_test_roundtrip";
  // Remove stale file to ensure save actually writes
  std::remove(filepath.c_str());

  TCICT_ASSERT_NOTHROW(tci::save(ctx, tensor, filepath));

  TenT loaded_tensor;
  TCICT_ASSERT_NOTHROW(loaded_tensor = tci::load<TenT>(ctx, filepath));

  TCICT_ASSERT(tci::shape(ctx, loaded_tensor) == tci::shape(ctx, tensor));

  auto val00 = tci::get_elem(ctx, loaded_tensor, {0, 0});
  auto val11 = tci::get_elem(ctx, loaded_tensor, {1, 1});
  TCICT_ASSERT_CLOSE(real_part<TenT>(val00), 1.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(val11), 1.0, eps);

  std::remove(filepath.c_str());
#else
  (void)fix;
#endif
}

// --- load verifies data integrity against original ---

template <typename TenT>
void test_load_data_integrity(tci_test_fixture<TenT>& fix) {
#if !defined(TCICT_SKIP_SAVE) && !defined(TCICT_SKIP_LOAD)
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  auto original = tci::eye<TenT>(ctx, 2);
  std::string filepath = "/tmp/tcict_test_load_integrity";
  // Remove stale file to ensure save actually writes
  std::remove(filepath.c_str());
  tci::save(ctx, original, filepath);

  TenT loaded;
  TCICT_ASSERT_NOTHROW(loaded = tci::load<TenT>(ctx, filepath));
  TCICT_ASSERT(tci::shape(ctx, original) == tci::shape(ctx, loaded));
  TCICT_ASSERT(tci::close(ctx, original, loaded, eps));

  std::remove(filepath.c_str());
#else
  (void)fix;
#endif
}

// --- save/load roundtrip: zeroth-order ---

template <typename TenT>
void test_save_load_zeroth_order(tci_test_fixture<TenT>& fix) {
#if !defined(TCICT_SKIP_SAVE) && !defined(TCICT_SKIP_LOAD) && \
    defined(TCICT_HAS_ZEROTH_ORDER_SWEEP)
  auto& ctx = fix.context();
  auto tensor = tci::zeros<TenT>(ctx, {});
  TCICT_SET_ZEROTH_ORDER(fix, tensor, -6.5);

  std::string filepath = "/tmp/tcict_test_zeroth_order_roundtrip";
  // Remove stale file to ensure save actually writes
  std::remove(filepath.c_str());
  TCICT_ASSERT_NOTHROW(tci::save(ctx, tensor, filepath));

  TenT loaded;
  TCICT_ASSERT_NOTHROW(loaded = tci::load<TenT>(ctx, filepath));
  TCICT_EXPECT_ZEROTH_ORDER(fix, loaded, -6.5);
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

#if !defined(TCICT_SKIP_SAVE) && !defined(TCICT_SKIP_LOAD) && \
    defined(TCICT_HAS_ELEM_SWEEP)
namespace io_check {

// Saves a {2, 3} ramp through `save_to(ctx, tensor, path)`, asserts that the
// file at `path` exists, loads it through `load_from(ctx, path)` and checks
// every element exactly; `stem` names the scratch file.
template <typename TenT, typename SaveTo, typename LoadFrom>
void expect_round_trip(tci_test_fixture<TenT> &fix, const char *stem,
                       SaveTo save_to, LoadFrom load_from) {
  auto &ctx = fix.context();
  const tci::shape_t<TenT> shape = {2, 3};
  auto at = ramp_over<TenT>(shape, 1.0);
  auto tensor = tci::zeros<TenT>(ctx, shape);
  TCICT_SET_ELEMS(fix, tensor, at);

  const detail::scratch_ten_file file(stem);
  TCICT_ASSERT_NOTHROW(save_to(ctx, tensor, file.str()));
  TCICT_ASSERT(std::filesystem::exists(file.str()));
  TenT loaded;
  TCICT_ASSERT_NOTHROW(loaded = load_from(ctx, file.str()));
  TCICT_EXPECT_ELEMS(fix, loaded, shape, at, tol_category::exact);
}

} // namespace io_check
#endif

template <typename TenT>
void test_save_load_string_view(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_SAVE) && !defined(TCICT_SKIP_LOAD) && \
    defined(TCICT_HAS_ELEM_SWEEP)
  // Each view ends before its buffer does. A backend reading the save view as
  // a null-terminated string through `data()` writes another file, so the
  // existence check fails; one reading the load view so looks for a file
  // that does not exist, so the load throws.
  io_check::expect_round_trip(
      fix, "string_view",
      [](auto &ctx, const TenT &t, const std::string &path) {
        const std::string buffer = path + ".not_this_one";
        tci::save(ctx, t, std::string_view(buffer).substr(0, path.size()));
        std::error_code ec;
        std::filesystem::remove(buffer, ec);
      },
      [](auto &ctx, const std::string &path) {
        const std::string buffer = path + ".not_this_one";
        return tci::load<TenT>(ctx,
                               std::string_view(buffer).substr(0, path.size()));
      });
#else
  (void)fix;
#endif
}

template <typename TenT>
void test_save_load_c_string(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_SAVE) && !defined(TCICT_SKIP_LOAD) && \
    defined(TCICT_HAS_ELEM_SWEEP)
  io_check::expect_round_trip(
      fix, "c_string",
      [](auto &ctx, const TenT &t, const std::string &path) {
        const char *c_path = path.c_str();
        tci::save(ctx, t, c_path);
      },
      [](auto &ctx, const std::string &path) {
        const char *c_path = path.c_str();
        return tci::load<TenT>(ctx, c_path);
      });
#else
  (void)fix;
#endif
}

template <typename TenT>
void test_save_load_string_rvalue(tci_test_fixture<TenT> &fix) {
#if !defined(TCICT_SKIP_SAVE) && !defined(TCICT_SKIP_LOAD) && \
    defined(TCICT_HAS_ELEM_SWEEP)
  io_check::expect_round_trip(
      fix, "string_rvalue",
      [](auto &ctx, const TenT &t, const std::string &path) {
        tci::save(ctx, t, std::string(path));
      },
      [](auto &ctx, const std::string &path) {
        return tci::load<TenT>(ctx, std::string(path));
      });
#else
  (void)fix;
#endif
}

}}  // namespace tcict::tests

// Bulk registration helper: invokes X(..., "category", test_fn) once per test.
// See include/tcict/adapters/doctest.h for usage.
#define TCICT_FOREACH_IO_OPERATIONS_TEST_ALL_TYPES(X, ...) \
  X(__VA_ARGS__, "io_operations", test_save_load_roundtrip) \
  X(__VA_ARGS__, "io_operations", test_load_data_integrity) \
  X(__VA_ARGS__, "io_operations", test_save_load_zeroth_order) \
  X(__VA_ARGS__, "io_operations", test_save_load_string_view) \
  X(__VA_ARGS__, "io_operations", test_save_load_c_string) \
  X(__VA_ARGS__, "io_operations", test_save_load_string_rvalue)
