#pragma once

#include <tcict/assertion.h>
#include <tcict/elem_helper.h>
#include <tcict/fixture.h>
#include <tcict/skip.h>

#include <cmath>
#include <functional>

namespace tcict { namespace tests {

// --- zeros ---

template <typename TenT>
void test_zeros(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_ZEROS
  return;
#endif
  auto& ctx = fix.context();
  tci::shape_t<TenT> shape = {2, 3};
  TenT tensor;
  TCICT_ASSERT_NOTHROW(tensor = tci::zeros<TenT>(ctx, shape));
  TCICT_ASSERT(tci::order(ctx, tensor) == 2);
  auto result_shape = tci::shape(ctx, tensor);
  TCICT_ASSERT(result_shape.size() == 2);
  TCICT_ASSERT(result_shape[0] == 2);
  TCICT_ASSERT(result_shape[1] == 3);
}

// --- eye ---

template <typename TenT>
void test_eye(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_EYE
  return;
#endif
  auto& ctx = fix.context();
  TenT identity;
  TCICT_ASSERT_NOTHROW(identity = tci::eye<TenT>(ctx, 3));
  TCICT_ASSERT(tci::order(ctx, identity) == 2);
  auto result_shape = tci::shape(ctx, identity);
  TCICT_ASSERT(result_shape.size() == 2);
  TCICT_ASSERT(result_shape[0] == 3);
  TCICT_ASSERT(result_shape[1] == 3);
}

// --- random (in-place) ---

template <typename TenT>
void test_random_inplace(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_RANDOM
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  tci::shape_t<TenT> shape = {2, 3};
  TenT tensor;
  std::size_t counter = 0;

  auto gen = [&]() -> tci::elem_t<TenT> {
    double val = static_cast<double>(counter++);
    return make_elem<TenT>(val, val + 0.5);
  };

  TCICT_ASSERT_NOTHROW(tci::random(ctx, shape, gen, tensor));
  TCICT_ASSERT(counter == 6);
  TCICT_ASSERT(tci::shape(ctx, tensor) == shape);

  auto elem_00 = tci::get_elem(ctx, tensor, {0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(elem_00), 0.0, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(elem_00), 0.5, eps);

  auto elem_01 = tci::get_elem(ctx, tensor, {0, 1});
  TCICT_ASSERT_CLOSE(real_part<TenT>(elem_01), 1.0, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(elem_01), 1.5, eps);

  auto elem_12 = tci::get_elem(ctx, tensor, {1, 2});
  TCICT_ASSERT_CLOSE(real_part<TenT>(elem_12), 5.0, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(elem_12), 5.5, eps);
}

// --- random (out-of-place) ---

template <typename TenT>
void test_random_outofplace(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_RANDOM
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  tci::shape_t<TenT> shape = {2, 2};
  std::size_t counter = 0;

  auto gen = [&]() -> tci::elem_t<TenT> {
    double val = static_cast<double>(counter++);
    return make_elem<TenT>(val, val + 0.5);
  };

  TenT tensor;
  TCICT_ASSERT_NOTHROW(tensor = tci::template random<TenT>(ctx, shape, gen));
  TCICT_ASSERT(counter == 4);
  TCICT_ASSERT(tci::shape(ctx, tensor) == shape);

  auto elem_11 = tci::get_elem(ctx, tensor, {1, 1});
  TCICT_ASSERT_CLOSE(real_part<TenT>(elem_11), 3.0, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(elem_11), 3.5, eps);
}

// --- copy (in-place) ---

template <typename TenT>
void test_copy_inplace(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_COPY
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  auto a = tci::zeros<TenT>(ctx, {2, 3});
  tci::set_elem(ctx, a, {0, 0}, make_elem<TenT>(42.0, 13.0));
  tci::set_elem(ctx, a, {1, 2}, make_elem<TenT>(-5.5, 7.7));

  TenT b;
  TCICT_ASSERT_NOTHROW(b = tci::copy(ctx, a));

  auto val1 = tci::get_elem(ctx, b, {0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(val1), 42.0, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(val1), 13.0, eps);

  auto val2 = tci::get_elem(ctx, b, {1, 2});
  TCICT_ASSERT_CLOSE(real_part<TenT>(val2), -5.5, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(val2), 7.7, eps);

  TCICT_ASSERT(tci::shape(ctx, a) == tci::shape(ctx, b));
  TCICT_ASSERT(tci::order(ctx, a) == tci::order(ctx, b));
}

// --- copy (out-of-place) ---

template <typename TenT>
void test_copy_outofplace(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_COPY
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  auto a = tci::zeros<TenT>(ctx, {3, 4, 2});
  tci::set_elem(ctx, a, {0, 1, 0}, make_elem<TenT>(1.23, 4.56));
  tci::set_elem(ctx, a, {2, 3, 1}, make_elem<TenT>(-9.87, 6.54));

  auto b = tci::copy(ctx, a);

  auto val1_a = tci::get_elem(ctx, a, {0, 1, 0});
  auto val1_b = tci::get_elem(ctx, b, {0, 1, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(val1_a), real_part<TenT>(val1_b), eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(val1_a), imag_part<TenT>(val1_b), eps);

  TCICT_ASSERT(tci::shape(ctx, a) == tci::shape(ctx, b));
  TCICT_ASSERT(tci::order(ctx, a) == tci::order(ctx, b));
}

// --- copy independence ---

template <typename TenT>
void test_copy_independence(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_COPY
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  auto a = tci::zeros<TenT>(ctx, {2, 2});
  tci::set_elem(ctx, a, {0, 0}, make_elem<TenT>(100.0));

  auto b = tci::copy(ctx, a);
  tci::set_elem(ctx, a, {0, 0}, make_elem<TenT>(999.0));

  auto copy_val = tci::get_elem(ctx, b, {0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(copy_val), 100.0, eps);

  auto orig_val = tci::get_elem(ctx, a, {0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(orig_val), 999.0, eps);
}

// --- copy single element ---

template <typename TenT>
void test_copy_single_element(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_COPY
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  auto a = tci::zeros<TenT>(ctx, {1});
  tci::set_elem(ctx, a, {0}, make_elem<TenT>(3.14, 2.71));

  auto b = tci::copy(ctx, a);

  auto val = tci::get_elem(ctx, b, {0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(val), 3.14, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(val), 2.71, eps);
}

// --- copy large tensor ---

template <typename TenT>
void test_copy_large(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_COPY
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  auto a = tci::zeros<TenT>(ctx, {10, 10, 10});
  tci::set_elem(ctx, a, {0, 0, 0}, make_elem<TenT>(1.0, 0.0));
  tci::set_elem(ctx, a, {9, 9, 9}, make_elem<TenT>(0.0, 1.0));

  auto b = tci::copy(ctx, a);

  auto corner1 = tci::get_elem(ctx, b, {0, 0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(corner1), 1.0, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(corner1), 0.0, eps);

  auto corner2 = tci::get_elem(ctx, b, {9, 9, 9});
  TCICT_ASSERT_CLOSE(real_part<TenT>(corner2), 0.0, eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(corner2), 1.0, eps);

  TCICT_ASSERT(tci::shape(ctx, a) == tci::shape(ctx, b));
}

// --- assign_from_range (row-major) ---

template <typename TenT>
void test_assign_from_range_row_major(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_ASSIGN_FROM_RANGE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  std::vector<tci::elem_t<TenT>> container = {
      make_elem<TenT>(1.0), make_elem<TenT>(2.0), make_elem<TenT>(3.0),
      make_elem<TenT>(4.0), make_elem<TenT>(5.0), make_elem<TenT>(6.0)};

  std::function<std::ptrdiff_t(const tci::elem_coors_t<TenT>&)> coors2idx
      = [](const tci::elem_coors_t<TenT>& coors) -> std::ptrdiff_t {
    return coors[0] * 3 + coors[1];
  };

  tci::shape_t<TenT> shape = {2, 3};
  TenT tensor;
  TCICT_ASSERT_NOTHROW(
      tensor = tci::template assign_from_range<TenT>(ctx, shape, container.begin(),
                                                     std::move(coors2idx)));

  TCICT_ASSERT(tci::order(ctx, tensor) == 2);
  auto result_shape = tci::shape(ctx, tensor);
  TCICT_ASSERT(result_shape[0] == 2);
  TCICT_ASSERT(result_shape[1] == 3);

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 1.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 1})), 2.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 0})), 4.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 2})), 6.0, eps);
}

// --- assign_from_range (column-major) ---

template <typename TenT>
void test_assign_from_range_column_major(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_ASSIGN_FROM_RANGE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();

  // Column-major layout: [1, 3, 2, 4]
  std::vector<tci::elem_t<TenT>> container = {make_elem<TenT>(1.0), make_elem<TenT>(3.0),
                                               make_elem<TenT>(2.0), make_elem<TenT>(4.0)};

  std::function<std::ptrdiff_t(const tci::elem_coors_t<TenT>&)> coors2idx
      = [](const tci::elem_coors_t<TenT>& coors) -> std::ptrdiff_t {
    return coors[1] * 2 + coors[0];
  };

  tci::shape_t<TenT> shape = {2, 2};
  TenT tensor;
  TCICT_ASSERT_NOTHROW(
      tensor = tci::template assign_from_range<TenT>(ctx, shape, container.begin(),
                                                     std::move(coors2idx)));

  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 0})), 1.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {0, 1})), 2.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 0})), 3.0, eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(tci::get_elem(ctx, tensor, {1, 1})), 4.0, eps);
}

// --- allocate ---

template <typename TenT>
void test_allocate_3d(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_ALLOCATE
  return;
#endif
  auto& ctx = fix.context();
  tci::shape_t<TenT> shape = {3, 4, 5};
  auto tensor = tci::allocate<TenT>(ctx, shape);

  auto tensor_shape = tci::shape(ctx, tensor);
  TCICT_ASSERT(tensor_shape.size() == 3);
  TCICT_ASSERT(tensor_shape[0] == 3);
  TCICT_ASSERT(tensor_shape[1] == 4);
  TCICT_ASSERT(tensor_shape[2] == 5);
  TCICT_ASSERT(tci::size(ctx, tensor) == 60);
}

template <typename TenT>
void test_allocate_2d(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_ALLOCATE
  return;
#endif
  auto& ctx = fix.context();
  tci::shape_t<TenT> shape = {2, 3};
  auto tensor = tci::allocate<TenT>(ctx, shape);

  auto tensor_shape = tci::shape(ctx, tensor);
  TCICT_ASSERT(tensor_shape.size() == 2);
  TCICT_ASSERT(tensor_shape[0] == 2);
  TCICT_ASSERT(tensor_shape[1] == 3);
  TCICT_ASSERT(tci::size(ctx, tensor) == 6);
}

template <typename TenT>
void test_allocate_1d(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_ALLOCATE
  return;
#endif
  auto& ctx = fix.context();
  tci::shape_t<TenT> shape = {10};
  auto tensor = tci::allocate<TenT>(ctx, shape);

  auto tensor_shape = tci::shape(ctx, tensor);
  TCICT_ASSERT(tensor_shape.size() == 1);
  TCICT_ASSERT(tensor_shape[0] == 10);
  TCICT_ASSERT(tci::size(ctx, tensor) == 10);
}

// --- clear ---

template <typename TenT>
void test_clear_basic(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_CLEAR
  return;
#endif
  auto& ctx = fix.context();
  auto tensor = tci::eye<TenT>(ctx, 3);
  TCICT_ASSERT(tci::size(ctx, tensor) == 9);
  TCICT_ASSERT_NOTHROW(tci::clear(ctx, tensor));
}

template <typename TenT>
void test_clear_empty(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_CLEAR
  return;
#endif
  auto& ctx = fix.context();
  TenT empty_tensor;
  TCICT_ASSERT_NOTHROW(tci::clear(ctx, empty_tensor));
}

template <typename TenT>
void test_clear_and_reallocate(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_CLEAR
  return;
#endif
  auto& ctx = fix.context();
  auto tensor = tci::eye<TenT>(ctx, 2);
  tci::clear(ctx, tensor);
  tensor = tci::allocate<TenT>(ctx, {2, 2});
}

// --- move (in-place) ---

template <typename TenT>
void test_move_inplace(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_MOVE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  auto source = tci::eye<TenT>(ctx, 3);

  auto source_size = tci::size(ctx, source);
  auto source_shape = tci::shape(ctx, source);
  auto original_elem = tci::get_elem(ctx, source, {0, 0});

  auto destination = tci::move(ctx, source);

  TCICT_ASSERT(tci::size(ctx, destination) == source_size);
  TCICT_ASSERT(tci::shape(ctx, destination) == source_shape);
  auto dest_elem = tci::get_elem(ctx, destination, {0, 0});
  TCICT_ASSERT_CLOSE(real_part<TenT>(dest_elem), real_part<TenT>(original_elem), eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(dest_elem), imag_part<TenT>(original_elem), eps);
}

// --- move (out-of-place) ---

template <typename TenT>
void test_move_outofplace(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_MOVE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  auto source = tci::eye<TenT>(ctx, 2);

  auto source_size = tci::size(ctx, source);
  auto source_shape = tci::shape(ctx, source);
  auto original_elem = tci::get_elem(ctx, source, {1, 1});

  auto result = tci::move(ctx, source);

  TCICT_ASSERT(tci::size(ctx, result) == source_size);
  TCICT_ASSERT(tci::shape(ctx, result) == source_shape);
  auto result_elem = tci::get_elem(ctx, result, {1, 1});
  TCICT_ASSERT_CLOSE(real_part<TenT>(result_elem), real_part<TenT>(original_elem), eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(result_elem), imag_part<TenT>(original_elem), eps);
}

// --- move empty tensor ---

template <typename TenT>
void test_move_empty(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_MOVE
  return;
#endif
  auto& ctx = fix.context();
  TenT empty_source;
  tci::move(ctx, empty_source);
}

// --- move preserves values ---

template <typename TenT>
void test_move_preserves_values(tci_test_fixture<TenT>& fix) {
#ifdef TCICT_SKIP_MOVE
  return;
#endif
  auto& ctx = fix.context();
  auto eps = fix.epsilon();
  auto source = tci::zeros<TenT>(ctx, {2, 3});

  auto val1 = make_elem<TenT>(2.5, 1.5);
  auto val2 = make_elem<TenT>(3.7, -2.1);
  tci::set_elem(ctx, source, {0, 1}, val1);
  tci::set_elem(ctx, source, {1, 2}, val2);

  auto destination = tci::move(ctx, source);

  auto moved_val1 = tci::get_elem(ctx, destination, {0, 1});
  auto moved_val2 = tci::get_elem(ctx, destination, {1, 2});

  TCICT_ASSERT_CLOSE(real_part<TenT>(moved_val1), real_part<TenT>(val1), eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(moved_val1), imag_part<TenT>(val1), eps);
  TCICT_ASSERT_CLOSE(real_part<TenT>(moved_val2), real_part<TenT>(val2), eps);
  TCICT_ASSERT_CLOSE(imag_part<TenT>(moved_val2), imag_part<TenT>(val2), eps);
}

}}  // namespace tcict::tests
