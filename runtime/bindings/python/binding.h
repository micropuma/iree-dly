// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_PYTHON_IREE_BINDING_H_
#define IREE_BINDINGS_PYTHON_IREE_BINDING_H_

#include <optional>
#include <vector>

#include "iree/base/api.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/string_view.h"
#include "nanobind/stl/vector.h"

// Uncomment the following to enable various noisy output to stderr for
// verifying sequencing and reference counting.
// #define IREE_PY_TRACE_ENABLED 1

#if IREE_PY_TRACE_ENABLED
#define IREE_PY_TRACEF(fmt, ...) \
  fprintf(stderr, "[[IREE_PY_TRACE]]: " fmt "\n", __VA_ARGS__)
#define IREE_PY_TRACE(msg) fprintf(stderr, "[[IREE_PY_TRACE]]: %s", msg)
#else
#define IREE_PY_TRACEF(...)
#define IREE_PY_TRACE(...)
#endif

namespace iree {
namespace python {

namespace py = nanobind;
using namespace nanobind::literals;

template <typename T>
struct ApiPtrAdapter {};

template <typename Self, typename T>
class ApiRefCounted {
 public:
  using RawPtrType = T*;
  ApiRefCounted() : instance_(nullptr) {}
  ApiRefCounted(ApiRefCounted& other) : instance_(other.instance_) { Retain(); }
  ApiRefCounted(ApiRefCounted&& other) : instance_(other.instance_) {
    other.instance_ = nullptr;
  }
  ApiRefCounted& operator=(ApiRefCounted&& other) {
    instance_ = other.instance_;
    other.instance_ = nullptr;
    return *this;
  }
  void operator=(const ApiRefCounted&) = delete;

  ~ApiRefCounted() { Release(); }

  // Steals the reference to the object referenced by the given raw pointer and
  // returns a wrapper (transfers ownership).
  static Self StealFromRawPtr(T* retained_inst) {
    auto self = Self();
    self.instance_ = retained_inst;
    return self;
  }

  // Retains the object referenced by the given raw pointer and returns
  // a wrapper.
  static Self BorrowFromRawPtr(T* non_retained_inst) {
    auto self = Self();
    self.instance_ = non_retained_inst;
    if (non_retained_inst) {
      ApiPtrAdapter<T>::Retain(non_retained_inst);
    }
    return self;
  }

  // Whether it is nullptr.
  operator bool() const { return instance_; }

  T* steal_raw_ptr() {
    T* ret = instance_;
    instance_ = nullptr;
    return ret;
  }

  T* raw_ptr() {
    if (!instance_) {
      throw std::invalid_argument("API object is null");
    }
    return instance_;
  }

  const T* raw_ptr() const {
    return const_cast<ApiRefCounted*>(this)->raw_ptr();
  }

  void Retain() {
    if (instance_) {
      ApiPtrAdapter<T>::Retain(instance_);
    }
  }
  void Release() {
    if (instance_) {
      ApiPtrAdapter<T>::Release(instance_);
    }
  }

 private:
  T* instance_;
};

// Pybind11 had an isintance for Python objects helper. Nanobind doesn't.
inline bool is_instance_of_type_object(py::handle inst,
                                       py::handle type_object) {
  int rc = PyObject_IsInstance(inst.ptr(), type_object.ptr());
  if (rc == -1) {
    throw py::python_error();
  }
  return static_cast<bool>(rc);
}

// Nanobind's tuple class has a default constructor that creates a nullptr
// tuple. Which is not really what one wants.
inline py::object create_empty_tuple() {
  return py::steal(py::handle(PyTuple_New(0)));
}

// For a bound class, binds the buffer protocol. This will result in a call
// to on the CppType:
//   HandleBufferProtocol(Py_buffer *view, int flags)
// This is a low level callback and must not raise any exceptions. If
// error conditions are warranted the usual PyErr_SetString approach must be
// used (and -1 returned). Return 0 on success.
template <typename CppType>
void BindBufferProtocol(py::handle clazz) {
  PyBufferProcs buffer_procs;
  memset(&buffer_procs, 0, sizeof(buffer_procs));
  buffer_procs.bf_getbuffer =
      // It is not legal to raise exceptions from these callbacks.
      +[](PyObject* raw_self, Py_buffer* view, int flags) -> int {
    if (view == NULL) {
      PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
      return -1;
    }

    // Cast must succeed due to invariants.
    auto self = py::cast<CppType*>(py::handle(raw_self));

    Py_INCREF(raw_self);
    view->obj = raw_self;
    return self->HandleBufferProtocol(view, flags);
  };
  buffer_procs.bf_releasebuffer =
      +[](PyObject* raw_self, Py_buffer* view) -> void {};
  auto heap_type = reinterpret_cast<PyHeapTypeObject*>(clazz.ptr());
  assert(heap_type->ht_type.tp_flags & Py_TPFLAGS_HEAPTYPE &&
         "must be heap type");
  heap_type->as_buffer = buffer_procs;
}

// Nanobind 2.0 had a backwards compatibility bug where it left out the
// def_static helper. For cases that use this, we patch it here. Note that
// any def_static must be called directly on the subclassed enum as the existing
// helpers do not chain to this subclass.
// See: https://github.com/wjakob/nanobind/issues/597
// It is likely that this is fixed in the next minor version, at which time,
// this check should be changed to select only the affected versions for
// special treatment.
#if NB_VERSION_MAJOR >= 2
template <typename T>
struct nanobind1_compat_enum_ : nanobind::enum_<T> {
  using py::enum_<T>::enum_;
  template <typename Func, typename... Extra>
  nanobind1_compat_enum_& def_static(const char* name_, Func&& f,
                                     const Extra&... extra) {
    static_assert(
        !std::is_member_function_pointer_v<Func>,
        "def_static(...) called with a non-static member function pointer");
    cpp_function_def((::nanobind::detail::forward_t<Func>)f,
                     nanobind::scope(*this), nanobind::name(name_), extra...);
    return *this;
  }
};
#else
template <typename T>
using nanobind1_compat_enum_ = nanobind::enum_<T>;
#endif

}  // namespace python
}  // namespace iree

#endif  // IREE_BINDINGS_PYTHON_IREE_BINDING_H_
