// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_SEMAPHORE_H_
#define IREE_HAL_SEMAPHORE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/queue.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_t iree_hal_device_t;

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// A bitmask of flags controlling the behavior of a semaphore.
enum iree_hal_semaphore_flag_bits_t {
  IREE_HAL_SEMAPHORE_FLAG_NONE = 0u,
};
typedef uint32_t iree_hal_semaphore_flags_t;

// The maximum valid payload value of an iree_hal_semaphore_t.
// Payload values larger than this indicate that the semaphore has failed.
//
// This originates from Vulkan having a lower-bound of INT_MAX for
// maxTimelineSemaphoreValueDifference and many Android devices only supporting
// that lower-bound. At ~100 signals per second it'll take 1.5+ years to
// saturate. We may increase this value at some point but so long as there are
// some devices in the wild that may have this limitation we can ensure better
// consistency across the backends by observing this.
//
// The major mitigation here is that in proper usage of IREE there are no
// semaphores that are implicitly referenced by multiple VMs (each creates their
// own internally) and in a multitenant system each session should have its own
// semaphores - so even if the process lives for years it's highly unlikely any
// particular session does. Whatever, 640K is enough for anyone.
//
// In the future we may try to back this out and go back to UINT64_MAX.
//
// See:
//   https://vulkan.gpuinfo.org/displayextensionproperty.php?name=maxTimelineSemaphoreValueDifference
#define IREE_HAL_SEMAPHORE_MAX_VALUE (2147483647ull - 1)

// The minimum value for a semaphore that indicates failure. Any value
// greater-than-or-equal-to (>=) this indicates the semaphore has failed.
//
// If the upper bit 63 is set then the value represents an iree_status_t.
// Use iree_hal_semaphore_failure_as_status to convert a payload value to a
// status. Not all implementations do (or can) support encoding statuses and may
// only ever be able to set a failing semaphore to this value.
#define IREE_HAL_SEMAPHORE_FAILURE_VALUE (IREE_HAL_SEMAPHORE_MAX_VALUE + 1)

// Bit indicating that a failing semaphore value can be interpreted as an
// iree_status_t.
#define IREE_HAL_SEMAPHORE_FAILURE_VALUE_STATUS_BIT 0x8000000000000000ull

// Returns a semaphore payload value that encodes the given |status|.
// Ownership of the status is transferred to the semaphore and it must be
// freed by a consumer. Not all implementations can support failure status
// payloads and this should only be used by those implementations that can.
static inline uint64_t iree_hal_status_as_semaphore_failure(
    iree_status_t status) {
  return IREE_HAL_SEMAPHORE_FAILURE_VALUE_STATUS_BIT |
         (((uint64_t)status) >> 1);
}

// Returns OK if the |value| does not indicate an error.
// Returns an error status if the semaphore payload value represents a failure.
// If the payload contains an encoded iree_status_t it will be cloned and the
// new copy will be returned to the caller.
static inline iree_status_t iree_hal_semaphore_failure_as_status(
    uint64_t value) {
  if (value >= IREE_HAL_SEMAPHORE_FAILURE_VALUE) {
    if (value & IREE_HAL_SEMAPHORE_FAILURE_VALUE_STATUS_BIT) {
      // The top bits of a pointer are sign-extended from bit 47 so we can
      // restore the top bit by left-shifting the upper bits and then
      // right-shifting with sign extension. We only use a single bit today and
      // so bit 62 should still be the original value of the pointer.
      // Note that if the status is just a code (no allocated pointer) this
      // clone is a no-op and the code will be returned without an allocation.
      //
      // See:
      // https://en.wikipedia.org/wiki/X86-64#Canonical_form_addresses
      return iree_status_clone((iree_status_t)(((int64_t)value << 1) >> 1));
    } else {
      return iree_status_from_code(IREE_STATUS_INTERNAL);
    }
  } else {
    return iree_ok_status();
  }
}

// Frees an iree_status_t encoded in a semaphore |value|, if any.
static inline void iree_hal_semaphore_failure_free(uint64_t value) {
  if (value & IREE_HAL_SEMAPHORE_FAILURE_VALUE_STATUS_BIT) {
    iree_status_free((iree_status_t)(((int64_t)value << 1) >> 1));
  }
}

//===----------------------------------------------------------------------===//
// iree_hal_semaphore_t
//===----------------------------------------------------------------------===//

// Synchronization mechanism for host->device, device->host, host->host,
// and device->device notification. Semaphores behave like Vulkan timeline
// semaphores (or D3D12 fences) and contain a monotonically increasing
// uint64_t payload. They may be waited on any number of times even if they
// have already been signaled for a particular value. They may also be waited
// on for a particular value prior to the signal for that value.
//
// A semaphore is updated to its new value after all prior commands have
// completed but the delay between completion and the host being woken varies.
// Some implementations may coalesce semaphores to avoid spurious waking while
// others will immediately synchronize with the host.
//
// One use of semaphores is for resource lifetime management: all resources used
// by a set of submission batches must be considered live until the semaphore
// attached to the submission has signaled.
//
// Another use of semaphores is device->device synchronization for setting up
// the DAG of command buffers across queue submissions. This allows devices to
// perform non-trivial scheduling behavior without the need to wake the host.
//
// Semaphores may be set to a permanently failed state by implementations when
// errors occur during asynchronous execution. Users are expected to propagate
// the failures and possibly reset the entire device that produced the error.
//
// For more information on semaphores see the following docs describing how
// timelines are generally used (specifically in the device->host case):
// https://www.youtube.com/watch?v=SpE--Rf516Y
// https://www.khronos.org/assets/uploads/developers/library/2018-xdc/Vulkan-Timeline-Semaphores-Part-1_Sep18.pdf
// https://docs.microsoft.com/en-us/windows/win32/direct3d12/user-mode-heap-synchronization
typedef struct iree_hal_semaphore_t iree_hal_semaphore_t;

// Creates a semaphore that can be used with command queues owned by this
// device. To use the semaphores with other devices or instances they must
// first be exported.
IREE_API_EXPORT iree_status_t iree_hal_semaphore_create(
    iree_hal_device_t* device, uint64_t initial_value,
    iree_hal_semaphore_flags_t flags, iree_hal_semaphore_t** out_semaphore);

// Retains the given |semaphore| for the caller.
IREE_API_EXPORT void iree_hal_semaphore_retain(iree_hal_semaphore_t* semaphore);

// Releases the given |semaphore| from the caller.
IREE_API_EXPORT void iree_hal_semaphore_release(
    iree_hal_semaphore_t* semaphore);

// Queries the current payload of the semaphore and stores the result in
// |out_value|. As the payload is monotonically increasing it is guaranteed that
// the value is at least equal to the previous result of a
// iree_hal_semaphore_query call and coherent with any waits for a
// specified value via iree_device_wait_all_semaphores.
//
// Returns the status at the time the method is called without blocking and as
// such is only valid after a semaphore has been signaled. The same failure
// status will be returned regardless of when in the timeline the error
// occurred.
IREE_API_EXPORT iree_status_t
iree_hal_semaphore_query(iree_hal_semaphore_t* semaphore, uint64_t* out_value);

// Signals the |semaphore| to the given payload value.
// The call is ignored if the current payload value exceeds |new_value|.
IREE_API_EXPORT iree_status_t
iree_hal_semaphore_signal(iree_hal_semaphore_t* semaphore, uint64_t new_value);

// Signals the |semaphore| with a failure. The |status| will be returned from
// iree_hal_semaphore_query and iree_hal_semaphore_signal for the lifetime
// of the semaphore. Ownership of the status transfers to the semaphore and
// callers must clone it if they wish to retain it.
IREE_API_EXPORT void iree_hal_semaphore_fail(iree_hal_semaphore_t* semaphore,
                                             iree_status_t status);

// Blocks the caller until the semaphore reaches or exceeds the specified
// payload |value| or the |timeout| elapses.
//
// Returns success if the wait is successful and the semaphore has met or
// exceeded the required payload value.
//
// Returns IREE_STATUS_DEADLINE_EXCEEDED if the |timeout| elapses without the
// semaphore reaching the required value. If an asynchronous failure occurred
// this will return the failure status that was set immediately.
//
// Returns IREE_STATUS_ABORTED if one or more semaphores has failed. Callers can
// use iree_hal_semaphore_query on the semaphores to find the ones that have
// failed and get the status.
IREE_API_EXPORT iree_status_t iree_hal_semaphore_wait(
    iree_hal_semaphore_t* semaphore, uint64_t value, iree_timeout_t timeout);

// Returns a wait source reference to |semaphore| after it reaches or exceeds
// the specified payload |value|.
IREE_API_EXPORT iree_wait_source_t
iree_hal_semaphore_await(iree_hal_semaphore_t* semaphore, uint64_t value);

//===----------------------------------------------------------------------===//
// iree_hal_semaphore_list_t
//===----------------------------------------------------------------------===//

// A list of semaphores and their corresponding payloads.
// When signaling each semaphore will be set to the new payload value provided.
// When waiting each semaphore must reach or exceed the payload value.
// This points at external storage and does not retain the semaphores itself.
typedef struct iree_hal_semaphore_list_t {
  iree_host_size_t count;
  iree_hal_semaphore_t** semaphores;
  uint64_t* payload_values;
} iree_hal_semaphore_list_t;

// Returns an empty semaphore list.
static inline iree_hal_semaphore_list_t iree_hal_semaphore_list_empty(void) {
  iree_hal_semaphore_list_t list = {0};
  return list;
}

// Signals each semaphore in |semaphore_list| to the defined timepoint.
IREE_API_EXPORT iree_status_t
iree_hal_semaphore_list_signal(iree_hal_semaphore_list_t semaphore_list);

// Signals each semaphore in |semaphore_list| to indicate failure with
// |signal_status|.
// Takes ownership of |signal_status|.
IREE_API_EXPORT void iree_hal_semaphore_list_fail(
    iree_hal_semaphore_list_t semaphore_list, iree_status_t signal_status);

// Blocks the caller until all semaphore timepoints are reached or the |timeout|
// elapses.
//
// Returns success if the wait is successful and all semaphores reached their
// timepoints successfully.
//
// Returns IREE_STATUS_DEADLINE_EXCEEDED if the |timeout| elapses without all
// timepoints being reached. If an asynchronous failure occurred on any timeline
// this will return the failure status that was set immediately.
//
// Returns IREE_STATUS_ABORTED if one or more semaphores has failed. Callers can
// use iree_hal_semaphore_query to get the status from each.
//
// NOTE: this is not the most optimal way to wait on semaphores; if at all
// possible use a single wait on a single semaphore to avoid additional
// overheads in multiplexing fences across device implementations. This list
// wait should be used to perform a join that will propagate failures from any
// semaphore used in timepoints.
IREE_API_EXPORT iree_status_t iree_hal_semaphore_list_wait(
    iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout);

//===----------------------------------------------------------------------===//
// iree_hal_semaphore_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_semaphore_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_semaphore_t* semaphore);

  iree_status_t(IREE_API_PTR* query)(iree_hal_semaphore_t* semaphore,
                                     uint64_t* out_value);
  iree_status_t(IREE_API_PTR* signal)(iree_hal_semaphore_t* semaphore,
                                      uint64_t new_value);
  void(IREE_API_PTR* fail)(iree_hal_semaphore_t* semaphore,
                           iree_status_t status);

  iree_status_t(IREE_API_PTR* wait)(iree_hal_semaphore_t* semaphore,
                                    uint64_t value, iree_timeout_t timeout);
} iree_hal_semaphore_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_semaphore_vtable_t);

IREE_API_EXPORT void iree_hal_semaphore_destroy(
    iree_hal_semaphore_t* semaphore);

// Erases the i-th semaphore from the list in-place with O(1).
// Expects that the index |i| is in bounds.
IREE_API_EXPORT void iree_hal_semaphore_list_erase(
    iree_hal_semaphore_list_t* semaphore_list, iree_host_size_t i);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_SEMAPHORE_H_