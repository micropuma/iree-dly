// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/hip_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/event_pool.h"
#include "iree/base/internal/math.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/hip/context_util.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/event_pool.h"
#include "iree/hal/drivers/hip/event_semaphore.h"
#include "iree/hal/drivers/hip/graph_command_buffer.h"
#include "iree/hal/drivers/hip/hip_allocator.h"
#include "iree/hal/drivers/hip/hip_buffer.h"
#include "iree/hal/drivers/hip/memory_pools.h"
#include "iree/hal/drivers/hip/nop_executable_cache.h"
#include "iree/hal/drivers/hip/rccl_channel.h"
#include "iree/hal/drivers/hip/rccl_dynamic_symbols.h"
#include "iree/hal/drivers/hip/status_util.h"
#include "iree/hal/drivers/hip/stream_command_buffer.h"
#include "iree/hal/drivers/hip/timepoint_pool.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/deferred_work_queue.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/memory_file.h"
#include "iree/hal/utils/stream_tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_hip_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_hip_device_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t block_pool;

  // Optional driver that owns the HIP symbols. We retain it for our lifetime
  // to ensure the symbols remains valid.
  iree_hal_driver_t* driver;

  const iree_hal_hip_dynamic_symbols_t* hip_symbols;
  const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols;

  // Parameters used to control device behavior.
  iree_hal_hip_device_params_t params;

  hipCtx_t hip_context;
  hipDevice_t hip_device;
  // TODO: Support multiple device streams.
  // The hipStream_t used to issue device kernels and allocations.
  hipStream_t hip_dispatch_stream;

  iree_hal_stream_tracing_context_t* tracing_context;

  iree_allocator_t host_allocator;

  // Host/device event pools, used for backing semaphore timepoints.
  iree_event_pool_t* host_event_pool;
  iree_hal_hip_event_pool_t* device_event_pool;
  // Timepoint pools, shared by various semaphores.
  iree_hal_hip_timepoint_pool_t* timepoint_pool;

  // A queue to order device workloads and relase to the GPU when constraints
  // are met. It buffers submissions and allocations internally before they
  // are ready. This queue couples with HAL semaphores backed by iree_event_t
  // and hipEvent_t objects.
  iree_hal_deferred_work_queue_t* work_queue;

  // Device memory pools and allocators.
  bool supports_memory_pools;
  iree_hal_hip_memory_pools_t memory_pools;
  iree_hal_allocator_t* device_allocator;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;
} iree_hal_hip_device_t;

static iree_hal_hip_device_t* iree_hal_hip_device_cast(
    iree_hal_device_t* base_value);

static const iree_hal_device_vtable_t iree_hal_hip_device_vtable;
static const iree_hal_deferred_work_queue_device_interface_vtable_t
    iree_hal_hip_deferred_work_queue_device_interface_vtable;

// We put a hipEvent_t into a iree_hal_deferred_work_queue_native_event_t.
static_assert(sizeof(hipEvent_t) <=
                  sizeof(iree_hal_deferred_work_queue_native_event_t),
              "Unexpected event size");
typedef struct iree_hal_hip_deferred_work_queue_device_interface_t {
  iree_hal_deferred_work_queue_device_interface_t base;
  iree_hal_device_t* device;
  hipDevice_t hip_device;
  hipCtx_t hip_context;
  hipStream_t dispatch_hip_stream;
  iree_allocator_t host_allocator;
  const iree_hal_hip_dynamic_symbols_t* hip_symbols;
} iree_hal_hip_deferred_work_queue_device_interface_t;

static void iree_hal_hip_deferred_work_queue_device_interface_destroy(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface) {
  iree_hal_hip_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_hip_deferred_work_queue_device_interface_t*)(base_device_interface);
  iree_allocator_free(device_interface->host_allocator, device_interface);
}

static iree_status_t
iree_hal_hip_deferred_work_queue_device_interface_bind_to_thread(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface) {
  iree_hal_hip_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_hip_deferred_work_queue_device_interface_t*)(base_device_interface);
  return IREE_HIP_RESULT_TO_STATUS(
      device_interface->hip_symbols,
      hipCtxSetCurrent(device_interface->hip_context), "hipCtxSetCurrent");
}

static iree_status_t
iree_hal_hip_deferred_work_queue_device_interface_wait_native_event(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_deferred_work_queue_native_event_t event) {
  iree_hal_hip_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_hip_deferred_work_queue_device_interface_t*)(base_device_interface);
  return IREE_HIP_RESULT_TO_STATUS(
      device_interface->hip_symbols,
      hipStreamWaitEvent(device_interface->dispatch_hip_stream,
                         (hipEvent_t)event, 0),
      "hipStreamWaitEvent");
}

static iree_status_t
iree_hal_hip_deferred_work_queue_device_interface_create_native_event(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_deferred_work_queue_native_event_t* out_event) {
  iree_hal_hip_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_hip_deferred_work_queue_device_interface_t*)(base_device_interface);
  return IREE_HIP_RESULT_TO_STATUS(device_interface->hip_symbols,
                                   hipEventCreate((hipEvent_t*)out_event),
                                   "hipEventCreate");
}
static iree_status_t
iree_hal_hip_deferred_work_queue_device_interface_record_native_event(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_deferred_work_queue_native_event_t event) {
  iree_hal_hip_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_hip_deferred_work_queue_device_interface_t*)(base_device_interface);
  return IREE_HIP_RESULT_TO_STATUS(
      device_interface->hip_symbols,
      hipEventRecord((hipEvent_t)event, device_interface->dispatch_hip_stream),
      "hipEventRecord");
}

static iree_status_t
iree_hal_hip_deferred_work_queue_device_interface_synchronize_native_event(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_deferred_work_queue_native_event_t event) {
  iree_hal_hip_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_hip_deferred_work_queue_device_interface_t*)(base_device_interface);
  return IREE_HIP_RESULT_TO_STATUS(device_interface->hip_symbols,
                                   hipEventSynchronize((hipEvent_t)event));
}

static iree_status_t
iree_hal_hip_deferred_work_queue_device_interface_destroy_native_event(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_deferred_work_queue_native_event_t event) {
  iree_hal_hip_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_hip_deferred_work_queue_device_interface_t*)(base_device_interface);
  return IREE_HIP_RESULT_TO_STATUS(device_interface->hip_symbols,
                                   hipEventDestroy((hipEvent_t)event));
}

static iree_status_t
iree_hal_hip_deferred_work_queue_device_interface_semaphore_acquire_timepoint_device_signal_native_event(
    iree_hal_deferred_work_queue_device_interface_t* device_interface,
    struct iree_hal_semaphore_t* semaphore, uint64_t value,
    iree_hal_deferred_work_queue_native_event_t* out_event) {
  return iree_hal_hip_event_semaphore_acquire_timepoint_device_signal(
      semaphore, value, (hipEvent_t*)out_event);
}

static bool
iree_hal_hip_deferred_work_queue_device_interface_acquire_host_wait_event(
    iree_hal_deferred_work_queue_device_interface_t* device_interface,
    struct iree_hal_semaphore_t* semaphore, uint64_t value,
    iree_hal_deferred_work_queue_host_device_event_t* out_event) {
  return iree_hal_hip_semaphore_acquire_event_host_wait(
      semaphore, value, (iree_hal_hip_event_t**)out_event);
}

static iree_status_t
iree_hal_hip_deferred_work_queue_device_interface_device_wait_on_host_event(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_deferred_work_queue_host_device_event_t wait_event) {
  iree_hal_hip_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_hip_deferred_work_queue_device_interface_t*)(base_device_interface);
  return IREE_HIP_RESULT_TO_STATUS(
      device_interface->hip_symbols,
      hipStreamWaitEvent(
          device_interface->dispatch_hip_stream,
          iree_hal_hip_event_handle((iree_hal_hip_event_t*)wait_event), 0),
      "hipStreamWaitEvent");
}

static void
iree_hal_hip_deferred_work_queue_device_interface_release_wait_event(
    iree_hal_deferred_work_queue_device_interface_t* device_interface,
    iree_hal_deferred_work_queue_host_device_event_t wait_event) {
  iree_hal_hip_event_release(wait_event);
}

static iree_hal_deferred_work_queue_native_event_t
iree_hal_hip_deferred_work_queue_device_interface_native_event_from_wait_event(
    iree_hal_deferred_work_queue_device_interface_t* device_interface,
    iree_hal_deferred_work_queue_host_device_event_t event) {
  iree_hal_hip_event_t* wait_event = (iree_hal_hip_event_t*)event;
  return iree_hal_hip_event_handle(wait_event);
}

static iree_status_t
iree_hal_hip_deferred_work_queue_device_interface_create_stream_command_buffer(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_command_buffer_mode_t mode, iree_hal_command_category_t categories,
    iree_hal_command_buffer_t** out) {
  iree_hal_hip_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_hip_deferred_work_queue_device_interface_t*)(base_device_interface);
  return iree_hal_hip_device_create_stream_command_buffer(
      device_interface->device, mode, categories, 0, out);
}

static iree_status_t
iree_hal_hip_deferred_work_queue_device_interface_submit_command_buffer(
    iree_hal_deferred_work_queue_device_interface_t* device_interface,
    iree_hal_command_buffer_t* command_buffer) {
  iree_hal_hip_deferred_work_queue_device_interface_t* table =
      (iree_hal_hip_deferred_work_queue_device_interface_t*)(device_interface);
  iree_status_t status = iree_ok_status();
  if (iree_hal_hip_stream_command_buffer_isa(command_buffer)) {
    // Stream command buffer so nothing to do but notify it was submitted.
    iree_hal_hip_stream_notify_submitted_commands(command_buffer);
  } else {
    hipGraphExec_t exec =
        iree_hal_hip_graph_command_buffer_handle(command_buffer);
    status = IREE_HIP_RESULT_TO_STATUS(
        table->hip_symbols, hipGraphLaunch(exec, table->dispatch_hip_stream));
    if (IREE_LIKELY(iree_status_is_ok(status))) {
      iree_hal_hip_graph_tracing_notify_submitted_commands(command_buffer);
    }
  }
  return status;
}

static iree_status_t
iree_hal_hip_deferred_work_queue_device_interface_async_alloc(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_buffer_t* buffer) {
  iree_hal_hip_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_hip_deferred_work_queue_device_interface_t*)
          base_device_interface;
  iree_hal_hip_device_t* device =
      iree_hal_hip_device_cast(device_interface->device);
  if (device->supports_memory_pools) {
    return iree_hal_hip_memory_pools_allocate_pointer(
        &device->memory_pools, buffer, device->hip_dispatch_stream,
        iree_hal_buffer_allocation_size(buffer));
  }

  return iree_hal_hip_allocator_alloc_async(
      iree_hal_device_allocator(device_interface->device),
      device->hip_dispatch_stream, buffer);
}

// Asynchronously frees a buffer.
static iree_status_t
iree_hal_hip_deferred_work_queue_device_interface_async_dealloc(
    iree_hal_deferred_work_queue_device_interface_t* base_device_interface,
    iree_hal_buffer_t* buffer) {
  iree_hal_hip_deferred_work_queue_device_interface_t* device_interface =
      (iree_hal_hip_deferred_work_queue_device_interface_t*)
          base_device_interface;
  iree_hal_hip_device_t* device =
      iree_hal_hip_device_cast(device_interface->device);
  if (device->supports_memory_pools) {
    return iree_hal_hip_memory_pools_deallocate(
        &device->memory_pools, device->hip_dispatch_stream, buffer);
  }
  return iree_hal_hip_allocator_free_async(
      iree_hal_device_allocator(device_interface->device),
      device->hip_dispatch_stream, buffer);
}

typedef struct iree_hal_hip_tracing_device_interface_t {
  iree_hal_stream_tracing_device_interface_t base;
  hipDevice_t device;
  hipCtx_t context;
  hipStream_t dispatch_stream;
  iree_allocator_t host_allocator;
  const iree_hal_hip_dynamic_symbols_t* hip_symbols;
} iree_hal_hip_tracing_device_interface_t;
static const iree_hal_stream_tracing_device_interface_vtable_t
    iree_hal_hip_tracing_device_interface_vtable_t;

void iree_hal_hip_tracing_device_interface_destroy(
    iree_hal_stream_tracing_device_interface_t* base_device_interface) {
  iree_hal_hip_tracing_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_interface_t*)base_device_interface;

  iree_allocator_free(device_interface->host_allocator, device_interface);
}

iree_status_t iree_hal_hip_tracing_device_interface_synchronize_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t base_event) {
  iree_hal_hip_tracing_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_interface_t*)base_device_interface;

  return IREE_HIP_RESULT_TO_STATUS(device_interface->hip_symbols,
                                   hipEventSynchronize((hipEvent_t)base_event));
}

iree_status_t iree_hal_hip_tracing_device_interface_create_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t* base_event) {
  iree_hal_hip_tracing_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_interface_t*)base_device_interface;

  return IREE_HIP_RESULT_TO_STATUS(
      device_interface->hip_symbols,
      hipEventCreateWithFlags((hipEvent_t*)base_event, hipEventDefault));
}

iree_status_t iree_hal_hip_tracing_device_interface_query_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t base_event) {
  iree_hal_hip_tracing_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_interface_t*)base_device_interface;

  return IREE_HIP_RESULT_TO_STATUS(device_interface->hip_symbols,
                                   hipEventQuery((hipEvent_t)base_event));
}

void iree_hal_hip_tracing_device_interface_event_elapsed_time(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    float* relative_millis, iree_hal_stream_tracing_native_event_t start_event,
    iree_hal_stream_tracing_native_event_t end_event) {
  iree_hal_hip_tracing_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_interface_t*)base_device_interface;

  IREE_HIP_IGNORE_ERROR(
      device_interface->hip_symbols,
      hipEventElapsedTime(relative_millis, (hipEvent_t)start_event,
                          (hipEvent_t)end_event));
}

void iree_hal_hip_tracing_device_interface_destroy_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t base_event) {
  iree_hal_hip_tracing_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_interface_t*)base_device_interface;

  IREE_HIP_IGNORE_ERROR(device_interface->hip_symbols,
                        hipEventDestroy((hipEvent_t)base_event));
}

iree_status_t iree_hal_hip_tracing_device_interface_record_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t base_event) {
  iree_hal_hip_tracing_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_interface_t*)base_device_interface;

  return IREE_HIP_RESULT_TO_STATUS(
      device_interface->hip_symbols,
      hipEventRecord((hipEvent_t)base_event,
                     (hipStream_t)device_interface->dispatch_stream));
}

iree_status_t iree_hal_hip_tracing_device_interface_add_graph_event_record_node(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_graph_node_t* out_node,
    iree_hal_stream_tracing_native_graph_t graph,
    iree_hal_stream_tracing_native_graph_node_t* dependency_nodes,
    size_t dependency_nodes_count,
    iree_hal_stream_tracing_native_event_t event) {
  iree_hal_hip_tracing_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_interface_t*)base_device_interface;

  return IREE_HIP_RESULT_TO_STATUS(
      device_interface->hip_symbols,
      hipGraphAddEventRecordNode((hipGraphNode_t*)out_node, (hipGraph_t)graph,
                                 (hipGraphNode_t*)dependency_nodes,
                                 dependency_nodes_count, (hipEvent_t)event));
}

static iree_hal_hip_device_t* iree_hal_hip_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hip_device_vtable);
  return (iree_hal_hip_device_t*)base_value;
}

static iree_hal_hip_device_t* iree_hal_hip_device_cast_unsafe(
    iree_hal_device_t* base_value) {
  return (iree_hal_hip_device_t*)base_value;
}

IREE_API_EXPORT void iree_hal_hip_device_params_initialize(
    iree_hal_hip_device_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  out_params->arena_block_size = 32 * 1024;
  out_params->event_pool_capacity = 32;
  out_params->queue_count = 1;
  out_params->command_buffer_mode = IREE_HAL_HIP_COMMAND_BUFFER_MODE_STREAM;
  out_params->stream_tracing = 0;
  out_params->async_allocations = true;
  out_params->allow_inline_execution = false;
}

static iree_status_t iree_hal_hip_device_check_params(
    const iree_hal_hip_device_params_t* params) {
  if (params->arena_block_size < 4096) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "arena block size too small (< 4096 bytes)");
  }
  if (params->queue_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at least one queue is required");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_device_create_internal(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_hip_device_params_t* params, hipDevice_t hip_device,
    hipStream_t dispatch_stream, hipCtx_t context,
    const iree_hal_hip_dynamic_symbols_t* symbols,
    const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_hip_device_t* device = NULL;
  iree_host_size_t total_size = iree_sizeof_struct(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&device));

  iree_hal_resource_initialize(&iree_hal_hip_device_vtable, &device->resource);
  iree_string_view_append_to_buffer(
      identifier, &device->identifier,
      (char*)device + iree_sizeof_struct(*device));
  iree_arena_block_pool_initialize(params->arena_block_size, host_allocator,
                                   &device->block_pool);
  device->driver = driver;
  iree_hal_driver_retain(device->driver);
  device->hip_symbols = symbols;
  device->nccl_symbols = nccl_symbols;
  device->params = *params;
  device->hip_context = context;
  device->hip_device = hip_device;
  device->hip_dispatch_stream = dispatch_stream;
  device->host_allocator = host_allocator;

  iree_hal_hip_deferred_work_queue_device_interface_t* device_interface;
  iree_status_t status = iree_allocator_malloc(
      host_allocator,
      sizeof(iree_hal_hip_deferred_work_queue_device_interface_t),
      (void**)&device_interface);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    iree_hal_device_release((iree_hal_device_t*)device);
    return status;
  }
  device_interface->base.vtable =
      &iree_hal_hip_deferred_work_queue_device_interface_vtable;
  device_interface->hip_context = context;
  device_interface->hip_symbols = symbols;
  device_interface->device = (iree_hal_device_t*)device;
  device_interface->hip_device = hip_device;
  device_interface->dispatch_hip_stream = dispatch_stream;
  device_interface->host_allocator = host_allocator;
  status = iree_hal_deferred_work_queue_create(
      (iree_hal_deferred_work_queue_device_interface_t*)device_interface,
      &device->block_pool, host_allocator, &device->work_queue);

  // Enable tracing for the (currently only) stream - no-op if disabled.
  if (iree_status_is_ok(status) && device->params.stream_tracing) {
    if (device->params.stream_tracing >=
            IREE_HAL_STREAM_TRACING_VERBOSITY_MAX ||
        device->params.stream_tracing < IREE_HAL_STREAM_TRACING_VERBOSITY_OFF) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "invalid stream_tracing argument: expected to be between %d and %d",
          IREE_HAL_STREAM_TRACING_VERBOSITY_OFF,
          IREE_HAL_STREAM_TRACING_VERBOSITY_MAX);
    }

    iree_hal_hip_tracing_device_interface_t* tracing_device_interface = NULL;
    status = iree_allocator_malloc(
        host_allocator, sizeof(iree_hal_hip_tracing_device_interface_t),
        (void**)&tracing_device_interface);

    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      iree_hal_device_release((iree_hal_device_t*)device);
      return status;
    }

    tracing_device_interface->base.vtable =
        &iree_hal_hip_tracing_device_interface_vtable_t;
    tracing_device_interface->context = context;
    tracing_device_interface->device = hip_device;
    tracing_device_interface->dispatch_stream = dispatch_stream;
    tracing_device_interface->host_allocator = host_allocator;
    tracing_device_interface->hip_symbols = symbols;

    status = iree_hal_stream_tracing_context_allocate(
        (iree_hal_stream_tracing_device_interface_t*)tracing_device_interface,
        device->identifier, device->params.stream_tracing, &device->block_pool,
        host_allocator, &device->tracing_context);
  }

  // Memory pool support is conditional.
  if (iree_status_is_ok(status) && params->async_allocations) {
    int supports_memory_pools = 0;
    status = IREE_HIP_RESULT_TO_STATUS(
        symbols,
        hipDeviceGetAttribute(&supports_memory_pools,
                              hipDeviceAttributeMemoryPoolsSupported,
                              hip_device),
        "hipDeviceGetAttribute");
    device->supports_memory_pools = supports_memory_pools != 0;
  }

  // Create memory pools first so that we can share them with the allocator.
  if (iree_status_is_ok(status) && device->supports_memory_pools) {
    status = iree_hal_hip_memory_pools_initialize(
        symbols, hip_device, context, &params->memory_pools, host_allocator,
        &device->memory_pools);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_allocator_create(
        symbols, hip_device, context, dispatch_stream,
        device->supports_memory_pools ? &device->memory_pools : NULL,
        host_allocator, &device->device_allocator);
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  return status;
}

iree_status_t iree_hal_hip_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_hip_device_params_t* params,
    const iree_hal_hip_dynamic_symbols_t* symbols,
    const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols, hipDevice_t device,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_hip_device_check_params(params);

  // Get the main context for the device.
  hipCtx_t context = NULL;
  if (iree_status_is_ok(status)) {
    status = IREE_HIP_RESULT_TO_STATUS(
        symbols, hipDevicePrimaryCtxRetain(&context, device));
  }
  if (iree_status_is_ok(status)) {
    status = IREE_HIP_RESULT_TO_STATUS(symbols, hipCtxSetCurrent(context));
  }

  // Create the default dispatch stream for the device.
  hipStream_t dispatch_stream = NULL;
  if (iree_status_is_ok(status)) {
    status = IREE_HIP_RESULT_TO_STATUS(
        symbols,
        hipStreamCreateWithFlags(&dispatch_stream, hipStreamNonBlocking));
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_device_create_internal(
        driver, identifier, params, device, dispatch_stream, context, symbols,
        nccl_symbols, host_allocator, out_device);
  } else {
    if (dispatch_stream) symbols->hipStreamDestroy(dispatch_stream);
    // NOTE: This function return hipSuccess though doesn't release the
    // primaryCtx by design on HIP/HCC path.
    if (context) symbols->hipDevicePrimaryCtxRelease(device);
  }

  iree_event_pool_t* host_event_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_event_pool_allocate(params->event_pool_capacity,
                                      host_allocator, &host_event_pool);
  }

  iree_hal_hip_event_pool_t* device_event_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_event_pool_allocate(
        symbols, context, params->event_pool_capacity, host_allocator,
        &device_event_pool);
  }

  iree_hal_hip_timepoint_pool_t* timepoint_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_timepoint_pool_allocate(
        host_event_pool, device_event_pool, params->event_pool_capacity,
        host_allocator, &timepoint_pool);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_hip_device_t* hip_device = iree_hal_hip_device_cast(*out_device);
    hip_device->host_event_pool = host_event_pool;
    hip_device->device_event_pool = device_event_pool;
    hip_device->timepoint_pool = timepoint_pool;
  } else {
    // Release resources we have accquired after HAL device creation.
    if (timepoint_pool) iree_hal_hip_timepoint_pool_free(timepoint_pool);
    if (device_event_pool) iree_hal_hip_event_pool_release(device_event_pool);
    if (host_event_pool) iree_event_pool_free(host_event_pool);
    // Release other resources via the HAL device.
    iree_hal_device_release(*out_device);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

hipCtx_t iree_hal_hip_device_context(iree_hal_device_t* base_device) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast_unsafe(base_device);
  return device->hip_context;
}

const iree_hal_hip_dynamic_symbols_t* iree_hal_hip_device_dynamic_symbols(
    iree_hal_device_t* base_device) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast_unsafe(base_device);
  return device->hip_symbols;
}

static void iree_hal_hip_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  const iree_hal_hip_dynamic_symbols_t* symbols = device->hip_symbols;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Destroy the pending workload queue.
  iree_hal_deferred_work_queue_destroy(device->work_queue);

  // There should be no more buffers live that use the allocator.
  iree_hal_allocator_release(device->device_allocator);

  // Buffers may have been retaining collective resources.
  iree_hal_channel_provider_release(device->channel_provider);

  // Destroy memory pools that hold on to reserved memory.
  iree_hal_hip_memory_pools_deinitialize(&device->memory_pools);

  iree_hal_stream_tracing_context_free(device->tracing_context);

  // Destroy various pools for synchronization.
  if (device->timepoint_pool) {
    iree_hal_hip_timepoint_pool_free(device->timepoint_pool);
  }
  if (device->device_event_pool) {
    iree_hal_hip_event_pool_release(device->device_event_pool);
  }
  if (device->host_event_pool) iree_event_pool_free(device->host_event_pool);

  IREE_HIP_IGNORE_ERROR(symbols, hipStreamDestroy(device->hip_dispatch_stream));

  // NOTE: This function return hipSuccess though doesn't release the
  // primaryCtx by design on HIP/HCC path.
  IREE_HIP_IGNORE_ERROR(symbols,
                        hipDevicePrimaryCtxRelease(device->hip_device));

  iree_arena_block_pool_deinitialize(&device->block_pool);

  // Finally, destroy the device.
  iree_hal_driver_release(device->driver);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_hip_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_hip_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_hip_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_hip_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_hip_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_hip_device_trim(iree_hal_device_t* base_device) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_hip_set_context(device->hip_symbols, device->hip_context));
  iree_arena_block_pool_trim(&device->block_pool);
  IREE_RETURN_IF_ERROR(iree_hal_allocator_trim(device->device_allocator));
  if (device->supports_memory_pools) {
    IREE_RETURN_IF_ERROR(iree_hal_hip_memory_pools_trim(
        &device->memory_pools, &device->params.memory_pools));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_device_query_attribute(
    iree_hal_hip_device_t* device, hipDeviceAttribute_t attribute,
    int64_t* out_value) {
  IREE_RETURN_IF_ERROR(
      iree_hal_hip_set_context(device->hip_symbols, device->hip_context));
  int value = 0;
  IREE_HIP_RETURN_IF_ERROR(
      device->hip_symbols,
      hipDeviceGetAttribute(&value, attribute, device->hip_device),
      "hipDeviceGetAttribute");
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_hip_set_context(device->hip_symbols, device->hip_context));
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value =
        iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    *out_value = iree_string_view_equal(key, IREE_SV("rocm-hsaco-fb")) ? 1 : 0;
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_hip_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_hip_set_context(device->hip_symbols, device->hip_context));

  if (!device->nccl_symbols || !device->nccl_symbols->dylib) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "RCCL runtime library version %d.%d and greater not available; "
        "ensure installed and the shared library (rccl.dll/librccl.so) "
        "is on your PATH/LD_LIBRARY_PATH.",
        NCCL_MAJOR, NCCL_MINOR);
  }

  // Today we only allow a single logical device per channel.
  // We could multiplex channels but it'd be better to surface that to the
  // compiler so that it can emit the right rank math.
  int requested_count = iree_math_count_ones_u64(queue_affinity);
  // TODO(#12206): properly assign affinity in the compiler.
  if (requested_count != 64 && requested_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "exactly one participant is allowed in a "
                            "channel but %d were specified",
                            requested_count);
  }

  // Ask the channel provider (if configured) for the default rank and count
  // if the user did not set them.
  if (device->channel_provider &&
      (params.rank == IREE_HAL_CHANNEL_RANK_DEFAULT ||
       params.count == IREE_HAL_CHANNEL_COUNT_DEFAULT)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_channel_provider_query_default_rank_and_count(
            device->channel_provider, &params.rank, &params.count),
        "querying default collective group rank and count");
  }

  // An ID is required to initialize NCCL. On the root it'll be the local ID and
  // on all other participants it'll be the root ID.
  iree_hal_hip_nccl_id_t id;
  memset(&id, 0, sizeof(id));
  if (iree_const_byte_span_is_empty(params.id)) {
    // User wants the default ID.
    if (!device->channel_provider) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "default collective channel ID requested but no channel provider has "
          "been set on the device to provide it");
    }
    if (params.rank == 0) {
      // Bootstrap NCCL to get the root ID.
      IREE_RETURN_IF_ERROR(
          iree_hal_hip_nccl_get_unique_id(device->nccl_symbols, &id),
          "bootstrapping NCCL root");
    }
    // Exchange NCCL ID with all participants.
    IREE_RETURN_IF_ERROR(iree_hal_channel_provider_exchange_default_id(
                             device->channel_provider,
                             iree_make_byte_span((void*)&id, sizeof(id))),
                         "exchanging NCCL ID with other participants");
  } else if (params.id.data_length != IREE_ARRAYSIZE(id.data)) {
    // User provided something but it's not what we expect.
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "NCCL ID must be %zu bytes matching the "
                            "ncclUniqueId struct but caller provided %zu bytes",
                            IREE_ARRAYSIZE(id.data), sizeof(id));
  } else {
    // User provided the ID - we treat it as opaque here and let NCCL validate.
    memcpy(id.data, params.id.data, IREE_ARRAYSIZE(id.data));
  }

  if (iree_hal_hip_nccl_id_is_empty(&id)) {
    // TODO: maybe this is ok? a localhost alias or something?
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no default NCCL ID specified (all zeros)");
  }

  // TODO: when we support multiple logical devices we'll want to pass in the
  // context of the device mapped to the queue_affinity. For now since this
  // implementation only supports one device we pass in the only one we have.
  return iree_hal_hip_nccl_channel_create(
      device->hip_symbols, device->nccl_symbols, &id, params.rank, params.count,
      device->host_allocator, out_channel);
}

iree_status_t iree_hal_hip_device_create_stream_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_hip_set_context(device->hip_symbols, device->hip_context));

  return iree_hal_hip_stream_command_buffer_create(
      iree_hal_device_allocator(base_device), device->hip_symbols,
      device->nccl_symbols, device->hip_context, device->tracing_context, mode,
      command_categories, binding_capacity, device->hip_dispatch_stream,
      &device->block_pool, device->host_allocator, out_command_buffer);
}

static iree_status_t iree_hal_hip_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_hip_set_context(device->hip_symbols, device->hip_context));

  if (device->params.allow_inline_execution &&
      iree_all_bits_set(mode,
                        IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION)) {
    // The caller has indicated the command buffer can be executed as it is
    // recorded, implying that the command buffer cannot be reused and doesn't
    // need to be persisted. This lets us lower the execution delay as we can
    // directly route commands to a HIP stream and let it eagerly flush.
    return iree_hal_hip_stream_command_buffer_create(
        iree_hal_device_allocator(base_device), device->hip_symbols,
        device->nccl_symbols, device->hip_context, device->tracing_context,
        mode, command_categories, binding_capacity, device->hip_dispatch_stream,
        &device->block_pool, device->host_allocator, out_command_buffer);
  }
  switch (device->params.command_buffer_mode) {
    case IREE_HAL_HIP_COMMAND_BUFFER_MODE_GRAPH:
      // TODO(indirect-cmd): when we can record indirect graphs we won't need
      // to use deferred command buffers - this is here to emulate indirect
      // command buffers.
      if (binding_capacity > 0) {
        return iree_hal_deferred_command_buffer_create(
            iree_hal_device_allocator(base_device), mode, command_categories,
            binding_capacity, &device->block_pool,
            iree_hal_device_host_allocator(base_device), out_command_buffer);
      } else {
        return iree_hal_hip_graph_command_buffer_create(
            iree_hal_device_allocator(base_device), device->hip_symbols,
            device->tracing_context, device->hip_context, mode,
            command_categories, queue_affinity, binding_capacity,
            &device->block_pool, device->host_allocator, out_command_buffer);
      }
    case IREE_HAL_HIP_COMMAND_BUFFER_MODE_STREAM:
      return iree_hal_deferred_command_buffer_create(
          iree_hal_device_allocator(base_device), mode, command_categories,
          binding_capacity, &device->block_pool,
          iree_hal_device_host_allocator(base_device), out_command_buffer);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid command buffer mode");
  }
}

static iree_status_t iree_hal_hip_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "event not yet implmeneted");
}

static iree_status_t iree_hal_hip_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_hip_set_context(device->hip_symbols, device->hip_context));

  if (iree_io_file_handle_type(handle) !=
      IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "implementation does not support the external file type");
  }
  return iree_hal_memory_file_wrap(
      queue_affinity, access, handle, iree_hal_device_allocator(base_device),
      iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_hip_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_hip_set_context(device->hip_symbols, device->hip_context));
  return iree_hal_hip_nop_executable_cache_create(
      identifier, device->hip_symbols, device->hip_device, device->hip_context,
      device->host_allocator, out_executable_cache);
}

static iree_status_t iree_hal_hip_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_flags_t flags, iree_hal_semaphore_t** out_semaphore) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_hip_set_context(device->hip_symbols, device->hip_context));

  return iree_hal_hip_event_semaphore_create(
      initial_value, device->hip_symbols, device->hip_context,
      device->timepoint_pool, device->work_queue, device->host_allocator,
      out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_hip_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  // TODO: implement HIP semaphores.
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_hip_device_pepare_async_alloc(
    iree_hal_hip_device_t* device, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)allocation_size);

  iree_hal_buffer_params_canonicalize(&params);

  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_hip_buffer_wrap(
      device->device_allocator, params.type, params.access, params.usage,
      allocation_size, /*byte_offset=*/0,
      /*byte_length=*/allocation_size, IREE_HAL_HIP_BUFFER_TYPE_ASYNC,
      /*device_ptr=*/NULL, /*host_ptr=*/NULL,
      iree_hal_buffer_release_callback_null(), device->host_allocator, &buffer);

  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  } else if (buffer) {
    iree_hal_hip_buffer_set_allocation_empty(buffer);
    iree_hal_buffer_release(buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// TODO: implement multiple streams; today we only have one and queue_affinity
//       is ignored.
// TODO: implement proper semaphores in HIP to ensure ordering and avoid
//       the barrier here.
static iree_status_t iree_hal_hip_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_hip_set_context(device->hip_symbols, device->hip_context));

  if (device->supports_memory_pools &&
      !iree_all_bits_set(params.type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    iree_hal_buffer_t* buffer = NULL;

    IREE_RETURN_IF_ERROR(iree_hal_hip_memory_pools_prepare_buffer(
        &device->memory_pools, device->hip_dispatch_stream, pool, params,
        allocation_size, &buffer));

    iree_status_t status = iree_hal_deferred_work_queue_enqueue_alloc(
        device->work_queue, wait_semaphore_list, signal_semaphore_list, buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_deferred_work_queue_issue(device->work_queue);
    }
    if (iree_status_is_ok(status)) {
      *out_buffer = buffer;
    } else {
      iree_hal_hip_buffer_set_allocation_empty(buffer);
      iree_hal_resource_release(&buffer->resource);
    }
    return status;
  } else if (!iree_all_bits_set(params.type,
                                IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) &&
             iree_hal_hip_allocator_isa(
                 iree_hal_device_allocator(base_device))) {
    iree_hal_buffer_t* buffer = NULL;

    IREE_RETURN_IF_ERROR(iree_hal_hip_device_pepare_async_alloc(
        device, params, allocation_size, &buffer));

    iree_status_t status = iree_hal_deferred_work_queue_enqueue_alloc(
        device->work_queue, wait_semaphore_list, signal_semaphore_list, buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_deferred_work_queue_issue(device->work_queue);
    }
    if (iree_status_is_ok(status)) {
      *out_buffer = buffer;
    } else {
      iree_hal_hip_buffer_set_allocation_empty(buffer);
      iree_hal_resource_release(&buffer->resource);
    }
    return status;
  }

  // NOTE: block on the semaphores here; we could avoid this by properly
  // sequencing device work with semaphores. The HIP HAL is not currently
  // asynchronous.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
                                                    iree_infinite_timeout()));

  // Allocate from the pool; likely to fail in cases of virtual memory
  // exhaustion but the error may be deferred until a later synchronization.
  // If pools are not supported we allocate a buffer as normal from whatever
  // allocator is set on the device.
  iree_status_t status =
      iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(base_device),
                                         params, allocation_size, out_buffer);

  // Only signal if not returning a synchronous error - synchronous failure
  // indicates that the stream is unchanged (it's not really since we waited
  // above, but we at least won't deadlock like this).
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(signal_semaphore_list);
  }
  return status;
}

// TODO: implement multiple streams; today we only have one and queue_affinity
//       is ignored.
// TODO: implement proper semaphores in HIP to ensure ordering and avoid
//       the barrier here.
static iree_status_t iree_hal_hip_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_hip_set_context(device->hip_symbols, device->hip_context));

  if (iree_hal_hip_allocator_isa(iree_hal_device_allocator(base_device))) {
    iree_status_t status = iree_hal_deferred_work_queue_enqueue_dealloc(
        device->work_queue, wait_semaphore_list, signal_semaphore_list, buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_deferred_work_queue_issue(device->work_queue);
    }
    return status;
  }

  // NOTE: block on the semaphores here; we could avoid this by properly
  // sequencing device work with semaphores. The HIP HAL is not currently
  // asynchronous.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
                                                    iree_infinite_timeout()));

  // Schedule the buffer deallocation if we got it from a pool and otherwise
  // drop it on the floor and let it be freed when the buffer is released.
  iree_status_t status = iree_ok_status();
  if (device->supports_memory_pools) {
    status = iree_hal_hip_memory_pools_deallocate(
        &device->memory_pools, device->hip_dispatch_stream, buffer);
  }

  // Only signal if not returning a synchronous error - synchronous failure
  // indicates that the stream is unchanged (it's not really since we waited
  // above, but we at least won't deadlock like this).
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(signal_semaphore_list);
  }
  return status;
}

static iree_status_t iree_hal_hip_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  // TODO: expose streaming chunk count/size options.
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      .loop = iree_loop_inline(&loop_status),
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_read_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_file, source_offset, target_buffer, target_offset, length, flags,
      options));
  return loop_status;
}

static iree_status_t iree_hal_hip_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  // TODO: expose streaming chunk count/size options.
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      .loop = iree_loop_inline(&loop_status),
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_write_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_buffer, source_offset, target_file, target_offset, length, flags,
      options));
  return loop_status;
}

static void iree_hal_hip_device_collect_tracing_context(void* user_data) {
  iree_hal_stream_tracing_context_collect(
      (iree_hal_stream_tracing_context_t*)user_data);
}

static iree_status_t iree_hal_hip_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hip_set_context(device->hip_symbols, device->hip_context));

  iree_status_t status = iree_hal_deferred_work_queue_enqueue(
      device->work_queue, iree_hal_hip_device_collect_tracing_context,
      device->tracing_context, wait_semaphore_list, signal_semaphore_list,
      command_buffer ? 1 : 0, command_buffer ? &command_buffer : NULL,
      &binding_table);
  if (iree_status_is_ok(status)) {
    // Try to advance the deferred work queue.
    status = iree_hal_deferred_work_queue_issue(device->work_queue);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  // Try to advance the deferred work queue.
  iree_status_t status = iree_hal_deferred_work_queue_issue(device->work_queue);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  IREE_RETURN_IF_ERROR(
      iree_hal_hip_set_context(device->hip_symbols, device->hip_context));

  return iree_hal_hip_semaphore_multi_wait(semaphore_list, wait_mode, timeout,
                                           &device->block_pool);
}

static iree_status_t iree_hal_hip_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_device_profiling_flush(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_device_profiling_end(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_hip_device_vtable = {
    .destroy = iree_hal_hip_device_destroy,
    .id = iree_hal_hip_device_id,
    .host_allocator = iree_hal_hip_device_host_allocator,
    .device_allocator = iree_hal_hip_device_allocator,
    .replace_device_allocator = iree_hal_hip_replace_device_allocator,
    .replace_channel_provider = iree_hal_hip_replace_channel_provider,
    .trim = iree_hal_hip_device_trim,
    .query_i64 = iree_hal_hip_device_query_i64,
    .create_channel = iree_hal_hip_device_create_channel,
    .create_command_buffer = iree_hal_hip_device_create_command_buffer,
    .create_event = iree_hal_hip_device_create_event,
    .create_executable_cache = iree_hal_hip_device_create_executable_cache,
    .import_file = iree_hal_hip_device_import_file,
    .create_semaphore = iree_hal_hip_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_hip_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_hip_device_queue_alloca,
    .queue_dealloca = iree_hal_hip_device_queue_dealloca,
    .queue_fill = iree_hal_device_queue_emulated_fill,
    .queue_update = iree_hal_device_queue_emulated_update,
    .queue_copy = iree_hal_device_queue_emulated_copy,
    .queue_read = iree_hal_hip_device_queue_read,
    .queue_write = iree_hal_hip_device_queue_write,
    .queue_execute = iree_hal_hip_device_queue_execute,
    .queue_flush = iree_hal_hip_device_queue_flush,
    .wait_semaphores = iree_hal_hip_device_wait_semaphores,
    .profiling_begin = iree_hal_hip_device_profiling_begin,
    .profiling_flush = iree_hal_hip_device_profiling_flush,
    .profiling_end = iree_hal_hip_device_profiling_end,
};

static const iree_hal_deferred_work_queue_device_interface_vtable_t
    iree_hal_hip_deferred_work_queue_device_interface_vtable = {
        .destroy = iree_hal_hip_deferred_work_queue_device_interface_destroy,
        .bind_to_thread =
            iree_hal_hip_deferred_work_queue_device_interface_bind_to_thread,
        .wait_native_event =
            iree_hal_hip_deferred_work_queue_device_interface_wait_native_event,
        .create_native_event =
            iree_hal_hip_deferred_work_queue_device_interface_create_native_event,
        .record_native_event =
            iree_hal_hip_deferred_work_queue_device_interface_record_native_event,
        .synchronize_native_event =
            iree_hal_hip_deferred_work_queue_device_interface_synchronize_native_event,
        .destroy_native_event =
            iree_hal_hip_deferred_work_queue_device_interface_destroy_native_event,
        .semaphore_acquire_timepoint_device_signal_native_event =
            iree_hal_hip_deferred_work_queue_device_interface_semaphore_acquire_timepoint_device_signal_native_event,
        .acquire_host_wait_event =
            iree_hal_hip_deferred_work_queue_device_interface_acquire_host_wait_event,
        .device_wait_on_host_event =
            iree_hal_hip_deferred_work_queue_device_interface_device_wait_on_host_event,
        .release_wait_event =
            iree_hal_hip_deferred_work_queue_device_interface_release_wait_event,
        .native_event_from_wait_event =
            iree_hal_hip_deferred_work_queue_device_interface_native_event_from_wait_event,
        .create_stream_command_buffer =
            iree_hal_hip_deferred_work_queue_device_interface_create_stream_command_buffer,
        .submit_command_buffer =
            iree_hal_hip_deferred_work_queue_device_interface_submit_command_buffer,
        .async_alloc =
            iree_hal_hip_deferred_work_queue_device_interface_async_alloc,
        .async_dealloc =
            iree_hal_hip_deferred_work_queue_device_interface_async_dealloc,
};

static const iree_hal_stream_tracing_device_interface_vtable_t
    iree_hal_hip_tracing_device_interface_vtable_t = {
        .destroy = iree_hal_hip_tracing_device_interface_destroy,
        .synchronize_native_event =
            iree_hal_hip_tracing_device_interface_synchronize_native_event,
        .create_native_event =
            iree_hal_hip_tracing_device_interface_create_native_event,
        .query_native_event =
            iree_hal_hip_tracing_device_interface_query_native_event,
        .event_elapsed_time =
            iree_hal_hip_tracing_device_interface_event_elapsed_time,
        .destroy_native_event =
            iree_hal_hip_tracing_device_interface_destroy_native_event,
        .record_native_event =
            iree_hal_hip_tracing_device_interface_record_native_event,
        .add_graph_event_record_node =
            iree_hal_hip_tracing_device_interface_add_graph_event_record_node,
};
