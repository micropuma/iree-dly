// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/graph_command_buffer.h"

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/cuda/cuda_buffer.h"
#include "iree/hal/drivers/cuda/cuda_dynamic_symbols.h"
#include "iree/hal/drivers/cuda/cuda_status_util.h"
#include "iree/hal/drivers/cuda/native_executable.h"
#include "iree/hal/utils/collective_batch.h"
#include "iree/hal/utils/resource_set.h"
#include "iree/hal/utils/stream_tracing.h"

// The maximal number of CUDA graph nodes that can run concurrently between
// barriers.
#define IREE_HAL_CUDA_MAX_CONCURRENT_GRAPH_NODE_COUNT 32

// Command buffer implementation that directly records into CUDA graphs.
// The command buffer records the commands on the calling thread without
// additional threading indirection.
typedef struct iree_hal_cuda_graph_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  const iree_hal_cuda_dynamic_symbols_t* symbols;

  // Per-stream CUDA tracing context.
  iree_hal_stream_tracing_context_t* tracing_context;
  iree_hal_stream_tracing_context_event_list_t tracing_event_list;

  // A resource set to maintain references to all resources used within the
  // command buffer.
  iree_hal_resource_set_t* resource_set;

  // Staging arena used for host->device transfers.
  // This is used for when we need CUDA to be able to reference memory as it
  // performs asynchronous operations.
  iree_arena_allocator_t arena;

  CUcontext cu_context;
  // The CUDA graph under construction.
  CUgraph cu_graph;
  CUgraphExec cu_graph_exec;

  // A node acting as a barrier for all commands added to the command buffer.
  CUgraphNode cu_barrier_node;

  // Nodes added to the command buffer after the last barrier.
  CUgraphNode cu_graph_nodes[IREE_HAL_CUDA_MAX_CONCURRENT_GRAPH_NODE_COUNT];
  iree_host_size_t graph_node_count;

  // Iteratively constructed batch of collective operations.
  iree_hal_collective_batch_t collective_batch;
} iree_hal_cuda_graph_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_cuda_graph_command_buffer_vtable;

static iree_status_t
iree_hal_cuda_graph_command_buffer_execution_barrier_internal(
    iree_hal_cuda_graph_command_buffer_t* command_buffer);

static iree_hal_cuda_graph_command_buffer_t*
iree_hal_cuda_graph_command_buffer_cast(iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_graph_command_buffer_vtable);
  return (iree_hal_cuda_graph_command_buffer_t*)base_value;
}

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

static void iree_cuda_graph_command_buffer_trace_zone_begin_external(
    iree_hal_cuda_graph_command_buffer_t* command_buffer, int32_t verbosity,
    const char* file_name, size_t file_name_length, uint32_t line,
    const char* function_name, size_t function_name_length, const char* name,
    size_t name_length) {
  // Make sure there are no new nodes after the last barrier.
  // Work should start after the event.
  if (IREE_UNLIKELY(command_buffer->graph_node_count != 0)) {
    iree_hal_cuda_graph_command_buffer_execution_barrier_internal(
        command_buffer);
  }

  CUgraphNode* tracing_event_node =
      &command_buffer->cu_graph_nodes[command_buffer->graph_node_count++];
  size_t dependency_count = command_buffer->cu_barrier_node ? 1 : 0;
  IREE_HAL_GRAPH_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, &command_buffer->tracing_event_list,
      (iree_hal_stream_tracing_native_graph_node_t*)tracing_event_node,
      (iree_hal_stream_tracing_native_graph_t*)command_buffer->cu_graph,
      verbosity,
      (iree_hal_stream_tracing_native_graph_node_t*)&command_buffer
          ->cu_barrier_node,
      dependency_count, file_name, file_name_length, line, function_name,
      function_name_length, name, name_length);

  // Move the barrier forward to make sure that the tracing event is recorded
  // before work starts.
  // Downstream operations will wait on the tracing node.
  command_buffer->cu_barrier_node = *tracing_event_node;
}

static void iree_cuda_graph_command_buffer_trace_zone_end(
    iree_hal_cuda_graph_command_buffer_t* command_buffer, int32_t verbosity) {
  // Make sure there are no new nodes after the last barrier.
  // Prior work should end before the tracing event is recorded.
  if (IREE_UNLIKELY(command_buffer->graph_node_count != 0)) {
    iree_hal_cuda_graph_command_buffer_execution_barrier_internal(
        command_buffer);
  }

  CUgraphNode* tracing_event_node =
      &command_buffer->cu_graph_nodes[command_buffer->graph_node_count++];
  size_t dependency_count = command_buffer->cu_barrier_node ? 1 : 0;
  IREE_ASSERT_GT(dependency_count, 0,
                 "ending a zone should at least depend on the beginning");
  IREE_HAL_GRAPH_TRACE_ZONE_END(
      command_buffer->tracing_context, &command_buffer->tracing_event_list,
      (iree_hal_stream_tracing_native_graph_node_t*)tracing_event_node,
      (iree_hal_stream_tracing_native_graph_t*)command_buffer->cu_graph,
      verbosity,
      (iree_hal_stream_tracing_native_graph_node_t*)&command_buffer
          ->cu_barrier_node,
      dependency_count);

  // We need to wait on the tracing end before other work starts.
  // GPU tracing zones are first-in, last-out.
  command_buffer->cu_barrier_node = *tracing_event_node;
}

#define IREE_CUDA_GRAPH_COMMAND_BUFFER_TRACE_ZONE_BEGIN_EXTERNAL(   \
    command_buffer, verbosity, file_name, file_name_length, line,   \
    function_name, function_name_length, name, name_length)         \
  iree_cuda_graph_command_buffer_trace_zone_begin_external(         \
      command_buffer, verbosity, file_name, file_name_length, line, \
      function_name, function_name_length, name, name_length)
#define IREE_CUDA_GRAPH_COMMAND_BUFFER_TRACE_ZONE_BEGIN(command_buffer, \
                                                        verbosity)      \
  IREE_CUDA_GRAPH_COMMAND_BUFFER_TRACE_ZONE_BEGIN_EXTERNAL(             \
      command_buffer, verbosity, /*file_name=*/NULL, 0, /*line=*/0,     \
      __FUNCTION__, strlen(__FUNCTION__), /*name=*/NULL, 0)
#define IREE_CUDA_GRAPH_COMMAND_BUFFER_TRACE_ZONE_END(command_buffer, \
                                                      verbosity)      \
  iree_cuda_graph_command_buffer_trace_zone_end(command_buffer, verbosity)

#else  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

#define IREE_CUDA_GRAPH_COMMAND_BUFFER_TRACE_ZONE_BEGIN_EXTERNAL( \
    command_buffer, verbosity, file_name, file_name_length, line, \
    function_name, function_name_length, name, name_length)
#define IREE_CUDA_GRAPH_COMMAND_BUFFER_TRACE_ZONE_BEGIN(command_buffer, \
                                                        verbosity)
#define IREE_CUDA_GRAPH_COMMAND_BUFFER_TRACE_ZONE_END(command_buffer, verbosity)
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

iree_status_t iree_hal_cuda_graph_command_buffer_create(
    iree_hal_allocator_t* device_allocator,
    const iree_hal_cuda_dynamic_symbols_t* cuda_symbols,
    iree_hal_stream_tracing_context_t* tracing_context, CUcontext context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_graph_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator,
                            sizeof(*command_buffer) +
                                iree_hal_command_buffer_validation_state_size(
                                    mode, binding_capacity),
                            (void**)&command_buffer));

  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, (uint8_t*)command_buffer + sizeof(*command_buffer),
      &iree_hal_cuda_graph_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->symbols = cuda_symbols;
  command_buffer->tracing_context = tracing_context;
  command_buffer->tracing_event_list.head = NULL;
  command_buffer->tracing_event_list.tail = NULL;
  iree_arena_initialize(block_pool, &command_buffer->arena);
  command_buffer->cu_context = context;
  command_buffer->cu_graph = NULL;
  command_buffer->cu_graph_exec = NULL;
  command_buffer->cu_barrier_node = NULL;
  command_buffer->graph_node_count = 0;

  iree_status_t status =
      iree_hal_resource_set_allocate(block_pool, &command_buffer->resource_set);

  if (iree_status_is_ok(status)) {
    iree_hal_collective_batch_initialize(&command_buffer->arena,
                                         command_buffer->resource_set,
                                         &command_buffer->collective_batch);
  }

  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_command_buffer_release(&command_buffer->base);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda_graph_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_stream_tracing_free(command_buffer->tracing_context,
                               &command_buffer->tracing_event_list);

  // Drop any pending collective batches before we tear things down.
  iree_hal_collective_batch_clear(&command_buffer->collective_batch);

  if (command_buffer->cu_graph != NULL) {
    IREE_CUDA_IGNORE_ERROR(command_buffer->symbols,
                           cuGraphDestroy(command_buffer->cu_graph));
    command_buffer->cu_graph = NULL;
  }
  if (command_buffer->cu_graph_exec != NULL) {
    IREE_CUDA_IGNORE_ERROR(command_buffer->symbols,
                           cuGraphExecDestroy(command_buffer->cu_graph_exec));
    command_buffer->cu_graph_exec = NULL;
  }
  command_buffer->cu_barrier_node = NULL;
  command_buffer->graph_node_count = 0;

  iree_hal_collective_batch_deinitialize(&command_buffer->collective_batch);
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_cuda_graph_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_cuda_graph_command_buffer_vtable);
}

CUgraphExec iree_hal_cuda_graph_command_buffer_handle(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);
  return command_buffer->cu_graph_exec;
}

void iree_hal_cuda_graph_tracing_notify_submitted_commands(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);
  if (!command_buffer->tracing_context) {
    return;
  }

  iree_hal_stream_tracing_notify_submitted(command_buffer->tracing_context,
                                           &command_buffer->tracing_event_list);
}

// Flushes any pending batched collective operations.
// Must be called before any other non-collective nodes are added to the graph
// or a barrier is encountered.
static iree_status_t iree_hal_cuda_graph_command_buffer_flush_collectives(
    iree_hal_cuda_graph_command_buffer_t* command_buffer) {
  // NOTE: we could move this out into callers by way of an always-inline shim -
  // that would make this a single compare against the command buffer state we
  // are likely to access immediately after anyway and keep overheads minimal.
  if (IREE_LIKELY(iree_hal_collective_batch_is_empty(
          &command_buffer->collective_batch))) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(#9580): use CUDA graph capture so that the NCCL calls end up in the
  // graph:
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html
  //
  // Something like:
  //  syms->cuStreamBeginCapture(nccl_stream);
  //  iree_hal_cuda_nccl_submit_batch(command_buffer->context,
  //                                  &command_buffer->collective_batch,
  //                                  nccl_stream);
  //  syms->cuStreamEndCapture(nccl_stream, &child_graph);
  //  syms->cuGraphAddChildGraphNode(..., child_graph);
  //  syms->cuGraphDestroy(child_graph);  // probably, I think it gets cloned
  //
  // Note that we'll want to create a scratch stream that we use to perform the
  // capture - we could memoize that on the command buffer or on the device
  // (though that introduces potential threading issues). There may be a special
  // stream mode for these capture-only streams that is lighter weight than a
  // normal stream.
  iree_status_t status = iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "CUDA graph capture of collective operations not yet implemented");

  iree_hal_collective_batch_clear(&command_buffer->collective_batch);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_cuda_graph_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);

  if (command_buffer->cu_graph != NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command buffer cannot be re-recorded");
  }

  // Create a new empty graph to record into.
  IREE_CUDA_RETURN_IF_ERROR(
      command_buffer->symbols,
      cuGraphCreate(&command_buffer->cu_graph, /*flags=*/0), "cuGraphCreate");

  IREE_CUDA_GRAPH_COMMAND_BUFFER_TRACE_ZONE_BEGIN(
      command_buffer, IREE_HAL_STREAM_TRACING_VERBOSITY_COARSE);

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_graph_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);

  // Flush any pending collective batches.
  IREE_RETURN_IF_ERROR(
      iree_hal_cuda_graph_command_buffer_flush_collectives(command_buffer));

  IREE_CUDA_GRAPH_COMMAND_BUFFER_TRACE_ZONE_END(
      command_buffer, IREE_HAL_STREAM_TRACING_VERBOSITY_COARSE);

  // Reset state used during recording.
  command_buffer->cu_barrier_node = NULL;
  command_buffer->graph_node_count = 0;

  // Compile the graph.
  CUgraphNode error_node = NULL;
  iree_status_t status = IREE_CURESULT_TO_STATUS(
      command_buffer->symbols,
      cuGraphInstantiate(&command_buffer->cu_graph_exec,
                         command_buffer->cu_graph, &error_node,
                         /*logBuffer=*/NULL,
                         /*bufferSize=*/0));
  if (iree_status_is_ok(status)) {
    // No longer need the source graph used for construction.
    IREE_CUDA_IGNORE_ERROR(command_buffer->symbols,
                           cuGraphDestroy(command_buffer->cu_graph));
    command_buffer->cu_graph = NULL;
  }

  iree_hal_resource_set_freeze(command_buffer->resource_set);

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_graph_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);

  (void)command_buffer;
  IREE_CUDA_GRAPH_COMMAND_BUFFER_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer, IREE_HAL_STREAM_TRACING_VERBOSITY_COARSE,
      location ? location->file.data : NULL, location ? location->file.size : 0,
      location ? location->line : 0,
      /*func_name=*/NULL, 0, label.data, label.size);

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_graph_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);
  (void)command_buffer;
  IREE_CUDA_GRAPH_COMMAND_BUFFER_TRACE_ZONE_END(
      command_buffer, IREE_HAL_STREAM_TRACING_VERBOSITY_COARSE);
  return iree_ok_status();
}

static iree_status_t
iree_hal_cuda_graph_command_buffer_execution_barrier_internal(
    iree_hal_cuda_graph_command_buffer_t* command_buffer) {
  IREE_RETURN_IF_ERROR(
      iree_hal_cuda_graph_command_buffer_flush_collectives(command_buffer));

  IREE_ASSERT_GT(command_buffer->graph_node_count, 0,
                 "expected at least one node before a barrier");

  // Use the last node as a barrier to avoid creating redundant empty nodes.
  if (IREE_LIKELY(command_buffer->graph_node_count == 1)) {
    command_buffer->cu_barrier_node = command_buffer->cu_graph_nodes[0];
    command_buffer->graph_node_count = 0;
    return iree_ok_status();
  }

  IREE_CUDA_RETURN_IF_ERROR(
      command_buffer->symbols,
      cuGraphAddEmptyNode(
          &command_buffer->cu_barrier_node, command_buffer->cu_graph,
          command_buffer->cu_graph_nodes, command_buffer->graph_node_count),
      "cuGraphAddEmptyNode");

  command_buffer->graph_node_count = 0;

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_graph_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_hal_cuda_graph_command_buffer_execution_barrier_internal(
          command_buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_cuda_graph_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_cuda_graph_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_cuda_graph_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_cuda_graph_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  // We could mark the memory as invalidated so that if this is a managed buffer
  // CUDA does not try to copy it back to the host.
  return iree_ok_status();
}

// Splats a pattern value of 1/2/4 bytes out to a 4 byte value.
static uint32_t iree_hal_cuda_splat_pattern(const void* pattern,
                                            size_t pattern_length) {
  switch (pattern_length) {
    case 1: {
      uint32_t pattern_value = *(const uint8_t*)(pattern);
      return (pattern_value << 24) | (pattern_value << 16) |
             (pattern_value << 8) | pattern_value;
    }
    case 2: {
      uint32_t pattern_value = *(const uint16_t*)(pattern);
      return (pattern_value << 16) | pattern_value;
    }
    case 4: {
      uint32_t pattern_value = *(const uint32_t*)(pattern);
      return pattern_value;
    }
    default:
      return 0;  // Already verified that this should not be possible.
  }
}

static iree_status_t iree_hal_cuda_graph_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_CUDA_GRAPH_COMMAND_BUFFER_TRACE_ZONE_BEGIN(
      command_buffer, IREE_HAL_STREAM_TRACING_VERBOSITY_FINE);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_graph_command_buffer_flush_collectives(command_buffer));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &target_ref.buffer));

  CUdeviceptr target_device_buffer = iree_hal_cuda_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  uint32_t pattern_4byte = iree_hal_cuda_splat_pattern(pattern, pattern_length);

  CUDA_MEMSET_NODE_PARAMS params = {
      .dst = target_device_buffer + target_offset,
      .elementSize = pattern_length,
      .pitch = 0,                                   // unused if height == 1
      .width = target_ref.length / pattern_length,  // element count
      .height = 1,
      .value = pattern_4byte,
  };

  if (command_buffer->graph_node_count >=
      IREE_HAL_CUDA_MAX_CONCURRENT_GRAPH_NODE_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "exceeded max concurrent node limit");
  }

  size_t dependency_count = command_buffer->cu_barrier_node ? 1 : 0;
  IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->symbols,
      cuGraphAddMemsetNode(
          &command_buffer->cu_graph_nodes[command_buffer->graph_node_count++],
          command_buffer->cu_graph, &command_buffer->cu_barrier_node,
          dependency_count, &params, command_buffer->cu_context),
      "cuGraphAddMemsetNode");

  IREE_CUDA_GRAPH_COMMAND_BUFFER_TRACE_ZONE_END(
      command_buffer, IREE_HAL_STREAM_TRACING_VERBOSITY_FINE);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_graph_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_CUDA_GRAPH_COMMAND_BUFFER_TRACE_ZONE_BEGIN(
      command_buffer, IREE_HAL_STREAM_TRACING_VERBOSITY_FINE);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_graph_command_buffer_flush_collectives(command_buffer));

  // Allocate scratch space in the arena for the data and copy it in.
  // The update buffer API requires that the command buffer capture the host
  // memory at the time the method is called in case the caller wants to reuse
  // the memory. Because CUDA memcpys are async if we didn't copy it's possible
  // for the reused memory to change before the stream reaches the copy
  // operation and get the wrong data.
  uint8_t* storage = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, target_ref.length,
                              (void**)&storage));
  memcpy(storage, (const uint8_t*)source_buffer + source_offset,
         target_ref.length);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &target_ref.buffer));

  CUdeviceptr target_device_buffer = iree_hal_cuda_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  CUDA_MEMCPY3D params = {
      .srcMemoryType = CU_MEMORYTYPE_HOST,
      .srcHost = storage,
      .dstMemoryType = CU_MEMORYTYPE_DEVICE,
      .dstDevice = target_device_buffer,
      .dstXInBytes =
          iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset,
      .WidthInBytes = target_ref.length,
      .Height = 1,
      .Depth = 1,
  };

  if (command_buffer->graph_node_count >=
      IREE_HAL_CUDA_MAX_CONCURRENT_GRAPH_NODE_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "exceeded max concurrent node limit");
  }

  size_t dependency_count = command_buffer->cu_barrier_node ? 1 : 0;
  IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->symbols,
      cuGraphAddMemcpyNode(
          &command_buffer->cu_graph_nodes[command_buffer->graph_node_count++],
          command_buffer->cu_graph, &command_buffer->cu_barrier_node,
          dependency_count, &params, command_buffer->cu_context),
      "cuGraphAddMemcpyNode");

  IREE_CUDA_GRAPH_COMMAND_BUFFER_TRACE_ZONE_END(
      command_buffer, IREE_HAL_STREAM_TRACING_VERBOSITY_FINE);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_graph_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_CUDA_GRAPH_COMMAND_BUFFER_TRACE_ZONE_BEGIN(
      command_buffer, IREE_HAL_STREAM_TRACING_VERBOSITY_FINE);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_graph_command_buffer_flush_collectives(command_buffer));

  const iree_hal_buffer_t* resources[2] = {source_ref.buffer,
                                           target_ref.buffer};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set,
                                       IREE_ARRAYSIZE(resources), resources));

  CUdeviceptr source_device_buffer = iree_hal_cuda_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(source_ref.buffer));
  iree_device_size_t source_offset =
      iree_hal_buffer_byte_offset(source_ref.buffer) + source_ref.offset;
  CUdeviceptr target_device_buffer = iree_hal_cuda_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;

  CUDA_MEMCPY3D params = {
      .srcMemoryType = CU_MEMORYTYPE_DEVICE,
      .srcDevice = source_device_buffer,
      .srcXInBytes = source_offset,
      .dstMemoryType = CU_MEMORYTYPE_DEVICE,
      .dstDevice = target_device_buffer,
      .dstXInBytes = target_offset,
      .WidthInBytes = target_ref.length,
      .Height = 1,
      .Depth = 1,
  };

  if (command_buffer->graph_node_count >=
      IREE_HAL_CUDA_MAX_CONCURRENT_GRAPH_NODE_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "exceeded max concurrent node limit");
  }

  size_t dependency_count = command_buffer->cu_barrier_node ? 1 : 0;
  IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->symbols,
      cuGraphAddMemcpyNode(
          &command_buffer->cu_graph_nodes[command_buffer->graph_node_count++],
          command_buffer->cu_graph, &command_buffer->cu_barrier_node,
          dependency_count, &params, command_buffer->cu_context),
      "cuGraphAddMemcpyNode");

  IREE_CUDA_GRAPH_COMMAND_BUFFER_TRACE_ZONE_END(
      command_buffer, IREE_HAL_STREAM_TRACING_VERBOSITY_FINE);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_graph_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);
  iree_hal_buffer_binding_t send_binding = {
      .buffer = send_ref.buffer,
      .offset = send_ref.offset,
      .length = send_ref.length,
  };
  iree_hal_buffer_binding_t recv_binding = {
      .buffer = recv_ref.buffer,
      .offset = recv_ref.offset,
      .length = recv_ref.length,
  };
  return iree_hal_collective_batch_append(&command_buffer->collective_batch,
                                          channel, op, param, send_binding,
                                          recv_binding, element_count);
}

static iree_status_t iree_hal_cuda_graph_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    const uint32_t workgroup_count[3], iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_cuda_graph_command_buffer_t* command_buffer =
      iree_hal_cuda_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_graph_command_buffer_flush_collectives(command_buffer));

  // Lookup kernel parameters used for side-channeling additional launch
  // information from the compiler.
  const iree_hal_cuda_kernel_params_t* kernel_params = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_native_executable_lookup_kernel_params(
              executable, entry_point, &kernel_params));

  IREE_CUDA_GRAPH_COMMAND_BUFFER_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer, IREE_HAL_STREAM_TRACING_VERBOSITY_FINE,
      kernel_params->debug_info.source_filename.data,
      kernel_params->debug_info.source_filename.size,
      kernel_params->debug_info.source_line,
      kernel_params->debug_info.function_name.data,
      kernel_params->debug_info.function_name.size, /*name=*/NULL, 0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &executable));
  // We append push constants to the end of descriptors to form a linear chain
  // of kernel arguments.
  iree_host_size_t kernel_params_count =
      kernel_params->binding_count + kernel_params->constant_count;
  iree_host_size_t kernel_params_length = kernel_params_count * sizeof(void*);

  // TODO: use packed parameters instead of the indirection mechanism - this
  // would avoid additional driver overhead to reflect and repack them all.
  //
  // Per CUDA API requirements, we need two levels of indirection for passing
  // kernel arguments in.
  //   "If the kernel has N parameters, then kernelParams needs to be an array
  //   of N pointers. Each pointer, from kernelParams[0] to kernelParams[N-1],
  //   points to the region of memory from which the actual parameter will be
  //   copied."
  //
  // (From the cuGraphAddKernelNode API doc in
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b)
  //
  // It means each kernel_params[i] is itself a pointer to the corresponding
  // element at the *second* inline allocation at the end of the current
  // segment.
  iree_host_size_t total_size = kernel_params_length * 2;
  uint8_t* storage_base = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, total_size,
                              (void**)&storage_base));
  void** params_ptr = (void**)storage_base;
  CUdeviceptr* payload_ptr =
      (CUdeviceptr*)((uint8_t*)params_ptr + kernel_params_length);
  for (size_t i = 0; i < kernel_params_count; i++) {
    params_ptr[i] = &payload_ptr[i];
  }
  for (iree_host_size_t i = 0; i < bindings.count; i++) {
    const iree_hal_buffer_ref_t* binding = &bindings.values[i];
    CUdeviceptr device_ptr = 0;
    if (binding->buffer) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                           &binding->buffer));
      CUdeviceptr device_buffer = iree_hal_cuda_buffer_device_pointer(
          iree_hal_buffer_allocated_buffer(binding->buffer));
      iree_device_size_t offset = iree_hal_buffer_byte_offset(binding->buffer);
      device_ptr = device_buffer + offset + binding->offset;
    }
    payload_ptr[i] = device_ptr;
  }

  // As commented in the above, what each kernel parameter points to is a
  // CUdeviceptr, which as the size of a pointer on the target machine. we are
  // just storing a 32-bit value for the push constant here instead. So we must
  // process one element each type, for 64-bit machines.
  for (iree_host_size_t i = 0; i < kernel_params->constant_count; i++) {
    *((uint32_t*)params_ptr[kernel_params->binding_count + i]) =
        ((const uint32_t*)constants.data)[i];
  }

  CUDA_KERNEL_NODE_PARAMS params = {
      .func = kernel_params->function,
      .blockDimX = kernel_params->block_dims[0],
      .blockDimY = kernel_params->block_dims[1],
      .blockDimZ = kernel_params->block_dims[2],
      .gridDimX = workgroup_count[0],
      .gridDimY = workgroup_count[1],
      .gridDimZ = workgroup_count[2],
      .kernelParams = params_ptr,
      .sharedMemBytes = kernel_params->block_shared_memory_size,
  };

  if (command_buffer->graph_node_count >=
      IREE_HAL_CUDA_MAX_CONCURRENT_GRAPH_NODE_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "exceeded max concurrent node limit");
  }

  size_t dependency_count = command_buffer->cu_barrier_node ? 1 : 0;
  IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->symbols,
      cuGraphAddKernelNode(
          &command_buffer->cu_graph_nodes[command_buffer->graph_node_count++],
          command_buffer->cu_graph, &command_buffer->cu_barrier_node,
          dependency_count, &params),
      "cuGraphAddKernelNode");

  IREE_CUDA_GRAPH_COMMAND_BUFFER_TRACE_ZONE_END(
      command_buffer, IREE_HAL_STREAM_TRACING_VERBOSITY_FINE);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_graph_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_ref_t workgroups_ref, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "indirect dispatch not yet implemented");
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_cuda_graph_command_buffer_vtable = {
        .destroy = iree_hal_cuda_graph_command_buffer_destroy,
        .begin = iree_hal_cuda_graph_command_buffer_begin,
        .end = iree_hal_cuda_graph_command_buffer_end,
        .begin_debug_group =
            iree_hal_cuda_graph_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_cuda_graph_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_cuda_graph_command_buffer_execution_barrier,
        .signal_event = iree_hal_cuda_graph_command_buffer_signal_event,
        .reset_event = iree_hal_cuda_graph_command_buffer_reset_event,
        .wait_events = iree_hal_cuda_graph_command_buffer_wait_events,
        .advise_buffer = iree_hal_cuda_graph_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_cuda_graph_command_buffer_fill_buffer,
        .update_buffer = iree_hal_cuda_graph_command_buffer_update_buffer,
        .copy_buffer = iree_hal_cuda_graph_command_buffer_copy_buffer,
        .collective = iree_hal_cuda_graph_command_buffer_collective,
        .dispatch = iree_hal_cuda_graph_command_buffer_dispatch,
        .dispatch_indirect =
            iree_hal_cuda_graph_command_buffer_dispatch_indirect,
};
