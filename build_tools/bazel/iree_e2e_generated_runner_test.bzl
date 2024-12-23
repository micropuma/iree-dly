# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Macros for defining tests that use the iree-e2e-${test_type}-test runner."""

load("//build_tools/bazel:iree_bytecode_module.bzl", "iree_bytecode_module")
load("//build_tools/bazel:native_binary.bzl", "native_test")

def iree_e2e_runner_test(
        name,
        test_type,
        tests_src,
        tests_vmfb,
        calls_src,
        calls_vmfb,
        target_backend,
        driver,
        test_runner,
        compiler_flags = [],
        runner_args = [],
        tags = [],
        timeout = None,
        **kwargs):
    """Creates a test using a specified test runner program.

    Args:
        name: Name of the target
        test_type: Name of the test (e.g., matmuls, conv2ds).
        tests_src: mlir source file with tests to be compiled.
        tests_vmfb: specifies the path to use for the generated IREE module.
        calls_src: mlir source file with calls to be compiled.
        calls_vmfb: specifies the path to use for the generated IREE module.
        target_backend: target backend to compile for.
        driver: driver to run the module with.
        compiler_flags: additional flags to pass to the compiler. Bytecode
            output format and backend flags are passed automatically.
        runner_args: additional args to pass to the test runner program. The
            driver and input file flags are passed automatically.
        tags: Additional labels to apply to the test. "driver=${DRIVER}" is
            added automatically.
        test_runner: test runner program to run.
        timeout: timeout for the generated tests.
        **kwargs: any additional attributes to pass to the underlying tests and
            test suite.
    """

    iree_bytecode_module(
        name = name + "_%s_module" % test_type,
        module_name = tests_vmfb,
        src = tests_src,
        flags = [
            "--iree-hal-target-backends=%s" % target_backend,
        ] + compiler_flags,
        visibility = ["//visibility:private"],
        testonly = True,
        **kwargs
    )

    iree_bytecode_module(
        name = name + "_calls_module",
        module_name = calls_vmfb,
        src = calls_src,
        flags = [
            "--iree-hal-target-backends=%s" % target_backend,
        ] + compiler_flags,
        visibility = ["//visibility:private"],
        testonly = True,
        **kwargs
    )

    native_test(
        name = name,
        args = [
            "--device=%s" % driver,
            "--module=$(location :%s)" % tests_vmfb,
            "--module=$(location :%s)" % calls_vmfb,
        ] + runner_args,
        data = [
            ":%s" % tests_vmfb,
            ":%s" % calls_vmfb,
        ],
        src = test_runner,
        tags = tags + ["driver=%s" % driver],
        timeout = timeout,
        **kwargs
    )

def iree_single_backend_e2e_runner_test(
        name,
        test_type,
        generator,
        test_runner,
        target_backend,
        driver,
        generator_args = [],
        compiler_flags = [],
        runner_args = [],
        tags = [],
        timeout = None,
        **kwargs):
    """Generates an iree_e2e_runner_test using a custom python generator script.

    The generator script produces .mlir sources which are compiled and passed to
    iree_e2e_runner_test.

    Args:
        name: Name of the target
        test_type: Name of the test (e.g., matmul, conv2d).
        generator: Target to run to generate the source MLIR files.
            It will be invoked with the following standard flags, in addition
            to generator_args:
            --output_${test_type}_mlir=(current binary dir)/name_${test_type}.mlir
            --output_calls_mlir=(current binary dir)/name_calls.mlir
        generator_args: additional args to pass to the generator program.
        target_backend: target backend to compile for.
        driver: driver to run the module with.
        compiler_flags: additional flags to pass to the compiler. Bytecode
            output format and backend flags are passed automatically.
        runner_args: additional args to pass to the test runner program. The
            driver and input file flags are passed automatically.
        tags: Additional labels to apply to the test. "driver=${DRIVER}" is
            added automatically.
        test_runner: test runner program to run.
        timeout: timeout for the generated tests.
        **kwargs: any additional attributes to pass to the underlying tests and
            test suite.
    """

    tests_src = "%s.mlir" % (name)
    tests_vmfb = "%s.vmfb" % (name)
    calls_src = "%s_calls.mlir" % (name)
    calls_vmfb = "%s_calls.vmfb" % (name)
    native.genrule(
        name = "%s_generate" % (name),
        outs = [tests_src, calls_src],
        cmd = " ".join([
            "$(location %s)" % (generator),
            " ".join([('"%s"' % arg) for arg in generator_args]),
            "--output_%s_mlir=$(location %s)" % (test_type, tests_src),
            "--output_calls_mlir=$(location %s)" % (calls_src),
        ] + [('"%s"' % arg) for arg in generator_args]),
        tools = [generator],
        message = "Generating code and calls for test %s..." % (name),
        output_to_bindir = 1,
        testonly = True,
        **kwargs
    )
    iree_e2e_runner_test(
        name = name,
        test_type = test_type,
        tests_src = tests_src,
        tests_vmfb = tests_vmfb,
        calls_src = calls_src,
        calls_vmfb = calls_vmfb,
        target_backend = target_backend,
        driver = driver,
        test_runner = test_runner,
        compiler_flags = compiler_flags,
        runner_args = runner_args,
        tags = tags,
        timeout = timeout,
        **kwargs
    )

def iree_generated_e2e_runner_test(
        name,
        test_type,
        generator,
        test_runner,
        target_backends_and_drivers,
        generator_args = [],
        compiler_flags = [],
        runner_args = [],
        tags = [],
        timeout = None,
        target_cpu_features_variants = [],
        **kwargs):
    """Generates a suite of iree_e2e_runner_test on multiple backends/drivers.

    Args:
        name: Name of the target
        test_type: Name of the test (e.g., matmul, conv2d).
        generator: Target to run to generate the source MLIR files.
            It will be invoked with the following standard flags, in addition
            to generator_args:
            --output_${test_type}_mlir=(current binary dir)/name_${test_type}.mlir
            --output_calls_mlir=(current binary dir)/name_calls.mlir
        generator_args: additional args to pass to the generator program.
        target_backends_and_drivers: backend/driver pairs to compile and run
            the module.
        compiler_flags: additional flags to pass to the compiler. Bytecode
            output format and backend flags are passed automatically.
        runner_args: additional args to pass to the test runner program. The
            driver and input file flags are passed automatically.
        tags: Additional labels to apply to the test. "driver=${DRIVER}" is
            added automatically.
        test_runner: test runner program to run.
        timeout: timeout for the generated tests.
        target_cpu_features_variants: ignored, assumed to be ["generic"] in this
            Bazel implementation. See the CMake implementation for what this does
            in general.
        **kwargs: any additional attributes to pass to the underlying tests and test suite.
    """

    # Like CMake, default to "generic". Unlike CMake, do not honor other values.
    generic_flags = compiler_flags + ["--iree-llvmcpu-target-cpu=generic"]

    tests = []
    for backend, driver in target_backends_and_drivers:
        # CUDA/ROCm backend/driver not supported by Bazel build.
        if backend == "cuda" or driver == "cuda" or backend == "rocm" or driver == "hip":
            continue
        suite_entry_name = "_".join([name, backend, driver])
        iree_single_backend_e2e_runner_test(
            name = suite_entry_name,
            test_type = test_type,
            generator = generator,
            test_runner = test_runner,
            driver = driver,
            target_backend = backend,
            generator_args = generator_args,
            compiler_flags = generic_flags,
            runner_args = runner_args,
            tags = tags,
            timeout = timeout,
            **kwargs
        )
        tests.append(suite_entry_name)
    native.test_suite(
        name = name,
        tests = tests,
        tags = tags,
        **kwargs
    )
