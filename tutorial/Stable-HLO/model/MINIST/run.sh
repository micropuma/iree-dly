# Compile the MNIST program.
iree-compile \
    ./mnist.mlir \
    --iree-input-type=stablehlo \
    --iree-hal-target-backends=llvm-cpu \
    -o /tmp/mnist_cpu.vmfb

# Convert the test image to the 1x28x28x1xf32 buffer format the program expects.
cat mnist_test.png | python3 convert_image.py > /tmp/mnist_test.bin

# Run the program, passing the path to the binary file as a function input.
iree-run-module \
  --module=/tmp/mnist_cpu.vmfb \
  --function=predict \
  --input=1x28x28x1xf32=@/tmp/mnist_test.bin

# Observe the results - a list of prediction confidence scores for each digit.
