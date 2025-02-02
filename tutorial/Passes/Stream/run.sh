# 测试IREE::Stream::createConvertToStreamPass
# 这个pass是最为关键的一个pass
iree-opt --iree-stream-conversion \
    toStream.mlir -o stream.mlir

# 测试IREE::Stream::createRefineUsagePass
iree-opt --iree-stream-refine-usage \
    refine-usage.mlir -o refine-usage-out.mlir

# 测试IREE::Stream::createScheduleExecutionPass
# iree-opt --iree-stream-schedule-execution \
#     schedule-exe.mlir -o schedule2.mlir