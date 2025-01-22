#  Model Exploerer
install:
```shell
pip install model-explorer
```

usage:
```shell
model-explorer ~/...
```
可选：
```shell
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```
目前，这个工具还是只能支持`Tensorflow`，`JAX`和`Pytorch`的模型，对于`mlir`方言的支持也只局限于mlir中对应的dialect，比如`LSTM.mlir`就无法解析，因为`cf` dialect不支持。
## References
1. [Model Exploerer documents](https://github.com/google-ai-edge/model-explorer)
2. [User guide](https://github.com/google-ai-edge/model-explorer/wiki/2.-User-Guide#select-models)
3. [Command line guide](https://github.com/google-ai-edge/model-explorer/wiki/3.-Command-Line-Guide)
