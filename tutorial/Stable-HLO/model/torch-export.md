# Torch-export
这个repo讲解如何从`pytorch`导入模型到`StableHLO dialect`并使用IREE做编译和deploy。
## Environment set
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow-cpu
pip install torch-xla
```
上述方法failed了。`torch-mlir`项目提供更加方便的模型导入，同时也支持了足够多的dialect。
参考[torch-mlir官网文档](https://github.com/llvm/torch-mlir/blob/main/docs/development.md)，或是直接理解`torch-mlir.ipynb`脚本即可。
## From pytorch to StableHLO
参考[官方教程即可](https://openxla.org/stablehlo/tutorials/pytorch-export)，详细接入script见`torch-export.ipynb`。
