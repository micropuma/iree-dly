#!/bin/bash

# 获取当前的 LD_LIBRARY_PATH
LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH)

# 删除所有包含 'vitis' 或 'vivado' 的路径
LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -vE '(/mnt/home/tools/xilinx/Vitis|/mnt/home/tools/xilinx/Vivado)' | tr '\n' ':')

# 去除结尾的多余冒号
LD_LIBRARY_PATH=${LD_LIBRARY_PATH%:}

# 设置新的 LD_LIBRARY_PATH
export LD_LIBRARY_PATH

# 打印新的 LD_LIBRARY_PATH
echo "Updated LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
