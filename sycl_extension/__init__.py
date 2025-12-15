import ctypes
import platform
from pathlib import Path

import torch

# 1. 获取当前目录
# 注意：原来的代码用的是 parent.parent，假设 __init__.py 在包内，而 build 在包外。
# 如果安装后文件结构变了（比如 build 目录没了，.pyd 直接在包里），你可能需要调整路径。
# 这里假设开发模式下结构不变。
current_dir = Path(__file__).parent.parent
build_dir = current_dir / "build"

# 2. 根据系统决定文件后缀
if platform.system() == 'Windows':
    # Windows 下 Python 扩展后缀是 .pyd
    # glob 模式匹配：在 build 目录下递归查找 .pyd
    file_pattern = "**/*.pyd"
else:
    # Linux 下是 .so
    file_pattern = "**/*.so"

lib_files = list(build_dir.glob(file_pattern))

# 3. 增强健壮性：如果 build 目录找不到，尝试在当前包目录下找（对应 pip install 后的情况）
if not lib_files:
    current_package_dir = Path(__file__).parent
    lib_files = list(current_package_dir.glob(file_pattern))

assert len(lib_files) > 0, f"Could not find any {file_pattern} file in {build_dir} or {current_dir}"
# 如果找到多个，通常取第一个，或者你可以加更严格的筛选逻辑
lib_file = lib_files[0]

# 4. 加载库
# torch._ops.dl_open_guard() 主要是处理 RTLD_GLOBAL (Linux专用)，
# 在 Windows 上通常不需要，但保留着也没坏处。
with torch._ops.dl_open_guard():
    # ctypes.CDLL 在 Windows 上也可以加载 .pyd (本质是 DLL)
    loaded_lib = ctypes.CDLL(str(lib_file))

from . import ops

__all__ = [
    "loaded_lib",
    "ops",
]