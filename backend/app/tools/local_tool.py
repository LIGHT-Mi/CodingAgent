"""所有本地执行工具共享的最小类型。"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar


class LocalTool(ABC):
    """本地工具注册表可保存的执行器基类。"""

    name: ClassVar[str]
    arguments_type: ClassVar[type[object]]
