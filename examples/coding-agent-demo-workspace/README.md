# Calculator Core

这是一个只使用 Python 标准库的微型计算模块，用于验证编程任务中的问题定位、精确修改和测试闭环。

## 项目结构

- `calculator/`：计算模块源码。
- `tests/`：单元测试。

## 运行测试

在当前目录执行：

```text
python3 -m unittest discover -s tests -p test_*.py
```

项目没有第三方依赖。请以测试输出和实际源码为依据定位问题，修改后重新运行同一测试命令确认结果。
