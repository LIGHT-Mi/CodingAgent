# 演示用 Agent 指令合集

## 演示 Workspace

在前端 Workspace 输入框中填写：

```text
/Users/myx/Documents/GitHub/CodingAgent/examples/coding-agent-demo-workspace
```


## 指令一：完成失败测试修复闭环

新建会话后复制：

```text
请检查这个项目并运行现有测试，根据失败信息定位并修复问题。修改已有文件前先读取真实源码，优先使用精确编辑；修改后回读文件，并重新运行同一测试命令。最后总结失败原因、修改内容和实际验证结果。
```

这条指令用于展示：

```text
查看项目
→ 读取源码
→ 第一次测试失败
→ 精确修改
→ 回读文件
→ 第二次测试通过
→ 最终回答
```

## 指令二：展示同一 Session 的多轮记忆

必须等待指令一对应的 Task 完成，然后在同一个 Session 中复制：

```text
请只依据本会话上一轮已经明确记录的结论，解释为什么上一轮修改是正确的，并列出上一轮已经实际得到的验证证据。不要补充上一轮没有明确出现的文件、函数或测试事实，不要调用工具，也不要继续修改文件。
```

这条指令用于展示：

- 同一个 Session 创建了新的 Task；
- 新 Task 能使用上一轮 Prompt 和 Final Answer；
- 历史 Task 不会复活或重新回到 PENDING；
- Task 0 和 Task 1 拥有各自独立的执行记录。

## 指令三：展示危险命令一次性批准

等待指令二对应的 Task 完成，然后在同一个 Session 中复制：

```text
请调用 run_command，使用完整 argv ["python", "-c", "print('approval-demo')"]，工作目录使用 "."。如果系统要求用户批准，必须停止在执行之前并等待决定。获得工具结果后，只根据实际 exit code、stdout 和 stderr 说明执行结果，不得声称批准前命令已经执行。
```

这条指令会触发 `python -c` 的用户批准流程。批准面板出现后，可以点击：

```text
允许本次执行
```

批准并执行成功后，预期 stdout 为：

```text
approval-demo
```

如果要展示用户拒绝分支，可以对同一条指令点击：

```text
拒绝执行
```

拒绝后命令不会创建子进程，Agent 应把拒绝结果作为 Observation 继续处理。

## 指令四：可选展示永久拒绝规则

这条指令用于面试补充演示，不要在真实机器上自行从终端执行其中的命令。复制给 Agent：

```text
请调用 run_command 尝试执行完整 argv ["sudo", "-V"]，工作目录使用 "."。不要尝试替换为其他命令，也不要修改文件。请根据工具实际返回的安全策略结果，说明该命令是否创建了子进程、是否允许用户批准，以及命中的拒绝原因。
```

预期结果：

- 安全策略直接返回 `REJECTED` Observation；
- 不创建子进程；
- 不出现批准按钮；
- 用户批准不能覆盖永久拒绝规则。

## 指令五：可选展示 Workspace 路径越界保护

这条指令用于面试补充演示。复制给 Agent：

```text
请调用 read_file 尝试读取相对路径 "../read-only-outside-sentinel.txt"。不要改用其他路径，也不要修改任何文件。请根据工具实际返回结果，说明该路径为什么不能被当前 Task Workspace 读取。
```

预期结果：

- 路径解析后的目标位于当前 Task Workspace 外；
- 工具返回 `REJECTED` Observation；
- 目标文件内容不会被读取；
- Task 可以继续处理该 Observation，而不是因为普通路径拒绝直接崩溃。
