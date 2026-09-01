"""本地命令的结构化安全决策与拒绝结果构造。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from app.agent.contracts import ToolResult
from app.tools.command_contracts import RUN_COMMAND_TOOL_NAME, RunCommandArguments
from app.tools.command_results import CommandResultBuilder


class CommandSafetyVerdict(str, Enum):
    """命令安全策略可以作出的互斥决定。"""

    ALLOW = "ALLOW"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    REJECT = "REJECT"


class CommandRiskLevel(str, Enum):
    """用于日志、Observation 和用户批准界面的风险等级。"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True, slots=True)
class CommandSafetyDecision:
    """CommandSafetyPolicy 返回的供应商无关结构化决定。"""

    verdict: CommandSafetyVerdict
    reason: str
    rule_id: str
    risk_level: CommandRiskLevel
    approval_eligible: bool

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, CommandSafetyVerdict):
            raise TypeError("verdict must be a CommandSafetyVerdict")
        _require_non_blank(self.reason, "reason")
        _require_non_blank(self.rule_id, "rule_id")
        if not isinstance(self.risk_level, CommandRiskLevel):
            raise TypeError("risk_level must be a CommandRiskLevel")
        if not isinstance(self.approval_eligible, bool):
            raise TypeError("approval_eligible must be a boolean")
        expected_approval_eligible = (
            self.verdict is CommandSafetyVerdict.REQUIRE_APPROVAL
        )
        if self.approval_eligible is not expected_approval_eligible:
            raise ValueError(
                "approval_eligible must be true only for REQUIRE_APPROVAL"
            )


class CommandSafetyPolicy:
    """区分自动允许、需要一次性批准和永久拒绝的命令。"""

    _PYTHON_EXECUTABLES = frozenset({"python", "python3"})
    _SAFE_PYTHON_MODULES = {
        "pytest": ("SAFE_PYTHON_TEST_MODULE", CommandRiskLevel.LOW),
        "unittest": ("SAFE_PYTHON_TEST_MODULE", CommandRiskLevel.LOW),
        "compileall": ("SAFE_PYTHON_BUILD_MODULE", CommandRiskLevel.LOW),
    }
    _PERMANENTLY_BLOCKED_EXECUTABLES = frozenset(
        {
            "dd",
            "diskutil",
            "fdisk",
            "halt",
            "mkfs",
            "poweroff",
            "reboot",
            "shutdown",
            "su",
            "sudo",
        }
    )
    _APPROVAL_ELIGIBLE_DESTRUCTIVE_EXECUTABLES = frozenset(
        {"chmod", "chown", "kill", "pkill", "rm", "rmdir"}
    )
    _SHELL_EXECUTABLES = frozenset(
        {
            "bash",
            "cmd",
            "csh",
            "dash",
            "fish",
            "ksh",
            "powershell",
            "pwsh",
            "sh",
            "tcsh",
            "zsh",
        }
    )
    _EXACT_SHELL_OPERATORS = frozenset(
        {"&", "&&", ";", "<", "<<", ">", ">>", "|", "||"}
    )

    def evaluate(
        self,
        arguments: RunCommandArguments,
        resolved_cwd: Path,
    ) -> CommandSafetyDecision:
        """只判断命令是否可自动执行，不创建或运行子进程。"""

        if not isinstance(arguments, RunCommandArguments):
            raise TypeError("arguments must be RunCommandArguments")
        _require_resolved_cwd(resolved_cwd)

        command = arguments.command
        executable = command[0]
        executable_name = Path(executable).name.lower()

        if _contains_shell_syntax(command):
            return _reject(
                reason=(
                    "shell operators and command substitution are not supported; "
                    "pass one executable and its argv directly"
                ),
                rule_id="UNSUPPORTED_SHELL_SYNTAX",
                risk_level=CommandRiskLevel.HIGH,
            )

        if _is_permanently_blocked_executable(executable_name):
            return _reject(
                reason=(
                    "privilege, system power, or disk management commands are "
                    "permanently blocked"
                ),
                rule_id="PERMANENTLY_BLOCKED_EXECUTABLE",
                risk_level=CommandRiskLevel.CRITICAL,
            )

        if executable_name in self._SHELL_EXECUTABLES:
            return _require_approval(
                reason="shell interpreters require explicit user approval",
                rule_id="SHELL_INTERPRETER_REQUIRES_APPROVAL",
                risk_level=CommandRiskLevel.HIGH,
            )

        if (
            executable_name
            in self._APPROVAL_ELIGIBLE_DESTRUCTIVE_EXECUTABLES
        ):
            return _require_approval(
                reason=(
                    "destructive process or file command requires user approval"
                ),
                rule_id="DESTRUCTIVE_COMMAND_REQUIRES_APPROVAL",
                risk_level=CommandRiskLevel.HIGH,
            )

        if executable != executable_name:
            return _require_approval(
                reason=(
                    "executable paths are not in the automatic command allowlist"
                ),
                rule_id="EXECUTABLE_PATH_REQUIRES_APPROVAL",
                risk_level=CommandRiskLevel.HIGH,
            )

        if executable_name == "pytest":
            return _allow(
                reason="pytest is allowed for Workspace test execution",
                rule_id="SAFE_PYTEST_COMMAND",
                risk_level=CommandRiskLevel.LOW,
            )

        if executable_name in self._PYTHON_EXECUTABLES:
            return self._evaluate_python(command)

        return _require_approval(
            reason="executable is not in the automatic command allowlist",
            rule_id="UNKNOWN_EXECUTABLE_REQUIRES_APPROVAL",
            risk_level=CommandRiskLevel.MEDIUM,
        )

    def _evaluate_python(
        self,
        command: tuple[str, ...],
    ) -> CommandSafetyDecision:
        if len(command) == 2 and command[1] in {"-V", "--version"}:
            return _allow(
                reason="Python version queries are allowed",
                rule_id="SAFE_PYTHON_VERSION",
                risk_level=CommandRiskLevel.LOW,
            )

        if len(command) >= 3 and command[1] == "-m":
            module = command[2]
            safe_module = self._SAFE_PYTHON_MODULES.get(module)
            if safe_module is not None:
                rule_id, risk_level = safe_module
                return _allow(
                    reason=f"python -m {module} is allowed for project verification",
                    rule_id=rule_id,
                    risk_level=risk_level,
                )
            return _require_approval(
                reason=f"python module {module!r} is not automatically allowed",
                rule_id="PYTHON_MODULE_REQUIRES_APPROVAL",
                risk_level=CommandRiskLevel.HIGH,
            )

        if len(command) >= 2 and _is_workspace_relative_python_script(command[1]):
            return _allow(
                reason="a Workspace-relative Python script is allowed to run",
                rule_id="SAFE_WORKSPACE_PYTHON_SCRIPT",
                risk_level=CommandRiskLevel.MEDIUM,
            )

        return _require_approval(
            reason=(
                "interactive Python, inline code, stdin code, and unsupported "
                "Python options are not automatically allowed"
            ),
            rule_id="PYTHON_MODE_REQUIRES_APPROVAL",
            risk_level=CommandRiskLevel.HIGH,
        )


def build_rejected_command_result(
    tool_call_id: str,
    arguments: RunCommandArguments,
    resolved_cwd: Path,
    decision: CommandSafetyDecision,
) -> ToolResult:
    """把 REJECT 决策转换为模型可见的普通 REJECTED Observation。"""

    _require_non_blank(tool_call_id, "tool_call_id")
    if not isinstance(arguments, RunCommandArguments):
        raise TypeError("arguments must be RunCommandArguments")
    _require_resolved_cwd(resolved_cwd)
    if not isinstance(decision, CommandSafetyDecision):
        raise TypeError("decision must be a CommandSafetyDecision")
    if decision.verdict is not CommandSafetyVerdict.REJECT:
        raise ValueError("only a REJECT decision can build a rejected result")

    fingerprint = command_fingerprint(arguments.command, resolved_cwd)
    return CommandResultBuilder().build_rejected(
        tool_call_id,
        decision.reason,
        argv=arguments.command,
        cwd=resolved_cwd,
        details=(
            f"rule: {decision.rule_id}\n"
            f"risk: {decision.risk_level.value}\n"
            "approval_eligible: "
            f"{str(decision.approval_eligible).lower()}"
        ),
        metadata={
            "argv": list(arguments.command),
            "cwd": str(resolved_cwd),
            "rule_id": decision.rule_id,
            "risk_level": decision.risk_level.value,
            "approval_eligible": decision.approval_eligible,
            "command_fingerprint": fingerprint,
        },
    )


def command_fingerprint(
    command: tuple[str, ...],
    resolved_cwd: Path,
) -> str:
    """为完整 argv 和规范 cwd 生成跨进程稳定的 SHA-256 指纹。"""

    normalized_command = _require_command_tuple(command)
    _require_resolved_cwd(resolved_cwd)
    payload = json.dumps(
        {
            "argv": normalized_command,
            "cwd": str(resolved_cwd),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _contains_shell_syntax(command: tuple[str, ...]) -> bool:
    return any(
        argument in CommandSafetyPolicy._EXACT_SHELL_OPERATORS
        or "$(" in argument
        or "`" in argument
        or "\n" in argument
        or "\r" in argument
        for argument in command
    )


def _is_permanently_blocked_executable(executable_name: str) -> bool:
    return (
        executable_name in CommandSafetyPolicy._PERMANENTLY_BLOCKED_EXECUTABLES
        or executable_name.startswith("mkfs.")
    )


def _is_workspace_relative_python_script(argument: str) -> bool:
    path = Path(argument)
    return (
        path.suffix == ".py"
        and not path.is_absolute()
        and ".." not in path.parts
    )


def _allow(
    *,
    reason: str,
    rule_id: str,
    risk_level: CommandRiskLevel,
) -> CommandSafetyDecision:
    return CommandSafetyDecision(
        verdict=CommandSafetyVerdict.ALLOW,
        reason=reason,
        rule_id=rule_id,
        risk_level=risk_level,
        approval_eligible=False,
    )


def _reject(
    *,
    reason: str,
    rule_id: str,
    risk_level: CommandRiskLevel,
) -> CommandSafetyDecision:
    return CommandSafetyDecision(
        verdict=CommandSafetyVerdict.REJECT,
        reason=reason,
        rule_id=rule_id,
        risk_level=risk_level,
        approval_eligible=False,
    )


def _require_approval(
    *,
    reason: str,
    rule_id: str,
    risk_level: CommandRiskLevel,
) -> CommandSafetyDecision:
    return CommandSafetyDecision(
        verdict=CommandSafetyVerdict.REQUIRE_APPROVAL,
        reason=reason,
        rule_id=rule_id,
        risk_level=risk_level,
        approval_eligible=True,
    )


def _require_command_tuple(command: object) -> tuple[str, ...]:
    if not isinstance(command, tuple):
        raise TypeError("command must be a tuple of strings")
    if not command or any(not isinstance(argument, str) for argument in command):
        raise ValueError("command must contain at least one string argument")
    return command


def _require_resolved_cwd(resolved_cwd: object) -> Path:
    if not isinstance(resolved_cwd, Path):
        raise TypeError("resolved_cwd must be a pathlib.Path")
    if not resolved_cwd.is_absolute():
        raise ValueError("resolved_cwd must be an absolute path")
    return resolved_cwd


def _require_non_blank(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
