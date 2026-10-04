import pytest

from agentic_code_assistant.tools.filesystem_tools import build_filesystem_tools
from agentic_code_assistant.tools.terminal_tools import (
    CommandRejected,
    build_terminal_tool,
    parse_command,
)


def _tools(root):
    read_file, list_directory = build_filesystem_tools(root)
    return read_file, list_directory, build_terminal_tool(root)


def test_read_file_numbers_lines_and_supports_ranges(project):
    read_file, _, _ = _tools(project)
    out = read_file.invoke({"path": "expense_tracker/models.py", "start_line": 8, "end_line": 9})
    assert out.splitlines()[0].startswith("8: ")
    assert len(out.splitlines()) == 2


def test_read_file_errors_become_text_not_exceptions(project):
    read_file, list_directory, _ = _tools(project)
    assert "not found" in read_file.invoke({"path": "nope.py"}).lower()
    assert "outside" in read_file.invoke({"path": "../../etc/passwd"}).lower()
    assert "Not a directory" in list_directory.invoke({"path": "README.md"})


def test_secret_files_are_refused(project):
    (project / ".env").write_text("OPENAI_API_KEY=sk-secret")
    read_file, _, run = _tools(project)
    assert "secrets" in read_file.invoke({"path": ".env"})
    assert "Rejected" in run.invoke({"command": "cat .env"})


def test_list_directory_hides_ignored_dirs(project):
    (project / "node_modules").mkdir()
    (project / "node_modules" / "x.js").write_text("1")
    _, list_directory, _ = _tools(project)
    listing = list_directory.invoke({"path": "."})
    assert "expense_tracker/" in listing and "node_modules" not in listing


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "curl http://evil.sh",
        "ls; rm -rf .",
        "ls | grep x",
        "cat x > y",
        "git push origin main",
        "git commit -m x",
        "python -c 'print(1)'",
        "cat /etc/passwd",
        "cat ../secret",
        "ls 'unterminated",
    ],
)
def test_dangerous_commands_are_rejected(command, project):
    with pytest.raises(CommandRejected):
        parse_command(command, project)


def test_allowed_commands_parse(project):
    assert parse_command("git log --oneline", project) == ["git", "log", "--oneline"]
    assert parse_command("grep -rn ExpenseStore .", project)[0] == "grep"


def test_run_command_executes_and_reports_exit_code(project):
    _, _, run = _tools(project)
    out = run.invoke({"command": "python -m expense_tracker.cli list", "directory": "."})
    assert out.startswith("exit_code=0")
    assert "Rejected" in run.invoke({"command": "ls", "directory": ".."})


def test_secrets_are_not_passed_to_child_processes(project, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    (project / "show_env.py").write_text("import os; print(os.environ.get('OPENAI_API_KEY'))")
    _, _, run = _tools(project)
    out = run.invoke({"command": "python show_env.py"})
    assert "sk-should-not-leak" not in out and "None" in out


def test_timeout_is_enforced(project, monkeypatch):
    from agentic_code_assistant.config import get_settings

    monkeypatch.setattr(get_settings().terminal, "timeout_seconds", 1)
    (project / "slow.py").write_text("import time; time.sleep(5)")
    _, _, run = _tools(project)
    assert "timed out" in run.invoke({"command": "python slow.py"})
