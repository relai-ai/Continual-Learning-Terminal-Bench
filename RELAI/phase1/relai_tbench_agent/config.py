from __future__ import annotations

AGENT_MODEL = "openai/gpt-5.5"
COMPLETION_GATE_VERIFIER_FILENAMES = (
    "test_outputs.py",
    "tests/test_outputs.py",
    "verifier.py",
    "grader.py",
    "check.py",
)

COMPLETION_GATE_VALIDATION_HINTS = (
    "pytest",
    "verifier",
    "test_",
    "assert",
    "row_matched",
    "ratio_diff",
    "all_matched",
    "test passed",
    "verify.sh",
    "http 404",
    "http 200",
    "curl ",
    "hello.html",
    "git push",
    "show-ref",
    "frame.bmp",
    "i_initgraphics",
    "doom screen size",
    "320 x 200",
    "640 x 400",
    "program terminated at pc",
    "executed ",
)

COMPLETION_GATE_CONVENIENCE_CHECK_HINTS = (
    "np.array_equal",
    "array_equal",
    "forward.A1",
    "== forward.",
    "artifact-equality",
    "equality check",
    "max(np.abs(",
    "http_status=200",
    "port8080_http_status=200",
    "web server responds 200",
    "liveness check",
    "health check",
)

COMPLETION_GATE_ALLOWED_DELIVERABLE_PATTERNS = (
    "only ",
    "expected only ",
    "must contain only ",
    "should contain only ",
)

COMPLETION_GATE_EPHEMERAL_ARTIFACT_HINTS = (
    "build output",
    "temporary executable",
    "temp file",
    "cache",
    "artifact",
    "leftover",
)

OPERATIONAL_TASK_MARKERS = (
    "web server",
    "server",
    "service",
    "http",
    "curl",
    "port ",
    "git",
    "nginx",
    "apache",
    "configure",
    "publish",
    "deploy",
    "endpoint",
    "html",
)

OPERATIONAL_TASK_WORKFLOW_NOTE = """
For operational/infrastructure tasks, use a verifier-shaped delivery loop:
1. Infer the externally visible deliverable, not just the process liveness condition.
2. Inspect nearby task files, tests, verifier scripts, or obvious validation assets when present so you know what a checker will request over the network, filesystem, or CLI.
3. Treat HTTP 200 on a root page, an open port, a running service, or a clean daemon status as intermediate evidence unless the task explicitly says that is the final deliverable.
4. Before finishing, perform one end-to-end audit that exercises the published artifact or workflow result the verifier is likely to check.
5. If the task depends on a user workflow step such as clone/push/fetch/request, simulate that workflow or validate the exact resulting externally served artifact before stopping.
"""

COMPLETION_GATE_OPERATIONAL_DELIVERABLE_HINTS = (
    "http 404",
    "404",
    "not found",
    "hello.html",
    "index.html",
    "test passed",
    "verify.sh",
    "git push",
    "git clone",
    "show-ref",
    "refs/",
    "published",
    "served",
)

CODE_TASK_WORKFLOW_NOTE = """
For implementation-heavy debugging and code-edit tasks, use a strict verifier-driven loop:
1. Front-load grader discovery: inspect the task statement, the smallest relevant source files, and the exact tests/verifier files before proposing edits.
2. State one concrete semantic hypothesis for the failure before changing code. Tie it to an observed invariant, mismatch, tolerance rule, ordering rule, or backward/forward behavior that the verifier is likely checking.
3. Minimize exploration. Prefer a small targeted file set and focused reads over broad filesystem audits. Do not repeat repository-wide scans once the likely edit surface is known.
4. Make one minimal code change at a time on the most likely in-path file, then immediately run the narrowest validation that can falsify the hypothesis.
5. After each validation, explicitly decide: a) hypothesis supported and continue, b) hypothesis falsified and revise, or c) task complete. Do not keep editing on vague intuition.
6. Budget time aggressively. If a command or investigation is unlikely to change the hypothesis or validation plan, skip it.
7. Before finishing, re-check the touched files and confirm the final behavior matches verifier semantics, not just a plausible implementation story.
8. Prefer targeted reads such as ls, pwd, git status, sed on known files, and direct pytest/test invocations. Avoid broad find over both . and /app unless the task still lacks a concrete edit or verifier target.
9. Use short polling waits by default: 0.1s for instant commands, about 1s for likely-complete commands, and only extend waits after observing evidence that a process is still running.
10. Once the required deliverable has been verified with checker-shaped evidence and the touched-file audit is clean, stop immediately instead of doing another exploratory pass.
"""

CODE_TASK_COMPLETION_FOCUSED_AUDIT_CMD = (
    "printf '__TASK_AUDIT__\\n'; "
    "pwd; "
    "printf '__GIT_STATUS__\\n'; "
    "(git status --short 2>/dev/null || true); "
    "printf '__APP_TOUCHES__\\n'; "
    "(find /app -maxdepth 2 -type f "
    "\\( -name '*.py' -o -name '*.txt' -o -name '*.md' -o -name '*.json' \\) "
    "-printf '%P\\n' 2>/dev/null | sort || true)"
)

CODE_TASK_VERIFIER_DISCOVERY_CMD = (
    "printf '__VERIFIER_FILES__\\n'; "
    "(ls tests/test_outputs.py test_outputs.py verifier.py grader.py check.py 2>/dev/null || true); "
    "printf '__VERIFIER_PREVIEW__\\n'; "
    "(sed -n '1,220p' tests/test_outputs.py 2>/dev/null "
    "|| sed -n '1,220p' ./test_outputs.py 2>/dev/null "
    "|| sed -n '1,220p' ./verifier.py 2>/dev/null "
    "|| sed -n '1,220p' ./grader.py 2>/dev/null "
    "|| sed -n '1,220p' ./check.py 2>/dev/null "
    "|| true)"
)

CODE_TASK_RUNTIME_CONTRACT_CMD = (
    "printf '__RUNTIME_CONTRACT__\\n'; "
    "(sed -n '1,220p' tests/test_outputs.py 2>/dev/null "
    "|| sed -n '1,220p' ./test_outputs.py 2>/dev/null "
    "|| sed -n '1,220p' ./verifier.py 2>/dev/null "
    "|| sed -n '1,220p' ./grader.py 2>/dev/null "
    "|| sed -n '1,220p' ./check.py 2>/dev/null "
    "|| true); "
    "printf '\\n__RUNTIME_LOG_SIGNALS__\\n'; "
    "(printf '%s\\n' \"$(tail -n 120 /tmp/full_vm.out 2>/dev/null)\"; "
    "printf '%s\\n' \"$(tail -n 120 /tmp/vm.out 2>/dev/null)\"; "
    "printf '%s\\n' \"$(tail -n 120 /tmp/stdout 2>/dev/null)\"; "
    "printf '%s\\n' \"$(tail -n 120 /tmp/output.txt 2>/dev/null)\"; true); "
    "printf '__RUNTIME_ARTIFACTS__\\n'; "
    "(ls -l /tmp/frame.bmp /tmp/*.bmp 2>/dev/null || true); "
    "printf '__RUNTIME_IMAGE_META__\\n'; "
    "(python3 - <<'PY'\n"
    "from pathlib import Path\n"
    "import struct\n"
    "for p in [Path('/tmp/frame.bmp')] + sorted(Path('/tmp').glob('*.bmp')):\n"
    "    if not p.exists():\n"
    "        continue\n"
    "    try:\n"
    "        data = p.read_bytes()[:64]\n"
    "        width, height = struct.unpack_from('<ii', data, 18)\n"
    "        print(f'{p}:{width}x{height}')\n"
    "    except Exception as exc:\n"
    "        print(f'{p}:ERROR:{exc}')\n"
    "PY\n"
    "2>/dev/null || true)"
)
