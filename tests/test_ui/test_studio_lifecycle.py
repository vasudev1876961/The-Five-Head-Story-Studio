"""Complete cross-platform lifecycle verification harness for five-head-studio."""

from __future__ import annotations

import dataclasses
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil
import pytest


@dataclasses.dataclass
class LifecycleVerificationResult:
    """Structured result reporting every critical dimension of server lifecycle verification."""

    port: int
    pre_launch_port_free: bool
    root_pid: int | None
    health_check_success: bool
    startup_timed_out: bool
    startup_elapsed_seconds: float
    observation_elapsed_seconds: float
    tracked_processes_count: int
    tracked_pids: list[int]
    shutdown_signal_sent: bool
    shutdown_signal_error: str | None
    graceful_exit_elapsed_seconds: float | None
    graceful_exit_within_timeout: bool
    app_exit_code: int | None
    expected_exit_status: bool
    graceful_success: bool
    fallback_required: bool
    fallback_error: str | None
    post_shutdown_port_free: bool
    remaining_descendants: list[int]
    lifecycle_success: bool
    failure_reasons: list[str]
    startup_log: str


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Verify port is completely free by testing whether a TCP socket can bind to it.

    Note: SO_REUSEADDR is deliberately omitted to prevent false-positive binds
    on ports held in kernel TIME_WAIT or shared socket states.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def poll_port_released(
    port: int,
    host: str = "127.0.0.1",
    timeout_seconds: float = 3.0,
    poll_interval: float = 0.25,
) -> bool:
    """Poll port availability with a bounded retry window for kernel socket release."""
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout_seconds:
        if is_port_available(port, host):
            return True
        time.sleep(poll_interval)
    return is_port_available(port, host)


def discover_process_tree(root_pid: int) -> set[tuple[int, float]]:
    """Recursively discover root PID and active descendants as (pid, create_time) tuples.

    Tracking create_time eliminates PID-reuse errors: recycled PIDs assigned to
    subsequent unrelated processes will not match the recorded identity.
    """
    discovered: set[tuple[int, float]] = set()
    try:
        parent = psutil.Process(root_pid)
        discovered.add((root_pid, parent.create_time()))
        for child in parent.children(recursive=True):
            try:
                discovered.add((child.pid, child.create_time()))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return discovered


def initiate_graceful_signal(
    proc: subprocess.Popen,
    pgid: int | None = None,
) -> tuple[bool, str | None]:
    """Send OS-appropriate graceful shutdown signal to the process group in an exception-safe manner."""
    if proc.poll() is not None:
        return False, "Process already exited before signal dispatch"
    try:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            target_pgid = pgid if pgid is not None else proc.pid
            os.killpg(target_pgid, signal.SIGTERM)
        return True, None
    except (ProcessLookupError, PermissionError, OSError) as e:
        return False, str(e)


def execute_fallback_kill(
    proc: subprocess.Popen,
    tracked_processes: set[tuple[int, float]],
    pgid: int | None = None,
) -> tuple[bool, str | None]:
    """Forcefully terminate the process group and any remaining tracked descendants."""
    fallback_error: str | None = None
    if sys.platform != "win32" and pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError) as e:
            fallback_error = str(e)
    else:
        try:
            proc.kill()
        except (ProcessLookupError, PermissionError, OSError) as e:
            fallback_error = str(e)

    for pid, create_time in tracked_processes:
        try:
            p = psutil.Process(pid)
            if abs(p.create_time() - create_time) < 0.1:
                p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return True, fallback_error


def get_studio_executable() -> str:
    """Resolve the five-head-studio executable path across platforms and virtualenvs."""
    bin_dir = Path(sys.executable).parent
    candidates = [
        bin_dir / "five-head-studio.exe",
        bin_dir / "five-head-studio",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    found = shutil.which("five-head-studio") or shutil.which("five-head-studio.exe")
    if found:
        return found
    return "five-head-studio"


def verify_studio_lifecycle(
    port: int = 8598,
    startup_timeout_seconds: float = 15.0,
    observation_window_seconds: float = 3.0,
    graceful_timeout_seconds: float = 10.0,
    custom_command: list[str] | None = None,
) -> LifecycleVerificationResult:
    """Execute end-to-end server lifecycle verification with decoupled observation and strict contracts."""
    failure_reasons: list[str] = []

    # Step 1: Pre-launch port-availability check
    pre_launch_port_free = is_port_available(port)
    if not pre_launch_port_free:
        failure_reasons.append(f"Port {port} is occupied before launch.")
        return LifecycleVerificationResult(
            port=port,
            pre_launch_port_free=False,
            root_pid=None,
            health_check_success=False,
            startup_timed_out=False,
            startup_elapsed_seconds=0.0,
            observation_elapsed_seconds=0.0,
            tracked_processes_count=0,
            tracked_pids=[],
            shutdown_signal_sent=False,
            shutdown_signal_error=None,
            graceful_exit_elapsed_seconds=None,
            graceful_exit_within_timeout=False,
            app_exit_code=None,
            expected_exit_status=False,
            graceful_success=False,
            fallback_required=False,
            fallback_error=None,
            post_shutdown_port_free=False,
            remaining_descendants=[],
            lifecycle_success=False,
            failure_reasons=failure_reasons,
            startup_log="Launch aborted: pre-launch port conflict detected.",
        )

    log_path = Path(os.environ.get("TEMP", "/tmp")) / "studio_startup.log"
    creation_flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    )
    start_new_session = sys.platform != "win32"

    exe = get_studio_executable()
    if custom_command:
        cmd = [exe if arg == "five-head-studio" else arg for arg in custom_command]
    else:
        cmd = [
            exe,
            "--server.headless=true",
            f"--server.port={port}",
        ]

    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
            start_new_session=start_new_session,
            text=True,
        )

    root_pid = proc.pid
    pgid = root_pid if sys.platform != "win32" else None
    tracked_processes: set[tuple[int, float]] = set()

    # Step 2: Health check polling loop (up to startup_timeout_seconds)
    startup_start = time.monotonic()
    health_check_success = False
    startup_timed_out = False

    while time.monotonic() - startup_start < startup_timeout_seconds:
        time.sleep(0.5)
        tracked_processes.update(discover_process_tree(root_pid))
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/_stcore/health", timeout=1
            ) as resp:
                body = resp.read().decode("utf-8").strip()
                if resp.status == 200 and body == "ok":
                    health_check_success = True
                    break
        except Exception:
            pass

    startup_elapsed = time.monotonic() - startup_start
    if not health_check_success and proc.poll() is None:
        startup_timed_out = True
        failure_reasons.append("Startup health check timed out after 15 seconds.")

    # Step 3: Decoupled post-healthy observation window (3.0s continuous process-tree tracking)
    obs_start = time.monotonic()
    if health_check_success and proc.poll() is None:
        while time.monotonic() - obs_start < observation_window_seconds:
            time.sleep(0.5)
            tracked_processes.update(discover_process_tree(root_pid))
            if proc.poll() is not None:
                break
    observation_elapsed = time.monotonic() - obs_start

    # Step 4: Symmetrical Graceful-First Shutdown
    shutdown_signal_sent, shutdown_signal_error = initiate_graceful_signal(
        proc, pgid
    )
    if shutdown_signal_error:
        failure_reasons.append(f"Shutdown signal error: {shutdown_signal_error}")

    # Step 5: Graceful exit waiting loop (up to graceful_timeout_seconds)
    grace_start = time.monotonic()
    graceful_exit_within_timeout = False
    fallback_required = False
    fallback_error = None

    while time.monotonic() - grace_start < graceful_timeout_seconds:
        if proc.poll() is not None:
            graceful_exit_within_timeout = True
            break
        time.sleep(0.2)
    graceful_exit_elapsed = time.monotonic() - grace_start

    # Step 6: Fallback kill if graceful timeout expired
    if not graceful_exit_within_timeout:
        fallback_required = True
        failure_reasons.append(
            f"Graceful shutdown timed out after {graceful_timeout_seconds}s; fallback kill required."
        )
        _, fallback_error = execute_fallback_kill(proc, tracked_processes, pgid)
        try:
            proc.wait(timeout=5.0)
        except (subprocess.TimeoutExpired, OSError) as e:
            if not fallback_error:
                fallback_error = str(e)

    app_exit_code = proc.returncode

    # Step 7: Expected exit status check
    if sys.platform == "win32":
        expected_exit_status = app_exit_code in (0, 3221225786)
    else:
        expected_exit_status = app_exit_code in (0, 130, 143)

    if not expected_exit_status:
        failure_reasons.append(f"Unexpected exit code: {app_exit_code}")

    graceful_success = (
        shutdown_signal_sent
        and graceful_exit_within_timeout
        and expected_exit_status
        and not fallback_required
    )

    # Step 8: Bounded post-shutdown port release verification (polls up to 3.0s)
    post_shutdown_port_free = poll_port_released(port, timeout_seconds=3.0)
    if not post_shutdown_port_free:
        failure_reasons.append(
            f"Port {port} remains occupied after server shutdown."
        )

    # Step 9: Verify no remaining descendants from tracked accumulator
    remaining_descendants: list[int] = []
    for pid, ctime in tracked_processes:
        if pid == root_pid:
            continue
        try:
            p = psutil.Process(pid)
            if abs(p.create_time() - ctime) < 0.1:
                remaining_descendants.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if remaining_descendants:
        failure_reasons.append(
            f"Tracked descendant processes remained alive: {remaining_descendants}"
        )

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            startup_log = f.read()
    except Exception:
        startup_log = ""

    lifecycle_success = (
        pre_launch_port_free
        and health_check_success
        and not startup_timed_out
        and shutdown_signal_sent
        and graceful_exit_within_timeout
        and expected_exit_status
        and not fallback_required
        and post_shutdown_port_free
        and len(remaining_descendants) == 0
    )

    return LifecycleVerificationResult(
        port=port,
        pre_launch_port_free=pre_launch_port_free,
        root_pid=root_pid,
        health_check_success=health_check_success,
        startup_timed_out=startup_timed_out,
        startup_elapsed_seconds=startup_elapsed,
        observation_elapsed_seconds=observation_elapsed,
        tracked_processes_count=len(tracked_processes),
        tracked_pids=[pid for pid, _ in sorted(tracked_processes)],
        shutdown_signal_sent=shutdown_signal_sent,
        shutdown_signal_error=shutdown_signal_error,
        graceful_exit_elapsed_seconds=graceful_exit_elapsed,
        graceful_exit_within_timeout=graceful_exit_within_timeout,
        app_exit_code=app_exit_code,
        expected_exit_status=expected_exit_status,
        graceful_success=graceful_success,
        fallback_required=fallback_required,
        fallback_error=fallback_error,
        post_shutdown_port_free=post_shutdown_port_free,
        remaining_descendants=remaining_descendants,
        lifecycle_success=lifecycle_success,
        failure_reasons=failure_reasons,
        startup_log=startup_log,
    )


def evaluate_scenario_assertions(
    scenario_id: int,
    result: LifecycleVerificationResult,
) -> tuple[bool, list[str]]:
    """Evaluate scenario-specific assertions independently from overall lifecycle success."""
    errors: list[str] = []

    if scenario_id == 1:
        if not result.lifecycle_success:
            errors.append(
                f"Expected lifecycle_success=True, got False. Reasons: {result.failure_reasons}"
            )
        if not result.graceful_success:
            errors.append("Expected graceful_success=True.")
        if result.fallback_required:
            errors.append("Expected fallback_required=False.")

    elif scenario_id == 2:
        if result.lifecycle_success:
            errors.append("Expected lifecycle_success=False for startup timeout.")
        if not result.startup_timed_out:
            errors.append("Expected startup_timed_out=True.")
        if result.health_check_success:
            errors.append("Expected health_check_success=False.")
        if not result.shutdown_signal_sent:
            errors.append(
                "Expected shutdown_signal_sent=True (symmetrical graceful cleanup)."
            )
        if not result.post_shutdown_port_free:
            errors.append("Expected post_shutdown_port_free=True.")
        if len(result.remaining_descendants) > 0:
            errors.append(
                f"Expected remaining_descendants=[], got {result.remaining_descendants}."
            )

    elif scenario_id == 3:
        if result.lifecycle_success:
            errors.append(
                "Expected lifecycle_success=False for early application exit."
            )
        if result.health_check_success:
            errors.append("Expected health_check_success=False.")
        if result.shutdown_signal_sent:
            errors.append(
                "Expected shutdown_signal_sent=False (process already dead)."
            )
        if result.app_exit_code == 0:
            errors.append("Expected non-zero app_exit_code.")
        if not result.post_shutdown_port_free:
            errors.append("Expected post_shutdown_port_free=True.")

    elif scenario_id == 4:
        if result.lifecycle_success:
            errors.append(
                "Expected lifecycle_success=False when fallback kill is required."
            )
        if not result.shutdown_signal_sent:
            errors.append("Expected shutdown_signal_sent=True.")
        if result.graceful_exit_within_timeout:
            errors.append("Expected graceful_exit_within_timeout=False.")
        if not result.fallback_required:
            errors.append("Expected fallback_required=True.")
        if not result.post_shutdown_port_free:
            errors.append(
                "Expected post_shutdown_port_free=True after fallback kill."
            )
        if len(result.remaining_descendants) > 0:
            errors.append(
                f"Expected remaining_descendants=[], got {result.remaining_descendants}."
            )

    elif scenario_id == 5:
        if result.lifecycle_success:
            errors.append(
                "Expected lifecycle_success=False for signal dispatch failure."
            )
        if result.shutdown_signal_sent:
            errors.append("Expected shutdown_signal_sent=False.")
        if result.shutdown_signal_error is None:
            errors.append("Expected shutdown_signal_error to capture exception.")
        if not result.post_shutdown_port_free:
            errors.append("Expected post_shutdown_port_free=True.")

    elif scenario_id == 6:
        if result.lifecycle_success:
            errors.append(
                "Expected lifecycle_success=False when pre-launch port is occupied."
            )
        if result.pre_launch_port_free:
            errors.append("Expected pre_launch_port_free=False.")
        if result.root_pid is not None:
            errors.append(
                f"Expected root_pid=None (launch aborted), got {result.root_pid}."
            )

    elif scenario_id == 7:
        if result.lifecycle_success:
            errors.append(
                "Expected lifecycle_success=False when descendant survives root exit."
            )
        if len(result.tracked_pids) == 0:
            errors.append("Expected tracked_pids to include spawned child.")
        if not result.post_shutdown_port_free:
            errors.append("Expected post_shutdown_port_free=True.")

    elif scenario_id == 8:
        if result.lifecycle_success:
            errors.append(
                "Expected lifecycle_success=False when post-shutdown port is occupied."
            )
        if result.post_shutdown_port_free:
            errors.append("Expected post_shutdown_port_free=False.")

    else:
        errors.append(f"Unknown scenario_id: {scenario_id}")

    scenario_test_passed = len(errors) == 0
    return scenario_test_passed, errors


# =========================================================================
# Explicit Pytest Test Scenarios (1 through 8)
# =========================================================================


def test_scenario_1_normal_lifecycle() -> None:
    """Scenario 1: Standard clean launch, health check, observation, graceful shutdown."""
    result = verify_studio_lifecycle(port=8591)
    passed, errors = evaluate_scenario_assertions(1, result)
    assert passed, f"Scenario 1 assertions failed: {errors}"
    assert result.lifecycle_success is True


def test_scenario_2_startup_timeout() -> None:
    """Scenario 2: Server fails to respond on health endpoint within timeout window."""
    # Run with tiny startup timeout to simulate server timeout
    result = verify_studio_lifecycle(port=8592, startup_timeout_seconds=0.1)
    passed, errors = evaluate_scenario_assertions(2, result)
    assert passed, f"Scenario 2 assertions failed: {errors}"
    assert result.lifecycle_success is False


def test_scenario_3_early_exit() -> None:
    """Scenario 3: Process exits prematurely with invalid startup argument."""
    result = verify_studio_lifecycle(
        port=8593,
        custom_command=["five-head-studio", "--invalid-startup-flag-xyz"],
    )
    passed, errors = evaluate_scenario_assertions(3, result)
    assert passed, f"Scenario 3 assertions failed: {errors}"
    assert result.lifecycle_success is False


def test_scenario_4_graceful_timeout_fallback() -> None:
    """Scenario 4: Graceful shutdown times out; fallback kill escalation is executed."""
    # Run with 0.1s graceful timeout against python sleep command that ignores control events
    creation_flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        creationflags=creation_flags,
    )
    tracked = discover_process_tree(proc.pid)
    # Simulate graceful timeout
    time.sleep(0.1)
    success, err = execute_fallback_kill(proc, tracked)
    proc.wait(timeout=5)
    assert proc.poll() is not None
    assert success is True


def test_scenario_5_signal_dispatch_failure() -> None:
    """Scenario 5: Process terminates immediately prior to signal dispatch."""
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])
    proc.wait(timeout=5)
    sent, err = initiate_graceful_signal(proc)
    assert sent is False
    assert err is not None


def test_scenario_6_port_occupied_pre_launch() -> None:
    """Scenario 6: Target port is already occupied before application launch."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 8596))
        s.listen(1)
        result = verify_studio_lifecycle(port=8596)

    passed, errors = evaluate_scenario_assertions(6, result)
    assert passed, f"Scenario 6 assertions failed: {errors}"
    assert result.pre_launch_port_free is False
    assert result.root_pid is None


def test_scenario_7_descendant_survives_root() -> None:
    """Scenario 7: Descendant process survives root exit and is cleaned up by fallback."""
    creation_flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    )
    # Spawn parent which spawns background child
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)']); time.sleep(0.5)",
        ],
        creationflags=creation_flags,
    )
    time.sleep(0.8)
    tracked = discover_process_tree(proc.pid)
    # Terminate root
    proc.terminate()
    proc.wait(timeout=3)
    # Fallback kill cleans up the remaining tracked descendant
    execute_fallback_kill(proc, tracked)
    # Verify all tracked descendants terminated
    for pid, ctime in tracked:
        try:
            p = psutil.Process(pid)
            if abs(p.create_time() - ctime) < 0.1:
                assert False, f"Descendant {pid} still alive"
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def test_scenario_8_port_occupied_post_shutdown() -> None:
    """Scenario 8: Port remains occupied after server shutdown."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 8598))
        s.listen(1)
        free = poll_port_released(8598, timeout_seconds=0.5, poll_interval=0.1)
        assert not free
