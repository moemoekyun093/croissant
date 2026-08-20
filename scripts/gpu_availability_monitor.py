#!/usr/bin/env python3
"""Check GPU availability on remote hosts and send state-change emails.

The monitor intentionally uses only the Python standard library. Remote checks
are performed with the local ``ssh`` client and NVIDIA's ``nvidia-smi``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import smtplib
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any


LOG = logging.getLogger("gpu-monitor")
NVIDIA_SMI_QUERY = (
    "nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu "
    "--format=csv,noheader,nounits"
)


@dataclass(frozen=True)
class GPU:
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_percent: int

    @property
    def memory_free_mb(self) -> int:
        return self.memory_total_mb - self.memory_used_mb


@dataclass(frozen=True)
class HostResult:
    host: str
    available_gpus: tuple[GPU, ...]
    all_gpus: tuple[GPU, ...]
    error: str | None = None

    @property
    def state_key(self) -> dict[str, Any]:
        return {
            "available_gpu_indexes": [gpu.index for gpu in self.available_gpus],
            "error": self.error,
        }


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    required = ("servers", "smtp", "email")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing required configuration keys: {', '.join(missing)}")
    if not config["servers"]:
        raise ValueError("servers must contain at least one SSH destination")
    labels = [server.get("name", server.get("host")) for server in config["servers"]]
    if any(not label for label in labels):
        raise ValueError("every server must have a host")
    if len(set(labels)) != len(labels):
        raise ValueError("server names must be unique")
    recipients = config["email"].get("to")
    if not isinstance(recipients, list) or not recipients:
        raise ValueError("email.to must be a non-empty list")
    return config


def parse_gpu_output(output: str) -> tuple[GPU, ...]:
    gpus: list[GPU] = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise ValueError(f"unexpected nvidia-smi output on line {line_number}: {line!r}")
        try:
            gpus.append(
                GPU(
                    index=int(fields[0]),
                    name=fields[1],
                    memory_total_mb=int(fields[2]),
                    memory_used_mb=int(fields[3]),
                    utilization_percent=int(fields[4]),
                )
            )
        except ValueError as exc:
            raise ValueError(
                f"invalid numeric value in nvidia-smi output on line {line_number}: {line!r}"
            ) from exc
    return tuple(gpus)


def check_host(
    server: dict[str, Any],
    defaults: dict[str, Any],
) -> HostResult:
    destination = server["host"]
    label = server.get("name", destination)
    port = str(server.get("port", 22))
    timeout = int(server.get("ssh_timeout_seconds", defaults.get("ssh_timeout_seconds", 10)))
    minimum_free_mb = int(
        server.get("minimum_free_memory_mb", defaults.get("minimum_free_memory_mb", 8192))
    )
    maximum_utilization = int(
        server.get("maximum_utilization_percent", defaults.get("maximum_utilization_percent", 10))
    )

    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={timeout}",
        "-p",
        port,
        destination,
        NVIDIA_SMI_QUERY,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout + 5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return HostResult(label, (), (), f"SSH check failed: {exc}")

    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"ssh exited with status {completed.returncode}"
        return HostResult(label, (), (), detail[-500:])

    try:
        all_gpus = parse_gpu_output(completed.stdout)
    except ValueError as exc:
        return HostResult(label, (), (), str(exc))

    available = tuple(
        gpu
        for gpu in all_gpus
        if gpu.memory_free_mb >= minimum_free_mb
        and gpu.utilization_percent <= maximum_utilization
    )
    return HostResult(label, available, all_gpus)


def build_message(
    config: dict[str, Any], results: list[HostResult], *, reminder: bool = False
) -> EmailMessage:
    available_count = sum(len(result.available_gpus) for result in results)
    error_count = sum(result.error is not None for result in results)
    prefix = config["email"].get("subject_prefix", "[GPU monitor]")
    subject = f"{prefix} {available_count} GPU(s) available"
    if error_count:
        subject += f", {error_count} server error(s)"
    if reminder:
        subject += " (reminder)"

    lines = [
        f"GPU availability checked at {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
    ]
    for result in results:
        if result.error:
            lines.extend([f"{result.host}: ERROR", f"  {result.error}", ""])
            continue
        lines.append(
            f"{result.host}: {len(result.available_gpus)}/{len(result.all_gpus)} GPU(s) available"
        )
        available_indexes = {gpu.index for gpu in result.available_gpus}
        for gpu in result.all_gpus:
            status = "AVAILABLE" if gpu.index in available_indexes else "busy"
            lines.append(
                f"  GPU {gpu.index} ({gpu.name}): {status}; "
                f"free {gpu.memory_free_mb}/{gpu.memory_total_mb} MiB; "
                f"utilization {gpu.utilization_percent}%"
            )
        lines.append("")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = config["email"]["from"]
    message["To"] = ", ".join(config["email"]["to"])
    message.set_content("\n".join(lines))
    return message


def send_email(config: dict[str, Any], message: EmailMessage) -> None:
    smtp_config = config["smtp"]
    host = smtp_config["host"]
    port = int(smtp_config.get("port", 587))
    timeout = int(smtp_config.get("timeout_seconds", 20))
    use_ssl = bool(smtp_config.get("ssl", False))

    smtp_class = smtplib.SMTP_SSL if use_ssl else smtplib.SMTP
    with smtp_class(host, port, timeout=timeout) as client:
        if smtp_config.get("starttls", not use_ssl):
            client.starttls()
        username = smtp_config.get("username")
        if username:
            password_env = smtp_config.get("password_env", "GPU_MONITOR_SMTP_PASSWORD")
            password = os.environ.get(password_env)
            if password is None:
                raise RuntimeError(f"SMTP password environment variable {password_env!r} is not set")
            client.login(username, password)
        client.send_message(message)


def read_state(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            state = json.load(handle)
            return state if isinstance(state, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def write_state(path: Path, results: list[HostResult], notified_at: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "hosts": {result.host: result.state_key for result in results},
        "last_notification_epoch": notified_at,
        "last_check_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_once(config: dict[str, Any], state_path: Path) -> bool:
    defaults = config.get("availability", {})
    results = [check_host(server, defaults) for server in config["servers"]]
    current_hosts = {result.host: result.state_key for result in results}
    old_state = read_state(state_path)
    previous_hosts = old_state.get("hosts")
    changed = previous_hosts != current_hosts

    reminder_hours = float(config.get("reminder_hours", 0))
    last_notification = float(old_state.get("last_notification_epoch", 0))
    reminder_due = bool(
        reminder_hours > 0
        and last_notification
        and time.time() - last_notification >= reminder_hours * 3600
    )
    notify_on_first_run = bool(config.get("notify_on_first_run", True))
    should_notify = (changed and (previous_hosts is not None or notify_on_first_run)) or reminder_due
    notified_at = last_notification

    for result in results:
        if result.error:
            LOG.warning("%s: %s", result.host, result.error)
        else:
            LOG.info(
                "%s: %d/%d GPU(s) available",
                result.host,
                len(result.available_gpus),
                len(result.all_gpus),
            )

    if should_notify:
        send_email(config, build_message(config, results, reminder=reminder_due and not changed))
        notified_at = time.time()
        LOG.info("notification sent")
    else:
        LOG.info("state unchanged; no notification sent")

    write_state(state_path, results, notified_at)
    return should_notify


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="path to JSON configuration")
    parser.add_argument("--once", action="store_true", help="check once instead of running forever")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        config = load_config(args.config)
        state_path = Path(config.get("state_file", "~/.local/state/gpu-monitor/state.json")).expanduser()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        LOG.error("cannot load configuration: %s", exc)
        return 2

    if args.once:
        try:
            check_once(config, state_path)
            return 0
        except Exception:
            LOG.exception("monitoring check failed")
            return 1

    interval = max(30, int(config.get("check_interval_seconds", 300)))
    stopping = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    LOG.info("monitor started; checking every %d seconds", interval)
    while not stopping:
        started = time.monotonic()
        try:
            check_once(config, state_path)
        except Exception:
            LOG.exception("monitoring check failed; will retry")
        remaining = max(0.0, interval - (time.monotonic() - started))
        deadline = time.monotonic() + remaining
        while not stopping and time.monotonic() < deadline:
            time.sleep(min(1.0, deadline - time.monotonic()))
    LOG.info("monitor stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
