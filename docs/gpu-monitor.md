# GPU availability email monitor

`scripts/gpu_availability_monitor.py` queries `nvidia-smi` on each configured
server through SSH. A GPU is available when it has at least the configured free
VRAM and no more than the configured utilization. Email is sent on the first
check, whenever available GPU indexes or server errors change, and periodically
when `reminder_hours` is nonzero.

## Prerequisites

- Python 3.9 or newer and an `ssh` client on the monitoring machine.
- `nvidia-smi` on every GPU server.
- SSH key authentication from the monitoring machine. Add each server's host
  key to `known_hosts` before running the monitor; interactive SSH prompts are
  disabled.
- An SMTP account. For Gmail, use an app password rather than the account's main
  password.

Use a dedicated, unprivileged SSH account when possible. It only needs permission
to run `nvidia-smi`.

## Configure and test

```bash
cp configs/gpu-monitor.example.json configs/gpu-monitor.json
chmod 600 configs/gpu-monitor.json
# Edit the server, threshold, SMTP, and recipient values.

export GPU_MONITOR_SMTP_PASSWORD='your SMTP password or app password'
python3 scripts/gpu_availability_monitor.py \
  --config configs/gpu-monitor.json --once --verbose
```

The password is read from the environment and should not be put in the JSON
file. The default state file remembers the last result so unchanged checks do
not generate repeated email.

To run the process directly instead of using a scheduler, omit `--once`:

```bash
python3 scripts/gpu_availability_monitor.py --config configs/gpu-monitor.json
```

## Run with systemd

The files in `deploy/` run a fresh one-shot check every five minutes. This is
usually more resilient than keeping a Python process alive.

1. Copy the script to `/opt/gpu-monitor/gpu_availability_monitor.py` and the
   edited config to `/etc/gpu-monitor/config.json`.
2. Set `state_file` in that config to `/var/lib/gpu-monitor/state.json`.
3. Create a `gpu-monitor` system user and ensure it has its SSH private key and
   `known_hosts` file.
4. Create `/etc/gpu-monitor/environment`, readable only by root, containing:

   ```text
   GPU_MONITOR_SMTP_PASSWORD=your-password
   ```

5. Copy `deploy/gpu-monitor.service` and `deploy/gpu-monitor.timer` to
   `/etc/systemd/system/`, then run:

   ```bash
   sudo install -d -o gpu-monitor -g gpu-monitor /var/lib/gpu-monitor
   sudo chmod 600 /etc/gpu-monitor/environment /etc/gpu-monitor/config.json
   sudo systemctl daemon-reload
   sudo systemctl enable --now gpu-monitor.timer
   systemctl list-timers gpu-monitor.timer
   journalctl -u gpu-monitor.service
   ```

Change `OnUnitActiveSec` in the timer if a different interval is wanted. When
using the timer, `check_interval_seconds` is ignored because `--once` is used.
