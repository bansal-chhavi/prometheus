Deployment notes — TrustSignal

This file contains quick examples for running TrustSignal under a supervisor.

1) systemd (Linux)

- Copy `deployment/trustsignal.service` to `/etc/systemd/system/trustsignal.service` and edit absolute paths:
  - `WorkingDirectory` should point to the repo root (where `src/` lives)
  - `ExecStart` should use the absolute `python3` in your virtualenv or system

Commands:
```bash
sudo cp deployment/trustsignal.service /etc/systemd/system/trustsignal.service
sudo systemctl daemon-reload
sudo systemctl enable --now trustsignal
sudo journalctl -u trustsignal -f
```

2) Windows (PowerShell)

- Use `scripts/install_windows_service.ps1` (run as Administrator). Adjust `PythonPath` and `WorkingDir` parameters.

Example (run as admin PowerShell):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\\scripts\\install_windows_service.ps1 -PythonPath "C:\\Users\\you\\AppData\\Local\\Programs\\Python\\Python39\\python.exe" -WorkingDir "C:\\path\\to\\repo"
Start-Service -Name TrustSignal
Get-Service -Name TrustSignal
```

3) Graceful shutdown

- The FastAPI app exposes `startup`/`shutdown` events; uvicorn responds to SIGTERM and will shut down gracefully.
- When using systemd, the `KillSignal=SIGTERM` and `TimeoutStopSec` settings in the unit control graceful termination.
- Monitor logs to ensure background NLI tasks finish promptly before process exit.

4) Health checks & timeouts

- Add an HTTP health endpoint (e.g., `/health`) if you need external load balancers to probe readiness.
- Use request timeouts and worker pool sizing for CPU-bound NLI work.

