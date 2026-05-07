"""IBM Bob CLI subprocess lifecycle."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from coral.agent.runtime import AgentHandle, write_coral_log_entry
from coral.workspace.repo import _clean_env

logger = logging.getLogger(__name__)


class BobRuntime:
    """Spawn and manage IBM Bob CLI agent subprocesses."""

    @property
    def instruction_filename(self) -> str:
        return "BOB.md"

    @property
    def shared_dir_name(self) -> str:
        return ".bob"

    def extract_session_id(self, log_path: Path) -> str | None:
        return None  # Bob doesn't expose session IDs in the same way

    def start(
        self,
        worktree_path: Path,
        coral_md_path: Path,
        model: str = "premium",
        runtime_options: dict[str, Any] | None = None,
        max_turns: int = 200,
        log_dir: Path | None = None,
        verbose: bool = False,
        resume_session_id: str | None = None,
        prompt: str | None = None,
        prompt_source: str | None = None,
        task_name: str | None = None,
        task_description: str | None = None,
        gateway_url: str | None = None,
        gateway_api_key: str | None = None,
    ) -> AgentHandle:
        """Start a Bob agent in the given worktree.

        Bob reads instructions from BOB.md via stdin and uses --yolo for auto-approve.
        """
        agent_id_file = worktree_path / ".coral_agent_id"
        agent_id = agent_id_file.read_text().strip() if agent_id_file.exists() else "unknown"

        if log_dir is None:
            log_dir = worktree_path / ".bob" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_idx = len(list(log_dir.glob(f"{agent_id}*.log")))
        log_path = log_dir / f"{agent_id}.{log_idx}.log"

        if prompt is None:
            prompt = "Begin."

        # Bob command: bob --yolo -o text < BOB.md
        # --yolo: auto-approve all actions (similar to dangerously-skip-permissions)
        # -o text: output format as text
        # Note: Bob picks its own model, so we don't pass --model flag
        cmd = [
            "bob",
            "--yolo",
            "-o", "text",
        ]

        logger.info(f"Starting Bob agent {agent_id} in {worktree_path}")
        logger.info(f"Command: {' '.join(cmd)} < {coral_md_path}")

        agent_env = _clean_env()
        worktree_venv = str(worktree_path / ".venv")
        agent_env["UV_PROJECT_ENVIRONMENT"] = worktree_venv
        # Set VIRTUAL_ENV so login shells (which reset PATH) can restore it
        # via /etc/profile.d/coral-venv.sh in Docker containers.
        agent_env["VIRTUAL_ENV"] = worktree_venv
        # Prepend .venv/bin to PATH for non-login shells
        venv_bin = str(worktree_path / ".venv" / "bin")
        agent_env["PATH"] = venv_bin + ":" + agent_env.get("PATH", "")

        # Route through gateway if configured
        # Bob may use different env vars - adjust as needed based on Bob's documentation
        if gateway_url:
            agent_env["BOB_BASE_URL"] = gateway_url
            logger.info(f"Bob agent {agent_id}: routing via gateway at {gateway_url}")
        if gateway_api_key:
            agent_env["BOB_API_KEY"] = gateway_api_key

        log_file = open(log_path, "w", buffering=1)

        write_coral_log_entry(
            log_file,
            prompt=prompt,
            source=prompt_source or "start",
            agent_id=agent_id,
            task_name=task_name,
            task_description=task_description,
        )

        # Bob reads instructions from stdin, so we need to pipe the BOB.md file
        try:
            instruction_file = open(coral_md_path, "r")
        except FileNotFoundError:
            logger.error(f"Instruction file not found: {coral_md_path}")
            raise

        if verbose:
            process = subprocess.Popen(
                cmd,
                cwd=str(worktree_path),
                stdin=instruction_file,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=agent_env,
            )

            def _tee_output(proc: subprocess.Popen, log_f, agent: str) -> None:
                try:
                    assert proc.stdout is not None
                    for line in iter(proc.stdout.readline, b""):
                        decoded = line.decode("utf-8", errors="replace")
                        sys.stdout.write(f"[{agent}] {decoded}")
                        sys.stdout.flush()
                        log_f.write(decoded)
                        log_f.flush()
                except Exception as e:
                    logger.error(f"Tee thread error: {e}")
                finally:
                    log_f.close()
                    if proc.stdout:
                        try:
                            proc.stdout.close()
                        except Exception:
                            pass

            tee_thread = threading.Thread(
                target=_tee_output,
                args=(process, log_file, agent_id),
                daemon=True,
            )
            tee_thread.start()
            log_file_ref = None
        else:
            process = subprocess.Popen(
                cmd,
                cwd=str(worktree_path),
                stdin=instruction_file,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=agent_env,
            )
            log_file_ref = log_file

        logger.info(f"Bob agent {agent_id} started with PID {process.pid}")

        return AgentHandle(
            agent_id=agent_id,
            process=process,
            worktree_path=worktree_path,
            log_path=log_path,
            _log_file=log_file_ref,
        )
