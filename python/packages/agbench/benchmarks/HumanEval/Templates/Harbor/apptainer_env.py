"""
ApptainerEnvironment: a Harbor `BaseEnvironment` backend that runs tasks inside
rootless apptainer instances instead of Docker.

This module is intentionally **standalone**: it imports nothing from agbench so
that it can be loaded by the harbor-env Python (which does not have agbench
installed). Harbor itself is the only third-party dependency, and Harbor's
runtime injects this class via `--environment-import-path`.

Mapping from BaseEnvironment to apptainer:

- `start()`  -> resolve image to a SIF (cached at ~/.cache/agbench-harbor/),
               then `apptainer instance start --writable-tmpfs --bind <mounts>
               <sif> <session>`. Bind-mounts the trial's verifier/agent/
               artifacts directories into /logs and a private scratch dir into
               /_hb_scratch (used as a transfer area for upload/download).
- `stop()`   -> `apptainer instance stop <session>`.
- `exec()`   -> `apptainer exec instance://<session> bash -c "..."`. Per-exec
               env vars are passed via APPTAINERENV_KEY=value in the subprocess
               environment so they appear inside the container.
- `upload_*` -> copy from host into the scratch bind mount, then `cp` it into
               the requested target path inside the container's writable tmpfs.
- `download_*` -> the inverse of upload.

Limitations / known caveats:

- Apptainer has no per-exec user switching; the `user` argument to `exec()` is
  ignored. Containers run as the host user.
- Internet isolation (`allow_internet = false`) is not supported by apptainer
  out of the box, so `can_disable_internet` returns False; tasks that require
  net isolation will fail validation early (this is intentional).
- GPU passthrough via `--nv` is supported but only if the host has CUDA
  drivers; it is enabled when the task requests `gpus > 0`.
- The cached SIF lives at ~/.cache/agbench-harbor/<sanitized>.sif and is
  *not* deleted on `stop(delete=True)` to avoid rebuilding on every trial.
  Pass `--ek delete_sif=true` to override.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths


_DEFAULT_IMAGE = "docker://python:3.12-slim"
_SIF_CACHE_DIR = Path(
    os.environ.get("AGBENCH_HARBOR_SIF_CACHE", str(Path.home() / ".cache" / "agbench-harbor"))
)
_SCRATCH_GUEST_PATH = "/_hb_scratch"


def _sanitize(name: str) -> str:
    """Make a string safe to use as a filename / instance name."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", name).lower()[:60]


class ApptainerEnvironment(BaseEnvironment):
    """A Harbor environment backed by rootless apptainer instances."""

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        logger: logging.Logger | None = None,
        # kwargs forwarded by the Harbor CLI via `--ek key=value`:
        image: str | None = None,
        delete_sif: bool = False,
        extra_binds: str | None = None,
        **kwargs,
    ):
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            logger=logger,
            **kwargs,
        )

        self._image = image or os.environ.get("AGBENCH_HARBOR_IMAGE", _DEFAULT_IMAGE)
        self._delete_sif = bool(delete_sif)
        self._extra_binds = [b for b in (extra_binds or "").split(",") if b]
        self._instance_name = f"hb_{_sanitize(session_id)}_{uuid.uuid4().hex[:6]}"
        self._sif_path: Path | None = None

        # Host-side scratch dir lives next to the trial's verifier/agent dirs.
        self._scratch_host = trial_paths.trial_dir / "_hb_scratch"
        # Host-backed workdir for apptainer's writable layer. Without this,
        # `--writable-tmpfs` falls back to a tiny RAM tmpfs (~16MB) which is
        # too small for installed agents that apt-get install build-essential.
        self._workdir_host = trial_paths.trial_dir / "_hb_workdir"

    # ------------------------------------------------------------------
    # Required class metadata
    # ------------------------------------------------------------------

    @staticmethod
    def type() -> EnvironmentType:
        # Harbor's enum has no APPTAINER member; DOCKER is the closest analog
        # and is only used for cosmetic error messages when import_path is set.
        return EnvironmentType.DOCKER

    @property
    def is_mounted(self) -> bool:
        # /logs/{agent,verifier,artifacts} are bind-mounted from the trial dir,
        # so Harbor doesn't need to upload/download those paths separately.
        return True

    @property
    def supports_gpus(self) -> bool:
        return True  # via `apptainer exec --nv`

    @property
    def can_disable_internet(self) -> bool:
        return False

    def _validate_definition(self):
        # Apptainer doesn't need a Dockerfile or compose file. The image is
        # configured globally (default _DEFAULT_IMAGE or via --ek image=...).
        if not self.environment_dir.exists():
            raise FileNotFoundError(
                f"environment_dir {self.environment_dir} does not exist"
            )

    # ------------------------------------------------------------------
    # Image resolution
    # ------------------------------------------------------------------

    def _resolve_sandbox_template(self) -> Path:
        """Build (or reuse) a cached read-only sandbox dir for the image.

        We use a sandbox directory rather than a SIF because rootless
        apptainer on this cluster has neither kernel overlayfs (blocked for
        non-root) nor fuse-overlayfs installed, so the only path to a
        writable container filesystem is `--writable <sandbox_dir>`. The
        cached sandbox is treated as a read-only template; each trial gets
        its own copy in start().
        """
        _SIF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(self._image.encode()).hexdigest()[:12]
        sandbox = _SIF_CACHE_DIR / f"{_sanitize(self._image)}-{digest}.sandbox"
        if not sandbox.exists():
            self.logger.info(f"Building apptainer sandbox for {self._image} -> {sandbox}")
            res = subprocess.run(
                ["apptainer", "build", "--sandbox", str(sandbox), self._image],
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                raise RuntimeError(
                    f"apptainer build --sandbox failed for {self._image}: "
                    f"{res.stderr or res.stdout}"
                )
        return sandbox

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, force_build: bool) -> None:
        # Ensure host-side mount points exist before binding.
        for d in (
            self.trial_paths.verifier_dir,
            self.trial_paths.agent_dir,
            self.trial_paths.artifacts_dir,
            self._scratch_host,
            self._workdir_host,
        ):
            d.mkdir(parents=True, exist_ok=True)

        if force_build:
            digest = hashlib.sha1(self._image.encode()).hexdigest()[:12]
            stale = _SIF_CACHE_DIR / f"{_sanitize(self._image)}-{digest}.sandbox"
            if stale.exists():
                shutil.rmtree(stale)
        sandbox_template = self._resolve_sandbox_template()

        # Per-trial writable copy of the sandbox so trials don't pollute the
        # shared template. The slim image is ~120 MB; copytree is a few seconds.
        sandbox_local = self._workdir_host / "sandbox"
        if sandbox_local.exists():
            shutil.rmtree(sandbox_local)
        shutil.copytree(sandbox_template, sandbox_local, symlinks=True)
        self._sif_path = sandbox_local

        # apptainer in --writable mode (no overlay) cannot create bind-mount
        # targets inside the container automatically. Pre-create them in the
        # sandbox so the bind options resolve.
        for guest_dir in (
            EnvironmentPaths.verifier_dir,
            EnvironmentPaths.agent_dir,
            EnvironmentPaths.artifacts_dir,
            _SCRATCH_GUEST_PATH,
        ):
            (sandbox_local / str(guest_dir).lstrip("/")).mkdir(parents=True, exist_ok=True)

        binds = [
            f"{self.trial_paths.verifier_dir.resolve()}:{EnvironmentPaths.verifier_dir}",
            f"{self.trial_paths.agent_dir.resolve()}:{EnvironmentPaths.agent_dir}",
            f"{self.trial_paths.artifacts_dir.resolve()}:{EnvironmentPaths.artifacts_dir}",
            f"{self._scratch_host.resolve()}:{_SCRATCH_GUEST_PATH}",
        ]
        binds.extend(self._extra_binds)

        cmd = [
            "apptainer", "instance", "start",
            # Sandbox dir copied per-trial above; --writable lets the agent
            # mutate it directly. No overlay needed (and none available on
            # this cluster — see _resolve_sandbox_template).
            "--writable",
            "--containall",
        ]
        if self.task_env_config.gpus and self.task_env_config.gpus > 0:
            cmd.append("--nv")
        for b in binds:
            cmd.extend(["--bind", b])
        cmd.append(str(self._sif_path))
        cmd.append(self._instance_name)

        self.logger.info(f"apptainer instance start: {self._instance_name}")
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"apptainer instance start failed (rc={proc.returncode}):\n"
                f"{out.decode(errors='replace')}"
            )

        # Make log mount points writable by everything (matches docker backend).
        await self.exec(
            f"mkdir -p {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir} "
            f"{EnvironmentPaths.artifacts_dir} && "
            f"chmod 777 {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir} "
            f"{EnvironmentPaths.artifacts_dir}"
        )

    async def stop(self, delete: bool):
        proc = await asyncio.create_subprocess_exec(
            "apptainer", "instance", "stop", self._instance_name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            self.logger.warning(
                f"apptainer instance stop failed: {out.decode(errors='replace')}"
            )

        if delete and self._delete_sif and self._sif_path and self._sif_path.exists():
            try:
                self._sif_path.unlink()
            except OSError as e:
                self.logger.warning(f"failed to remove SIF {self._sif_path}: {e}")

        # Cleanup the host scratch dir; trial verifier/agent/artifacts are
        # owned by Harbor and persisted intentionally.
        for d in (self._scratch_host, self._workdir_host):
            if d.exists():
                try:
                    shutil.rmtree(d)
                except OSError as e:
                    self.logger.warning(f"failed to remove {d}: {e}")

    # ------------------------------------------------------------------
    # exec
    # ------------------------------------------------------------------

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,  # ignored: apptainer runs as host user
    ) -> ExecResult:
        merged_env = self._merge_env(env) or {}
        # Apptainer reads APPTAINERENV_<KEY>=value from the host environment
        # and exposes it as <KEY>=value inside the container.
        host_env = os.environ.copy()
        for k, v in merged_env.items():
            host_env[f"APPTAINERENV_{k}"] = str(v)

        wrapped = command if not cwd else f"cd {shlex.quote(cwd)} && {command}"
        cmd = [
            "apptainer", "exec",
            f"instance://{self._instance_name}",
            "bash", "-lc", wrapped,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=host_env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            if timeout_sec:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_sec
                )
            else:
                stdout_b, stderr_b = await proc.communicate()
        except asyncio.TimeoutError:
            proc.kill()
            stdout_b, stderr_b = await proc.communicate()
            raise RuntimeError(f"apptainer exec timed out after {timeout_sec}s")

        return ExecResult(
            stdout=stdout_b.decode(errors="replace") if stdout_b else None,
            stderr=stderr_b.decode(errors="replace") if stderr_b else None,
            return_code=proc.returncode or 0,
        )

    # ------------------------------------------------------------------
    # File transfer (via the bound scratch dir)
    # ------------------------------------------------------------------

    def _scratch_pair(self) -> tuple[Path, str]:
        """Allocate a unique (host_path, guest_path) pair under the scratch mount."""
        token = uuid.uuid4().hex
        return self._scratch_host / token, f"{_SCRATCH_GUEST_PATH}/{token}"

    async def upload_file(self, source_path: Path | str, target_path: str):
        host_p, guest_p = self._scratch_pair()
        host_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(source_path), host_p)
        res = await self.exec(
            f"mkdir -p {shlex.quote(str(Path(target_path).parent))} && "
            f"cp {shlex.quote(guest_p)} {shlex.quote(target_path)}"
        )
        if res.return_code != 0:
            raise RuntimeError(
                f"upload_file failed for {source_path} -> {target_path}: {res.stderr}"
            )

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        host_p, guest_p = self._scratch_pair()
        # Copy whole tree under a unique scratch slot.
        shutil.copytree(str(source_dir), host_p)
        res = await self.exec(
            f"mkdir -p {shlex.quote(target_dir)} && "
            f"cp -a {shlex.quote(guest_p)}/. {shlex.quote(target_dir)}/"
        )
        if res.return_code != 0:
            raise RuntimeError(
                f"upload_dir failed for {source_dir} -> {target_dir}: {res.stderr}"
            )

    async def download_file(self, source_path: str, target_path: Path | str):
        host_p, guest_p = self._scratch_pair()
        res = await self.exec(
            f"mkdir -p {shlex.quote(str(Path(guest_p).parent))} && "
            f"cp {shlex.quote(source_path)} {shlex.quote(guest_p)}"
        )
        if res.return_code != 0:
            raise RuntimeError(
                f"download_file failed for {source_path}: {res.stderr}"
            )
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(host_p, target)

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        host_p, guest_p = self._scratch_pair()
        res = await self.exec(
            f"mkdir -p {shlex.quote(guest_p)} && "
            f"cp -a {shlex.quote(source_dir)}/. {shlex.quote(guest_p)}/"
        )
        if res.return_code != 0:
            raise RuntimeError(
                f"download_dir failed for {source_dir}: {res.stderr}"
            )
        target = Path(target_dir)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(host_p, target)
