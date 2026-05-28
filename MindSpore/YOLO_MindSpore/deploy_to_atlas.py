#!/usr/bin/env python3
"""Deploy the Atlas runtime bundle to a board over SSH/SFTP."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
LOCAL_PY_PKGS = PROJECT_ROOT / ".python_packages" / "pc_deploy"
if LOCAL_PY_PKGS.exists():
    sys.path.insert(0, str(LOCAL_PY_PKGS))

import paramiko


DEFAULT_HOST = "192.168.137.100"
DEFAULT_USER = "HwHiAiUser"
DEFAULT_PASSWORD = "Mind@123"
DEFAULT_REMOTE_DIR = "/home/HwHiAiUser/pig_counting"

DEPLOY_LOCAL = SCRIPT_DIR / "deploy_atlas"
LOCAL_MODEL = PROJECT_ROOT / "yolov8n_pig_fp16.om"
DATASET_DIRS = {
    "4": SCRIPT_DIR / "数据集" / "四",
    "5": SCRIPT_DIR / "数据集" / "五",
}
REMOTE_DATASETS = {
    "4": "group4",
    "5": "group5",
}


def ssh_cmd(ssh: paramiko.SSHClient, cmd: str, timeout: int = 120) -> tuple[str, str, int]:
    print(f"  [CMD] {cmd}")
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out.strip():
        print(f"  [OUT] {out.strip()[:240]}")
    if err.strip():
        print(f"  [ERR] {err.strip()[:240]}")
    return out, err, code


def ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = []
    current = remote_dir
    while current not in ("", "/"):
        parts.append(current)
        current = current.rsplit("/", 1)[0]
    for path in reversed(parts):
        try:
            sftp.stat(path)
        except FileNotFoundError:
            sftp.mkdir(path)


def safe_remove_remote_tree(ssh: paramiko.SSHClient, remote_dir: str) -> None:
    if not remote_dir.startswith("/home/"):
        raise ValueError(f"Refusing to clean unsafe path: {remote_dir}")
    ssh_cmd(ssh, f"mkdir -p {remote_dir}")
    ssh_cmd(ssh, f"find {remote_dir} -mindepth 1 -maxdepth 1 -exec rm -rf {{}} +")


def sftp_upload_file(sftp: paramiko.SFTPClient, local_path: Path, remote_path: str) -> None:
    ensure_remote_dir(sftp, remote_path.rsplit("/", 1)[0])
    size_mb = local_path.stat().st_size / 1024 / 1024
    print(f"  Upload file: {local_path.name} ({size_mb:.1f} MB) -> {remote_path}")
    sftp.put(str(local_path), remote_path)


def sftp_upload_dir(
    sftp: paramiko.SFTPClient,
    local_dir: Path,
    remote_dir: str,
    *,
    skip_names: set[str] | None = None,
) -> None:
    skip_names = skip_names or set()
    ensure_remote_dir(sftp, remote_dir)
    for item in sorted(local_dir.iterdir(), key=lambda p: p.name):
        if item.name in skip_names:
            continue
        remote_path = f"{remote_dir}/{item.name}"
        if item.is_dir():
            sftp_upload_dir(sftp, item, remote_path, skip_names=skip_names)
        else:
            sftp_upload_file(sftp, item, remote_path)


def upload_dataset_group(
    ssh: paramiko.SSHClient,
    sftp: paramiko.SFTPClient,
    group_id: str,
    remote_dir: str,
) -> None:
    local_dir = DATASET_DIRS[group_id]
    if not local_dir.exists():
        print(f"  Skip dataset group {group_id}: local dir missing -> {local_dir}")
        return

    remote_name = REMOTE_DATASETS[group_id]
    remote_dataset_dir = f"{remote_dir}/datasets/{remote_name}"
    ssh_cmd(ssh, f"mkdir -p {remote_dataset_dir}")
    print(f"\n[Dataset {group_id}] Uploading *.mp4 to {remote_dataset_dir}")
    for video in sorted(local_dir.glob("*.mp4"), key=lambda p: p.name):
        sftp_upload_file(sftp, video, f"{remote_dataset_dir}/{video.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy pig counter to Atlas board")
    parser.add_argument("--host", default=os.getenv("ATLAS_HOST", DEFAULT_HOST))
    parser.add_argument("--user", default=os.getenv("ATLAS_USER", DEFAULT_USER))
    parser.add_argument("--password", default=os.getenv("ATLAS_PASS", DEFAULT_PASSWORD))
    parser.add_argument("--remote-dir", default=os.getenv("ATLAS_REMOTE_DIR", DEFAULT_REMOTE_DIR))
    parser.add_argument(
        "--model",
        default=str(LOCAL_MODEL),
        help="Local OM model path. Defaults to the repo root model.",
    )
    parser.add_argument(
        "--dataset-groups",
        nargs="*",
        default=["4", "5"],
        choices=sorted(DATASET_DIRS.keys()),
        help="Dataset groups to upload. Use none to skip datasets.",
    )
    parser.add_argument(
        "--skip-datasets",
        action="store_true",
        help="Do not upload any local videos.",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Upload files only, without running bootstrap_board.sh on the board.",
    )
    parser.add_argument(
        "--keep-remote",
        action="store_true",
        help="Do not clear the remote directory before uploading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()

    if not DEPLOY_LOCAL.exists():
        raise FileNotFoundError(f"Deploy bundle missing: {DEPLOY_LOCAL}")
    if not model_path.exists():
        raise FileNotFoundError(f"OM model missing: {model_path}")

    print("=" * 60)
    print("Deploying to Atlas 200I DK A2")
    print(f"Host: {args.host}")
    print(f"Remote dir: {args.remote_dir}")
    print(f"Model: {model_path}")
    print("=" * 60)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(args.host, username=args.user, password=args.password, timeout=10)
    sftp = ssh.open_sftp()

    try:
        print("\n[Step 1] Prepare remote directory")
        ensure_remote_dir(sftp, args.remote_dir)
        if not args.keep_remote:
            safe_remove_remote_tree(ssh, args.remote_dir)
        ssh_cmd(
            ssh,
            f"mkdir -p {args.remote_dir}/models {args.remote_dir}/datasets "
            f"{args.remote_dir}/output {args.remote_dir}/logs",
        )

        print("\n[Step 2] Upload deploy bundle")
        sftp_upload_dir(
            sftp,
            DEPLOY_LOCAL,
            args.remote_dir,
            skip_names={"__pycache__"},
        )

        print("\n[Step 3] Upload OM model")
        sftp_upload_file(sftp, model_path, f"{args.remote_dir}/models/{model_path.name}")

        if not args.skip_datasets:
            print("\n[Step 4] Upload datasets")
            for group_id in args.dataset_groups:
                upload_dataset_group(ssh, sftp, group_id, args.remote_dir)
        else:
            print("\n[Step 4] Skip dataset upload")

        if not args.skip_bootstrap:
            print("\n[Step 5] Run bootstrap script on board")
            ssh_cmd(ssh, f"chmod +x {args.remote_dir}/bootstrap_board.sh")
            ssh_cmd(ssh, f"cd {args.remote_dir} && ./bootstrap_board.sh", timeout=180)
        else:
            print("\n[Step 5] Skip bootstrap")

        print("\n[Step 6] Verify remote files")
        ssh_cmd(ssh, f"ls -la {args.remote_dir}")
        ssh_cmd(ssh, f"ls -la {args.remote_dir}/models")
        for group_id in args.dataset_groups if not args.skip_datasets else []:
            remote_name = REMOTE_DATASETS[group_id]
            ssh_cmd(ssh, f"find {args.remote_dir}/datasets/{remote_name} -maxdepth 1 -name '*.mp4' | wc -l")

        print("\n" + "=" * 60)
        print("Deployment complete.")
        print(f"Board shell: ssh {args.user}@{args.host}")
        print(f"Board dir:   cd {args.remote_dir}")
        print(
            "Run demo:    ./bootstrap_board.sh && "
            "python3 web_monitor.py --video datasets/group4/1-12头.mp4 "
            f"--om models/{model_path.name}"
        )
        print("=" * 60)
    finally:
        sftp.close()
        ssh.close()


if __name__ == "__main__":
    main()
