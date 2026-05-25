"""一次性脚本：

1. 用密码登录 Atlas 板，把本机 ~/.ssh/id_ed25519.pub 追加到板上 authorized_keys
2. SFTP 上传 weight_regressor.onnx
3. 远端运行 ATC 转 .om
4. 把 .om 拉回本地

之后 ssh/scp 即可免密。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import paramiko

HOST = "192.168.137.100"
USER = "HwHiAiUser"
PASS = "Mind@123"
PUBKEY = Path.home() / ".ssh" / "id_ed25519.pub"

LOCAL_ONNX = Path("E:/pig_couter/runs/20260525-165217/weight_regressor.onnx")
REMOTE_DIR = "/home/HwHiAiUser/atlas_demo/weight"
REMOTE_ONNX = f"{REMOTE_DIR}/weight_regressor.onnx"
REMOTE_OM = f"{REMOTE_DIR}/weight_regressor_fp16"  # ATC adds .om
LOCAL_OM_PULL = Path("E:/pig_couter/runs/20260525-165217/weight_regressor_fp16.om")


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 1800) -> tuple[int, str, str]:
    """Run cmd on board, stream stdout to console, return (exit_code, stdout, stderr)."""
    print(f"[remote] $ {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout, get_pty=True)
    chan = stdout.channel
    out_chunks: list[str] = []
    err_chunks: list[str] = []
    # Stream both stdout (via pty) and stderr until channel closes
    while True:
        if chan.recv_ready():
            data = chan.recv(4096).decode("utf-8", errors="replace")
            sys.stdout.write(data)
            sys.stdout.flush()
            out_chunks.append(data)
        if chan.recv_stderr_ready():
            data = chan.recv_stderr(4096).decode("utf-8", errors="replace")
            sys.stderr.write(data)
            sys.stderr.flush()
            err_chunks.append(data)
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
        time.sleep(0.05)
    rc = chan.recv_exit_status()
    return rc, "".join(out_chunks), "".join(err_chunks)


def main():
    pubkey = PUBKEY.read_text().strip()
    print(f"[local] pubkey: {pubkey[:60]}...")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"[local] connecting {USER}@{HOST} (password)...")
    client.connect(HOST, username=USER, password=PASS, timeout=10, look_for_keys=False, allow_agent=False)
    print("[local] connected.")

    # 1. install pubkey idempotently
    cmd = (
        "mkdir -p ~/.ssh && touch ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys && "
        f"grep -qxF '{pubkey}' ~/.ssh/authorized_keys || echo '{pubkey}' >> ~/.ssh/authorized_keys && "
        "echo PUBKEY_INSTALLED"
    )
    rc, out, _ = run(client, cmd)
    assert rc == 0 and "PUBKEY_INSTALLED" in out, "pubkey install failed"

    # 2. mkdir remote
    rc, _, _ = run(client, f"mkdir -p {REMOTE_DIR} && ls -la {REMOTE_DIR}")
    assert rc == 0

    # 3. SFTP upload
    print(f"[local] SFTP put {LOCAL_ONNX} -> {REMOTE_ONNX}")
    sftp = client.open_sftp()
    sftp.put(str(LOCAL_ONNX), REMOTE_ONNX, confirm=True)
    attr = sftp.stat(REMOTE_ONNX)
    print(f"[local] uploaded: {attr.st_size} bytes")

    # 4. Check ATC availability + run
    # Source CANN env first (path may differ; try common locations)
    atc_setup = (
        "[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && "
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh || true ; "
        "[ -f $HOME/Ascend/ascend-toolkit/set_env.sh ] && "
        "source $HOME/Ascend/ascend-toolkit/set_env.sh || true ; "
        "which atc || (echo 'ATC not found' && exit 1)"
    )
    rc, _, _ = run(client, atc_setup)
    if rc != 0:
        print("[!] ATC not on PATH. Aborting.")
        return

    atc_cmd = (
        f"cd {REMOTE_DIR} && "
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null; "
        "source $HOME/Ascend/ascend-toolkit/set_env.sh 2>/dev/null; "
        f"atc --model=weight_regressor.onnx "
        f"--framework=5 --output=weight_regressor_fp16 "
        f"--soc_version=Ascend310B4 --precision_mode=force_fp16 "
        f"--input_shape='input:1,3,224,224' 2>&1"
    )
    rc, _, _ = run(client, atc_cmd, timeout=1800)
    if rc != 0:
        print(f"[!] ATC exit code {rc}")
        client.close()
        sys.exit(rc)

    # 5. list result + pull back
    rc, out, _ = run(client, f"ls -la {REMOTE_DIR}/*.om")
    assert rc == 0

    LOCAL_OM_PULL.parent.mkdir(parents=True, exist_ok=True)
    print(f"[local] SFTP get {REMOTE_OM}.om -> {LOCAL_OM_PULL}")
    sftp.get(f"{REMOTE_OM}.om", str(LOCAL_OM_PULL))
    print(f"[local] pulled {LOCAL_OM_PULL.stat().st_size} bytes")

    sftp.close()
    client.close()
    print("[DONE]")


if __name__ == "__main__":
    main()
