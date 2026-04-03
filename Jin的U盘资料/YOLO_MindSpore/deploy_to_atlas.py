#!/usr/bin/env python3
"""Deploy pig counting system to Atlas 200I DK A2."""
import paramiko
import os
import sys
import stat

HOST = '192.168.137.100'
USER = 'HwHiAiUser'
PASS = 'Mind@123'
REMOTE_DIR = '/home/HwHiAiUser/pig_counting'
BACKUP_DIR = '/home/HwHiAiUser/pig_counting_old_backup'

DEPLOY_LOCAL = r'C:\Users\Skyha\Desktop\pig_couter\Jin的U盘资料\YOLO_MindSpore\deploy_atlas'
VIDEO_DIR_4 = r'C:\Users\Skyha\Desktop\pig_couter\Jin的U盘资料\YOLO_MindSpore\数据集\四'
VIDEO_DIR_5 = r'C:\Users\Skyha\Desktop\pig_couter\Jin的U盘资料\YOLO_MindSpore\数据集\五'


def ssh_cmd(ssh, cmd, timeout=60):
    print(f'  [CMD] {cmd}')
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    code = stdout.channel.recv_exit_status()
    if out.strip():
        print(f'  [OUT] {out.strip()[:200]}')
    if code != 0 and err.strip():
        print(f'  [ERR] {err.strip()[:200]}')
    return out, err, code


def sftp_upload_dir(sftp, local_dir, remote_dir, skip_pycache=True):
    """Recursively upload a directory."""
    try:
        sftp.mkdir(remote_dir)
    except:
        pass
    for item in os.listdir(local_dir):
        if skip_pycache and item == '__pycache__':
            continue
        local_path = os.path.join(local_dir, item)
        remote_path = remote_dir + '/' + item
        if os.path.isdir(local_path):
            sftp_upload_dir(sftp, local_path, remote_path, skip_pycache)
        else:
            print(f'  Upload: {item} -> {remote_path}')
            sftp.put(local_path, remote_path)


def main():
    print('=' * 60)
    print('Deploying to Atlas 200I DK A2')
    print('=' * 60)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=10)
    sftp = ssh.open_sftp()

    # Step 1: Backup old .om model
    print('\n[Step 1] Backup old models...')
    ssh_cmd(ssh, f'mkdir -p {BACKUP_DIR}')
    ssh_cmd(ssh, f'cp -r {REMOTE_DIR}/models/* {BACKUP_DIR}/ 2>/dev/null; echo done')

    # Step 2: Clean old content
    print('\n[Step 2] Clean old content...')
    ssh_cmd(ssh, f'rm -rf {REMOTE_DIR}/*')
    ssh_cmd(ssh, f'mkdir -p {REMOTE_DIR}/models {REMOTE_DIR}/trackers {REMOTE_DIR}/datasets {REMOTE_DIR}/output')

    # Step 3: Upload deploy scripts
    print('\n[Step 3] Upload deploy scripts...')
    sftp_upload_dir(sftp, DEPLOY_LOCAL, REMOTE_DIR)

    # Step 4: Restore .om model from backup
    print('\n[Step 4] Restore .om model...')
    ssh_cmd(ssh, f'cp {BACKUP_DIR}/yolov8n_pig_fp16.om {REMOTE_DIR}/models/')

    # Step 5: Upload video datasets
    print('\n[Step 5] Upload dataset 四...')
    remote_ds4 = f'{REMOTE_DIR}/datasets/si'
    ssh_cmd(ssh, f'mkdir -p {remote_ds4}')
    for f in sorted(os.listdir(VIDEO_DIR_4)):
        if f.endswith('.mp4'):
            local_f = os.path.join(VIDEO_DIR_4, f)
            remote_f = f'{remote_ds4}/{f}'
            size_mb = os.path.getsize(local_f) / 1024 / 1024
            print(f'  Upload: {f} ({size_mb:.1f} MB)')
            sftp.put(local_f, remote_f)

    print('\n[Step 6] Upload dataset 五...')
    remote_ds5 = f'{REMOTE_DIR}/datasets/wu'
    ssh_cmd(ssh, f'mkdir -p {remote_ds5}')
    for f in sorted(os.listdir(VIDEO_DIR_5)):
        if f.endswith('.mp4'):
            local_f = os.path.join(VIDEO_DIR_5, f)
            remote_f = f'{remote_ds5}/{f}'
            size_mb = os.path.getsize(local_f) / 1024 / 1024
            print(f'  Upload: {f} ({size_mb:.1f} MB)')
            sftp.put(local_f, remote_f)

    # Step 7: Verify
    print('\n[Step 7] Verify deployment...')
    out, _, _ = ssh_cmd(ssh, f'ls -la {REMOTE_DIR}/')
    out, _, _ = ssh_cmd(ssh, f'ls -la {REMOTE_DIR}/models/')
    out, _, _ = ssh_cmd(ssh, f'ls {REMOTE_DIR}/datasets/si/ | wc -l')
    print(f'  Dataset 四: {out.strip()} videos')
    out, _, _ = ssh_cmd(ssh, f'ls {REMOTE_DIR}/datasets/wu/ | wc -l')
    print(f'  Dataset 五: {out.strip()} videos')

    sftp.close()
    ssh.close()

    print('\n' + '=' * 60)
    print('Deployment complete!')
    print(f'Run on board: cd {REMOTE_DIR} && python3 track_and_count_npu.py --video datasets/si/1-12头.mp4 --output_dir output/test --no_timestamp')
    print('=' * 60)


if __name__ == '__main__':
    main()
