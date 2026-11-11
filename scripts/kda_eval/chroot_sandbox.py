"""Root-only fallback for containers that prohibit namespace creation.

Agents run in a chroot containing only a toolchain snapshot and the allowed
task workspace, as a dedicated unprivileged UID with no capabilities, no-new-
privileges, and a seccomp ban on Internet sockets. No host /proc is mounted.
System files are root-owned hard links, never writable by the agent. Temporary
files and all processes belonging to the sandbox UID are cleared per tool call.
"""
import ctypes
import errno
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / ".cache/kda_chroot_base"
VERSION = "6"


def _link_or_copy(source, dest):
    try:
        return os.link(source, dest)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        return shutil.copy2(source, dest)


def _copy_tree(source, dest):
    source, dest = Path(source), Path(dest)
    if source.is_symlink():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(os.readlink(source))
    elif source.is_dir():
        shutil.copytree(source, dest, copy_function=_link_or_copy, symlinks=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        _link_or_copy(source, dest)


def prepare_base():
    if os.geteuid() != 0:
        raise RuntimeError("chroot sandbox requires root with CAP_SYS_CHROOT/CAP_MKNOD")
    BASE.parent.mkdir(exist_ok=True)
    with (BASE.parent / "kda_chroot_base.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = BASE / ".prepared"
        if marker.exists() and marker.read_text() == VERSION:
            return
        if BASE.exists():
            shutil.rmtree(BASE)
        BASE.mkdir()
        # Exclude /usr/local Python packages and all user-controlled directories.
        for name in ["bin", "sbin", "lib", "lib32", "lib64", "libx32", "libexec", "include", "share"]:
            source = Path("/usr") / name
            if source.exists():
                _copy_tree(source, BASE / "usr" / name)
        toolkit = Path("/usr/local/cuda").resolve()
        _copy_tree(toolkit, BASE / str(toolkit).lstrip("/"))
        if Path("/opt/nvidia/nsight-compute").exists():
            _copy_tree("/opt/nvidia/nsight-compute", BASE / "opt/nvidia/nsight-compute")
        (BASE / "usr/local/cuda").symlink_to(toolkit.name)
        for name in ["bin", "sbin", "lib", "lib32", "lib64", "libx32"]:
            (BASE / name).symlink_to("usr/" + name)
        for name in ["dev", "proc", "sys", "tmp", "var/tmp", "home/agent", "run/kh", "etc", "opt"]:
            (BASE / name).mkdir(parents=True, exist_ok=True)
        for name in ["ld.so.cache", "ld.so.conf", "ld.so.conf.d", "alternatives"]:
            source = Path("/etc") / name
            if source.exists():
                _copy_tree(source, BASE / "etc" / name)
        (BASE / "etc/passwd").write_text("root:x:0:0:root:/root:/bin/bash\nagent:x:61090:61090:agent:/home/agent:/bin/bash\n")
        (BASE / "etc/group").write_text("root:x:0:\nagent:x:61090:\n")
        # A small read-only snapshot of driver discovery files; never host PIDs.
        proc_driver = Path("/proc/driver/nvidia")
        for source in [proc_driver / "version", proc_driver / "params",
                       proc_driver / "registry", *proc_driver.glob("gpus/*/information")]:
            if source.exists():
                dest = BASE / "proc/driver/nvidia" / source.relative_to(proc_driver)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(source.read_bytes())
        for source in ["/proc/cpuinfo","/proc/stat","/proc/meminfo","/proc/sys/vm/mmap_min_addr",
                       "/sys/devices/system/cpu/online"]:
            dest=BASE/source.lstrip("/");dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(Path(source).read_bytes())
        compat=BASE/"opt/kda_compat";compat.mkdir()
        subprocess.run(["gcc","-shared","-fPIC","-O2",str(HERE/"proc_compat.c"),
                        "-o",str(compat/"libproc_compat.so"),"-ldl"],check=True)
        marker.write_text(VERSION)


def prepare(spec):
    """Called before the task clock starts; return its chroot path and UID."""
    prepare_base()
    root = Path(spec.workspace).parent / ".sandbox_root"
    uid = 61090 + spec.gpu_index
    if not root.exists():
        shutil.copytree(BASE, root, copy_function=os.link, symlinks=True)
        workspace = Path(spec.workspace)
        workspace.rename(root / "workspace")
        workspace.symlink_to(".sandbox_root/workspace")
        for path in (root / "workspace").rglob("*"):
            if not path.is_symlink():
                os.chown(path, uid, uid)
        os.chown(root / "workspace", uid, uid)
        for source, target in [(spec.opt_kh, "/opt/kh"), *spec.ro_binds]:
            _copy_tree(source, root / target.lstrip("/"))
        cutlass = ROOT / "reference_sources/cutlass"
        for name in ["include", "tools/util/include", "examples/common"]:
            _copy_tree(cutlass / name, root / "opt/cutlass" / name)
        devices = ["/dev/null", "/dev/zero", "/dev/random", "/dev/urandom",
                   "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools",
                   f"/dev/nvidia{spec._minor}"]
        for device in devices:
            source = Path(device)
            if source.exists():
                info = source.stat()
                os.mknod(root / device.lstrip("/"), stat.S_IFCHR | 0o666, info.st_rdev)
                os.chmod(root / device.lstrip("/"), 0o666)
    return root, uid


def reset_tool(spec):
    root, uid = prepare(spec)
    for relative in ["tmp", "var/tmp", "home/agent"]:
        path = root / relative
        shutil.rmtree(path)
        path.mkdir()
        os.chown(path, uid, uid)
    source = Path(spec.sock_dir) / "judge.sock"
    dest = root / "run/kh/judge.sock"
    if source.exists() and not dest.exists():
        os.chown(source, 0, uid)
        os.chmod(source, 0o660)
        os.link(source, dest)
    return root, uid


def kill_processes(uid,exclude=None):
    # Dedicated UID has no host account and is used by one serialized GPU worker.
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if entry.stat().st_uid == uid and int(entry.name)!=exclude:
                    os.kill(int(entry.name), signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError):
                pass


def cleanup(spec):
    with (BASE.parent/f"kda_sandbox_gpu{spec.gpu_index}.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _cleanup(spec)


def _cleanup(spec):
    root = Path(spec.workspace).parent / ".sandbox_root"
    if root.exists():
        kill_processes(61090 + spec.gpu_index)
        Path(spec.workspace).unlink()
        (root / "workspace").rename(spec.workspace)
        shutil.rmtree(root)


def run(spec, command, stdin, timeout_s, env):
    BASE.parent.mkdir(exist_ok=True)
    with (BASE.parent/f"kda_sandbox_gpu{spec.gpu_index}.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _run(spec,command,stdin,timeout_s,env)


def _run(spec, command, stdin, timeout_s, env):
    root, uid = reset_tool(spec)
    argv = [sys.executable, str(Path(__file__).resolve()), "--exec", str(root), str(uid),
            json.dumps(env), *command]
    if os.environ.get("KH_SANDBOX_TRACE"):
        argv = ["strace","-f","-o",os.environ["KH_SANDBOX_TRACE"],*argv]
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, start_new_session=True,
                            env={"PATH":"/usr/bin:/bin","LANG":"C.UTF-8",
                                 "KH_PROC_TRACE":os.environ.get("KH_PROC_TRACE","")})
    timed_out = False
    try:
        try:
            out, err = proc.communicate(stdin.encode(), timeout=max(timeout_s, 0.1))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            kill_processes(uid)
            out, err = proc.communicate(timeout=10)
        return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), timed_out
    finally:
        kill_processes(uid)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def _exec(root, uid, env, command):
    proxy=Path(root)/"run/kh/proc.sock"
    if proxy.exists():proxy.unlink()
    server=socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET)
    previous_cwd=os.getcwd()
    try:
        os.chdir(proxy.parent)
        server.bind(proxy.name)  # sockaddr_un is limited to 108 bytes; task paths can be longer.
    finally:
        os.chdir(previous_cwd)
    os.chown(proxy,uid,uid);os.chmod(proxy,0o600)
    server.listen(32);server.settimeout(0.1)
    child=os.fork()
    if child:
        os.setgroups([]);os.setgid(uid);os.setuid(uid)
        try:
            while True:
                ended,status=os.waitpid(child,os.WNOHANG)
                if ended:
                    kill_processes(uid,exclude=os.getpid())
                    raise SystemExit(os.waitstatus_to_exitcode(status))
                try:conn,_=server.accept()
                except socket.timeout:continue
                with conn:
                    try:
                        conn.settimeout(1)
                        peer,peer_uid,_=struct.unpack("3i",conn.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
                        request=conn.recv(512).decode()
                        link=request.startswith("LINK:")
                        path=request[5:] if link else request
                        parts=path.split("/")
                        if peer_uid!=uid or len(parts)!=4 or parts[1]!="proc":raise ValueError("bad proc request")
                        pid=peer if parts[2]=="self" else int(parts[2])
                        if Path(f"/proc/{pid}").stat().st_uid!=uid:raise ValueError("foreign uid")
                        name=parts[3]
                        if name not in ["maps","status","stat","exe","cwd"]:raise ValueError("not allowed")
                        source=Path(f"/proc/{pid}/{name}")
                        if link:
                            data=os.readlink(source).removeprefix(root).encode()
                        else:
                            if name not in ["maps","status","stat"]:raise ValueError("not readable")
                            data=source.read_bytes().replace(root.encode(),b"")
                        fd=os.memfd_create("sandbox-proc",os.MFD_CLOEXEC)
                        try:
                            os.write(fd,data);os.lseek(fd,0,os.SEEK_SET)
                            conn.sendmsg([b"F"],[(socket.SOL_SOCKET,socket.SCM_RIGHTS,struct.pack("i",fd))])
                        finally:os.close(fd)
                    except Exception as error:
                        if os.environ.get("KH_PROC_TRACE"):
                            print(f"proc broker: {request}: {type(error).__name__}: {error}",file=sys.stderr,flush=True)
                        try:conn.send(b"E")
                        except OSError:pass
        finally:server.close()
    server.close()
    env=dict(env,LD_PRELOAD="/opt/kda_compat/libproc_compat.so")
    libc = ctypes.CDLL(None, use_errno=True)
    seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    class Compare(ctypes.Structure):
        _fields_ = [("arg",ctypes.c_uint),("op",ctypes.c_int),
                    ("datum_a",ctypes.c_uint64),("datum_b",ctypes.c_uint64)]
    context = seccomp.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("seccomp_init failed")
    deny = 0x00050000 | errno.EPERM
    socket_nr = seccomp.seccomp_syscall_resolve_name(b"socket")
    for family in [2, 10]:  # AF_INET / AF_INET6; AF_UNIX judge RPC is permitted.
        if seccomp.seccomp_rule_add(context, deny, socket_nr, 1, Compare(0,4,family,0)) != 0:
            raise RuntimeError("Cannot install network filter")
    for name in [b"io_uring_setup",b"io_uring_enter",b"io_uring_register",
                 b"ptrace",b"process_vm_readv",b"process_vm_writev",b"bpf"]:
        nr = seccomp.seccomp_syscall_resolve_name(name)
        if nr >= 0 and seccomp.seccomp_rule_add(context, deny, nr, 0) != 0:
            raise RuntimeError("Cannot install process filter")
    os.chroot(root)
    os.chdir("/workspace")
    os.setgroups([])
    os.setgid(uid)
    os.setuid(uid)
    if libc.prctl(4,1,0,0,0) != 0:  # PR_SET_DUMPABLE: trusted parent reads procfs.
        raise OSError(ctypes.get_errno(), "dumpable")
    if libc.prctl(38,1,0,0,0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "no_new_privs")
    if seccomp.seccomp_load(context) != 0:
        raise RuntimeError("seccomp_load failed")
    seccomp.seccomp_release(context)
    os.umask(0o022)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    if len(sys.argv) > 5 and sys.argv[1] == "--exec":
        _exec(sys.argv[2], int(sys.argv[3]), json.loads(sys.argv[4]), sys.argv[5:])
    else:
        prepare_base()
