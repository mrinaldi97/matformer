# matformer/remote/ssh.py
"""SSH integration for the console.

Two backends: paramiko when it is installed, otherwise the system `ssh` binary. The fallback keeps
the automatic launch working on machines that cannot install Python packages (an externally managed
Python is the normal case). `ensure()` returns the (host, port) the plain TCP client should dial:
the direct address when it is already reachable, otherwise a local tunnel after starting
`python -m matformer.server` on the far side when configured to.
"""
import os
import re
import select
import shlex
import shutil
import socket
import subprocess
import threading
import time

_SSH = None


def paramiko():
    global _SSH
    if _SSH is None:
        try:
            import paramiko
            _SSH = paramiko
        except ImportError:
            _SSH = False
    return _SSH


def reachable(host, port, timeout=1.5):
    try:
        socket.create_connection((host, int(port)), timeout=timeout).close()
        return True
    except OSError:
        return False


def is_matformer(host, port, timeout=2.0):
    """True when the TCP service answers the Matformer RPC line protocol."""
    import json
    try:
        with socket.create_connection((host, int(port)), timeout=timeout) as sock:
            sock.sendall(b'{"id":1,"method":"rpc.list","params":{}}\n')
            sock.settimeout(timeout)
            line = sock.makefile("rb").readline()
        message = json.loads(line)
        return isinstance(message.get("result"), list)
    except Exception:
        return False


def _user_host(spec):
    user, _, host = spec.rpartition("@")
    return user or os.environ.get("USER", "root"), host or spec


def _ssh_args(ssh_cfg):
    user, host = _user_host(ssh_cfg["host"])
    args = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=15"]
    if ssh_cfg.get("key"):
        args += ["-i", os.path.expanduser(ssh_cfg["key"])]
    if ssh_cfg.get("port"):
        args += ["-p", str(int(ssh_cfg["port"]))]
    args.append(f"{user}@{host}")
    return args


def _connect(ssh_cfg):
    p = paramiko()
    if not p:
        raise RuntimeError("no paramiko and no usable ssh command")
    user, host = _user_host(ssh_cfg["host"])
    client = p.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(p.AutoAddPolicy())
    client.connect(host, port=int(ssh_cfg.get("port", 22)), username=user,
                   key_filename=os.path.expanduser(ssh_cfg["key"]) if ssh_cfg.get("key") else None,
                   timeout=10)
    return client


def _remote_command(ssh_cfg, name="remote"):
    root = ssh_cfg.get("root", ".")
    python = ssh_cfg.get("python", "python3")
    preferred = int(ssh_cfg.get("remote_port", 8765))
    log = f"/tmp/matformer_server_{name}.log"
    inner = (
        "import os, socket, subprocess; "
        f"os.chdir({root!r}); "
        "probe = socket.socket(); "
        "probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
        "preferred = " + str(preferred) + "; "
        "\ntry:\n"
        "    probe.bind(('127.0.0.1', preferred)); probe.close(); port = preferred\n"
        "except OSError:\n"
        "    probe = socket.socket(); probe.bind(('127.0.0.1', 0)); "
        "port = probe.getsockname()[1]; probe.close()\n"
        f"env = dict(os.environ, MATFORMER_SERVER='1'); "
        f"env.setdefault('MATFORMER_ASSETS', {ssh_cfg.get('assets', '')!r}); "
        f"subprocess.Popen([{python!r}, '-m', 'matformer.server', '--port', str(port)], "
        "stdin=open('/dev/null', 'rb'), "
        f"stdout=open({log!r}, 'ab'), stderr=subprocess.STDOUT, "
        "env=env, start_new_session=True); print('PORT=' + str(port))"
    )
    return f"{shlex.quote(python)} -c {shlex.quote(inner)}"


def start_remote(ssh_cfg, name="remote"):
    """Launch the matformer server on the far side, detached. Returns the port it bound.

    A busy preferred port is not an error: the remote helper binds a free one and reports it.
    """
    command = _remote_command(ssh_cfg, name)
    output = ""
    if paramiko():
        client = _connect(ssh_cfg)
        try:
            stdin, stdout, stderr = client.exec_command(command, timeout=20)
            output = stdout.read().decode("utf-8", "replace")
            error = stderr.read().decode("utf-8", "replace").strip()
            if error and "PORT=" not in output:
                raise RuntimeError(error)
        finally:
            client.close()
    else:
        if not shutil.which("ssh"):
            raise RuntimeError("neither paramiko nor the ssh command is available")
        result = subprocess.run(_ssh_args(ssh_cfg) + [command],
                                capture_output=True, text=True, timeout=20)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip() or "ssh start failed")
        output = result.stdout
    match = re.search(r"PORT=(\d+)", output or "")
    return int(match.group(1)) if match else int(ssh_cfg.get("remote_port", 8765))


def _free_port():
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
    finally:
        probe.close()


class Tunnel(threading.Thread):
    """Local TCP listener forwarding to 127.0.0.1:remote_port through SSH."""

    def __init__(self, ssh_cfg, name="remote"):
        super().__init__(daemon=True, name=f"ssh-tunnel-{name}")
        self.remote_port = int(ssh_cfg.get("remote_port", 8765))
        self.ssh_cfg = ssh_cfg
        self.name = name
        self.local_port = _free_port()
        self.client, self.proc, self.error, self._stop = None, None, None, threading.Event()

    def run(self):
        if paramiko():
            self._run_paramiko()
        else:
            self._run_external()

    def _run_paramiko(self):
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", self.local_port))
        listener.listen(8)
        try:
            self.client = _connect(self.ssh_cfg)
            transport = self.client.get_transport()
        except Exception as problem:
            self.error = problem
            listener.close()
            return
        while not self._stop.is_set():
            try:
                local, _ = listener.accept()
                remote = transport.open_channel("direct-tcpip", ("127.0.0.1", self.remote_port),
                                                local.getsockname(), timeout=10)
            except OSError:
                break
            threading.Thread(target=self._pump, args=(local, remote), daemon=True).start()
        listener.close()

    def _run_external(self):
        if not shutil.which("ssh"):
            self.error = RuntimeError("the ssh command is not available")
            return
        log = open(f"/tmp/matformer_tunnel_{self.name}.log", "ab", buffering=0)
        args = _ssh_args(self.ssh_cfg) + ["-N", "-L",
                                          f"{self.local_port}:127.0.0.1:{self.remote_port}"]
        self.proc = subprocess.Popen(args, stdout=log, stderr=log)
        deadline = time.time() + 15
        while time.time() < deadline and not self._stop.is_set():
            if reachable("127.0.0.1", self.local_port, timeout=0.5):
                break
            if self.proc.poll() is not None:
                self.error = RuntimeError(f"ssh tunnel exited: see /tmp/matformer_tunnel_{self.name}.log")
                return
            time.sleep(0.2)
        else:
            self.error = RuntimeError(f"ssh tunnel did not listen: see /tmp/matformer_tunnel_{self.name}.log")
            return
        self.proc.wait()

    @staticmethod
    def _pump(a, b):
        try:
            while True:
                r, _, _ = select.select([a, b], [], [], 1)
                for src in r:
                    data = src.recv(65536)
                    if not data:
                        return
                    (b if src is a else a).sendall(data)
        except OSError:
            pass
        finally:
            for s in (a, b):
                try:
                    s.close()
                except OSError:
                    pass

    def close(self):
        self._stop.set()
        if self.client:
            self.client.close()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


TUNNELS = {}


def ensure(ssh_cfg, name="remote"):
    """The (host, port) to dial for a server with an [ssh] block: direct when already reachable,
    otherwise through a tunnel, booting the remote server first when `start = true`."""
    if name in TUNNELS and TUNNELS[name].is_alive() and not TUNNELS[name].error:
        return "127.0.0.1", TUNNELS[name].local_port
    host, rport = _user_host(ssh_cfg["host"])[1], int(ssh_cfg.get("remote_port", 8765))
    if not ssh_cfg.get("force_tunnel", False) and is_matformer(host, rport):
        return host, rport
    if ssh_cfg.get("start", True):
        # Reuse a server that is already listening on the preferred port before spawning a new one.
        probe = Tunnel(ssh_cfg, name + "-probe")
        probe.start()
        probe.join(10)
        if not probe.error and is_matformer("127.0.0.1", probe.local_port):
            TUNNELS[name] = probe
            return "127.0.0.1", probe.local_port
        probe.close()
        rport = start_remote(ssh_cfg, name)
    tunnel = Tunnel({**ssh_cfg, "remote_port": rport}, name)
    tunnel.start()
    tunnel.join(15)
    if tunnel.error:
        raise RuntimeError(f"ssh tunnel to {ssh_cfg['host']}: {tunnel.error}")
    TUNNELS[name] = tunnel
    return "127.0.0.1", tunnel.local_port
