# matformer/remote/ssh.py
"""SSH integration for the console (optional: everything degrades gracefully without paramiko).

A saved server may describe the far side fully:

    [[server]]
    name = "gpu-box"
    [server.ssh]
    host = "matteo@10.0.0.4"        # user optional (defaults to the local one)
    port = 22
    remote_port = 8765              # where the matformer server listens on the far side
    python = "python"               # interpreter on the far side
    root = "~/Matformer/matformer"  # repo root the server starts from
    start = true                    # boot the server if nothing answers

ensure() returns the (host, port) the plain TCP client should dial: the direct address when it is
already reachable (manual `ssh -L`, LAN exposure), otherwise a freshly opened local tunnel —
after starting `python -m matformer.server` remotely when configured to. remote.net stays pure
stdlib: the tunnel just hands it a local socket."""
import os
import select
import socket
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


def _user_host(spec):
    user, _, host = spec.rpartition("@")
    return user or os.environ.get("USER", "root"), host or spec


def _connect(ssh_cfg):
    p = paramiko()
    if not p:
        raise RuntimeError("paramiko is not installed: use `ssh -L` manually or pip install paramiko")
    user, host = _user_host(ssh_cfg["host"])
    client = p.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(p.AutoAddPolicy())
    client.connect(host, port=int(ssh_cfg.get("port", 22)), username=user, timeout=10)
    return client


def start_remote(ssh_cfg, name="remote"):
    """Launch the matformer server on the far side, detached: it survives this session."""
    client = _connect(ssh_cfg)
    try:
        cmd = (f"cd {ssh_cfg.get('root', '.')} && MATFORMER_SERVER=1 nohup "
               f"{ssh_cfg.get('python', 'python')} -m matformer.server "
               f"--port {int(ssh_cfg.get('remote_port', 8765))} "
               f"> /tmp/matformer_server_{name}.log 2>&1 & echo started")
        client.exec_command(cmd, timeout=15)[1].read()
    finally:
        client.close()


class Tunnel(threading.Thread):
    """Local TCP listener forwarding to 127.0.0.1:remote_port through an SSH direct-tcpip channel."""

    def __init__(self, ssh_cfg, name="remote"):
        super().__init__(daemon=True, name=f"ssh-tunnel-{name}")
        self.remote_port = int(ssh_cfg.get("remote_port", 8765))
        self.ssh_cfg = ssh_cfg
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.local_port = self.listener.getsockname()[1]
        self.client, self.error, self._stop = None, None, threading.Event()

    def run(self):
        try:
            self.client = _connect(self.ssh_cfg)
            transport = self.client.get_transport()
        except Exception as problem:
            self.error = problem
            return
        while not self._stop.is_set():
            try:
                local, _ = self.listener.accept()
                remote = transport.open_channel("direct-tcpip", ("127.0.0.1", self.remote_port),
                                                local.getsockname(), timeout=10)
            except OSError:
                return
            threading.Thread(target=self._pump, args=(local, remote), daemon=True).start()

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
        self.listener.close()
        if self.client:
            self.client.close()


TUNNELS = {}


def ensure(ssh_cfg, name="remote"):
    """The (host, port) to dial for a server with an [ssh] block: direct when already reachable,
    otherwise through a tunnel, booting the remote server first when `start = true`."""
    if name in TUNNELS and TUNNELS[name].is_alive() and not TUNNELS[name].error:
        return "127.0.0.1", TUNNELS[name].local_port
    host, rport = _user_host(ssh_cfg["host"])[1], ssh_cfg.get("remote_port", 8765)
    if reachable(host, rport) and not ssh_cfg.get("force_tunnel", False):
        return host, int(rport)
    if ssh_cfg.get("start", True):
        start_remote(ssh_cfg, name)
    tunnel = Tunnel(ssh_cfg, name)
    tunnel.start()
    tunnel.join(12)
    if tunnel.error:
        raise RuntimeError(f"ssh tunnel to {ssh_cfg['host']}: {tunnel.error}")
    TUNNELS[name] = tunnel
    return "127.0.0.1", tunnel.local_port
