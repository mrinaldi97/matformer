# matformer/remote/__init__.py
"""The renderer-agnostic client layer of the Matformer suite.

Everything a UI (Tk today, ncurses/web tomorrow) needs to talk to one or more matformer-servers:

    remote.net.Server        NDJSON-RPC connection: call(), event queue, auto-reconnect
    remote.ssh               optional paramiko helpers: reachability probe, direct-tcpip tunnel,
                             remote server startup
    remote.connect()         the one entry point: given a server config, return a host:port that
                             is actually reachable (tunneling and/or booting the far side first)

No imports from matformer.gui and none from matformer.server: this layer speaks the wire protocol
only, so it can be reused verbatim by any renderer or script."""
from matformer.remote.net import Server  # noqa: F401


def connect(cfg, events=None):
    """cfg: {"name", "url"|"ssh": {host, user, port, remote_port, python, matformer_root,
    start: bool}} -> (host, port) the GUI should dial, after any tunnel/server bootstrap.
    Plain urls pass through untouched."""
    url = cfg.get("url")
    if not cfg.get("ssh"):
        return url.split(":")[0], int(url.rsplit(":", 1)[1])
    from matformer.remote import ssh
    return ssh.ensure(cfg["ssh"], cfg.get("name", "remote"))
