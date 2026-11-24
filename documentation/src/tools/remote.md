---
title: Remote client
sources:
  - matformer/remote/__init__.py
  - matformer/remote/net.py
  - matformer/remote/ssh.py
---
# Remote client

`matformer.remote` is the client layer used by interfaces to reach one or more servers. `remote.net.Server` keeps an NDJSON-RPC connection with automatic reconnection, a queue for calls made before the connection is ready, and an event queue for server pushes.

`remote.connect(cfg)` accepts a plain `host:port` URL or an SSH configuration. In the SSH case it checks reachability, opens a local tunnel with paramiko, and starts `python -m matformer.server` on the remote side when `start` is set.

This module has no dependency on the GUI or the server implementation. Any renderer can use it. The protocol and the list of RPC methods are in [[tools/commands | Commands]].
