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

## Command line

For scripts and agents, `matformer.remote.cli` is a synchronous one-shot client over the same protocol:

```bash
python -m matformer.remote.cli rpc.list
python -m matformer.remote.cli task.state
python -m matformer.remote.cli task.run --out /tmp/run.json
python -m matformer.remote.cli runs.get '{"run":"project/run","db":"/tmp/run.db"}' --select metrics
python -m matformer.remote.cli train.task '{"id":"job-120000","type":"static_model_analysis"}'
```

`--select` walks slash-separated keys and returns one value, which avoids dumping a large summary;
`--out` writes the complete answer. The client handles a closed pipe (`| head`) as a normal exit.
