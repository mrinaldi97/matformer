---
title: Server
sources:
  - matformer/server/__init__.py
  - matformer/server/core.py
  - matformer/server/server.toml
---
# Server

Matformer starts a small server when the package is imported. It speaks line-delimited JSON over TCP: one object per line, with a request id, a method, and a parameters object. The server answers with a result or an error, and can push events on the same connection.

The default address is `127.0.0.1:8765`. The server is meant for a local GUI or for an SSH tunnel; it is not a public service. It is disabled with `MATFORMER_SERVER=0` and the port is overridden with `MATFORMER_SERVER_PORT`.

The configuration is in `matformer/server/server.toml`: host, port, name, API key, browsable roots, run database roots and the favourites file. A missing value keeps its default. When `MATFORMER_ASSETS` is configured, its root is added as the first browsable root and its `checkpoints` and `experiments` folders are added to the run roots.

Authentication is optional. If `api_key` is set, the first line from a client must be `{"auth": "<key>"}`; other messages are rejected until the key is accepted.

The server RPC commands are listed in [[tools/commands | Commands]]. The [[tools/gui | GUI]] and the [[tools/remote | remote client]] use the same protocol.
