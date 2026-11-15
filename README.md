# pi-embedded-ai

Local AI manager (llamafile-based) for [pi](https://github.com/earendil-works/pi)
(and the pi-rust port). Discovers `.llamafile` executables on the host, serves
them as a local OpenAI-compatible provider, and registers the selected one as
the out-of-the-box local model.

**No weights are bundled.** Models ship separately (see [awesome-llamafiles](https://github.com/andrea-tomassi/awesome-llamafiles))
and are discovered at runtime. This package is MIT.

## Install

From a local checkout (works fully offline):

```sh
pi install ./pi-embedded-ai/
```

## How it works

On load the extension scans, in order:

1. `PI_LLAMAFILE_DIR` (explicit override)
2. `~/.pi/agent/models/`
3. `~/models/` (one level of subdirectories)
4. the package directory itself

The first valid `.llamafile` is started (`--server --host 127.0.0.1
--port $PI_LLAMAFILE_PORT --ctx-size $PI_LLAMAFILE_CTX`, defaults 8080 and
128k) and registered as the `embedded-ai` provider. A port held by anything else
is reported, never reused or killed. Crash leftovers are adopted through a PID
file; stopping the extension (or the host) terminates the server.

## Commands

```
/embedded-ai list          discovered models ([preferred], [active])
/embedded-ai status        servers: running, loading or stopped (all discovered models)
/embedded-ai start [name] serve a model (default: preferred or first discovered)
/embedded-ai use <name>    switch to a model (restarts the server, remembers the choice)
/embedded-ai stop          terminate the managed server
/embedded-ai ctx <n>       set context size (min 1024) and restart the server
/embedded-ai port <n>      set port (1-65535) and restart the server
/embedded-ai logs [n]      last n lines of the server log (default 30)
```

Settings persist in `~/.pi/agent/llamafile.json` (`port`, `ctxSize`,
`modelId`); `PI_LLAMAFILE_PORT`, `PI_LLAMAFILE_CTX`, `PI_LLAMAFILE_MODEL`
env vars override them (the commands say so when this happens).

## Settings

| Variable           | Default | Meaning                              |
|--------------------|---------|--------------------------------------|
| `PI_LLAMAFILE_DIR` | —       | extra directory to discover models in |
| `PI_LLAMAFILE_PORT`| 8080    | port the managed server listens on    |
| `PI_LLAMAFILE_CTX` | 131072  | context size (KV cache ≈ RAM use)     |

## Runtime

QuickJS-compatible: only `node:child_process` (spawn), `node:fs`, `node:os`,
`node:path` and global `fetch`. No Node.js required on the host.
