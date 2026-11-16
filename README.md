# pi-embedded-ai

Local AI manager for [pi](https://github.com/earendil-works/pi) (and the
pi-rust port). Two engines share one lifecycle, each with its own provider, so
they can run side by side:

| Engine | Shape | Provider | Command |
|--------|-------|----------|---------|
| **llamafile** | a `.llamafile` executable that *is* the model | `embedded-ai` | `/embedded-llama` |
| **cortiq** | a `cortiq` binary plus `.cmf` model files | `cortiq` | `/embedded-cortiq` |

**No weights are bundled.** Models ship separately (see [awesome-llamafiles](https://github.com/andrea-tomassi/awesome-llamafiles)
for llamafiles, any CMF distribution for cortiq) and are discovered at runtime.
This package is MIT.

## Install

From a local checkout (works fully offline):

```sh
pi install ./pi-embedded-ai/
```

## How it works

On load the extension scans, in order:

1. `PI_LLAMAFILE_DIR` / `PI_CORTIQ_DIR` (explicit override, per engine)
2. `~/.pi/agent/models/`
3. `~/models/` (one level of subdirectories)
4. the package directory itself

The first valid `.llamafile` is started (`--server --host 127.0.0.1
--port $PI_LLAMAFILE_PORT --ctx-size $PI_LLAMAFILE_CTX`, defaults 8080 and
128k) and registered as the `embedded-ai` provider. `.cmf` files are served by
the `cortiq` binary (`cortiq serve <model> --host 127.0.0.1 --port
$PI_CORTIQ_PORT`, defaults 8081, context through `CMF_MAX_SEQ`) and registered
as the `cortiq` provider.

A port held by anything else is reported, never reused or killed. Crash
leftovers are adopted through a PID file; stopping the extension (or the host)
terminates the server.

The cortiq binary is looked up as `PI_CORTIQ_BIN`, then `cortiq`, then
`cortiq-cpu` in `PATH` — the airgapped bundle installs it under the second name.

## Commands

Type `/embedded-ai` with no argument to see the status and the usage recap;
each subcommand reports its result as a notification.

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
`modelId`) and `~/.pi/agent/cortiq.json` (`port`, `ctxSize`, `modelId`, `gpu`);
the `PI_LLAMAFILE_*` / `PI_CORTIQ_*` env vars override them (the commands say so
when this happens).

### cortiq

```
/embedded-cortiq list      discovered .cmf models ([preferred], [active], gpu mode)
/embedded-cortiq status    servers: running, loading or stopped
/embedded-cortiq set <m>   start and use a model (TAB completes: lfm3-1.7b-q4tp, lfm3-1.7b-q8_2f, ...)
/embedded-cortiq start     serve the preferred or first discovered model
/embedded-cortiq stop      terminate the managed server
/embedded-cortiq ctx <n>   set context size (min 1024, passed as CMF_MAX_SEQ) and restart
/embedded-cortiq port <n>  set port (1-65535) and restart the server
/embedded-cortiq gpu <m>   default | cpu | cuda | wgpu (sets CMF_GPU) and restart
/embedded-cortiq logs [n]  last n lines of the server log (default 30)
```

`use <model>` still works as an alias of `set <model>`. Model ids are the file
stem slug (e.g. `lfm3-1.7b-q4tp`), so `/scoped-models` shows every quant as a
distinct model; requests carry that id and both engines accept any model field
on the wire (single-model servers).

`gpu cpu` is passed to the engine as `CMF_GPU=0`: the engine reads only `0` (or `off`)
as CPU and treats any other value as "use the wgpu GPU". `cuda` needs a binary built
with `--features cuda`; the bundled CPU build has no GPU backend at all.

## Settings

| Variable           | Default | Meaning                              |
|--------------------|---------|--------------------------------------|
| `PI_LLAMAFILE_DIR` | —       | extra directory to discover models in |
| `PI_LLAMAFILE_PORT`| 8080    | port the managed server listens on    |
| `PI_LLAMAFILE_CTX` | 131072  | context size (KV cache ≈ RAM use)     |
| `PI_CORTIQ_DIR`    | —       | extra directory to discover `.cmf` in |
| `PI_CORTIQ_BIN`    | —       | cortiq executable (else `cortiq`, `cortiq-cpu` from `PATH`) |
| `PI_CORTIQ_PORT`   | 8081    | port the cortiq server listens on     |
| `PI_CORTIQ_CTX`    | 131072  | context size, passed as `CMF_MAX_SEQ` |
| `PI_CORTIQ_GPU`    | default | `CMF_GPU`: `default`, `cpu`, `cuda`, `wgpu` |

## Runtime

QuickJS-compatible: only `node:child_process` (spawn), `node:fs`, `node:os`,
`node:path` and global `fetch`. No Node.js required on the host.
