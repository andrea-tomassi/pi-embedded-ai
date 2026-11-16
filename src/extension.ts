// pi-embedded-ai/src/extension.ts
//
// Local AI manager for pi. Two engines are served through one lifecycle:
//   - llamafile: a `.llamafile` executable that *is* the model (`--server`);
//   - cortiq: a `cortiq` binary driving `.cmf` model files (`cortiq serve`).
// Each engine registers its own OpenAI-compatible provider, so both can run at once.
// Weights ship separately (awesome-llamafiles for llamafiles, CMF files for cortiq)
// and are discovered at runtime.
//
// Runs on QuickJS as well as Node: only `node:child_process` (spawn), `node:fs`,
// `node:os`, `node:path` and global `fetch` are used.

import { spawn } from "node:child_process";
import { existsSync, readdirSync, readFileSync, readlinkSync, statSync, unlinkSync, writeFileSync, mkdirSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join } from "node:path";

const MANAGER_ID = "pi-embedded-ai";
const HEALTH_TIMEOUT_MS = 180000;

type ExtensionAPI = any;
type EngineId = "llamafile" | "cortiq";

interface ManagedModel {
  id: string;
  path: string;
  engine: EngineId;
}

interface ServerState {
  pid: number | null;
  port: number;
  modelId: string;
  path: string;
}

interface Settings {
  port: number;
  ctxSize: number;
  modelId: string | null;
  /** cortiq only: `default`, `cpu` or `cuda`, mapped to CMF_GPU. */
  gpu: string;
}

interface EngineSpec {
  id: EngineId;
  /** Provider id registered in models.json and with the host. */
  providerId: string;
  /** Older provider keys this engine may have used, migrated on first write. */
  legacyProviderIds: string[];
  providerName: string;
  /** File extension that marks a model of this engine. */
  extension: string;
  /** Health endpoint served by the engine (`/health` vs `/healthz`). */
  healthPath: string;
  settingsFile: string;
  /** Env prefix for the settings overrides (`PI_LLAMAFILE` / `PI_CORTIQ`). */
  envPrefix: string;
  defaultPort: number;
  defaultCtx: number;
  /** Whether `ctx` maps to a server flag or to the CMF_MAX_SEQ env var. */
  ctxIsEnv: boolean;
  label: string;
}

const ENGINES: Record<EngineId, EngineSpec> = {
  llamafile: {
    id: "llamafile",
    providerId: "embedded-ai",
    legacyProviderIds: ["llamafile"],
    providerName: "embedded AI (local)",
    extension: ".llamafile",
    healthPath: "/health",
    settingsFile: "llamafile.json",
    envPrefix: "PI_LLAMAFILE",
    defaultPort: 8080,
    defaultCtx: 131072,
    ctxIsEnv: false,
    label: "llamafile",
  },
  cortiq: {
    id: "cortiq",
    providerId: "cortiq",
    legacyProviderIds: [],
    providerName: "cortiq (local)",
    extension: ".cmf",
    healthPath: "/healthz",
    settingsFile: "cortiq.json",
    envPrefix: "PI_CORTIQ",
    defaultPort: 8081,
    defaultCtx: 131072,
    ctxIsEnv: true,
    label: "cortiq",
  },
};

const ENGINE_IDS: EngineId[] = ["llamafile", "cortiq"];

const active: Record<EngineId, ServerState | null> = { llamafile: null, cortiq: null };
let packageDir = "";
let api: ExtensionAPI | null = null;

/** Serialize lifecycle operations: auto-start and commands never race each other. */
let busy: Promise<unknown> = Promise.resolve();
function serialize<T>(fn: () => Promise<T>): Promise<T> {
  const run = busy.then(fn, fn);
  busy = run.catch(() => {});
  return run;
}

function log(message: string) {
  console.error(`[pi-embedded-ai] ${message}`);
}

function envName(engine: EngineSpec, suffix: string): string {
  return `${engine.envPrefix}_${suffix}`;
}

// ---------------------------------------------------------------------------
// Discovery: `.llamafile` executables and `.cmf` model files under the same roots.
// ---------------------------------------------------------------------------

/** Candidate roots, in priority order. */
function searchRoots(engine: EngineSpec): string[] {
  const home = homedir();
  const roots: string[] = [];
  const fromEnv = process.env[envName(engine, "DIR")];
  if (fromEnv) roots.push(fromEnv);
  roots.push(join(home, ".pi", "agent", "models"));
  roots.push(join(home, "models"));
  roots.push(packageDir);
  return roots.filter((root) => {
    try {
      return statSync(root).isDirectory();
    } catch {
      return false;
    }
  });
}

/** Non-recursive scan of `dir` plus one level of subdirectories for the engine's extension. */
function scanDir(engine: EngineSpec, dir: string, depth: number): ManagedModel[] {
  const found: ManagedModel[] = [];
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return found;
  }
  for (const entry of entries) {
    const full = join(dir, entry);
    let stat;
    try {
      stat = statSync(full);
    } catch {
      continue;
    }
    if (stat.isFile() && entry.endsWith(engine.extension)) {
      found.push({
        id: entry.slice(0, entry.length - engine.extension.length),
        path: full,
        engine: engine.id,
      });
    } else if (stat.isDirectory() && depth > 0 && !entry.startsWith(".")) {
      found.push(...scanDir(engine, full, depth - 1));
    }
  }
  return found;
}

export function discoverModels(engineId: EngineId = "llamafile"): ManagedModel[] {
  const engine = ENGINES[engineId];
  const seen = new Set<string>();
  const models: ManagedModel[] = [];
  for (const root of searchRoots(engine)) {
    for (const model of scanDir(engine, root, 1)) {
      if (!seen.has(model.path)) {
        seen.add(model.path);
        models.push(model);
      }
    }
  }
  return models;
}

/**
 * The cortiq executable. The airgapped bundle installs it as `cortiq-cpu`, so both names are
 * tried before giving up; `PI_CORTIQ_BIN` wins over everything.
 */
export function cortiqBinary(): string | null {
  const explicit = process.env["PI_CORTIQ_BIN"];
  if (explicit && existsSync(explicit)) return explicit;
  const pathDirs = (process.env["PATH"] ?? "").split(":").filter(Boolean);
  for (const name of ["cortiq", "cortiq-cpu"]) {
    for (const dir of pathDirs) {
      const candidate = join(dir, name);
      if (existsSync(candidate)) return candidate;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Settings: ~/.pi/agent/<engine>.json, overridden by PI_<ENGINE>_* env.
// ---------------------------------------------------------------------------

function settingsPath(engine: EngineSpec): string {
  return join(homedir(), ".pi", "agent", engine.settingsFile);
}

function readSettingsFile(engine: EngineSpec): any {
  try {
    return JSON.parse(readFileSync(settingsPath(engine), "utf8"));
  } catch {
    return {};
  }
}

function num(envValue: string | undefined, fallback: number, min: number, max = Number.MAX_SAFE_INTEGER): number {
  const parsed = envValue ? parseInt(envValue, 10) : NaN;
  if (Number.isInteger(parsed) && parsed >= min && parsed <= max) return parsed;
  return fallback;
}

export function settings(engineId: EngineId = "llamafile"): Settings {
  const engine = ENGINES[engineId];
  const file = readSettingsFile(engine);
  const port = num(
    process.env[envName(engine, "PORT")],
    num(String(file.port ?? ""), engine.defaultPort, 1, 65535),
    1,
    65535,
  );
  const ctxSize = num(
    process.env[envName(engine, "CTX")],
    num(String(file.ctxSize ?? ""), engine.defaultCtx, 1024),
    1024,
  );
  const modelId =
    process.env[envName(engine, "MODEL")] ??
    (typeof file.modelId === "string" && file.modelId.length > 0 ? file.modelId : null);
  const gpu = process.env[envName(engine, "GPU")] ?? (typeof file.gpu === "string" ? file.gpu : "default");
  return { port, ctxSize, modelId, gpu };
}

export function saveSettings(patch: Partial<Settings>, engineId: EngineId = "llamafile"): Settings {
  const engine = ENGINES[engineId];
  const current = settings(engineId);
  const next = { ...current, ...patch };
  // Persist only what env does not override.
  const file: any = {};
  if (!process.env[envName(engine, "PORT")]) file.port = next.port;
  if (!process.env[envName(engine, "CTX")]) file.ctxSize = next.ctxSize;
  if (!process.env[envName(engine, "MODEL")] && next.modelId) file.modelId = next.modelId;
  if (!process.env[envName(engine, "GPU")] && engine.id === "cortiq" && next.gpu) file.gpu = next.gpu;
  try {
    mkdirSync(dirname(settingsPath(engine)), { recursive: true });
    writeFileSync(settingsPath(engine), JSON.stringify(file, null, 2) + "\n");
  } catch (error) {
    log(`could not save settings: ${error instanceof Error ? error.message : String(error)}`);
  }
  return next;
}

// ---------------------------------------------------------------------------
// Server lifecycle.
// ---------------------------------------------------------------------------

async function healthy(engine: EngineSpec, baseUrl: string): Promise<boolean> {
  try {
    const response = await fetch(`${baseUrl}${engine.healthPath}`, { signal: AbortSignal.timeout(3000) });
    return response.ok;
  } catch {
    return false;
  }
}

async function waitForHealth(engine: EngineSpec, baseUrl: string, onProgress: (message: string) => void): Promise<void> {
  const started = Date.now();
  let announced = false;
  while (Date.now() - started < HEALTH_TIMEOUT_MS) {
    if (await healthy(engine, baseUrl)) return;
    if (!announced && Date.now() - started > 5000) {
      announced = true;
      onProgress(`the ${engine.label} model is loading; this can take a minute`);
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`the ${engine.label} server did not become healthy within ${HEALTH_TIMEOUT_MS / 1000}s`);
}

function pidFile(engine: EngineSpec, modelId: string): string {
  return join(tmpdir(), `pi-embedded-ai-${engine.id}-${modelId}.pid`);
}

function logFile(engine: EngineSpec, modelId: string): string {
  return join(homedir(), ".pi", "agent", "models", `${engine.id}-${modelId}.log`);
}

function readPid(engine: EngineSpec, modelId: string): number | null {
  try {
    const raw = readFileSync(pidFile(engine, modelId), "utf8").trim();
    const pid = parseInt(raw, 10);
    return Number.isInteger(pid) && pid > 0 ? pid : null;
  } catch {
    return null;
  }
}

function processMatchesCmdline(pid: number, expectedPath: string): boolean {
  try {
    const cmdline = readFileSync(`/proc/${pid}/cmdline`, "utf8").replace(/\0/g, " ");
    return cmdline.includes(expectedPath);
  } catch {
    return false;
  }
}

function clearPid(engine: EngineSpec, modelId: string): void {
  try {
    unlinkSync(pidFile(engine, modelId));
  } catch {
    // Already gone.
  }
}

/**
 * The PID of the process listening on `port`, resolved from `/proc` (Linux). A llamafile run
 * through `sh` re-execs itself through the APE loader, so the spawned PID is not always the
 * socket owner: the PID file must record this one for `stop` to reach the real server.
 */
function resolveListenerPid(port: number): number | null {
  const hexPort = port.toString(16).toUpperCase().padStart(4, "0");
  let inode: string | null = null;
  for (const file of ["/proc/net/tcp", "/proc/net/tcp6"]) {
    let text: string;
    try {
      text = readFileSync(file, "utf8");
    } catch {
      continue;
    }
    for (const line of text.split("\n").slice(1)) {
      const fields = line.trim().split(/\s+/);
      if (fields.length < 10) continue;
      const local = fields[1] ?? "";
      if (!local.endsWith(`:${hexPort}`)) continue;
      if (fields[3] !== "0A") continue; // 0A == LISTEN
      inode = fields[9] ?? null;
      break;
    }
    if (inode) break;
  }
  if (!inode) return null;
  let pids: string[];
  try {
    pids = readdirSync("/proc");
  } catch {
    return null;
  }
  for (const pid of pids) {
    if (!/^\d+$/.test(pid)) continue;
    let fds: string[];
    try {
      fds = readdirSync(`/proc/${pid}/fd`);
    } catch {
      continue;
    }
    for (const fd of fds) {
      try {
        if (readlinkSync(`/proc/${pid}/fd/${fd}`) === `socket:[${inode}]`) return Number(pid);
      } catch {
        // fd closed while scanning.
      }
    }
  }
  return null;
}

/** Exclusive cross-process claim on (engine, model, port), so two sessions cannot both spawn. */
function lockFile(engine: EngineSpec, modelId: string, port: number): string {
  return join(tmpdir(), `pi-embedded-ai-${engine.id}-${modelId}-${port}.lock`);
}

function tryLock(engine: EngineSpec, modelId: string, port: number): boolean {
  try {
    writeFileSync(lockFile(engine, modelId, port), String(process.pid), { flag: "wx" });
    return true;
  } catch {
    // A hard-killed session leaves its lock behind; a lock whose owner is gone is not a lock.
    try {
      const owner = parseInt(readFileSync(lockFile(engine, modelId, port), "utf8").trim(), 10);
      if (Number.isInteger(owner) && owner > 0 && !processAlive(owner)) {
        unlinkSync(lockFile(engine, modelId, port));
        writeFileSync(lockFile(engine, modelId, port), String(process.pid), { flag: "wx" });
        return true;
      }
    } catch {
      // Someone else won the retry: fall through and wait for them.
    }
    return false;
  }
}

function processAlive(pid: number): boolean {
  // /proc is already the source of truth for every other check in this file.
  try {
    return statSync(`/proc/${pid}`).isDirectory();
  } catch {
    return false;
  }
}

function releaseLock(engine: EngineSpec, modelId: string, port: number): void {
  try {
    unlinkSync(lockFile(engine, modelId, port));
  } catch {
    // Already gone.
  }
}

function signalPid(pid: number, signal: string): void {
  try {
    process.kill(pid, signal);
  } catch {
    // Already gone.
  }
}

/**
 * Terminate the server of `model`, whether or not this session started it: the PID file and the
 * listening socket are the sources of truth, because another tab may own the running process.
 */
export async function stopModel(model: ManagedModel, port = settings(model.engine).port): Promise<number | null> {
  const engine = ENGINES[model.engine];
  const candidates = new Set<number>();
  const recorded = readPid(engine, model.id);
  if (recorded !== null && processMatchesCmdline(recorded, model.path)) candidates.add(recorded);
  const listener = resolveListenerPid(port);
  if (listener !== null && processMatchesCmdline(listener, model.path)) candidates.add(listener);

  for (const pid of candidates) signalPid(pid, "SIGTERM");
  if (candidates.size > 0) await waitForDown(engine, `http://127.0.0.1:${port}`);
  // Backstop: anything still holding the port for this model gets SIGKILL.
  const stubborn = resolveListenerPid(port);
  if (stubborn !== null && processMatchesCmdline(stubborn, model.path)) {
    signalPid(stubborn, "SIGKILL");
    candidates.add(stubborn);
    await waitForDown(engine, `http://127.0.0.1:${port}`);
  }
  clearPid(engine, model.id);
  releaseLock(engine, model.id, port);
  return candidates.size > 0 ? [...candidates][0]! : null;
}

/** The command line that serves `model` on `port`, per engine. */
function spawnSpec(model: ManagedModel, port: number, ctx: number, gpu: string): { cmd: string; args: string[]; env: Record<string, string> } {
  if (model.engine === "llamafile") {
    return {
      cmd: "sh",
      args: [model.path, "--server", "--host", "127.0.0.1", "--port", String(port), "--ctx-size", String(ctx)],
      env: {},
    };
  }
  const binary = cortiqBinary();
  if (!binary) {
    throw new Error("cortiq not found: put it in PATH, install it as cortiq-cpu, or set PI_CORTIQ_BIN");
  }
  const env: Record<string, string> = { CMF_MAX_SEQ: String(ctx) };
  // The engine reads `CMF_GPU=0` as "CPU" and treats *any* other value as "use the wgpu GPU",
  // so a literal `cpu` would silently try Vulkan and drop the request on a host without one.
  if (gpu === "cpu") env["CMF_GPU"] = "0";
  else if (gpu === "cuda" || gpu === "wgpu") env["CMF_GPU"] = gpu;
  return {
    cmd: binary,
    args: ["serve", model.path, "--host", "127.0.0.1", "--port", String(port)],
    env,
  };
}

export async function startServer(
  model: ManagedModel,
  opts?: { port?: number; ctxSize?: number },
  onProgress: (message: string) => void = log,
): Promise<string> {
  const engine = ENGINES[model.engine];
  const current = settings(model.engine);
  const serverPort = opts?.port ?? current.port;
  const ctx = opts?.ctxSize ?? current.ctxSize;
  const baseUrl = `http://127.0.0.1:${serverPort}`;
  if (await healthy(engine, baseUrl)) {
    const pid = readPid(engine, model.id);
    if (pid !== null && processMatchesCmdline(pid, model.path)) {
      active[model.engine] = { pid, port: serverPort, modelId: model.id, path: model.path };
      onProgress(`adopting ${engine.label} server on ${baseUrl} (pid ${pid})`);
      return baseUrl;
    }
    throw new Error(
      `port ${serverPort} is already served by another process this manager did not start; stop it or pick another port with /${engine.providerId} port <n>`,
    );
  }
  // Adopt a live leftover of ours (still loading) or spawn fresh.
  const leftover = readPid(engine, model.id);
  if (leftover !== null && processMatchesCmdline(leftover, model.path)) {
    onProgress(`waiting for the leftover ${engine.label} server (pid ${leftover})`);
    await waitForHealth(engine, baseUrl, onProgress);
    active[model.engine] = { pid: leftover, port: serverPort, modelId: model.id, path: model.path };
    return baseUrl;
  }
  // Another session may be loading this model right now: wait for it instead of spawning a
  // second server that would lose the port and then record its own dead PID.
  if (!tryLock(engine, model.id, serverPort)) {
    onProgress(`another session is starting ${model.id}; waiting for ${baseUrl}`);
    await waitForHealth(engine, baseUrl, onProgress);
    const pid = resolveListenerPid(serverPort);
    if (pid !== null && !processMatchesCmdline(pid, model.path)) {
      throw new Error(`port ${serverPort} is served by a process that is not ${model.id}`);
    }
    const adopted = pid ?? readPid(engine, model.id);
    if (adopted !== null) {
      try {
        writeFileSync(pidFile(engine, model.id), String(adopted));
      } catch {
        // Non-fatal.
      }
    }
    active[model.engine] = { pid: adopted, port: serverPort, modelId: model.id, path: model.path };
    return baseUrl;
  }

  const spec = spawnSpec(model, serverPort, ctx, current.gpu);
  onProgress(`starting ${model.id} on ${baseUrl} (ctx ${ctx})`);
  const child = spawn(spec.cmd, spec.args, {
    stdio: ["ignore", "pipe", "pipe"],
    detached: true,
    env: { ...process.env, ...spec.env },
  });
  if (child.pid === undefined) {
    releaseLock(engine, model.id, serverPort);
    throw new Error(`could not start the ${engine.label} server`);
  }
  const detachLog = attachLogTap(engine, model.id, child);
  (child as any).unref?.();
  try {
    await waitForHealth(engine, baseUrl, onProgress);
  } catch (error) {
    // A server that never came up has no logs worth following; a healthy one keeps its tap.
    detachLog();
    releaseLock(engine, model.id, serverPort);
    throw error;
  }
  // The socket owner is the real server; `sh` may have re-execed through the APE loader.
  const owner = resolveListenerPid(serverPort) ?? child.pid;
  if (owner !== child.pid && !processMatchesCmdline(owner, model.path)) {
    signalPid(child.pid, "SIGTERM");
    releaseLock(engine, model.id, serverPort);
    throw new Error(`port ${serverPort} is served by a process that is not ${model.id}`);
  }
  try {
    writeFileSync(pidFile(engine, model.id), String(owner));
  } catch {
    // Non-fatal: without a PID file a crash leftover just won't be adopted.
  }
  releaseLock(engine, model.id, serverPort);
  active[model.engine] = { pid: owner, port: serverPort, modelId: model.id, path: model.path };
  return baseUrl;
}

// ---------------------------------------------------------------------------
// Server log: piped stdout/stderr into a bounded buffer, flushed to disk.
// ---------------------------------------------------------------------------

const LOG_LINES = 500;
const logBuffers = new Map<string, string[]>();
const logFlushed = new Map<string, number>();

function bufferKey(engine: EngineSpec, modelId: string): string {
  return `${engine.id}:${modelId}`;
}

function appendLogLine(engine: EngineSpec, modelId: string, chunk: any): void {
  const key = bufferKey(engine, modelId);
  const text = typeof chunk === "string" ? chunk : String(chunk);
  const buffer = logBuffers.get(key) ?? [];
  for (const line of text.split("\n")) {
    if (line.length > 0) buffer.push(line);
  }
  while (buffer.length > LOG_LINES) buffer.shift();
  logBuffers.set(key, buffer);
}

function flushLog(engine: EngineSpec, modelId: string): void {
  const key = bufferKey(engine, modelId);
  const buffer = logBuffers.get(key) ?? [];
  const done = logFlushed.get(key) ?? 0;
  if (buffer.length <= done) return;
  const fresh = buffer.slice(done).join("\n") + "\n";
  let previous = "";
  try {
    previous = readFileSync(logFile(engine, modelId), "utf8");
  } catch {
    // First write.
  }
  try {
    mkdirSync(dirname(logFile(engine, modelId)), { recursive: true });
    writeFileSync(logFile(engine, modelId), previous + fresh);
    logFlushed.set(key, buffer.length);
  } catch (error) {
    log(`could not write server log: ${error instanceof Error ? error.message : String(error)}`);
  }
}

/**
 * Mirror the server's stdout/stderr into its log file for the whole session.
 *
 * The pipes are `unref`'d, not destroyed: destroying them made the next write by the server fail
 * with EPIPE, which cortiq turns into a dropped connection mid-request (the log line a request
 * emits is enough to kill its own response). Unref'd handles do not keep the hosting pi process
 * alive, which was the reason the tap used to be dropped once the server was healthy.
 */
function attachLogTap(engine: EngineSpec, modelId: string, child: any): () => void {
  const key = bufferKey(engine, modelId);
  logBuffers.set(key, []);
  logFlushed.set(key, 0);
  const listeners: Array<{ stream: any; onData: (chunk: any) => void }> = [];
  const tap = (stream: any) => {
    const onData = (chunk: any) => appendLogLine(engine, modelId, chunk);
    try {
      stream?.on?.("data", onData);
      // Keep reading, but never hold the host process open for a server it does not own.
      stream?.unref?.();
      listeners.push({ stream, onData });
    } catch {
      // Streams without events: logging is best-effort.
    }
  };
  tap(child.stdout);
  tap(child.stderr);
  const pump = setInterval(() => flushLog(engine, modelId), 2000);
  (pump as any).unref?.();
  return () => {
    clearInterval(pump);
    for (const { stream, onData } of listeners) {
      try {
        stream?.off?.("data", onData);
      } catch {
        // Already closed.
      }
    }
    flushLog(engine, modelId);
  };
}

async function waitForDown(engine: EngineSpec, baseUrl: string, timeoutMs = 20000): Promise<void> {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (!(await healthy(engine, baseUrl))) return;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`the ${engine.label} server on ${baseUrl} did not stop within ${timeoutMs / 1000}s`);
}

export async function stopServer(engineId: EngineId = "llamafile"): Promise<number | null> {
  const engine = ENGINES[engineId];
  const current = active[engineId];
  const modelId = current?.modelId ?? settings(engineId).modelId ?? discoverModels(engineId)[0]?.id;
  const port = current?.port ?? settings(engineId).port;
  active[engineId] = null;
  if (modelId) flushLog(engine, modelId);
  const model = discoverModels(engineId).find((candidate) => candidate.id === modelId);
  if (!model) {
    if (!current || current.pid === null) return null;
    signalPid(current.pid, "SIGTERM");
    clearPid(engine, modelId ?? "");
    await waitForDown(engine, `http://127.0.0.1:${current.port}`);
    return current.pid;
  }
  return stopModel(model, port);
}

/** Probe every discovered model: running (healthy), loading (PID but unhealthy), or stopped. */
export async function serverStatus(engineId: EngineId = "llamafile"): Promise<string> {
  const engine = ENGINES[engineId];
  const models = discoverModels(engineId);
  if (models.length === 0) return `no ${engine.extension} discovered`;
  const lines: string[] = [];
  for (const model of models) {
    const pid = readPid(engine, model.id);
    const alive = pid !== null && processMatchesCmdline(pid, model.path);
    const mark = active[engineId]?.modelId === model.id ? " [active]" : "";
    if (!alive) {
      lines.push(`${model.id}${mark}: stopped`);
      continue;
    }
    // Find which port answers: ours, by matching the PID's listening socket is racy;
    // probe the configured port and report health.
    const baseUrl = `http://127.0.0.1:${settings(engineId).port}`;
    const up = await healthy(engine, baseUrl);
    lines.push(`${model.id}${mark}: pid ${pid}, ${up ? `healthy on ${baseUrl}` : "loading..."}`);
  }
  return lines.join("\n");
}

/** One line describing a model's engine and, for cortiq, the GPU mode in effect. */
function describeModel(model: ManagedModel): string {
  if (model.engine !== "cortiq") return "";
  const gpu = settings("cortiq").gpu;
  return gpu === "default" ? " [cmf]" : ` [cmf, gpu=${gpu}]`;
}

// ---------------------------------------------------------------------------
// Provider registration (models.json persistence for OOTB default).
// ---------------------------------------------------------------------------

function agentModelsPath(): string {
  // The default agent dir; a custom PI_CONFIG_DIR build resolves elsewhere (documented limit).
  return join(homedir(), ".pi", "agent", "models.json");
}

function readModelsFile(): any {
  try {
    return JSON.parse(readFileSync(agentModelsPath(), "utf8"));
  } catch {
    return {};
  }
}

function managedEntry(engine: EngineSpec): any | null {
  const file = readModelsFile();
  const entry = file?.providers?.[engine.providerId];
  if (entry && (entry.managedBy === MANAGER_ID || entry.managedBy === undefined)) {
    return { ...entry, models: normalizeModels(engine, entry.models) };
  }
  // Migrate a provider persisted by an older key (pi-llamafile rename): same shape, new key.
  for (const legacy of engine.legacyProviderIds) {
    const previous = file?.providers?.[legacy];
    if (previous && (previous.managedBy === "pi-llamafile" || previous.managedBy === undefined)) {
      return { ...previous, models: normalizeModels(engine, previous.models) };
    }
  }
  return null;
}

function normalizeModels(engine: EngineSpec, models: any[]): any[] {
  return (models ?? []).map((model) => ({ ...model, provider: engine.providerId }));
}

function writeManagedProvider(engine: EngineSpec, models: any[]): void {
  const file = readModelsFile();
  file.providers = file.providers ?? {};
  const existing = file.providers[engine.providerId];
  if (existing && existing.managedBy !== MANAGER_ID && existing.managedBy !== undefined) {
    log(`models.json has a user-managed ${engine.providerId} provider; leaving it alone`);
    return;
  }
  const { port } = settings(engine.id);
  for (const legacy of engine.legacyProviderIds) delete file.providers[legacy];
  file.providers[engine.providerId] = {
    name: engine.providerName,
    baseUrl: `http://127.0.0.1:${port}/v1`,
    apiKey: "local",
    api: "openai-completions",
    models,
    managedBy: MANAGER_ID,
  };
  try {
    mkdirSync(dirname(agentModelsPath()), { recursive: true });
  } catch {
    // Directory already there.
  }
  writeFileSync(agentModelsPath(), JSON.stringify(file, null, 2) + "\n");
}

async function fetchModels(engine: EngineSpec, baseUrl: string, ctx: number, reasoning: boolean): Promise<any[]> {
  // llamafile answers both `/models` and `/v1/models`; cortiq only serves the versioned route.
  let response = await fetch(`${baseUrl}/v1/models`);
  if (response.status === 404) response = await fetch(`${baseUrl}/models`);
  if (!response.ok) throw new Error(`${engine.label} server returned HTTP ${response.status}`);
  const payload: any = await response.json();
  const entries: any[] = payload?.data ?? payload?.models ?? [];
  return entries
    .map((entry) => {
      const id = entry?.id ?? entry?.name ?? entry?.model;
      if (typeof id !== "string" || id.length === 0) return null;
      return {
        id,
        name: id,
        api: "openai-completions",
        provider: engine.providerId,
        baseUrl: `${baseUrl}/v1`,
        reasoning,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: ctx,
        maxTokens: ctx,
        // cortiq takes `enable_thinking` and ignores reasoning_effort, so the host's thinking
        // level drives the chat-template flag: off means no `<think>`, every other level means
        // think. Without this the model always thought, whatever level the user picked.
        ...(engine.id === "cortiq"
          ? {
              compat: {
                thinkingFormat: "chat-template",
                chatTemplateKwargs: { enable_thinking: { $var: "thinking.enabled" } },
                supportsReasoningEffort: false,
              },
            }
          : {}),
      };
    })
    .filter(Boolean);
}

function providerConfig(engine: EngineSpec, baseUrl: string, models: any[]) {
  return {
    id: engine.providerId,
    name: engine.providerName,
    apiKey: "local",
    baseUrl: `${baseUrl}/v1`,
    models,
    refreshModels: async () => fetchModels(engine, baseUrl, settings(engine.id).ctxSize, true),
  };
}

async function registerRunning(model: ManagedModel, baseUrl: string): Promise<string> {
  const engine = ENGINES[model.engine];
  const { ctxSize } = settings(model.engine);
  // Both engines expose thinking controls to the host: llamafile through llama.cpp's own
  // reasoning fields, cortiq through the chat-template flag wired in fetchModels.
  const models = await fetchModels(engine, baseUrl, ctxSize, true);
  writeManagedProvider(engine, models);
  api?.registerProvider(engine.providerId, providerConfig(engine, baseUrl, models));
  return `${model.id} ready at ${baseUrl}`;
}

async function switchTo(model: ManagedModel): Promise<string> {
  return serialize(async () => {
    if (active[model.engine] && active[model.engine]!.modelId !== model.id) await stopServer(model.engine);
    saveSettings({ modelId: model.id }, model.engine);
    const baseUrl = await startServer(model);
    return registerRunning(model, baseUrl);
  });
}

function usageFor(engine: EngineSpec): string {
  const head = `usage: /${engine.providerId} [list|status|start [name]|use <name>|stop|ctx <n>|port <n>|logs [n]${engine.id === "cortiq" ? "|gpu <default|cpu|cuda|wgpu>" : ""}]`;
  const lines = [
    `  list          discovered ${engine.id} models`,
    "  status        servers: running, loading or stopped",
    "  start [name]  serve a model (default: preferred or first discovered)",
    "  use <name>    switch to a model (restarts the server, remembers the choice)",
    "  stop          terminate the managed server",
    `  ctx <n>       set context size (min 1024)${engine.ctxIsEnv ? " — passed as CMF_MAX_SEQ" : ""} and restart the server`,
    "  port <n>      set port (1-65535) and restart the server",
    "  logs [n]      last n lines of the server log (default 30)",
  ];
  if (engine.id === "cortiq") {
    lines.push("  gpu <mode>    default | cpu | cuda | wgpu (sets CMF_GPU) and restart the server");
  }
  return [head, ...lines].join("\n");
}

/** Build the command handler for one engine; both commands share every step. */
function commandHandler(engineId: EngineId) {
  const engine = ENGINES[engineId];
  return async (rawArgs: string, ctx: any) => {
    const argv = String(rawArgs ?? "").trim().split(/\s+/).filter(Boolean);
    const [sub, ...rest] = argv;
    const say = (text: string) => ctx?.ui?.notify?.(text, "info");
    const fail = (text: string) => ctx?.ui?.notify?.(text, "error");
    const usage = usageFor(engine);
    try {
      const models = discoverModels(engineId);
      if (!sub) {
        say(`${await serverStatus(engineId)}\n\n${usage}`);
        return;
      }
      if (sub === "list") {
        if (models.length === 0) {
          fail(`no ${engine.extension} found (${envName(engine, "DIR")}, ~/.pi/agent/models, ~/models)`);
          return;
        }
        if (engineId === "cortiq" && cortiqBinary() === null) {
          fail("cortiq binary not found: put it in PATH, install it as cortiq-cpu, or set PI_CORTIQ_BIN");
        }
        const preferred = settings(engineId).modelId;
        say(
          models
            .map(
              (model) =>
                `${model.id}${model.id === preferred ? " [preferred]" : ""}${active[engineId]?.modelId === model.id ? " [active]" : ""}${describeModel(model)}\n  ${model.path}`,
            )
            .join("\n"),
        );
        return;
      }
      if (sub === "status") {
        say(await serverStatus(engineId));
        return;
      }
      if (sub === "logs") {
        const n = parseInt(rest[0] ?? "30", 10);
        const target = active[engineId]?.modelId ?? settings(engineId).modelId ?? models[0]?.id;
        if (!target) {
          fail("no model selected yet");
          return;
        }
        try {
          const content = readFileSync(logFile(engine, target), "utf8").trim().split("\n");
          say(`log ${target} (last ${n}):\n${content.slice(-(Number.isInteger(n) && n > 0 ? n : 30)).join("\n")}`);
        } catch {
          fail(`(no log file for ${target} yet)`);
        }
        return;
      }
      if (sub === "start") {
        const wanted = rest[0];
        const preferred = settings(engineId).modelId;
        const model =
          (wanted && models.find((candidate) => candidate.id === wanted)) ??
          models.find((candidate) => candidate.id === preferred) ??
          models[0];
        if (!model) {
          fail(wanted ? `no ${engineId} model named ${wanted}` : `no ${engine.extension} found`);
          return;
        }
        if (active[engineId]?.modelId === model.id && (await healthy(engine, `http://127.0.0.1:${active[engineId]!.port}`))) {
          say(`${model.id} already running`);
          return;
        }
        say("starting...");
        say(await serialize(async () => registerRunning(model, await startServer(model))));
        return;
      }
      if (sub === "use") {
        const wanted = rest[0];
        if (!wanted) {
          fail(`usage: /${engine.providerId} use <name>`);
          return;
        }
        const model = models.find((candidate) => candidate.id === wanted);
        if (!model) {
          fail(`no ${engineId} model named ${wanted}`);
          return;
        }
        say(`switching to ${model.id}...`);
        say(await switchTo(model));
        return;
      }
      if (sub === "stop") {
        const was = active[engineId]?.modelId;
        await serialize(async () => stopServer(engineId));
        api?.unregisterProvider(engine.providerId);
        say(was ? `${was} stopped` : "no managed server was running");
        return;
      }
      if (sub === "ctx" || sub === "port") {
        const value = parseInt(rest[0] ?? "", 10);
        if (sub === "ctx" && (!Number.isInteger(value) || value < 1024)) {
          fail(`usage: /${engine.providerId} ctx <n>  (n >= 1024)`);
          return;
        }
        if (sub === "port" && (!Number.isInteger(value) || value < 1 || value > 65535)) {
          fail(`usage: /${engine.providerId} port <n>  (1-65535)`);
          return;
        }
        const patch = sub === "ctx" ? { ctxSize: value } : { port: value };
        const next = saveSettings(patch, engineId);
        const overridden =
          (sub === "ctx" && process.env[envName(engine, "CTX")] !== undefined) ||
          (sub === "port" && process.env[envName(engine, "PORT")] !== undefined)
            ? ` (note: the ${engine.envPrefix}_* env var takes precedence over the saved setting)`
            : "";
        const current =
          (active[engineId] && models.find((candidate) => candidate.id === active[engineId]?.modelId)) ??
          models.find((candidate) => candidate.id === next.modelId) ??
          models[0];
        if (!current) {
          fail(`no ${engine.extension} found`);
          return;
        }
        say(`${sub === "ctx" ? "restarting with ctx " + next.ctxSize : "restarting on port " + next.port}...`);
        const done = await serialize(async () => {
          await stopServer(engineId);
          if (sub === "port") api?.unregisterProvider(engine.providerId);
          const baseUrl = await startServer(current, sub === "port" ? { port: next.port } : { ctxSize: next.ctxSize });
          return registerRunning(current, baseUrl);
        });
        say(done + overridden);
        return;
      }
      if (sub === "gpu" && engine.id === "cortiq") {
        const mode = (rest[0] ?? "").toLowerCase();
        if (!["default", "cpu", "cuda", "wgpu"].includes(mode)) {
          fail(`usage: /${engine.providerId} gpu <default|cpu|cuda|wgpu>`);
          return;
        }
        const next = saveSettings({ gpu: mode }, engineId);
        const overridden =
          process.env[envName(engine, "GPU")] !== undefined
            ? ` (note: ${envName(engine, "GPU")} takes precedence over the saved setting)`
            : "";
        const current =
          (active[engineId] && models.find((candidate) => candidate.id === active[engineId]?.modelId)) ??
          models.find((candidate) => candidate.id === next.modelId) ??
          models[0];
        if (!current) {
          say(`gpu set to ${mode}${overridden}`);
          return;
        }
        say(`restarting with CMF_GPU=${mode}...`);
        const done = await serialize(async () => {
          await stopServer(engineId);
          const baseUrl = await startServer(current);
          return registerRunning(current, baseUrl);
        });
        say(`${done} (gpu ${mode})${overridden}`);
        return;
      }
      fail(usage);
    } catch (error) {
      fail(error instanceof Error ? error.message : String(error));
    }
  };
}

/** Start the model a session should serve, per engine, without surprising the other engine. */
async function autoStart() {
  // A persisted provider means this engine ran last time on this host: restore it.
  for (const engineId of ENGINE_IDS) {
    const engine = ENGINES[engineId];
    const persisted = managedEntry(engine);
    if (!persisted?.models?.length) continue;
    api?.registerProvider(engine.providerId, providerConfig(engine, `http://127.0.0.1:${settings(engineId).port}`, persisted.models));
    const pick =
      discoverModels(engineId).find((candidate) => candidate.id === settings(engineId).modelId) ??
      discoverModels(engineId)[0];
    if (!pick) continue;
    return { engineId, model: pick };
  }
  // Nothing persisted: an engine with an explicit model choice wins over the default order, so a
  // `/cortiq use <name>` (or a distro install that pins one) is not silently overridden by a
  // leftover model of the other engine. Then the historical default: llamafile, then cortiq.
  for (const engineId of ENGINE_IDS) {
    const wanted = settings(engineId).modelId;
    const pick = discoverModels(engineId).find((candidate) => candidate.id === wanted);
    if (pick) return { engineId, model: pick };
  }
  for (const engineId of ENGINE_IDS) {
    const pick = discoverModels(engineId)[0];
    if (pick) return { engineId, model: pick };
  }
  return null;
}

export default function piEmbeddedAiExtension(pi: ExtensionAPI) {
  api = pi;
  packageDir = dirname(new URL(import.meta.url).pathname);
  pi.registerCommand("embedded-ai", {
    description: "Manage local llamafile models: list, start, use, stop, ctx, port, status, logs",
    handler: commandHandler("llamafile"),
  });
  pi.registerCommand("cortiq", {
    description: "Manage local cortiq (.cmf) models: list, start, use, stop, ctx, port, gpu, status, logs",
    handler: commandHandler("cortiq"),
  });

  // Auto-start the preferred (or first discovered) model and register it as the local provider.
  // A missing model is not fatal: the extension loads, start is manual via /embedded-ai or /cortiq.
  // A previously persisted provider registers synchronously so the first model
  // resolution already sees it; the running server is ensured in the background.
  const persisted = managedEntry(ENGINES.llamafile);
  if (persisted?.models?.length) {
    api?.registerProvider(ENGINES.llamafile.providerId, providerConfig(ENGINES.llamafile, `http://127.0.0.1:${settings("llamafile").port}`, persisted.models));
  }
  const persistedCortiq = managedEntry(ENGINES.cortiq);
  if (persistedCortiq?.models?.length) {
    api?.registerProvider(ENGINES.cortiq.providerId, providerConfig(ENGINES.cortiq, `http://127.0.0.1:${settings("cortiq").port}`, persistedCortiq.models));
  }
  // A delegated subagent shares the parent's model server: spawning one here would keep the
  // child process alive besides being pointless. pi-open-agents marks depth in the environment.
  const depth = Number(process.env["PI_OPEN_AGENTS_DEPTH"] ?? "0");
  if (Number.isFinite(depth) && depth > 0) {
    log("subagent process: model server managed by the parent session");
    return;
  }
  autoStart()
    .then(async (choice) => {
      if (!choice) {
        log("no .llamafile or .cmf discovered; use /embedded-ai list or /cortiq list after placing one");
        return;
      }
      const baseUrl = await startServer(choice.model);
      await registerRunning(choice.model, baseUrl);
    })
    .catch((error: unknown) => log(`auto-start skipped: ${error instanceof Error ? error.message : String(error)}`));
}
