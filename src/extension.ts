// pi-embedded-ai/src/extension.ts
//
// Local AI manager for pi (llamafile-based). Discovers `.llamafile` executables on the host,
// serves the selected one as a local OpenAI-compatible provider, and registers it
// as the out-of-the-box local model.
// Weights ship separately (see the awesome-llamafiles distro) and are discovered at runtime.
//
// Runs on QuickJS as well as Node: only `node:child_process` (spawn), `node:fs`,
// `node:os`, `node:path` and global `fetch` are used.

import { spawn } from "node:child_process";
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync, mkdirSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join } from "node:path";

const PROVIDER_ID = "embedded-ai";
const LEGACY_PROVIDER_ID = "llamafile";
const MANAGER_ID = "pi-embedded-ai";
const DEFAULT_PORT = 8080;
const DEFAULT_CTX_SIZE = 131072;
const HEALTH_TIMEOUT_MS = 180000;

type ExtensionAPI = any;

interface ManagedModel {
  id: string;
  path: string;
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
}

let active: ServerState | null = null;
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

/** Candidate roots, in priority order. */
function searchRoots(): string[] {
  const home = homedir();
  const roots: string[] = [];
  const fromEnv = process.env["PI_LLAMAFILE_DIR"];
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

/** Non-recursive scan of `dir` plus one level of subdirectories for `*.llamafile`. */
function scanDir(dir: string, depth: number): ManagedModel[] {
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
    if (stat.isFile() && entry.endsWith(".llamafile")) {
      found.push({ id: entry.replace(/\.llamafile$/, ""), path: full });
    } else if (stat.isDirectory() && depth > 0 && !entry.startsWith(".")) {
      found.push(...scanDir(full, depth - 1));
    }
  }
  return found;
}

export function discoverModels(): ManagedModel[] {
  const seen = new Set<string>();
  const models: ManagedModel[] = [];
  for (const root of searchRoots()) {
    for (const model of scanDir(root, 1)) {
      if (!seen.has(model.path)) {
        seen.add(model.path);
        models.push(model);
      }
    }
  }
  return models;
}

// ---------------------------------------------------------------------------
// Settings: ~/.pi/agent/llamafile.json, overridden by PI_LLAMAFILE_* env.
// ---------------------------------------------------------------------------

function settingsPath(): string {
  return join(homedir(), ".pi", "agent", "llamafile.json");
}

function readSettingsFile(): any {
  try {
    return JSON.parse(readFileSync(settingsPath(), "utf8"));
  } catch {
    return {};
  }
}

function num(envValue: string | undefined, fallback: number, min: number, max = Number.MAX_SAFE_INTEGER): number {
  const parsed = envValue ? parseInt(envValue, 10) : NaN;
  if (Number.isInteger(parsed) && parsed >= min && parsed <= max) return parsed;
  return fallback;
}

export function settings(): Settings {
  const file = readSettingsFile();
  const port = num(
    process.env["PI_LLAMAFILE_PORT"],
    num(String(file.port ?? ""), DEFAULT_PORT, 1, 65535),
    1,
    65535,
  );
  const ctxSize = num(
    process.env["PI_LLAMAFILE_CTX"],
    num(String(file.ctxSize ?? ""), DEFAULT_CTX_SIZE, 1024),
    1024,
  );
  const modelId =
    process.env["PI_LLAMAFILE_MODEL"] ??
    (typeof file.modelId === "string" && file.modelId.length > 0 ? file.modelId : null);
  return { port, ctxSize, modelId };
}

export function saveSettings(patch: Partial<Settings>): Settings {
  const current = settings();
  const next = { ...current, ...patch };
  // Persist only what env does not override.
  const file: any = {};
  if (!process.env["PI_LLAMAFILE_PORT"]) file.port = next.port;
  if (!process.env["PI_LLAMAFILE_CTX"]) file.ctxSize = next.ctxSize;
  if (!process.env["PI_LLAMAFILE_MODEL"] && next.modelId) file.modelId = next.modelId;
  try {
    mkdirSync(dirname(settingsPath()), { recursive: true });
    writeFileSync(settingsPath(), JSON.stringify(file, null, 2) + "\n");
  } catch (error) {
    log(`could not save settings: ${error instanceof Error ? error.message : String(error)}`);
  }
  return next;
}

// ---------------------------------------------------------------------------
// Server lifecycle.
// ---------------------------------------------------------------------------

async function healthy(baseUrl: string): Promise<boolean> {
  try {
    const response = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(3000) });
    return response.ok;
  } catch {
    return false;
  }
}

async function waitForHealth(baseUrl: string, onProgress: (message: string) => void): Promise<void> {
  const started = Date.now();
  let announced = false;
  while (Date.now() - started < HEALTH_TIMEOUT_MS) {
    if (await healthy(baseUrl)) return;
    if (!announced && Date.now() - started > 5000) {
      announced = true;
      onProgress("the llamafile model is loading; this can take a minute");
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`the llamafile server did not become healthy within ${HEALTH_TIMEOUT_MS / 1000}s`);
}

function pidFile(modelId: string): string {
  return join(tmpdir(), `pi-embedded-ai-${modelId}.pid`);
}

function logFile(modelId: string): string {
  return join(homedir(), ".pi", "agent", "models", `llamafile-${modelId}.log`);
}

function readPid(modelId: string): number | null {
  try {
    const raw = readFileSync(pidFile(modelId), "utf8").trim();
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

function clearPid(modelId: string): void {
  try {
    writeFileSync(pidFile(modelId), "");
  } catch {
    // Best effort.
  }
}

export async function startServer(
  model: ManagedModel,
  opts?: { port?: number; ctxSize?: number },
  onProgress: (message: string) => void = log,
): Promise<string> {
  const serverPort = opts?.port ?? settings().port;
  const ctx = opts?.ctxSize ?? settings().ctxSize;
  const baseUrl = `http://127.0.0.1:${serverPort}`;
  if (await healthy(baseUrl)) {
    const pid = readPid(model.id);
    if (pid !== null && processMatchesCmdline(pid, model.path)) {
      active = { pid, port: serverPort, modelId: model.id, path: model.path };
      onProgress(`adopting llamafile server on ${baseUrl} (pid ${pid})`);
      return baseUrl;
    }
    throw new Error(
      `port ${serverPort} is already served by another process this manager did not start; stop it or pick another port with /embedded-ai port <n>`,
    );
  }
  // Adopt a live leftover of ours (still loading) or spawn fresh.
  const leftover = readPid(model.id);
  if (leftover !== null && processMatchesCmdline(leftover, model.path)) {
    onProgress(`waiting for the leftover llamafile server (pid ${leftover})`);
    await waitForHealth(baseUrl, onProgress);
    active = { pid: leftover, port: serverPort, modelId: model.id, path: model.path };
    return baseUrl;
  }
  onProgress(`starting ${model.id} on ${baseUrl} (ctx ${ctx})`);
  const child = spawn(
    "sh",
    [model.path, "--server", "--host", "127.0.0.1", "--port", String(serverPort), "--ctx-size", String(ctx)],
    { stdio: ["ignore", "pipe", "pipe"], detached: true },
  );
  if (child.pid === undefined) throw new Error("could not start the llamafile server");
  attachLogTap(model.id, child);
  try {
    writeFileSync(pidFile(model.id), String(child.pid));
  } catch {
    // Non-fatal: without a PID file a crash leftover just won't be adopted.
  }
  (child as any).unref?.();
  await waitForHealth(baseUrl, onProgress);
  active = { pid: child.pid, port: serverPort, modelId: model.id, path: model.path };
  return baseUrl;
}

// ---------------------------------------------------------------------------
// Server log: piped stdout/stderr into a bounded buffer, flushed to disk.
// ---------------------------------------------------------------------------

const LOG_LINES = 500;
const logBuffers = new Map<string, string[]>();
const logFlushed = new Map<string, number>();
let logTimer: any = null;

function appendLogLine(modelId: string, chunk: any): void {
  const text = typeof chunk === "string" ? chunk : String(chunk);
  const buffer = logBuffers.get(modelId) ?? [];
  for (const line of text.split("\n")) {
    if (line.length > 0) buffer.push(line);
  }
  while (buffer.length > LOG_LINES) buffer.shift();
  logBuffers.set(modelId, buffer);
}

function flushLog(modelId: string): void {
  const buffer = logBuffers.get(modelId) ?? [];
  const done = logFlushed.get(modelId) ?? 0;
  if (buffer.length <= done) return;
  const fresh = buffer.slice(done).join("\n") + "\n";
  let previous = "";
  try {
    previous = readFileSync(logFile(modelId), "utf8");
  } catch {
    // First write.
  }
  try {
    mkdirSync(dirname(logFile(modelId)), { recursive: true });
    writeFileSync(logFile(modelId), previous + fresh);
    logFlushed.set(modelId, buffer.length);
  } catch (error) {
    log(`could not write server log: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function attachLogTap(modelId: string, child: any): void {
  logBuffers.set(modelId, []);
  logFlushed.set(modelId, 0);
  const tap = (stream: any) => {
    try {
      stream?.on?.("data", (chunk: any) => appendLogLine(modelId, chunk));
    } catch {
      // Streams without events: logging is best-effort.
    }
  };
  tap(child.stdout);
  tap(child.stderr);
  const pump = () => {
    flushLog(modelId);
    if (active?.modelId === modelId) {
      logTimer = setTimeout(pump, 2000);
    }
  };
  logTimer = setTimeout(pump, 2000);
}

async function waitForDown(baseUrl: string, timeoutMs = 20000): Promise<void> {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (!(await healthy(baseUrl))) return;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`the llamafile server on ${baseUrl} did not stop within ${timeoutMs / 1000}s`);
}

export async function stopServer(): Promise<void> {
  if (!active) return;
  const { pid, modelId, port } = active;
  active = null;
  flushLog(modelId);
  if (pid === null) return;
  try {
    process.kill(pid, "SIGTERM");
  } catch {
    // Already gone: nothing to do.
  }
  clearPid(modelId);
  await waitForDown(`http://127.0.0.1:${port}`);
}

/** Probe every discovered model: running (healthy), loading (PID but unhealthy), or stopped. */
export async function serverStatus(): Promise<string> {
  const models = discoverModels();
  if (models.length === 0) return "no .llamafile discovered";
  const lines: string[] = [];
  for (const model of models) {
    const pid = readPid(model.id);
    const alive = pid !== null && processMatchesCmdline(pid, model.path);
    const mark = active?.modelId === model.id ? " [active]" : "";
    if (!alive) {
      lines.push(`${model.id}${mark}: stopped`);
      continue;
    }
    // Find which port answers: ours, by matching the PID's listening socket is racy;
    // probe the configured port and report health.
    const baseUrl = `http://127.0.0.1:${settings().port}`;
    const up = await healthy(baseUrl);
    lines.push(`${model.id}${mark}: pid ${pid}, ${up ? `healthy on ${baseUrl}` : "loading..."}`);
  }
  return lines.join("\n");
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

function normalizeModels(models: any[]): any[] {
  return (models ?? []).map((model) => ({ ...model, provider: PROVIDER_ID }));
}

function managedEntry(): any | null {
  const file = readModelsFile();
  const entry = file?.providers?.[PROVIDER_ID];
  if (entry && (entry.managedBy === MANAGER_ID || entry.managedBy === undefined)) {
    return { ...entry, models: normalizeModels(entry.models) };
  }
  // Migrate a provider persisted by pi-llamafile (pre-rename): same shape, new key.
  const legacy = file?.providers?.[LEGACY_PROVIDER_ID];
  if (legacy && (legacy.managedBy === "pi-llamafile" || legacy.managedBy === undefined)) {
    return { ...legacy, models: normalizeModels(legacy.models) };
  }
  return null;
}

function writeManagedProvider(models: any[]): void {
  const file = readModelsFile();
  file.providers = file.providers ?? {};
  const existing = file.providers[PROVIDER_ID];
  if (existing && existing.managedBy !== MANAGER_ID && existing.managedBy !== undefined) {
    log("models.json has a user-managed embedded-ai provider; leaving it alone");
    return;
  }
  const { port } = settings();
  delete file.providers[LEGACY_PROVIDER_ID];
  file.providers[PROVIDER_ID] = {
    name: "embedded AI (local)",
    baseUrl: `http://127.0.0.1:${port}/v1`,
    apiKey: "local",
    api: "openai-completions",
    models,
    managedBy: MANAGER_ID,
  };
  writeFileSync(agentModelsPath(), JSON.stringify(file, null, 2) + "\n");
}

async function fetchModels(baseUrl: string, ctx: number): Promise<any[]> {
  const response = await fetch(`${baseUrl}/models`);
  if (!response.ok) throw new Error(`llamafile server returned HTTP ${response.status}`);
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
        provider: PROVIDER_ID,
        baseUrl: `${baseUrl}/v1`,
        reasoning: true,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: ctx,
        maxTokens: ctx,
      };
    })
    .filter(Boolean);
}

function providerConfig(baseUrl: string, models: any[]) {
  return {
    id: PROVIDER_ID,
    name: "embedded AI (local)",
    apiKey: "local",
    baseUrl: `${baseUrl}/v1`,
    models,
    refreshModels: async () => fetchModels(baseUrl, settings().ctxSize),
  };
}

async function registerRunning(model: ManagedModel, baseUrl: string): Promise<string> {
  const { ctxSize } = settings();
  const models = await fetchModels(baseUrl, ctxSize);
  writeManagedProvider(models);
  api?.registerProvider(PROVIDER_ID, providerConfig(baseUrl, models));
  return `${model.id} ready at ${baseUrl}`;
}

async function switchTo(model: ManagedModel): Promise<string> {
  return serialize(async () => {
    if (active && active.modelId !== model.id) await stopServer();
    saveSettings({ modelId: model.id });
    const baseUrl = await startServer(model);
    return registerRunning(model, baseUrl);
  });
}

const USAGE = `usage: /embedded-ai [list|status|start [name]|use <name>|stop|ctx <n>|port <n>|logs [n]]
  list          discovered llamafile models
  status        servers: running, loading or stopped
  start [name]  serve a model (default: preferred or first discovered)
  use <name>    switch to a model (restarts the server, remembers the choice)
  stop          terminate the managed server
  ctx <n>       set context size (min 1024) and restart the server
  port <n>      set port (1-65535) and restart the server
  logs [n]      last n lines of the server log (default 30)`;

export default function piEmbeddedAiExtension(pi: ExtensionAPI) {
  api = pi;
  packageDir = dirname(new URL(import.meta.url).pathname);
  pi.registerCommand("embedded-ai", {
    description: "Manage local AI models: list, start, use, stop, ctx, port, status, logs",
    // The host passes the raw argument string plus a command context; output goes to ui.notify.
    handler: async (rawArgs: string, ctx: any) => {
      const argv = String(rawArgs ?? "").trim().split(/\s+/).filter(Boolean);
      const [sub, ...rest] = argv;
      const say = (text: string) => ctx?.ui?.notify?.(text, "info");
      const fail = (text: string) => ctx?.ui?.notify?.(text, "error");
      try {
        const models = discoverModels();
        if (!sub) {
          const status = await serverStatus();
          say(`${status}\n\n${USAGE}`);
          return;
        }
        if (sub === "list") {
          if (models.length === 0) {
            fail("no .llamafile found (PI_LLAMAFILE_DIR, ~/.pi/agent/models, ~/models)");
            return;
          }
          const preferred = settings().modelId;
          say(
            models
              .map(
                (model) =>
                  `${model.id}${model.id === preferred ? " [preferred]" : ""}${active?.modelId === model.id ? " [active]" : ""}\n  ${model.path}`,
              )
              .join("\n"),
          );
          return;
        }
        if (sub === "status") {
          say(await serverStatus());
          return;
        }
        if (sub === "logs") {
          const n = parseInt(rest[0] ?? "30", 10);
          const target = active?.modelId ?? settings().modelId ?? models[0]?.id;
          if (!target) {
            fail("no model selected yet");
            return;
          }
          try {
            const content = readFileSync(logFile(target), "utf8").trim().split("\n");
            say(`log ${target} (last ${n}):\n${content.slice(-(Number.isInteger(n) && n > 0 ? n : 30)).join("\n")}`);
          } catch {
            fail(`(no log file for ${target} yet)`);
          }
          return;
        }
        if (sub === "start") {
          const wanted = rest[0];
          const preferred = settings().modelId;
          const model =
            (wanted && models.find((candidate) => candidate.id === wanted)) ??
            models.find((candidate) => candidate.id === preferred) ??
            models[0];
          if (!model) {
            fail(wanted ? `no llamafile named ${wanted}` : "no .llamafile found");
            return;
          }
          if (active && active.modelId === model.id && (await healthy(`http://127.0.0.1:${active.port}`))) {
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
            fail("usage: /embedded-ai use <name>");
            return;
          }
          const model = models.find((candidate) => candidate.id === wanted);
          if (!model) {
            fail(`no llamafile named ${wanted}`);
            return;
          }
          say(`switching to ${model.id}...`);
          say(await switchTo(model));
          return;
        }
        if (sub === "stop") {
          const was = active?.modelId;
          await serialize(async () => stopServer());
          api?.unregisterProvider(PROVIDER_ID);
          say(was ? `${was} stopped` : "no managed server was running");
          return;
        }
        if (sub === "ctx" || sub === "port") {
          const value = parseInt(rest[0] ?? "", 10);
          if (sub === "ctx" && (!Number.isInteger(value) || value < 1024)) {
            fail("usage: /embedded-ai ctx <n>  (n >= 1024)");
            return;
          }
          if (sub === "port" && (!Number.isInteger(value) || value < 1 || value > 65535)) {
            fail("usage: /embedded-ai port <n>  (1-65535)");
            return;
          }
          const patch = sub === "ctx" ? { ctxSize: value } : { port: value };
          const next = saveSettings(patch);
          const overridden =
            (sub === "ctx" && process.env["PI_LLAMAFILE_CTX"] !== undefined) ||
            (sub === "port" && process.env["PI_LLAMAFILE_PORT"] !== undefined)
              ? " (note: the PI_LLAMAFILE_* env var takes precedence over the saved setting)"
              : "";
          const current =
            (active && models.find((candidate) => candidate.id === active?.modelId)) ??
            models.find((candidate) => candidate.id === next.modelId) ??
            models[0];
          if (!current) {
            fail("no .llamafile found");
            return;
          }
          say(`${sub === "ctx" ? "restarting with ctx " + next.ctxSize : "restarting on port " + next.port}...`);
          const done = await serialize(async () => {
            await stopServer();
            if (sub === "port") api?.unregisterProvider(PROVIDER_ID);
            const baseUrl = await startServer(current, sub === "port" ? { port: next.port } : { ctxSize: next.ctxSize });
            return registerRunning(current, baseUrl);
          });
          say(done + overridden);
          return;
        }
        fail(USAGE);
      } catch (error) {
        fail(error instanceof Error ? error.message : String(error));
      }
    },
  });

  // Auto-start the preferred (or first discovered) model and register it as the local provider.
  // A missing model is not fatal: the extension loads, start is manual via /embedded-ai.
  // A previously persisted provider registers synchronously so the first model
  // resolution already sees it; the running server is ensured in the background.
  const persisted = managedEntry();
  if (persisted?.models?.length) {
    api?.registerProvider(PROVIDER_ID, providerConfig(`http://127.0.0.1:${settings().port}`, persisted.models));
  }
  const { modelId } = settings();
  const preferred =
    discoverModels().find((candidate) => candidate.id === modelId) ?? discoverModels()[0];
  if (preferred) {
    serialize(() => startServer(preferred))
      .then(async (baseUrl) => {
        await registerRunning(preferred, baseUrl);
      })
      .catch((error: unknown) => log(`auto-start skipped: ${error instanceof Error ? error.message : String(error)}`));
  } else {
    log("no .llamafile discovered; use /embedded-ai list after placing one in ~/.pi/agent/models or ~/models");
  }
}
