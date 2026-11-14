// pi-llamafile/src/extension.ts
//
// Local llamafile manager for pi. Discovers `.llamafile` executables on the host,
// serves them as local OpenAI-compatible providers, and registers the selected one
// as the out-of-the-box local model. No bundled weights: models ship separately
// (see the pi-llamafile-models distro) and are discovered at runtime.
//
// Runs on QuickJS as well as Node: only `node:child_process` (spawn), `node:fs`,
// `node:os`, `node:path` and global `fetch` are used.

import { spawn } from "node:child_process";
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync, mkdirSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";

const PROVIDER_ID = "llamafile";
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

let active: ServerState | null = null;

function log(message: string) {
  console.error(`[pi-llamafile] ${message}`);
}

/** Candidate roots, in priority order. */
function searchRoots(packageDir: string): string[] {
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

export function discoverModels(packageDir: string): ManagedModel[] {
  const seen = new Set<string>();
  const models: ManagedModel[] = [];
  for (const root of searchRoots(packageDir)) {
    for (const model of scanDir(root, 1)) {
      if (!seen.has(model.path)) {
        seen.add(model.path);
        models.push(model);
      }
    }
  }
  return models;
}

function port(): number {
  const raw = process.env["PI_LLAMAFILE_PORT"];
  const parsed = raw ? parseInt(raw, 10) : NaN;
  return Number.isInteger(parsed) && parsed > 0 && parsed < 65536 ? parsed : DEFAULT_PORT;
}

function ctxSize(): number {
  const raw = process.env["PI_LLAMAFILE_CTX"];
  const parsed = raw ? parseInt(raw, 10) : NaN;
  return Number.isInteger(parsed) && parsed >= 1024 ? parsed : DEFAULT_CTX_SIZE;
}

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
  return join(tmpdir(), `pi-llamafile-${modelId}.pid`);
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

export async function startServer(
  model: ManagedModel,
  onProgress: (message: string) => void = log,
): Promise<string> {
  const serverPort = port();
  const baseUrl = `http://127.0.0.1:${serverPort}`;
  if (await healthy(baseUrl)) {
    const pid = readPid(model.id);
    if (pid !== null && processMatchesCmdline(pid, model.path)) {
      active = { pid, port: serverPort, modelId: model.id, path: model.path };
      onProgress(`adopting llamafile server on ${baseUrl} (pid ${pid})`);
      return baseUrl;
    }
    throw new Error(
      `port ${serverPort} is already served by another process this manager did not start; stop it or set PI_LLAMAFILE_PORT`,
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
  onProgress(`starting ${model.id} on ${baseUrl} (ctx ${ctxSize()})`);
  const child = spawn("sh", [model.path, "--server", "--host", "127.0.0.1", "--port", String(serverPort), "--ctx-size", String(ctxSize())], {
    stdio: "ignore",
    detached: true,
  });
  if (child.pid === undefined) throw new Error("could not start the llamafile server");
  try {
    writeFileSync(pidFile(model.id), String(child.pid));
  } catch {
    // Non-fatal: without a PID file a crash leftover just won't be adopted.
  }
  child.unref?.();
  await waitForHealth(baseUrl, onProgress);
  active = { pid: child.pid, port: serverPort, modelId: model.id, path: model.path };
  return baseUrl;
}

export function stopServer(): void {
  if (!active) return;
  const { pid } = active;
  active = null;
  if (pid === null) return;
  try {
    process.kill(pid, "SIGTERM");
  } catch {
    // Already gone: nothing to do.
  }
  try {
    writeFileSync(pidFile(pid.toString()), "");
  } catch {
    // Best effort cleanup.
  }
}

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

function managedEntry(): any | null {
  const file = readModelsFile();
  const entry = file?.providers?.[PROVIDER_ID];
  if (entry && (entry.managedBy === "pi-llamafile" || entry.managedBy === undefined)) return entry;
  return null;
}

function writeManagedProvider(models: any[]): void {
  const file = readModelsFile();
  file.providers = file.providers ?? {};
  const existing = file.providers[PROVIDER_ID];
  if (existing && existing.managedBy !== "pi-llamafile" && existing.managedBy !== undefined) {
    log("models.json has a user-managed llamafile provider; leaving it alone");
    return;
  }
  file.providers[PROVIDER_ID] = {
    name: "llamafile (local)",
    baseUrl: `http://127.0.0.1:${port()}/v1`,
    apiKey: "local",
    api: "openai-completions",
    models,
    managedBy: "pi-llamafile",
  };
  writeFileSync(agentModelsPath(), JSON.stringify(file, null, 2) + "\n");
}

async function fetchModels(baseUrl: string): Promise<any[]> {
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
        contextWindow: ctxSize(),
        maxTokens: ctxSize(),
      };
    })
    .filter(Boolean);
}

function providerConfig(baseUrl: string, models: any[]) {
  return {
    id: PROVIDER_ID,
    name: "llamafile (local)",
    apiKey: "local",
    baseUrl: `${baseUrl}/v1`,
    models,
    refreshModels: async () => fetchModels(baseUrl),
  };
}

export default function piLlamafileExtension(pi: ExtensionAPI) {
  const packageDir = dirname(new URL(import.meta.url).pathname);
  pi.registerCommand("llamafile", {
    description: "Manage local llamafile models: list, start, stop, status",
    handler: async (args: string[]) => {
      const [sub, ...rest] = args;
      const models = discoverModels(packageDir);
      if (sub === "list" || !sub) {
        if (models.length === 0) return "no .llamafile found (PI_LLAMAFILE_DIR, ~/.pi/agent/models, ~/models)";
        return models.map((model) => `${model.id}\n  ${model.path}`).join("\n");
      }
      if (sub === "status") {
        if (!active) return "no llamafile server managed by this extension";
        const up = await healthy(`http://127.0.0.1:${active.port}`);
        return `${active.modelId} on port ${active.port} (pid ${active.pid}): ${up ? "healthy" : "not responding"}`;
      }
      if (sub === "start") {
        const wanted = rest[0];
        const model = wanted ? models.find((candidate) => candidate.id === wanted) : models[0];
        if (!model) return wanted ? `no llamafile named ${wanted}` : "no .llamafile found";
        const baseUrl = await startServer(model);
        pi.registerProvider(PROVIDER_ID, providerConfig(baseUrl, await fetchModels(baseUrl)));
        return `${model.id} ready at ${baseUrl}`;
      }
      if (sub === "stop") {
        stopServer();
        pi.unregisterProvider(PROVIDER_ID);
        return "llamafile server stopped";
      }
      return "usage: /llamafile [list|start [name]|stop|status]";
    },
  });

  // Auto-start the first discovered model and register it as the local provider.
  // A missing model is not fatal: the extension loads, start is manual via /llamafile.
  // A previously persisted provider registers synchronously so the first model
  // resolution already sees it; the running server is ensured in the background.
  const persisted = managedEntry();
  if (persisted?.models?.length) {
    pi.registerProvider(PROVIDER_ID, providerConfig(`http://127.0.0.1:${port()}`, persisted.models));
  }
  const [first] = discoverModels(packageDir);
  if (first) {
    startServer(first)
      .then(async (baseUrl) => {
        const models = await fetchModels(baseUrl);
        writeManagedProvider(models);
        pi.registerProvider(PROVIDER_ID, providerConfig(baseUrl, models));
      })
      .catch((error: unknown) => log(`auto-start skipped: ${error instanceof Error ? error.message : String(error)}`));
  } else {
    log("no .llamafile discovered; use /llamafile list after placing one in ~/.pi/agent/models or ~/models");
  }
}
