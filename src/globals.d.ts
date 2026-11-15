// Minimal ambient declarations so `tsc --noEmit` passes without @types/node.
// The pi QuickJS/Node shims provide these at runtime.
declare module "node:child_process" {
  export function spawn(cmd: string, args?: string[], options?: any): any;
}
declare module "node:fs" {
  export function existsSync(p: string): boolean;
  export function readdirSync(p: string): string[];
  export function readlinkSync(p: string): string;
  export function unlinkSync(p: string): void;
  export function readFileSync(p: string, enc?: string): string;
  export function writeFileSync(p: string, data: string, options?: any): void;
  export function mkdirSync(p: string, opts?: any): void;
  export function statSync(p: string): { isDirectory(): boolean; isFile(): boolean };
}
declare module "node:os" {
  export function homedir(): string;
  export function tmpdir(): string;
}
declare module "node:path" {
  export function dirname(p: string): string;
  export function join(...parts: string[]): string;
  export function resolve(...parts: string[]): string;
}
declare const process: {
  env: Record<string, string | undefined>;
  pid: number;
  kill(pid: number, signal?: string): void;
};
