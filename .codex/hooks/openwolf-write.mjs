// Adapt Codex 0.163 apply_patch events to the installed OpenWolf file protocol.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

export function fileEvents(input) {
  const original = input.tool_input ?? {};
  if (original.file_path || original.path) return [{
    ...input, tool_input: { ...original, file_path: original.file_path || original.path },
  }];
  const patch = original.command;
  if (typeof patch !== 'string') throw new Error('Missing apply_patch command');
  const changes = [];
  let change;
  for (const line of patch.split('\n')) {
    const header = line.match(/^\*\*\* (Add|Update|Delete) File: (.+)$/);
    if (header) {
      change = { kind: header[1], from: header[2], to: header[2], old: [], added: [] };
      changes.push(change);
    } else if (change && line.startsWith('*** Move to: ')) {
      change.to = line.slice('*** Move to: '.length);
    } else if (change && line.startsWith('+')) {
      change.added.push(line.slice(1));
    } else if (change && line.startsWith('-')) {
      change.old.push(line.slice(1));
    }
  }
  if (!changes.length) throw new Error('No file headers in apply_patch command');
  return changes.map((c, index) => ({
    ...input,
    tool_use_id: input.tool_use_id ? `${input.tool_use_id}:${index}` : undefined,
    tool_name: c.kind === 'Add' ? 'Write' : 'Edit',
    tool_input: {
      file_path: c.to,
      content: c.kind === 'Add' ? c.added.join('\n') : '',
      old_string: c.old.join('\n'),
      new_string: c.kind === 'Add' ? '' : c.added.join('\n'),
    },
    removed_path: c.kind === 'Delete' || c.from !== c.to ? c.from : undefined,
  }));
}

// The installed post-write hook handles existing files only. Use its own locked
// anatomy APIs to retire deleted/moved entries rather than creating ghost files.
async function retireRemoved(root, events) {
  const removed = events.map(e => e.removed_path).filter(Boolean);
  if (!removed.length) return;
  const base = path.join(root, '.wolf/hooks');
  const { withAnatomyLock, HOOK_LOCK_BUDGET_MS } = await import(path.join(base, 'anatomy-lock.js'));
  const { loadStoreReconciled, saveStore, renderToFile } = await import(path.join(base, 'anatomy-store.js'));
  const wolf = path.join(root, '.wolf');
  const result = withAnatomyLock(wolf, HOOK_LOCK_BUDGET_MS, () => {
    const store = loadStoreReconciled(wolf, root);
    let changed = false;
    for (const name of removed) {
      const absolute = path.resolve(root, name);
      const relative = path.relative(root, absolute).split(path.sep).join('/');
      if (relative.startsWith('../') || path.isAbsolute(relative) || fs.existsSync(absolute)) continue;
      if (Object.hasOwn(store.files, relative)) {
        delete store.files[relative];
        changed = true;
      }
    }
    if (changed) {
      renderToFile(wolf, store);
      saveStore(wolf, store);
    }
    return true;
  });
  if (result === null) throw new Error('OpenWolf anatomy lock busy; deletion retirement not recorded');
}

export function invocationRoot(input, fallback) {
  let current = path.resolve(input.cwd || fallback);
  while (true) {
    if (fs.existsSync(path.join(current, '.wolf'))) return current;
    const parent = path.dirname(current);
    if (fs.existsSync(path.join(current, '.git')) || parent === current) break;
    current = parent;
  }
  throw new Error('Invocation checkout has no OpenWolf context; configure its hooks before editing');
}

export async function forward(input, phase, root, run = spawnSync) {
  if (!['pre', 'post'].includes(phase)) throw new Error('Expected pre or post');
  const base = path.resolve(input.cwd || root);
  const events = fileEvents(input).map(event => ({
    ...event,
    tool_input: { ...event.tool_input, file_path: path.resolve(base, event.tool_input.file_path) },
    removed_path: event.removed_path ? path.resolve(base, event.removed_path) : undefined,
  }));
  const contexts = [];
  const warnings = [];
  let denied;
  for (const event of events) {
    const absolute = path.resolve(root, event.tool_input.file_path);
    if (phase === 'post' && !fs.existsSync(absolute)) continue;
    const result = run(process.execPath, [path.join(root, `.wolf/hooks/${phase}-write.js`)], {
      input: JSON.stringify(event), encoding: 'utf8', timeout: 8000,
      env: { ...process.env, CODEX_PROJECT_ROOT: root },
    });
    if (result.error || result.status !== 0) throw new Error(result.error?.message || result.stderr || 'OpenWolf write hook failed');
    if (!result.stdout.trim()) continue;
    const output = JSON.parse(result.stdout);
    if (output.systemMessage) warnings.push(output.systemMessage);
    const hook = output.hookSpecificOutput ?? {};
    if (hook.additionalContext) contexts.push(hook.additionalContext);
    if (hook.permissionDecision === 'deny' || output.decision === 'block') {
      denied = hook.permissionDecisionReason || output.reason || 'OpenWolf denied this write';
    }
  }
  if (phase === 'post') await retireRemoved(root, events);
  const hook = { hookEventName: phase === 'pre' ? 'PreToolUse' : 'PostToolUse' };
  if (contexts.length) hook.additionalContext = [...new Set(contexts)].join('\n');
  if (denied && phase === 'pre') Object.assign(hook, { permissionDecision: 'deny', permissionDecisionReason: denied });
  return {
    ...(warnings.length ? { systemMessage: [...new Set(warnings)].join('\n') } : {}),
    hookSpecificOutput: hook,
  };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const input = JSON.parse(fs.readFileSync(0, 'utf8'));
    const fallback = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
    const root = invocationRoot(input, fallback);
    process.stdout.write(JSON.stringify(await forward(input, process.argv[2], root)));
  } catch (error) {
    // Surface the integration failure. Never turn an advisory hook into approval.
    process.stdout.write(JSON.stringify({ systemMessage: `OpenWolf write adapter: ${error.message}` }));
    process.exitCode = 1;
  }
}
