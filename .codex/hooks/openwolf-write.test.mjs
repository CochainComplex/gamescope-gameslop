import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fileEvents, forward, invocationRoot } from './openwolf-write.mjs';

const input = patch => ({ tool_use_id: 'patch-id', tool_name: 'apply_patch', tool_input: { command: patch } });

test('native multi-file patches preserve changed content, moves and event identity', () => {
  const events = fileEvents(input('*** Begin Patch\n*** Add File: a.txt\n+hello\n*** Update File: b.txt\n*** Move to: c.txt\n@@\n-before\n+after\n*** Delete File: d.txt\n*** End Patch'));
  assert.deepEqual(events.map(e => [e.tool_name, e.tool_input.file_path, e.removed_path, e.tool_use_id]), [
    ['Write', 'a.txt', undefined, 'patch-id:0'],
    ['Edit', 'c.txt', 'b.txt', 'patch-id:1'],
    ['Edit', 'd.txt', 'd.txt', 'patch-id:2'],
  ]);
  assert.equal(events[0].tool_input.content, 'hello');
  assert.equal(events[1].tool_input.old_string, 'before');
  assert.equal(events[1].tool_input.new_string, 'after');
});

test('forward contexts without approving a tool and preserve denial', async () => {
  const received = [];
  const result = await forward(input('*** Add File: a\n+one\n*** Add File: b\n+two'), 'pre', '/tmp', (_cmd, _args, options) => {
    received.push(JSON.parse(options.input));
    return { status: 0, stdout: JSON.stringify({ hookSpecificOutput: {
      additionalContext: 'same note', permissionDecision: 'deny', permissionDecisionReason: 'blocked',
    } }) };
  });
  assert.equal(received.length, 2);
  assert.equal(result.hookSpecificOutput.additionalContext, 'same note');
  assert.equal(result.hookSpecificOutput.permissionDecision, 'deny');
  const allowed = await forward(input('*** Add File: a\n+x'), 'pre', '/tmp', () => ({ status: 0, stdout: '' }));
  assert.equal(allowed.hookSpecificOutput.permissionDecision, undefined);
});

test('post hooks see final disk files and skip absent output', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'gamescope-hook-test-'));
  try {
    fs.writeFileSync(path.join(root, 'present'), 'final');
    const seen = [];
    await forward(input('*** Update File: present\n-old\n+final\n*** Add File: absent\n+x'), 'post', root, (_cmd, _args, options) => {
      seen.push(JSON.parse(options.input).tool_input.file_path);
      return { status: 0, stdout: '' };
    });
    assert.deepEqual(seen, [path.join(root, 'present')]);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('legacy path aliases work in pre and post events', async () => {
  for (const phase of ['pre', 'post']) {
    let seen;
    await forward({ tool_input: { path: '/tmp', content: 'x' } }, phase, '/tmp', (_cmd, _args, options) => {
      seen = JSON.parse(options.input).tool_input.file_path;
      return { status: 0, stdout: '' };
    });
    assert.equal(seen, '/tmp');
  }
});

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'gamescope-openwolf-'));
  const hooks = path.join(root, '.wolf/hooks');
  fs.mkdirSync(hooks, { recursive: true });
  const source = fileURLToPath(new URL('../..', import.meta.url));
  for (const name of fs.readdirSync(path.join(source, '.wolf/hooks'))) {
    if (name.endsWith('.js') || name === 'package.json') {
      fs.symlinkSync(path.join(source, '.wolf/hooks', name), path.join(hooks, name));
    }
  }
  fs.writeFileSync(path.join(root, '.wolf/config.json'), JSON.stringify({ openwolf: { enabled: true } }));
  return root;
}

test('invocation worktree and subdirectory paths are independent of adapter checkout', async () => {
  const root = fixture();
  try {
    const cwd = path.join(root, 'subdir');
    fs.mkdirSync(cwd);
    fs.writeFileSync(path.join(cwd, 'created.txt'), 'created\n');
    const event = { ...input('*** Add File: created.txt\n+created'), cwd, session_id: 'isolated-fixture' };
    assert.equal(invocationRoot(event, '/different/adapter/checkout'), root);
    await forward(event, 'pre', root);
    await forward(event, 'post', root);
    const store = JSON.parse(fs.readFileSync(path.join(root, '.wolf/anatomy-index.json')));
    assert.ok(store.files['subdir/created.txt']);
    assert.match(fs.readFileSync(path.join(root, '.wolf/memory.md'), 'utf8'), /subdir\/created.txt/);
    fs.writeFileSync(path.join(cwd, '.git'), 'gitdir: somewhere');
    assert.throws(() => invocationRoot(event, '/different/adapter/checkout'), /no OpenWolf/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('deletion and move retire old entries using the actual OpenWolf store APIs', async () => {
  const root = fixture();
  try {
    const added = { ...input('*** Add File: old.txt\n+content'), cwd: root };
    fs.writeFileSync(path.join(root, 'old.txt'), 'content');
    await forward(added, 'post', root);
    fs.renameSync(path.join(root, 'old.txt'), path.join(root, 'new.txt'));
    await forward({ ...input('*** Update File: old.txt\n*** Move to: new.txt\n-content\n+new'), cwd: root }, 'post', root);
    let store = JSON.parse(fs.readFileSync(path.join(root, '.wolf/anatomy-index.json')));
    assert.ok(store.files['new.txt']);
    assert.equal(store.files['old.txt'], undefined);
    fs.unlinkSync(path.join(root, 'new.txt'));
    await forward({ ...input('*** Delete File: new.txt'), cwd: root }, 'post', root);
    store = JSON.parse(fs.readFileSync(path.join(root, '.wolf/anatomy-index.json')));
    assert.equal(store.files['new.txt'], undefined);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('lock contention is surfaced rather than silently claiming retirement', async () => {
  const root = fixture();
  try {
    fs.writeFileSync(path.join(root, '.wolf/anatomy-index.lock'), JSON.stringify({
      pid: process.pid, hostname: os.hostname(), acquiredAt: Date.now(), nonce: 'test',
    }));
    await assert.rejects(forward(input('*** Delete File: missing.txt'), 'post', root), /lock busy/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('unsupported payloads and subprocess failures are visible', async () => {
  assert.throws(() => fileEvents({ tool_input: {} }), /Missing/);
  await assert.rejects(forward(input('*** Add File: a\n+x'), 'pre', '/tmp', () => ({ status: 1, stderr: 'broken runtime' })), /broken runtime/);
});
