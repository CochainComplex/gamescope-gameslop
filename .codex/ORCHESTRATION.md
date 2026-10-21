# gamescope-gameslop orchestration

This reference complements `AGENTS.md`; current user instructions and runtime
permissions remain authoritative. OpenWolf stores handoffs/lessons, Repowise
provides graph/history and live retrieval, and native Codex owns agent lifecycles.

## Situated project map

Observed on 2026-10-21 in the dirty `framegen-tune` checkout at `a2122191a`:

| Responsibility | Evidence / boundary |
|---|---|
| Player outcome | `README.md`: vendor/game-independent finished-frame generation, preferably on a second present GPU; experimental, no new input in generated frames |
| Deterministic policy | `src/framegen/{policy,scheduling,deadline,dispatch_policy,temporal}.hpp`; header contracts tested independently of Vulkan |
| Mutable GPU execution | `src/rendervulkan.cpp`, `src/framegen/device.cpp`: resource/history ownership, queue/timestamp execution, dispatch ordering |
| Presentation / lifecycle | `src/steamcompmgr.cpp`, `src/Backends`, `src/wlserver.cpp`: arbitration, physical/nested outputs, client/input teardown |
| Pixel contracts | `src/shaders/cs_framegen_*.comp`, framegen ABI/layout/profile headers: precision, formats, history and learned-state identity |
| Validation | `tests/meson.build`: framegen, deadline, swapchain and vblank contracts; `scripts/` and testing docs: offline/runtime evidence |
| Secondary tools | Vapor launcher/menu and NPU experiments are present in the working tree; read their current files before routing a change there |

`doc/framegen-maintenance.md` explains why tightly coupled dispatch state stays
together. Repowise identifies renderer/scheduler hotspots; its suggested splits
are hypotheses, not permission to violate that ownership boundary. The build is
C++20/Meson/Ninja with optional PipeWire/OpenVR/backend features and pinned
subprojects. Pure contracts are cheaper than a full compositor build but cannot
prove queue, shutdown or physical-display behavior.

The OpenWolf handoff describes a newer integrated line in another worktree. It
does not make that code the contents of this root checkout. Check base, dirty
state, optional features and hardware before reusing any result. HDR, native
DRM/VRR, mixed-vendor behavior and perceptual improvements require their own
evidence. Current handoff also defers timing while desktop clock behavior is
unresolved; this setup runs no compositor performance tests.

## Small specialist set

| Native agent | Default model / effort | Use |
|---|---|---|
| `repo_worker` | `gpt-6-luna` / `low` | Locate a bounded path or make explicitly owned routine edits |
| `compositor_reviewer` | `gpt-6.1-sol` / `high` | Independent invariants, regressions and failure-mode analysis |
| `validation` | `gpt-6.1-sol` / `medium` | Design/interpret checks and execute only assigned coordinated validation |

The primary normally implements. A separate persistent implementer, researcher,
deep-reasoner and per-component team would duplicate responsibilities. Use a
bounded unnamed worker or an existing skill for a one-off need. Never delegate
just to occupy slots. Start with zero or one child, ordinarily no more than two;
three is the configured ceiling excluding the primary. Only the primary spawns.
These depth/work-ownership limits are instructions, not sandbox boundaries.

## Routing and escalation

Consider difficulty, ambiguity, affected boundaries, risk/reversibility, domain
expertise, benefit of independence and context/compute cost. Use the highest
material concern; a one-line lifetime change can be critical.

| Tier | Typical task | Route |
|---|---|---|
| 0 trivial | Exact prose/comment correction | Primary directly; no review agent |
| 1 routine | Bounded lookup, mechanical launcher/documentation edit | Direct, or `repo_worker` if isolating the work is useful |
| 2 moderate | Local policy/refactor with a clear contract | Primary or dynamic `gpt-6.1-sol`/`medium`; `validation` when useful |
| 3 complex | Cadence math, multi-queue debugging, profile/ABI change | `gpt-6.1-sol`/`high`; independent `compositor_reviewer` for risky implementation |
| 4 critical | Resource races, shutdown, native presentation/HDR regressions | High/xhigh reasoning, separate correctness and validation passes, explicit failure modes and relevant runtime evidence |

For a hard unresolved Tier 3/4 hypothesis, a bounded `gpt-6-astra`/`high` analysis
is available if the live catalog permits it. Increase effort/model after evidence
of an invalid premise, failed check or reasoning limit, not after repeated blind
retries. Stop nonconverging review loops and find the missing invariant. Never
substitute an unavailable model silently. Preserve the user's primary model and
effort; project config sets child defaults only.

Codex 0.163.0 discovers standalone `.codex/agents/*.toml` with `name`,
`description` and `developer_instructions`. Fixed model/effort in a custom file
take precedence over spawn arguments. For dynamic escalation, use a plain worker
with explicit model/effort and the relevant specialist brief instead of trying to
override a pinned custom agent. In a tool surface without custom-agent selection,
pass that file's instructions as the bounded brief. On the current collaboration
surface explicit overrides require `fork_turns="none"`; check the actual tool
schema in future runtimes. Reuse an existing child for related follow-up work.

## Thinking tools

The existing user configuration already provides `tractatus_thinking` and
`sequential-thinking` MCP servers; all native specialists inherit them. Both
tool interfaces were discovered and a bounded call to each succeeded during
setup. Preserve those inherited settings; no project duplicate is needed.

Use **Tractatus Thinking** to clarify what must hold simultaneously: for example
image ownership, cross-queue dependencies and reset retirement, or which pieces
of a timing claim are observations versus assumptions. Use **Sequential Thinking**
to organize how to test a hypothesis or execute a dependent change, including
revisions when evidence invalidates a step. Choose according to the uncertainty;
when both are useful, clarify structure before process. Skip both for exact
lookups and mechanical edits. Start with a small bounded session and stop when
the decision and discriminating check are clear. These tools do not execute
checks or independently corroborate a claim; verify with live source and tests.
Pass only decision summaries and evidence to specialists/OpenWolf, not full
thinking logs. The primary remains accountable for conclusions.

## Brief and integration contract

Every brief includes KISS/YAGNI/SOLID/DRY and mandatory hook/MCP/distill use, plus:

```text
Objective and tier; selected model/effort and why delegation helps.
Checkout/worktree/base and current dirty files relevant to the task.
Evidence packet: paths/symbols, established invariants, known uncertainties.
Scope: owned files; analysis-only or permitted edits; actions forbidden.
Dependencies: what must finish first; coordination/measurement lock owner.
Acceptance: concrete result and discriminating checks; authorized runtime scope.
Report: result/findings; file/symbol evidence; changes; checks; limits/escalation.
No children; no edits to shared OpenWolf state. Report durable findings to primary.
```

Don't copy the full chat or repeat broad orientation when a short evidence packet
suffices. Agents share files; use disjoint ownership or an explicitly coordinated
worktree. One owner integrates shared renderer/shader/header changes. The primary
checks returned evidence, resolves contradictions, reviews the integrated diff
and reruns affected checks. A second agent agreeing is not proof without evidence.

## Validation discipline

Use the maintenance guide's change-class table. Typical commands (under the
required `repowise distill` wrapper) include `bash -n`, `shellcheck -x`, Python
syntax/fixture checks, standalone framegen contracts, a successful build and
targeted `meson test`. Select actual paths/build options from the current
checkout. Avoid building application code for an instruction-only edit.

Queue/resource/shader changes need nested smoke and native DRM validation in
addition to CPU tests. Quality changes follow
`.claude/reviews/framegen-method-playbook.md` including its five-POV review:
instrument parity first, then offline gates, metrics, blind OFF-inclusive A/B,
then real-game confirmation. Never promote a proxy score into a visual or input
latency claim. Record an unrun hardware check explicitly.

Builds/tests and timing are mutually exclusive across every agent. Coordinate
the existing `.claude/reviews/tools/spatial-learning/gpu-owner.lock`, its owning
campaign and the exact harness; do not introduce a second lock or acquire it in
both a parent and a child that blocks on that same lock. Use traps/finally for
probe cleanup. Desktop/native interaction requires applicable user authorization.

## Mandatory hook integration and limits

- Existing `.codex/config.toml` keeps hooks enabled and Repowise MCP connected.
  The user-level `~/.codex/hooks.json` runs `repowise-rewrite --agent codex` for
  shell tools. Keep it enabled; manual distill use remains mandatory because
  rewriting is selective. A second project copy would run the hook twice.
- `.codex/hooks.json` retains OpenWolf session, shell, read, compaction and end
  hooks. Its write adapter translates native `apply_patch` payloads into the
  installed OpenWolf per-file protocol. Hook invocation is direct, not distilled.
  Invocation cwd selects the worktree's context; a worktree without OpenWolf
  reports an integration error instead of recording into another checkout.
- Review new/changed entries with `/hooks`. Codex trusts an exact definition hash;
  project trust alone is insufficient. Never edit user trust records or use a
  trust-bypass flag to make a smoke test pass. Restart/resume to load new agents.
- Hooks provide context/bookkeeping, not a security boundary. Custom-agent
  instructions cannot make a child read-only. Native children inherit parent
  permissions, provider and MCP configuration; this setup does not broaden them.
- Account model/effort compatibility was checked in the live 0.163.0 account
  catalog on 2026-10-21; refresh it after account/runtime changes. Catalog
  presence is not a successful inference test or a measured cost comparison.
- In 0.163.0, a native child launched from `codex exec --ephemeral` failed with
  `failed to load model context ... no rollout found for thread id`. The same
  custom-agent smoke test passed from a persisted CLI session. Use a durable
  parent for native delegation until that runtime behavior is fixed; don't
  retry the failed ephemeral setup or disable hooks to work around it.

Official configuration sources checked during setup:
[subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents),
[configuration](https://learn.chatgpt.com/docs/config-file/config-reference),
[hooks](https://learn.chatgpt.com/docs/hooks). The installed runtime is the final
compatibility check; don't reintroduce old role tables or guessed security keys.

## Adaptation

After a meaningful delegation, record a compact outcome in the OpenWolf handoff:
task/tier, role/model/effort, accepted findings, failed attempts/rework and observed
wall time/usage if available. Prefer updating an existing relevant cerebrum entry
for a recurring lesson. Don't log routine commands or claim unmeasured savings.
Promote/demote routing only after repeated outcomes, retain at most the useful
recent evidence, and add a persistent role only for a recurring distinct need.

## Bootstrap evidence, 2026-10-21

- TOML and JSON parsed; native `codex app-server --strict-config` plus
  `config/read` and thread initialization passed without configuration warnings.
  Effective root remains GPT-6.1 Sol/xhigh; child default Sol/medium, cap three.
- Native `model/list` confirmed Luna, Sol and Astra and the configured efforts.
  Native custom `repo_worker` invocation returned `REPO_WORKER_OK`. A separate
  Luna/low bounded source-mapping delegation returned verified CPU test targets.
  Sol/high independently reviewed the new setup; three adapter findings were
  fixed and its recheck found no remaining material issue. The validation role
  and Astra escalation have catalog/configuration evidence, not an inference test.
- `repowise distill node .codex/hooks/openwolf-write.test.mjs`: **8/8 PASS**,
  including actual OpenWolf pre/post forwarding in temporary worktree fixtures,
  path aliases, move/delete retirement and contended-lock reporting. Node child
  creation required execution outside this session's filesystem sandbox; use
  direct test-file execution here, since `node --test` did not expose the child
  subprocess failure reliably inside the sandbox. No application suites ran.
- Native `hooks/list`: no hook configuration errors/warnings. At bootstrap the
  two changed write registrations required review; a subsequent2026-10-21 check
  reports all12 root registrations enabled/trusted. Eight fixtures and a live
  native add/delete probe verify forwarding and automatic anatomy/memory update
  plus retirement. See `.wolf/hook-awareness-20261021.json`. Separate wt-spatial/
  wt-exit worktrees need their own context before source edits; root activation
  does not prove configuration in a separate Git worktree.
- Both inherited thinking-tool calls succeeded. They supplied structured
  acceptance/planning support, not independent correctness evidence.
- The successful native smoke root reported 86,072 cumulative input tokens
  (59,392 cached), 374 output tokens; child usage was not separately reported.
  These are bootstrap observations, not efficiency gains. No comparative cost
  or quality measurement establishes a best model beyond this initial routing.
- The user-requested orchestration commit explicitly versions `AGENTS.md` and
  the selected `.codex/` configuration, specialist, reference and hook files.
  Existing ignore rules still exclude other local tooling state, including
  `.wolf/` and `.codex/prompts/`. The configuration retains its existing
  machine-specific OpenWolf and Repowise integration paths. No unrelated
  settings, application source, GPU campaigns, permissions or user hook trust
  records were changed.
