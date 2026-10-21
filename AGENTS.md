<!-- openwolf:begin -->
# OpenWolf

This project uses OpenWolf for context management. Read and follow .wolf/OPENWOLF.md at session start. Check .wolf/cerebrum.md before generating code. Grep .wolf/anatomy.md for a file's path before reading it (never read the whole index).
<!-- openwolf:end -->

# Project-native orchestration

We maintain gamescope-gameslop: a C++20 Wayland/Vulkan compositor with
experimental frame generation. Correct presentation, resource lifetime, HDR,
sharp images and low latency matter more than generated-FPS counts.

## Required context and hooks

- Follow OpenWolf above; read `.wolf/STATUS.md` as historical handoff and verify
  the actual checkout, worktree and dirty files before acting. Read relevant
  cerebrum entries before coding. Preserve other sessions' work.
- **OpenWolf and Repowise hooks are mandatory for the primary and all agents.**
  Keep `.codex/hooks.json`, `[features].hooks` and the inherited user Repowise
  hook enabled. Inspect `/hooks` after configuration changes; new/changed
  definitions require user trust. Never bypass hook trust or sandbox controls.
  If hooks are skipped or failing, report it and repair the integration; don't
  silently describe bookkeeping as active.
- Use Repowise MCP first for source discovery: `get_overview` once for an
  unfamiliar checkout, `get_answer` for how/where/why, `search_codebase` for
  symbols/paths, and batched `get_context`. Use `get_symbol` only for a returned
  symbol/range or omitted bytes. Check anatomy for a path before reading it;
  never load the whole index. If MCP cannot serve live bytes, use a bounded
  direct read and state the gap. Verify generated summaries against live code.
- Run **every shell command whose output you consume** through
  `repowise distill <command>`, including builds, tests, Git and log reads.
  Recover omitted evidence with `repowise expand <ref>` instead of rerunning.
  Native lifecycle hook commands are registered directly, without this wrapper.
- Before a substantive change to a shared/hotspot file, use `get_risk`; before
  changing an established pattern, use `get_why`; before renaming/updating all
  callers, retrieve references. Index facts can predate dirty files. Refresh
  with `repowise update --working-tree` when the affected graph must be current;
  don't treat a score or generated recommendation as a project requirement.

## Engineering contracts

- Apply KISS, YAGNI, SOLID and DRY in every agent brief. Reuse the existing path;
  avoid speculative options, duplicate pipelines and copied implementation tests.
- Keep deterministic framegen policy in `src/framegen` headers; keep mutable GPU
  resources, history and dispatch ordering with their explicit lifetime owner.
  Respect the narrow `device.cpp` exception. Don't split hot paths just to
  improve a size/complexity score. See `doc/framegen-maintenance.md`.
- Causal real frames take priority; normal presentation never waits for framegen.
  Bidir is a distinct ordered timeline. Preserve one batch in flight, queue
  dependencies, image ownership, reset retirement and CPU/shader ABI alignment.
  Generated frames must not manufacture client commits or presentation feedback.
- Preserve HDR precision and the color pipeline, capability fallbacks, Wayland
  lock ordering and shutdown ownership. Don't infer dead code from a build that
  omitted optional `HAVE_*` features.
- Follow surrounding C++/GLSL/shell/Python style. Documentation uses stable
  symbols; update `doc/framegen-reference.md` when changing a control/default,
  HUD row or metrics field. Keep proposals distinct from shipped behavior.

## Adaptive routing

The primary agent owns intent, architecture, integration and final validation.
Work directly on trivial or tightly coupled tasks. **Delegate bounded independent
work when it materially improves completion time or confidence**, using the
smallest useful team. This instruction requests such conditional delegation.
Read [.codex/ORCHESTRATION.md](.codex/ORCHESTRATION.md) for routing, specialist
briefs, validation and capability limits.

- Assess difficulty, uncertainty, scope, risk/reversibility, domain expertise,
  independent-validation benefit and context/compute overhead. Route by the
  highest material risk, not changed-line count.
- Tier 0: direct. Tier 1: direct or `repo_worker` (GPT-6 Luna/low). Tier 2:
  primary or a bounded GPT-6.1 Sol/medium worker; `validation` for test evidence.
  Tier 3: GPT-6.1 Sol/high, with `compositor_reviewer` for independent analysis.
  Tier 4: rigorous high/xhigh reasoning, independent correctness and validation
  passes; escalate a difficult unresolved hypothesis to GPT-6 Astra/high only
  when available and justified. Check live model/effort availability first.
- Use the inherited Tractatus Thinking tool for unclear concepts, coupled
  constraints and missing invariants; use Sequential Thinking for a bounded
  hypothesis/check/implementation sequence with branches or revisions. Use
  either when it helps Tier 2–4 work, both only when structure and process each
  need attention. They organize reasoning; they provide no source or runtime
  proof. Keep tool sessions bounded and summarize decisions/evidence, not raw
  thinking logs. Don't invoke them for routine work or create thinking agents.
- Independent review is required for nontrivial GPU lifetime/synchronization,
  scheduling, Wayland teardown, HDR/format/ABI and learned-profile correctness
  changes. Framegen quality changes also follow the five-POV review and evidence
  ladder in `.claude/reviews/framegen-method-playbook.md` before code/commit.
  Several explicit perspectives can share one reviewer; add agents only for
  useful independence. Skip independent review for trivial prose/mechanical edits.
- Default to at most two specialists; native cap is three child threads. Only
  the primary delegates (no recursive teams). Assign disjoint file ownership;
  never edit the same file concurrently. Serialize integration and dependent work.
- Include objective, checkout/base, relevant evidence, owned files, forbidden
  actions, acceptance checks and concise report shape in each brief. Pass a
  bounded context packet; avoid copying the whole conversation by default.
- Escalate after a concrete wrong premise, missing invariant or failed check;
  stop retrying an unsuitable model. If review rounds stop converging, revisit
  the invariant and test premise before adding fixes.

## Evidence and completion

- Match checks to the change using `doc/framegen-maintenance.md` and
  `doc/framegen-testing.md`. Policy changes need contract/deadline tests;
  queue/resource/shader changes need a successful build, relevant Meson tests,
  nested smoke and native DRM evidence. A failed build invalidates claims based
  on stale test binaries, including `meson test --no-rebuild`.
- **Never build, compile or run test suites concurrently with timing/A-B runs.**
  Coordinate all agents and use the existing shared GPU/build campaign lock
  (`.claude/reviews/tools/spatial-learning/gpu-owner.lock`) for those campaigns.
  Don't launch desktop/fullscreen/native tests without the user's applicable
  authorization; clean up every probe child with traps/finally.
- Quality changes: instrument parity → offline gates → metrics → matched blind
  A/B with framegen OFF → real-game confirmation. The user's visual verdict
  outranks a proxy score. Nested runs don't prove native DRM/VRR/HDR behavior.
- State observed facts, hypotheses and untested behavior separately. The primary
  reconciles contradictory findings and reruns only affected checks after edits.
  Record substantive routing outcomes (model/effort, useful findings, retries,
  rework and available time/usage) in a compact OpenWolf handoff note; don't
  invent savings or keep a command diary. Update existing guidance only when
  repeated evidence justifies it. Refresh `.wolf/STATUS.md` at completion without
  replacing another active campaign's state; let hooks maintain anatomy/memory.
