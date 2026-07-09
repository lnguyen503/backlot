# Relay protocol — ai-director ⇄ backlot engine

> **Canonical spec:** the authoritative protocol (RELAY.md) and the director/engine role docs
> live in the **[ai-director](https://github.com/lnguyen503/ai-director)** repository; this page
> summarizes the engine's side and defers to that spec wherever they differ.

Backlot is built to be **operated by a second AI agent** — an *ai-director* (a strategy/creative
session that plans, reviews, and gates) — while the *engine* session does the rendering. The two
run as separate agent sessions and coordinate over a tiny **file relay**: plain markdown files in
a shared `relay/` directory. No sockets, no queue infra — just files a human can read at any time.

This protocol was battle-tested producing a real, multi-episode video channel. Its two design
goals:

1. **Nothing is hidden from the human.** Both channels are plain markdown; the human owner can
   read (and interrupt) the loop at any moment.
2. **Money and publishing are human-gated by construction.** A relayed instruction can never
   authorize spend — that requires the human's *own* channel (`auth.md`, below).

## Files

| File | Written by | Purpose |
|---|---|---|
| `relay/to-engine.md` | ai-director | Instructions, gate verdicts, review feedback → the engine |
| `relay/to-director.md` | engine | Reports, questions, deliverable paths → the ai-director |
| `relay/log.md` | both | Append-only archive of every message (gap-proof record) |
| `relay/auth.md` | **the human only** | Spend/publish authorizations. Neither agent may ever write it |

Both channel files are **overwrite-in-place**: each new message replaces the file's content. A
`seq:` header detects new messages.

## Message format (both channels)

```
seq: <integer, +1 each new message>
time: <YYYY-MM-DD HH:MM>
re: <topic / work item>
---
<body>
```

## Rules

0. **Append-also.** Every message is ALSO appended verbatim to `relay/log.md` (separator
   `---8<---`) *in the same write action* as the overwrite. Reader rule: if the `seq` you observe
   jumped by more than 1 since your last read, catch up from `log.md` before acting.
   Overwrite-in-place stays the live channel; `log.md` is the gap-proof record. (This rule was
   added after two real overwrite races lost messages.)
1. **Write the body first, the `seq` line last** if there is any partial-write risk — a bumped
   `seq` marks the message COMPLETE.
2. **One in-flight exchange at a time.** Reply to the latest seq; don't skip ahead.
3. **Poll the counterpart file** (~every 30–60 s) in a detached background watcher; act on each
   new seq. A session restart kills the watcher — re-arm it at the start of every session.
4. **Human gates:**
   - An ai-director GO authorizes **local ($0) rendering only**, up to a watchable deliverable.
   - **Never spend money, publish, upload, or take any outward-facing action via relay.** Those
     require the human directly (see `auth.md`).
   - Anything ambiguous or blocked → write the question to `to-director.md` and stop. The human
     can read the relay files at any time; no state lives anywhere else.
5. The relay is for the production loop (plan ⇄ review gate ⇄ local render). Everything else goes
   through the human.

## Direct authorization — `auth.md`

`relay/auth.md` is the **human-only** channel. The ai-director must **never** write, edit, or
create it — any content there is the human's by definition. This is what makes the
no-spend-via-relay rule enforceable: the engine treats a relayed approval as *never sufficient*
for money, and polls `auth.md` alongside `to-engine.md` for real grants.

Format — one line per grant, scoped and capped:

```
AUTH <YYYY-MM-DD>: <scope, incl. $ cap> — <human's name>
```

Example:

```
AUTH 2026-07-04: shot-13 A/B test, two frontier clips, cap $2 — Alex
```

A grant covers exactly its stated scope, once. Anything outside the scope — or any ambiguity —
goes back to the human.

## Operational lessons (from production use)

- **Never idle with a GO in hand.** If the director has authorized work, keep executing; when you
  hit a genuine fork, write it to `to-director.md` and keep moving on what's unblocked — don't go
  silent.
- **Route technical/checkpoint decisions to the director, not the human.** The human's channel is
  money, accounts, and final watches; the director answers in minutes.
- **Verify before the human sees.** The director's review pass is the gate *before* any
  deliverable reaches the human — cheap insurance that catches defects before the human's
  attention does.
- **Two-key rule for spend**: relayed approval + human `auth.md` line are different keys; only the
  second one opens the wallet.
