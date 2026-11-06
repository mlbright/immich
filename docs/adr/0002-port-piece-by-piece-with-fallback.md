---
status: accepted
---

# Port the server piece by piece, with Fallback

The Rust server replaces the Reference Implementation one route group and one job queue at a time instead of in a single switch-over, because Upstream keeps changing the Wire Contract and a months-long rewrite with nothing deployable would never catch up. During the transition the Rust binary is the container's main process: it listens on Upstream's port, starts the Reference Implementation as a child on an internal port, and forwards to it every route and queue that is not yet Ported or that has open entries in the Porting Backlog (a Fallback).

## Consequences

- `server/` and `machine-learning/` stay byte-identical to Upstream in git, so merges into them are always clean and their diffs form the Porting Backlog. Rust code lives in `services/`.
- Where the Reference Implementation must behave differently during the transition, the change is a small patch in `services/patches/` applied only when the image is built. The first one lets the TypeScript job worker skip the queues Rust owns, through a fork-only environment variable.
- `services/PORTING-BACKLOG.md` is generated at each Upstream merge and read by the build: open entries cause Fallback while the Reference Implementation ships, and block a release once it no longer does.
- Until the last piece is Ported, the image carries both Node and Rust and is larger than Upstream's.
- The Rust code must interoperate with Reference Implementation internals during the transition: BullMQ's Redis format (see ADR-0004) and emitting socket.io events through the Redis adapter.
