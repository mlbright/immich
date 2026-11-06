---
status: accepted
---

# Keep BullMQ's Redis format permanently

The Rust server keeps Redis and stores jobs in BullMQ's format by running BullMQ's own Lua scripts, taken from the BullMQ version Upstream pins, even after no TypeScript is left. Waiting jobs, pause state and the counts on the admin Jobs page then survive a switch between Upstream and the fork in either direction, as ADR-0001 requires. Dropping Redis would save only 10–20 MB.

## Considered Options

- **A Rust-native queue in Redis.** Rejected: simpler code, but waiting jobs vanish when switching.
- **A queue in Postgres, with no Redis.** Rejected: removes `REDIS_*` from the Operator Contract and loses waiting jobs when switching.

## Consequences

- The fork depends on BullMQ's internal Redis key layout, which BullMQ does not promise to keep stable. A BullMQ upgrade in Upstream is a Porting Backlog entry to test, not a free merge.
