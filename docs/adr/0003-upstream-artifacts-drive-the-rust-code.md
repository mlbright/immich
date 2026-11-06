---
status: accepted
---

# Upstream's artifacts drive the Rust code

Keeping up with Upstream is a permanent cost, so the Rust server is shaped to make each Porting Backlog entry mechanical rather than idiomatic Rust all the way down. Its modules mirror Upstream's controllers, Domain Services and repositories one to one, with the same names and method names. Request and response types are generated from Upstream's OpenAPI spec. Database queries are copied from Upstream's generated SQL snapshots in `server/src/queries/` and checked at compile time with `sqlx`; only dynamic queries are built in code, following the Kysely chain line by line.

## Considered Options

- **Code-first OpenAPI (handlers emit a spec, CI diffs it against Upstream's).** Rejected: drift surfaces as a failed diff to chase instead of a compile error at the exact place to fix.
- **An ORM or a query builder for every query.** Rejected: the link to Upstream's SQL is lost, and Upstream relies heavily on CTEs and `json_agg`.
- **An idiomatic Rust design organized around Rust's own boundaries.** Rejected: Upstream diffs would no longer map to a place in the Rust code.

## Consequences

- The Rust server inherits Upstream's layering, including its awkward spots, such as Domain Services that talk to each other through events. Do not "clean this up" without accepting a higher porting cost on every Upstream merge.
- The OpenAPI spec does not cover socket.io events, some validation rules, error shapes or cookies; those rely on the end-to-end and differential tests.
