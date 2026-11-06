---
status: accepted
---

# Fork the Services' code, follow Upstream's contracts

The two Services are rewritten in Rust to cut memory and CPU use, ship as a Single Binary and gain memory and type safety. The fork stops tracking Upstream's server and machine-learning code, but the Wire Contract, Operator Contract (including the database schema, migration for migration) and Plugin Contract all keep following Upstream, because the mobile app installed from the app stores updates itself to Upstream's latest release whether we pull it or not. Every other part of the repository, mobile included, keeps merging from Upstream unchanged.

## Considered Options

- **Freeze the API at the fork point.** Rejected: newer mobile apps would call endpoints the fork lacks and fail partway.
- **Diverge the schema after adopting it once.** Rejected: switching from Upstream would be one-way, with no path back if the fork misbehaves.
- **Track Upstream's server code commit by commit.** Rejected: around 300 commits a month cannot be ported line for line while the rewrite is underway.

## Consequences

- Upstream vX and fork vX can replace each other on the same database, files and configuration, in either direction. The fork reports Upstream's version with a `+rust` build marker.
- The fork may add to the Operator Contract (for example Embedded Machine Learning, off by default) but never change or remove anything in it.
- The machine-learning `/predict` protocol is part of the Wire Contract, so a Rust server works with Upstream's Python machine learning and the other way round.
