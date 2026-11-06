# Immich Rust Fork

A fork of Immich whose server-side Services are rewritten in Rust, while clients and every other part keep following Upstream.

## Language

### Upstream and the fork

**Upstream**:
The immich-app/immich project this repository is forked from.
_Avoid_: origin, mainline, the original

**Reference Implementation**:
Upstream's unmodified code for a Service, kept in the repository as the authority on how the Rust code must behave.
_Avoid_: legacy code, old server, TypeScript server

**Porting Backlog**:
The changes Upstream has made to the Reference Implementation that the Rust code does not yet match.
_Avoid_: tech debt, sync queue, catch-up work

**Ported**:
Said of part of a Service when its Rust code passes the end-to-end suite and behaves identically to the Reference Implementation when both are given the same requests from the same starting data.
_Avoid_: done, migrated, converted, rewritten

**Fallback**:
Serving part of a Service from the Reference Implementation because its Rust code is not Ported for the current Upstream release. Only possible while the Reference Implementation still ships.
_Avoid_: rollback, passthrough, proxying

### Services

**Service**:
A separately deployable process. There are two: the server and machine learning.
_Avoid_: microservice, app, backend

**Domain Service**:
A unit of business logic inside the server that owns one area of the domain, such as assets or albums.
_Avoid_: service (unqualified)

**Single Binary**:
The server delivered as one executable that needs no Node or Python runtime. The media tools it calls may still ship alongside it.
_Avoid_: static binary, monolith, all-in-one

**Embedded Machine Learning**:
Machine learning running inside the server process instead of as its own Service. Off unless the operator turns it on.
_Avoid_: in-process ML, bundled ML, all-in-one

### Contracts

**Wire Contract**:
Everything a client can observe about a Service over the network: endpoints, request and response shapes, real-time events and stream formats. It follows Upstream even though the Services' code does not.
_Avoid_: API, protocol, spec

**Operator Contract**:
Everything a person running a Service configures or relies on outside the network: environment variables, configuration, stored data, files on disk, admin commands and the reported version. Like the Wire Contract, it follows Upstream; the fork may add to it but never change or remove anything.
_Avoid_: deployment config, settings

**Plugin Contract**:
Everything a plugin can observe about the server that hosts it: the functions it may call, their inputs and outputs, the plugin description format and the triggers that run it. Like the other contracts, it follows Upstream.
_Avoid_: plugin API, host interface, ABI
