---
status: accepted
---

# Pre-render emails from Upstream's React templates

Upstream's email templates are React components rendered at send time, and the admin template preview returns that HTML as part of the Wire Contract. Rust cannot run React, so a Node step at build time renders each template once for every combination of its branches (custom template or not, image attachment or not), with placeholder markers that become Rust templates built into the binary. Upstream changes to the emails then reach the fork on the next build with nothing to port, and Node is needed only to build, as it already is for the web app.

## Considered Options

- **Rewrite the templates in a Rust template engine.** Rejected: every Upstream email change becomes a Porting Backlog entry, and the look drifts.
- **Embed a JavaScript engine to render React at runtime.** Rejected: brings a JavaScript runtime back into the Single Binary.

## Consequences

- Values substituted into the placeholders must be HTML-escaped exactly as React does, both to match the Reference Implementation and to avoid injection. The differential tests compare rendered emails.
