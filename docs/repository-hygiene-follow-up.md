---
type: canonical
source: none
sync: none
sla: none
---

# Repository hygiene follow-up

The repository intentionally keeps historical planning and editor/tooling
artifacts under `.claude/`, `.cursor/`, `.outpost/`, and `docs/superpowers/`.
They were not removed in the P0 release-readiness repair because their public
status and downstream workflow dependencies have not been confirmed.

Before the next public release, a maintainer should:

1. Confirm which of these artifacts are intended to remain public and useful
   to contributors.
2. Review them for internal-only instructions, environment assumptions, and
   credentials; remove only files that are approved as non-public.
3. Add ignore rules for any generated state that is not already excluded.
4. Review the private-token workflow dependency separately and document its
   public CI behavior or replace it with a public equivalent.
5. Record the decision in a pull request or maintainer note.

This checklist is deliberately narrower than a repository-wide history rewrite:
no tags, history, repository settings, or existing workflow artifacts were
changed by this repair.
