"""Canonical destructive/risky verb taxonomy shared across safety detectors.

Before this module: each safety detector defined its own keyword list, with
overlapping but slightly different sets. Adding a new dangerous verb required
updating 4+ files. Now there is one source of truth here, and detectors compose
the categories they care about.

The taxonomy is organized by *intent*, not by *risk level*. Detectors decide
which intents matter for their context (e.g., cowork_safety only cares about
DELETE intents on cloud-synced paths; approval_bypass cares about DELETE +
DEPLOY + EXECUTE).
"""

import re
from typing import Iterable

# ── Canonical verb categories ────────────────────────────────────────────────

# Irreversible deletion of data, files, records, etc.
DELETE_VERBS: frozenset = frozenset(
    {
        "delete",
        "remove",
        "rm",
        "rm -rf",
        "rmdir",
        "unlink",
        "drop",
        "destroy",
        "purge",
        "truncate",
        "wipe",
        "erase",
        "overwrite",
    }
)

# Modification or in-place writes (less destructive than DELETE, still mutates)
WRITE_VERBS: frozenset = frozenset(
    {
        "write",
        "create",
        "save",
        "modify",
        "update",
        "patch",
        "put",
        "insert",
        "append",
        "merge",
    }
)

# External communication / network side effects
SEND_VERBS: frozenset = frozenset(
    {
        "send",
        "email",
        "notify",
        "publish",
        "broadcast",
        "post",
    }
)

# Production deployment
DEPLOY_VERBS: frozenset = frozenset(
    {
        "deploy",
        "push",
        "release",
        "ship",
        "promote",
    }
)

# Arbitrary code/command execution
EXECUTE_VERBS: frozenset = frozenset(
    {
        "execute",
        "run",
        "invoke",
        "trigger",
        "fire",
        "exec",
        "eval",
        "shell",
        "system",
        "subprocess",
        "command",
        "run_code",
    }
)

# Permission/privilege changes
PERMISSION_VERBS: frozenset = frozenset(
    {
        "grant",
        "revoke",
        "chmod",
        "chown",
        "permission",
        "privilege",
        "escalate",
        "elevate",
        "role",
    }
)

# Admin / moderation actions
ADMIN_VERBS: frozenset = frozenset(
    {
        "ban",
        "suspend",
        "block",
        "terminate",
        "kill",
        "shutdown",
        "rollback",
    }
)

# Bulk data operations (export/dump/migrate)
BULK_DATA_VERBS: frozenset = frozenset(
    {
        "bulk",
        "export",
        "dump",
        "migrate",
    }
)

# Financial actions
FINANCIAL_VERBS: frozenset = frozenset(
    {
        "transfer",
        "pay",
        "charge",
        "refund",
    }
)


# ── Per-detector compositions ────────────────────────────────────────────────
# Each composition is a frozenset that captures one detector's full verb scope.
# Behavior change vs. the old per-detector lists is intentional and small:
# detectors that previously omitted a verb in the same intent class now match it.
# The recalibration step at the end of this round validates no F1 regression.

# cowork_safety cares about destructive ops on cloud-synced paths.
# Original set: delete, remove, rm, rm -rf, rmdir, unlink, drop, overwrite, destroy, purge, truncate, wipe
COWORK_DESTRUCTIVE_VERBS: frozenset = DELETE_VERBS

# approval_bypass cares about high-risk irreversible actions needing approval.
# Original set: delete, drop, rm, remove, transfer, deploy, push --force, force-push,
#               send email, execute payment, shutdown, kill, truncate, format, purge,
#               rollback, revoke
APPROVAL_HIGH_RISK_VERBS: frozenset = (
    DELETE_VERBS
    | DEPLOY_VERBS
    | EXECUTE_VERBS
    | ADMIN_VERBS
    | FINANCIAL_VERBS
    | frozenset({"format", "revoke"})
)

# exploration_safety cares about anything irreversible during trial-and-error.
# Original (8 categories of patterns): delete/remove/drop/...; write/create/...;
# send/email/...; deploy/push/...; execute/run/...; transfer/move/migrate;
# grant/revoke/chmod/chown; insert/append/merge
EXPLORATION_DANGEROUS_VERBS: frozenset = (
    DELETE_VERBS
    | WRITE_VERBS
    | SEND_VERBS
    | DEPLOY_VERBS
    | EXECUTE_VERBS
    | PERMISSION_VERBS
    | frozenset({"transfer", "move", "migrate", "commit"})
)

# openclaw_elevated_risk uses a categorized dict structure (the keys map to
# severity reasons). Recreate the same shape so callers can keep using
# `RISKY_KEYWORDS["admin_actions"]` etc.
OPENCLAW_RISKY_KEYWORDS: dict = {
    "admin_actions": ADMIN_VERBS | frozenset({"delete", "revoke"}),
    "permission_ops": PERMISSION_VERBS,
    "data_operations": BULK_DATA_VERBS | frozenset({"truncate", "drop"}),
    "credential_ops": frozenset(
        {
            "password",
            "reset_password",
            "credential",
            "token",
            "secret",
        }
    ),
    "system_commands": EXECUTE_VERBS - frozenset({"run", "invoke", "trigger", "fire"}),
}


# ── Pattern compilation helper ───────────────────────────────────────────────

# Snake_case-aware boundaries: standard \b doesn't match between letters and
# underscores, so r"\bdelete\b" misses "delete_file". These boundaries treat
# `_` as a word separator.
_WB_START = r"(?:^|[_\W])"
_WB_END = r"(?:[_\W]|$)"


def make_verb_pattern(verbs: Iterable[str]) -> re.Pattern[str]:
    """Compile a single regex matching any verb in *verbs* with snake_case boundaries.

    Sorts verbs longest-first so multi-word verbs like "rm -rf" match before "rm".
    """
    sorted_verbs = sorted({v for v in verbs}, key=len, reverse=True)
    alts = "|".join(re.escape(v) for v in sorted_verbs)
    return re.compile(f"{_WB_START}(?:{alts}){_WB_END}", re.IGNORECASE)
