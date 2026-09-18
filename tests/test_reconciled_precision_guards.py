"""Behavioral contracts for the shared precision guards.

The guards answer two questions a rule tier must ask before counting a
surface token as evidence of a failure: does the token assert its own absence
(polarity), and did the underlying operation actually succeed (outcome).
Every guard is deliberately conservative, so each behavior is tested with a
risky input that must be suppressed AND a look-alike that must be left alone.
These are pure helpers, exercised directly with concrete payloads.

Vocabulary tables (negators, success words, policy keys, ...) are written out
as literal lists here rather than read back from the module, so removing a
word from the source fails a test instead of silently shrinking the table.
"""

import sys

import pytest

from pisama_detectors.detection.loop import StateSnapshot
from pisama_detectors.detection.precision_guards import (
    all_operations_permitted,
    asserts_absence,
    conforms_to_declared_type,
    driven_by_distinct_inputs,
    filter_absent_matches,
    is_lossless_coercion,
    negates_problem,
    outcome_is_success,
    policy_decision,
    reports_clean_outcome,
    reports_explicit_failure,
    workflow_reached_success,
)


def _absent(text: str, token: str, **kwargs) -> bool:
    start = text.index(token)
    return asserts_absence(text, start, start + len(token), **kwargs)


# ── Polarity: asserts_absence ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "start,end",
    [(-6, 9), (3, 100), (3, 3), (3, 2)],
    ids=["negative-start", "end-past-text", "empty-span", "inverted-span"],
)
def test_asserts_absence_rejects_invalid_spans(start, end):
    # Every span below starts right after "no ", so an unguarded lookup would
    # read it as a negated token. An unusable span must never be read as an
    # assertion of absence.
    text = "no errors"
    assert asserts_absence(text, 3, 9) is True  # the same text, valid span
    assert asserts_absence(text, start, end) is False


@pytest.mark.parametrize(
    "text,token,expected",
    [
        ("The run finished with no errors", "errors", True),
        ("The run finished with 3 errors", "errors", False),
        ("Reported: zero unexpected failures", "failures", True),
        ("Reported: 2 unexpected failures", "failures", False),
        ("Ran without any warnings", "warnings", True),
        ("Ran with several warnings", "warnings", False),
        ("The dependency tree is free of vulnerabilities", "vulnerabilities", True),
    ],
)
def test_asserts_absence_leading_negator(text, token, expected):
    assert _absent(text, token) is expected


@pytest.mark.parametrize(
    "text,token",
    [
        ("There were no errors", "errors"),
        ("It was not blocked", "blocked"),
        ("a non critical issue", "issue"),
        ("none failed", "failed"),
        ("It never failed", "failed"),
        ("neither warnings nor errors", "warnings"),
        ("neither warnings nor errors", "errors"),
        ("zero failures", "failures"),
        ("Ran without warnings", "warnings"),
        ("Shipped sans errors", "errors"),
        ("absent any warnings", "warnings"),
        ("nothing failed", "failed"),
        ("free of errors", "errors"),
        ("clear of errors", "errors"),
        ("devoid of errors", "errors"),
        ("lack of errors", "errors"),
        ("n/a alerts", "alerts"),
        ("There were 0 errors", "errors"),
    ],
)
def test_asserts_absence_recognises_every_negator_form(text, token):
    assert _absent(text, token) is True


@pytest.mark.parametrize(
    "text",
    [
        "no known errors",
        "no other errors",
        "no further errors",
        "no new errors",
        "no remaining errors",
        "no outstanding errors",
        "no unexpected errors",
        "no apparent errors",
        "no obvious errors",
        "no reported errors",
        "no detected errors",
        "no observed errors",
        "no open errors",
        "no active errors",
        "no critical errors",
        "no major errors",
        "no minor errors",
        "no real errors",
        "no other known critical errors",
    ],
)
def test_asserts_absence_skips_filler_words_between_negator_and_token(text):
    assert _absent(text, "errors") is True


def test_asserts_absence_only_skips_known_filler_words():
    # An unrecognised qualifier is not skipped: conservative, keeps the fire.
    assert _absent("no meaningful errors", "errors") is False


@pytest.mark.parametrize(
    "text",
    ["10 errors", "100 errors", "Casino errors"],
)
def test_asserts_absence_negator_must_be_a_whole_word(text):
    # "10" ends in a zero and "Casino" ends in "no", but neither is a negator.
    assert _absent(text, "errors") is False


def test_asserts_absence_negation_does_not_leak_across_a_line_break():
    # "no" on a previous line must not negate a token on the next one, but the
    # same words on one line are a genuine absence claim.
    assert _absent("Checked: no\nerrors reported by the linter", "errors") is False
    assert _absent("Checked: no errors reported by the linter", "errors") is True


def test_asserts_absence_window_bounds_the_lookback_and_lookahead():
    text = "no other known critical errors"
    assert _absent(text, "errors") is True
    assert _absent(text, "errors", window=5) is False  # negator out of range
    padded = "errors:        0"
    assert _absent(padded, "errors") is True
    assert _absent(padded, "errors", window=3) is False  # zero out of range


@pytest.mark.parametrize(
    "text,token,expected",
    [
        ("errors: 0", "errors", True),
        ("critical alerts: none", "alerts", True),
        ("warnings = none", "warnings", True),
        ("errors: 3", "errors", False),
        ("errors = 12", "errors", False),
        # A few words may sit between the token and the colon.
        ("risk flags: none", "risk", True),
        ("risk flags: high", "risk", False),
    ],
)
def test_asserts_absence_zero_valued_key(text, token, expected):
    assert _absent(text, token) is expected


@pytest.mark.parametrize(
    "text",
    [
        "errors: 0.0",
        "errors: none",
        "errors: nil",
        "errors: null",
        "errors: n/a",
        "errors: no",
        "errors: false",
        "errors: clean",
        "errors: ok",
        "warnings: empty",
    ],
)
def test_asserts_absence_recognises_every_zero_like_value(text):
    token = text.split(":")[0]
    assert _absent(text, token) is True


@pytest.mark.parametrize(
    "text,token,expected",
    [
        ("timeout: not reached", "timeout", True),
        ("limit never exceeded", "limit", True),
        ("alert was not triggered", "alert", True),
        ("timeout: reached", "timeout", False),
        ("limit exceeded", "limit", False),
        ("alert was triggered", "alert", False),
    ],
)
def test_asserts_absence_negated_outcome_verb(text, token, expected):
    assert _absent(text, token) is expected


@pytest.mark.parametrize(
    "negated,affirmed,token",
    [
        ("alert wasn't triggered", "alert was triggered", "alert"),
        ("limit has not been exceeded", "limit has been exceeded", "limit"),
        ("limits are never breached", "limits are breached", "limits"),
        ("quota is not breached", "quota is breached", "quota"),
        ("threshold was not hit", "threshold was hit", "threshold"),
        ("threshold was not met", "threshold was met", "threshold"),
        ("anomaly was not detected", "anomaly was detected", "anomaly"),
        ("regression not found", "regression found", "regression"),
        ("issues not present", "issues present", "issues"),
        ("errors were not encountered", "errors were encountered", "errors"),
        ("retry was not needed", "retry was needed", "retry"),
        ("fallback not required", "fallback required", "fallback"),
        ("fallback was not used", "fallback was used", "fallback"),
        ("exception was not raised", "exception was raised", "exception"),
        ("error was never thrown", "error was thrown", "error"),
        ("drift was not observed", "drift was observed", "drift"),
        ("incident was not reported", "incident was reported", "incident"),
        ("conflict never seen", "conflict seen", "conflict"),
        ("SLA was not violated", "SLA was violated", "SLA"),
        ("boundary was not crossed", "boundary was crossed", "boundary"),
    ],
)
def test_asserts_absence_negated_outcome_verbs_are_paired_with_their_affirmation(
    negated, affirmed, token
):
    assert _absent(negated, token) is True
    assert _absent(affirmed, token) is False


@pytest.mark.parametrize(
    "text,token,expected",
    [
        ("Validation passed for all items", "Validation", True),
        ("checks: ok", "checks", True),
        ("Validation failed for all items", "Validation", False),
        ("the deploy failed", "failed", False),
    ],
)
def test_asserts_absence_success_verb_after_token(text, token, expected):
    assert _absent(text, token) is expected


@pytest.mark.parametrize(
    "text,token",
    [
        ("Validation passed", "Validation"),
        ("tests passing", "tests"),
        ("build succeeded", "build"),
        ("deploy success", "deploy"),
        ("deploy successful", "deploy"),
        ("checks ok", "checks"),
        ("status: green", "status"),
        ("health healthy", "health"),
        ("linter clean", "linter"),
        ("incident resolved", "incident"),
        ("alert cleared", "alert"),
        ("SLA satisfied", "SLA"),
        ("tests: all passed", "tests"),
    ],
)
def test_asserts_absence_recognises_every_success_verb(text, token):
    assert _absent(text, token) is True


@pytest.mark.parametrize(
    "text,token",
    [
        ("tests failing", "tests"),
        ("tests blocked", "tests"),
        ("Validation failed", "Validation"),
        ("checks: red", "checks"),
    ],
)
def test_asserts_absence_failure_verb_after_token_is_not_absence(text, token):
    assert _absent(text, token) is False


def test_asserts_absence_all_items_present_clause():
    # The success verb ("present") is not in the trailing-verb list, so this
    # is only recognised by the whole-line "N of N" scan.
    assert _absent("412 of 412 daily partitions present", "partitions") is True
    assert _absent("3 of 412 daily partitions present", "partitions") is False


@pytest.mark.parametrize(
    "line",
    [
        "88 of 88 nightly regression tests passed",
        "12 of 12 nightly batch jobs ran and succeeded",
        "12 of 12 nightly batch jobs ran and finished ok",
        "412 of 412 nightly partitions were present",
    ],
)
def test_asserts_absence_all_items_clause_accepts_each_success_verb(line):
    # The token is the leading count, too far from the verb for the local
    # trailing-verb check, so only the whole-line scan can recognise it.
    assert _absent(line, line.split()[0]) is True
    # The same clause with unequal counts is a partial result, not absence.
    first, _, rest = line.partition(" of ")
    partial = f"{first} of {int(first) + 1}{rest[len(first):]}"
    assert _absent(partial, first) is False


def test_asserts_absence_all_items_clause_is_scoped_to_its_own_line():
    text = "3 of 412 daily partitions present\n88 of 88 daily partitions present"
    first = text.index("partitions")
    second = text.rindex("partitions")
    assert asserts_absence(text, first, first + len("partitions")) is False
    assert asserts_absence(text, second, second + len("partitions")) is True


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("no errors", True),
        ("zero regressions", True),
        ("nothing missing", True),
        ("0 failures", True),
        ("never failed", True),
        ("   No warnings!", True),
        ("without errors", True),
        ("none blocked", True),
        ("neither error nor warning", True),
        ("nor timeouts", True),
        ("not failing", True),
        ("no unexpected errors", True),
        ("no known critical errors", True),
        ("no timeout", True),
        # A genuine negative finding is not good news about a problem.
        ("no access", False),
        ("no matching records", False),
        # The problem noun must be near the negator, not a distant word.
        ("no results returned for errors", False),
        # No leading negator at all, or a numeral that merely ends in zero.
        ("errors found", False),
        ("3 failures", False),
        ("10 failures", False),
        ("", False),
    ],
)
def test_negates_problem(phrase, expected):
    assert negates_problem(phrase) is expected


def test_negates_problem_tolerates_a_missing_phrase():
    assert negates_problem(None) is False


def test_filter_absent_matches_keeps_only_genuine_findings():
    text = "errors: 0\n3 warnings\nno failures\ntimeouts: 5"
    spans = [
        (text.index(token), text.index(token) + len(token))
        for token in ("errors", "warnings", "failures", "timeouts")
    ]
    kept = filter_absent_matches(text, iter(spans))  # any iterable is accepted
    kept_tokens = [text[s:e] for s, e in kept]
    assert kept_tokens == ["warnings", "timeouts"]


# ── Outcome: outcome_is_success ─────────────────────────────────────────────


def test_outcome_boolean_flag_is_authoritative_and_must_be_a_bool():
    assert outcome_is_success({"ok": False}) is False
    assert outcome_is_success({"success": True}) is True
    # A boolean flag beats a contradictory status string.
    assert outcome_is_success({"ok": True, "status": "failed"}) is True
    # A non-bool "ok" is not a flag; the status decides.
    assert outcome_is_success({"ok": "yes", "status": "failed"}) is False


@pytest.mark.parametrize(
    "flag", ["ok", "success", "succeeded", "passed", "allowed", "permitted"]
)
def test_outcome_every_boolean_flag_key_is_read(flag):
    assert outcome_is_success({flag: True}) is True
    assert outcome_is_success({flag: False}) is False


@pytest.mark.parametrize(
    "blob,expected",
    [
        ({"exit_code": 0}, True),
        ({"exit_code": 2}, False),
        ({"returncode": 0.0}, True),
        ({"return_code": 0}, True),
        ({"return_code": 3}, False),
        ({"status_code": 200}, True),
        ({"status_code": 302}, True),
        ({"status_code": 404}, False),
        ({"status_code": 500}, False),
        # The HTTP range is 200 through 399 inclusive.
        ({"status_code": 199}, False),
        ({"status_code": 399}, True),
        ({"status_code": 400}, False),
        # A small "code" is an exit code, not an HTTP status.
        ({"code": 0}, True),
        ({"code": 1}, False),
        ({"code": 204}, True),
        # Only status_code / code are read as HTTP; an exit code is not.
        ({"exit_code": 200}, False),
        ({"returncode": 404}, False),
        # A numeric code outranks a contradictory status string.
        ({"exit_code": 1, "status": "success"}, False),
        # A non-numeric code is not a code; the status decides.
        ({"exit_code": "0", "status": "failed"}, False),
    ],
)
def test_outcome_numeric_codes(blob, expected):
    assert outcome_is_success(blob) is expected


@pytest.mark.parametrize(
    "blob,expected",
    [
        ({"status": "Completed."}, True),
        ({"state": "succeeded"}, True),
        ({"result": "timed out"}, False),
        ({"outcome": "Failed"}, False),
        ({"verdict": "banana"}, None),
        ({"status": "  ok  "}, True),
    ],
)
def test_outcome_status_words_are_normalised(blob, expected):
    assert outcome_is_success(blob) is expected


@pytest.mark.parametrize(
    "key",
    [
        "status", "state", "result", "outcome", "verdict",
        "disposition", "exit_status", "run_state", "phase",
    ],
)
def test_outcome_every_status_key_is_read(key):
    assert outcome_is_success({key: "completed"}) is True
    assert outcome_is_success({key: "failed"}) is False


@pytest.mark.parametrize(
    "word",
    [
        "success", "succeeded", "successful", "ok", "okay", "pass", "passed",
        "passing", "complete", "completed", "completed_successfully", "done",
        "finished", "healthy", "green", "allowed", "permitted", "granted",
        "accepted", "applied", "committed", "resolved",
    ],
)
def test_outcome_success_vocabulary(word):
    assert outcome_is_success(word) is True
    assert outcome_is_success({"status": word}) is True


@pytest.mark.parametrize(
    "word",
    [
        "error", "errored", "failure", "failed", "failing", "denied", "blocked",
        "rejected", "refused", "forbidden", "timeout", "timed_out", "cancelled",
        "canceled", "aborted", "crashed", "exception", "unauthorized", "red",
    ],
)
def test_outcome_failure_vocabulary(word):
    assert outcome_is_success(word) is False
    assert outcome_is_success({"status": word}) is False


@pytest.mark.parametrize(
    "blob,expected",
    [
        ({"error": None}, True),
        ({"error": ""}, True),
        ({"errors": []}, True),
        ({"exception": {}}, True),
        ({"failure": 0}, True),
        ({"failure": False}, True),
        ({"error": "connection reset"}, False),
        ({"errors": ["a"]}, False),
        ({"exception": {"type": "ValueError"}}, False),
        ({"failure": True}, False),
        ({}, None),
        ({"payload": {"rows": 3}}, None),
    ],
)
def test_outcome_error_field_presence(blob, expected):
    assert outcome_is_success(blob) is expected


def test_outcome_explicit_status_outranks_an_empty_error_field():
    # An empty error field is weak evidence; an explicit failed status is not.
    assert outcome_is_success({"status": "failed", "error": None}) is False
    # An unrecognised status defers to the error field.
    assert outcome_is_success({"status": "banana", "error": "boom"}) is False
    assert outcome_is_success({"status": "banana", "error": None}) is True


def test_outcome_tri_state_for_strings_and_other_types():
    assert outcome_is_success("OK") is True
    assert outcome_is_success("Completed.") is True
    assert outcome_is_success("timed out") is False
    # Unknown is None, never False: the caller must not infer a failure.
    assert outcome_is_success("banana") is None
    assert outcome_is_success(None) is None
    assert outcome_is_success(42) is None
    assert outcome_is_success(["ok"]) is None


# ── Outcome: policy_decision / all_operations_permitted ─────────────────────


def test_policy_decision_ignores_non_dicts_and_missing_keys():
    assert policy_decision("allowed") is None
    assert policy_decision(None) is None
    assert policy_decision({"tool": "read_file"}) is None


def test_policy_decision_boolean_value_is_returned_as_is():
    assert policy_decision({"permission": True}) is True
    assert policy_decision({"authorization": False}) is False


@pytest.mark.parametrize(
    "key",
    [
        "sandbox_policy", "policy", "policy_decision", "permission", "decision",
        "authorization", "authz", "acl", "guard", "enforcement",
    ],
)
def test_policy_decision_reads_every_policy_key(key):
    assert policy_decision({key: "allowed"}) is True
    assert policy_decision({key: "denied"}) is False


@pytest.mark.parametrize(
    "value",
    [
        "sandbox_policy_denied",
        "operation not permitted",
        "permission denied",
        "not allowed",
        "request blocked by sandbox",
        "sandbox escape attempt",
        "deny",
        "rejected",
        "refused by the sandbox",
        "forbidden",
        "policy violation",
        "unauthorized",
        "prohibited",
        "disallowed",
        "never granted",
        "not approved",
    ],
)
def test_policy_decision_deny_language(value):
    assert policy_decision({"sandbox_policy": value}) is False


@pytest.mark.parametrize(
    "value",
    [
        "allowed: scratch read",
        "allow",
        "approved",
        "granted",
        "permitted by sandbox",
        "sandbox allowed the read",
        "sanctioned",
        "accepted",
        "ok",
    ],
)
def test_policy_decision_allow_language(value):
    assert policy_decision({"sandbox_policy": value}) is True


def test_policy_decision_negated_allow_never_reads_as_allow():
    # "permitted" is an allow word, but "not permitted" must be a denial.
    assert policy_decision({"decision": "operation not permitted"}) is False
    assert policy_decision({"decision": "operation permitted"}) is True
    assert policy_decision({"decision": "not approved"}) is False
    assert policy_decision({"decision": "approved"}) is True


def test_policy_decision_unrecognised_value_falls_through_to_next_key():
    assert policy_decision({"policy": "pending", "decision": "denied"}) is False
    assert policy_decision({"policy": "pending"}) is None


def test_all_operations_permitted_when_nothing_was_stopped():
    events = [
        {"tool": "read_file", "status": "ok"},
        {"tool": "write_scratch", "exit_code": 0, "denied": False},
    ]
    assert all_operations_permitted(events) is True
    assert all_operations_permitted([]) is True
    assert all_operations_permitted(None) is True


def test_all_operations_permitted_treats_an_event_with_no_signal_as_permitted():
    # "Unknown" is not "stopped": an event that records no outcome at all
    # must not turn a clean run into a violation.
    assert all_operations_permitted([{"tool": "read_file"}]) is True
    assert all_operations_permitted([{"tool": "read_file", "note": "n/a"}]) is True


@pytest.mark.parametrize(
    "stopper",
    [
        {"tool": "shell", "status": "denied"},
        {"tool": "shell", "exit_code": 126},
        {"tool": "shell", "error": "EPERM"},
        {"tool": "shell", "denied": True},
        {"tool": "shell", "blocked": True},
        {"tool": "shell", "rejected": "policy"},
        {"tool": "shell", "violation": "egress"},
        {"tool": "shell", "escaped": True},
    ],
)
def test_all_operations_permitted_false_if_any_event_was_stopped(stopper):
    events = [{"tool": "read_file", "status": "ok"}, stopper]
    assert all_operations_permitted(events) is False


def test_all_operations_permitted_skips_non_dict_events():
    assert all_operations_permitted(["free-text note", 7, {"status": "ok"}]) is True


# ── Outcome: free-text guards ───────────────────────────────────────────────


def test_reports_explicit_failure_empty_text():
    assert reports_explicit_failure("") is False
    assert reports_clean_outcome("") is False


def test_reports_explicit_failure_type_mismatch_vs_matching_types():
    assert reports_explicit_failure("Expected string but got number") is True
    assert reports_explicit_failure("Expected array, received object") is True
    assert reports_explicit_failure("Expected object but got array") is True
    assert reports_explicit_failure("Expected number, found string") is True
    assert reports_explicit_failure("Expected string but got a number") is True
    # The same declared and actual type is not a mismatch, in any casing.
    assert reports_explicit_failure("Expected string but got string") is False
    assert reports_explicit_failure("Expected String but got string") is False


@pytest.mark.parametrize(
    "text,expected",
    [
        ("3 errors reported", True),
        ("2 timeouts while calling the API", True),
        ("Traceback (most recent call last)", True),
        ("no errors reported", False),
        ("errors: 0", False),
        ("Result: success", False),
        # One absent token must not hide a genuine one later in the message.
        ("no errors, but 2 timeouts", True),
    ],
)
def test_reports_explicit_failure_polarity_of_failure_words(text, expected):
    assert reports_explicit_failure(text) is expected


@pytest.mark.parametrize("noun", ["errors", "failures", "exceptions", "timeouts"])
def test_reports_explicit_failure_reads_each_failure_noun_with_polarity(noun):
    assert reports_explicit_failure(f"3 {noun} logged") is True
    assert reports_explicit_failure(f"no {noun} logged") is False
    assert reports_explicit_failure(f"{noun}: 0") is False


@pytest.mark.parametrize(
    "text",
    [
        "Cannot read property 'id' of the response",
        "fetchData is not defined",
        "Result was undefined",
        "Schema mismatches on column price",
        "stacktrace dumped to stderr",
        "Unhandled promise in worker",
        "Uncaught in the main loop",
        "KeyError on lookup",
        "errno 13 opening the file",
        "ECONNREFUSED 127.0.0.1:5432",
        "ETIMEDOUT while connecting",
        "The required field is missing",
        "The required field is absent",
        "The required column is missing",
        "A required key not provided",
        "ParseException at line 3",
        "A required key not present in the payload",
        "Record is missing a required column",
        "the job errored out",
        "the worker crashed",
        "the call timed out",
        "the job failed",
    ],
)
def test_reports_explicit_failure_recognises_each_failure_marker(text):
    assert reports_explicit_failure(text) is True


@pytest.mark.parametrize(
    "text",
    ["error: 0", "failure: none", "exception: null", "errors: 0"],
)
def test_reports_explicit_failure_ignores_zero_valued_error_fields(text):
    assert reports_explicit_failure(text) is False


def test_reports_explicit_failure_reads_a_populated_error_field():
    assert reports_explicit_failure("error: connection reset") is True
    assert reports_explicit_failure("exception = ValueError") is True


def test_reports_clean_outcome_requires_success_and_no_failure():
    assert reports_clean_outcome(
        "Validation passed for all 3 items. Every required field is present."
    ) is True
    assert reports_clean_outcome("Job completed successfully") is True
    # No positive success assertion: not a clean run, just silence.
    assert reports_clean_outcome("Processed the batch of rows") is False


@pytest.mark.parametrize(
    "text",
    [
        "Validation passed",
        "Validation succeeded",
        "validation ok",
        "All checks passed",
        "Every record valid",
        "Every column populated",
        "All inputs provided",
        "All items present",
        "Check passed for all shards",
        "Job completed successfully",
        "Job finished successfully",
        "Job ran successfully",
        "status: success",
        "state = ok",
        "result: passed",
        "outcome = completed",
        "workflow_status: success",
        "No errors were found",
        "no failures found",
        "No issues detected",
        "no problems reported",
        "no warnings raised",
        "no errors encountered",
        "5 of 5 checks passed",
    ],
)
def test_reports_clean_outcome_recognises_each_success_assertion(text):
    assert reports_clean_outcome(text) is True


def test_reports_clean_outcome_is_vetoed_by_any_explicit_failure():
    assert reports_clean_outcome(
        "validation passed for step 1; step 2 threw a TypeError"
    ) is False
    assert reports_clean_outcome(
        "Validation passed. Expected number but received string"
    ) is False
    # A clean-looking clause does not excuse a genuine failure token.
    assert reports_clean_outcome("no errors, but 2 timeouts") is False
    assert reports_clean_outcome("All checks passed except one: KeyError") is False


# ── Loop guard: driven_by_distinct_inputs ───────────────────────────────────


def test_distinct_intervening_inputs_are_a_fan_out_not_a_loop():
    windows = [
        ["Screening contract A", "Cleared: no issues"],
        ["Screening contract B"],
        ["Screening contract C"],
    ]
    assert driven_by_distinct_inputs(windows) is True
    assert driven_by_distinct_inputs([["only one window"]]) is True


def test_identical_or_empty_intervening_windows_are_a_real_loop():
    assert driven_by_distinct_inputs([["retry the call"], ["retry the call"]]) is False
    # Nothing arrived between repeats.
    assert driven_by_distinct_inputs([["contract A"], [], ["contract B"]]) is False
    assert driven_by_distinct_inputs([["contract A"], ["  ", ""]]) is False
    assert driven_by_distinct_inputs([]) is False


def test_windows_differing_only_by_case_or_whitespace_are_not_distinct():
    assert driven_by_distinct_inputs([["contract   a"], ["contract a"]]) is False
    assert driven_by_distinct_inputs([["Contract A"], ["contract a"]]) is False
    assert driven_by_distinct_inputs([["Contract   A"], ["contract a"]]) is False


def test_a_window_repeated_after_a_different_one_is_still_a_loop():
    # A, B, A is not a fan-out: the third repetition saw the first input again.
    assert driven_by_distinct_inputs([["contract A"], ["contract B"], ["contract A"]]) is False
    assert driven_by_distinct_inputs([["contract A"], ["contract B"], ["contract C"]]) is True


def test_changing_failure_prose_is_not_progress():
    windows = [["Attempt 1: connection timed out"], ["Attempt 2: connection refused"]]
    assert driven_by_distinct_inputs(windows) is False
    # Same shape without a failure marker is distinct input.
    assert driven_by_distinct_inputs(
        [["Attempt 1: connected"], ["Attempt 2: connected to replica"]]
    ) is True


def test_a_failure_marker_in_any_window_vetoes_the_fan_out():
    windows = [["Screened A: cleared"], ["Screened B: cleared"], ["Screened C: job failed"]]
    assert driven_by_distinct_inputs(windows) is False
    # ...but an absent failure word is not a failure marker.
    clean = [["Screened A: errors: 0"], ["Screened B: no errors"]]
    assert driven_by_distinct_inputs(clean) is True


# ── Type-drift guards ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "previous,current",
    [
        ("68.00", 68.0),
        ("68.00", "68.0"),
        ("2", 2),
        (" 7 ", 7),
        ("1e3", 1000),
        ("0.5", 0.5),
        ("true", True),
        ("No", False),
        ("f", False),
        ("t", True),
        ("false", False),
        ("no", "false"),
        ("yes", "true"),
        (True, "yes"),
        (3, 3.0),
        ("abc", " abc "),
    ],
)
def test_lossless_coercion_preserves_the_value(previous, current):
    assert is_lossless_coercion(previous, current) is True


@pytest.mark.parametrize(
    "previous,current",
    [
        ("68.00", "N/A"),
        (68.0, 0),
        ("2", 3),
        ("true", False),
        ("abc", "abd"),
        ("abc", 5),
        # A bool and the number 1 are not interchangeable.
        (True, 1),
    ],
)
def test_lossless_coercion_rejects_value_changes(previous, current):
    assert is_lossless_coercion(previous, current) is False


def test_lossless_coercion_leading_zero_string_is_not_provably_numeric():
    # Postal codes and account numbers depend on their leading zeroes.
    assert is_lossless_coercion("00501", 501) is False
    assert is_lossless_coercion("007", 7.0) is False
    assert is_lossless_coercion("-007", -7) is False
    assert is_lossless_coercion("+007", 7) is False
    assert is_lossless_coercion("007.5", 7.5) is False
    assert is_lossless_coercion("007e2", 700) is False
    # Unchanged text is trivially lossless, and a single zero is just zero.
    assert is_lossless_coercion("00501", "00501") is True
    assert is_lossless_coercion("00501", "0501") is False
    assert is_lossless_coercion("0.5", 0.5) is True


def test_lossless_coercion_ambiguous_values_return_false():
    assert is_lossless_coercion(None, None) is False
    assert is_lossless_coercion([1], [1]) is False
    assert is_lossless_coercion({"a": 1}, {"a": 1}) is False
    # Non-finite numbers cannot be compared, in string or float form, even
    # when both sides are identical (an equal pair would otherwise pass).
    assert is_lossless_coercion("NaN", "NaN") is False
    assert is_lossless_coercion("Infinity", "Infinity") is False
    assert is_lossless_coercion(float("inf"), float("inf")) is False
    assert is_lossless_coercion("Infinity", float("inf")) is False


@pytest.mark.parametrize(
    "value,declared,expected",
    [
        ("text", "str", True),
        ("text", "string", True),
        ("text", "text", True),
        (3, "int", True),
        (3, "integer", True),
        (3, "number", True),
        (3.5, "number", True),
        (3.5, "float", True),
        ([1], "list", True),
        ([1], "array", True),
        ([1], "sequence", True),
        ({"a": 1}, "dict", True),
        ({"a": 1}, "object", True),
        ({"a": 1}, "mapping", True),
        (True, "bool", True),
        (True, "boolean", True),
        ("text", "Sequence", False),
        # Wrong type for the declaration.
        ("3", "number", False),
        (3, "str", False),
        (3, "float", False),
        (3.5, "int", False),
        ({"a": 1}, "array", False),
        ([1], "object", False),
        # bool is an int subclass but must not satisfy a numeric declaration.
        (True, "int", False),
        (True, "number", False),
        (1, "bool", False),
    ],
)
def test_conforms_to_declared_type_plain_types(value, declared, expected):
    assert conforms_to_declared_type(value, declared) is expected


@pytest.mark.parametrize(
    "declared",
    ["Optional[list]", "optional[LIST]", "list | None", "List|None"],
)
def test_conforms_to_declared_type_optional_forms(declared):
    # Filling in an optional field is the schema working, not corruption.
    assert conforms_to_declared_type([1, 2], declared) is True
    assert conforms_to_declared_type(None, declared) is True
    # ...but the wrong type inside the Optional is still a violation.
    assert conforms_to_declared_type("not a list", declared) is False


def test_conforms_to_declared_type_none_needs_an_optional_declaration():
    assert conforms_to_declared_type(None, "list") is None
    # None is valid for any Optional, even one wrapping a type we do not parse.
    assert conforms_to_declared_type(None, "Optional[Foo]") is True


@pytest.mark.parametrize("declared", [None, "", 5, ["list"], "MyCustomModel", "Optional[Foo]"])
def test_conforms_to_declared_type_undeclared_or_unparsed_is_none(declared):
    # An undeclared or unparseable schema entry gives no verdict at all.
    assert conforms_to_declared_type([1], declared) is None


# ── Run-level outcome: workflow_reached_success ─────────────────────────────


def _snapshot(seq: int, delta: dict) -> StateSnapshot:
    return StateSnapshot(agent_id="a", state_delta=delta, content="", sequence_num=seq)


def test_workflow_success_reads_the_final_explicit_verdict_on_snapshots():
    ok = [_snapshot(0, {"step": "fetch"}), _snapshot(1, {"status": "success"})]
    failed = [_snapshot(0, {"status": "success"}), _snapshot(1, {"status": "failed"})]
    assert workflow_reached_success(ok) is True
    # A later failure overrides an earlier success.
    assert workflow_reached_success(failed) is False


def test_workflow_success_accepts_dict_states():
    assert workflow_reached_success([{"state_delta": {"exit_code": 0}}]) is True
    assert workflow_reached_success([{"state_delta": {"exit_code": 1}}]) is False


def test_workflow_success_requires_an_explicit_signal():
    assert workflow_reached_success([]) is False
    assert workflow_reached_success(None) is False
    # No verdict anywhere: unknown is not success.
    assert workflow_reached_success([_snapshot(0, {"step": "fetch"})]) is False
    assert workflow_reached_success([{"step": "fetch"}, {"state_delta": "raw"}]) is False


def test_workflow_success_only_reads_dict_deltas():
    # A bare string that happens to be a success word is not a structured verdict.
    assert workflow_reached_success([{"state_delta": "success"}]) is False
    assert workflow_reached_success([{"state_delta": ["ok"]}]) is False


def test_workflow_success_skips_trailing_states_without_a_verdict():
    states = [
        _snapshot(0, {"result": "completed"}),
        _snapshot(1, {"note": "cleanup"}),
        {"state_delta": "not a dict"},
    ]
    assert workflow_reached_success(states) is True


def test_lossless_coercion_number_too_large_to_stringify_is_ambiguous():
    # Python caps int->str conversion; a value that cannot be canonicalised is
    # reported as not-provably-lossless rather than raising.
    limit = getattr(sys, "get_int_max_str_digits", lambda: 0)()
    if not limit or limit > 100_000:
        pytest.skip("interpreter does not cap int-to-str conversion")
    huge = 10**limit
    assert is_lossless_coercion(huge, huge) is False
    assert is_lossless_coercion(huge, "1") is False
