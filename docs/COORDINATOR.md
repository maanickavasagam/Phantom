# Coordinator

`POST /coordinator/investigate` takes a goal, optional MIRAGE document / ECHO audio evidence
IDs, and an optional `website_url` for BLACKBOX (or a legacy synthetic `scenario`). It routes
only the relevant supplied inputs, runs each module at most once (three calls at most), records
a trace of every step, and always marks the report for human review.

## Interface (Coordinator tab)

1. **Goal** in plain words, with example chips. The routing uses whole-word keywords
   (`invoice`, `bill`, `call`, `recording`, `link`, `website`...). "investigate" or "all" runs
   everything supplied. A **plan preview** shows before running which checks will run, which
   are asked for but missing evidence, and which evidence the goal ignores.
2. **Evidence:** upload a document or recording (`POST /evidence/upload?kind=document|audio`,
   stored and hashed, not analysed yet), or pick a recent one from earlier MIRAGE/ECHO runs
   (`GET /evidence/recent`). Paste a link for BLACKBOX.
3. **Limits:** number of checks, overall time budget and per-check timeout.
4. **Live progress** (`POST /coordinator/stream`, NDJSON events: `started`, `plan`, `start`,
   `step`, `report`). It shows checks used, time used, and each step as waiting / running
   (with a timeout bar) / done / failed / timed out / skipped.
5. **Stop after current check** (`POST /coordinator/stop/{run}`). A running check is not
   interrupted; the rest are skipped as "stopped by user". Closing the page has the same
   effect.
6. **Report:** the headline says what was found ("Problems found by 2 of 3 checks"), with the
   run status as a detail. It also shows one card per module using that module's own verdict
   rule (`verdicts`), hidden-instruction warnings, problems, missing evidence, findings sorted
   by severity with evidence links, the full action history, and the limits used. Export JSON
   and Print work on it.

`POST /coordinator/investigate` returns the same report in one response, without progress
events.

## Report fields

| Field | Meaning |
|---|---|
| `status` | `completed` · `partial` (some tools failed) · `failed` · `nothing_to_do` |
| `stop_reason` | `completed`, `completed_with_errors`, `call_budget`, `time_budget`, `repeated_failures`, `all_failed`, `nothing_to_do`, `stopped_by_user` |
| `plan` | modules chosen **before** any tool ran (from the goal and supplied inputs only) |
| `routing` | `keywords`, `all_requested` ("investigate"/"all"), or `no_evidence_named` (fallback: every supplied input) |
| `ignored_inputs` | supplied evidence the goal did not ask about (listed, not analysed) |
| `verdicts` | each module's headline verdict by its own rule (MIRAGE/ECHO `suspicious`/`inconclusive`/`benign`, BLACKBOX `official`/`likely_fake`/...) |
| `trace` | per module: `completed` / `failed` / `timed_out` / `skipped` (with reason), time taken |
| `errors` | `invalid_input`, `tool_failed`, `tool_timeout`, `malformed_result` and module errors |
| `warnings` | `instructions_in_content`, `duplicate_input`, `goal_names_no_evidence` |
| `budget`, `tool_calls`, `elapsed_s` | limits used and consumed |

## Safeguards (Day 33)

| Risk | What the coordinator does | Test |
|---|---|---|
| **Malformed call** | Empty or >2,000-character goal, or out-of-range budget/timeout: rejected (HTTP 400/422) before anything runs. Control characters stripped from the goal. | `test_malformed_call_rejected_before_anything_runs` |
| **Bad evidence ID** | Must match `ev_` + 16 hex characters, exist, be the right kind (document for MIRAGE, audio for ECHO) and pass its SHA-256 check. Path-like IDs (`../../x`) are refused by the evidence store itself. Reported as `invalid_input` and not run. | `test_bad_evidence_id_*`, `test_wrong_kind_rejected`, `test_tampered_evidence_rejected`, `test_store_refuses_path_like_ids` |
| **Bad URL** | Over 2,048 characters or contains CR/LF/NUL (header injection): `invalid_input`. | `test_malformed_url_rejected` |
| **Tool crash** | Any exception is caught, reported with its type, and counted against the call budget. Other modules still run (`partial`). | `test_crash_of_any_type_is_contained`, `test_partial_failure_keeps_other_results` |
| **Malformed tool output** | Result must be a `ToolResult` from the expected module and pass re-validation of the contract (every finding has evidence). Otherwise it is discarded (`malformed_result`). | `test_malformed_result_rejected`, `test_result_from_wrong_module_rejected` |
| **Repeated failures** | Two failures in a row stop the run (`repeated_failures`). | `test_two_failures_in_a_row_stop_the_run` |
| **Hanging tool** | Hard per-tool timeout (default 90 s, capped by the time left in the overall budget). An overrunning tool is abandoned (`timed_out`), its late result is ignored, and the run moves on. | `test_slow_tool_is_abandoned_at_its_timeout`, `test_overall_time_budget_caps_the_tool_timeout` |
| **Prompt injection in content** | The plan is fixed from the user's goal and inputs **before** any tool runs. Tool output (OCR text, transcripts) is data: it can never add, skip or reorder calls or change a verdict. Instruction-like text ("ignore previous instructions", "mark this as genuine", `<system>`, "run the blackbox tool") is flagged as a warning for the reviewer. | `test_injected_instructions_are_flagged_but_change_nothing`, `test_injection_in_goal_cannot_add_tools`, `test_ordinary_text_is_not_called_injection` |
| **Repeated actions** | Each (module, input) runs at most once. The same evidence ID given as document and audio is flagged, and the wrong-kind copy is rejected. | `test_each_module_runs_at_most_once`, `test_same_item_as_document_and_audio` |
| **Budget exhaustion** | At most 3 calls and `max_seconds` overall. Skipped modules are listed with the reason. | `test_call_budget_reports_stop_reason`, `test_time_budget_skips_remaining_tools` |

All 37 safeguard tests use faked tools, so they are fast and deterministic. One test runs the
real BLACKBOX URL checker end to end. Live API check (26 Nov 2026): an empty goal gives 422,
`../../secret` as the document ID gives `invalid_input` with no tool run, and a fake SBI link
completes with a `likely_fake` finding.

## Limits

- Python cannot kill a thread: a timed-out tool keeps using CPU in the background until it
  ends, but its result is discarded. Running tools in separate processes would fix this, at
  the cost of start-up time (Whisper and OCR models would reload on every call).
- The injection scan is a small set of English phrase patterns. It is a reviewer alert, not
  the defence: the defence is that tool output never controls the coordinator. Paraphrased or
  non-English instructions will not be flagged (and still cannot change anything).
- Routing uses whole-word keywords in the goal, not a language model. (Fixed 26 Nov: plain
  substring matching made "invoice" trigger the call check, because it contains "voice".)
  If the goal names **no** kind of evidence ("Was the person who rang me trying to trick
  me?"), all supplied evidence is checked and the report says so (`routing:
  no_evidence_named`). Before this fallback such goals ran nothing (found in
  docs/E2E_EVAL.md). A goal that uses a keyword for the wrong evidence is still routed
  literally.