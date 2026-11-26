# Coordinator

`POST /coordinator/investigate` accepts a goal, optional document/audio
evidence IDs, and an optional local synthetic scenario. It routes only
relevant supplied inputs, limits work to three module calls, records a
per-tool trace, counts failed calls against the budget, combines evidence IDs
and findings, and always marks the
report for human review. Missing inputs, module errors, limitations, and
elapsed time remain visible in the response.

Run MIRAGE and ECHO first to obtain evidence IDs, then submit them with a
synthetic scenario for a mixed investigation. The coordinator never
interprets the content of uploaded documents or transcripts as instructions.
BLACKBOX accepts only local synthetic pages.

Local smoke check on 25 November 2026: one genuine invoice, one scripted
call, and the fee scenario completed all three modules with seven evidence
IDs, five findings, and no module errors. This verifies routing and
report assembly on those inputs, not accuracy across a broader population.

The time budget is checked before each module call; an individual module
call is not interrupted mid-run. A hard per-tool timeout and more adversarial
prompt-injection tests remain needed before unattended use.
