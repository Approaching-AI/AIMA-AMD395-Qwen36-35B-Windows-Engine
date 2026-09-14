# Agent integration and document checks

The service proposes tool calls; the caller executes them and verifies their
results. A response saying that a script or document was created does not
establish that the script ran or that every required output exists.

## Tool repair and completion

Append the issued assistant call and its actual tool result, retaining the
unique call ID. Completed repairs allow a fresh verification window for the
same command. An explicit successful exit from a different command can
confirm a repair even when stdout is empty. Unchanged repeated failures and
output-free calls remain bounded. Late or parallel results are ordered by
the issuing assistant turn; replaying a result ID is rejected.

Treat `error.code: "tool_call_no_progress"` as a failed turn. Nonstreaming
returns HTTP 400. Streaming can retain HTTP 200 headers but emits an error
then `[DONE]`, with no successful terminal finish reason. Read the complete
SSE stream and its errors. If another proposed call remains admissible, it
is returned normally. The retry diagnostics and response fields are described
in [API.md](API.md).

## Check every required DOCX

The archive includes a caller-side helper requiring Python 3.10 or newer.
Python is optional for this helper and is not an inference runtime dependency.
From the extracted archive, run for example:

```powershell
py -3 .\scripts\check-agent-documents.py .\report.docx .\evidence.docx
```

The helper checks that each input is a regular file, that its ZIP container
is intact and within inspection limits, and that its OOXML content types,
main-document relationship and XML structure agree. Text renamed to `.docx`,
a missing file or a malformed container returns exit code 2. `qualified`
in the JSON report means that these container checks passed for every input.

Optional text recovery writes new Markdown files:

```powershell
py -3 .\scripts\check-agent-documents.py .\report.docx --markdown-dir .\recovered
```

Recovery preserves readable text only; it does not convert Markdown to Word
or preserve images, automatic numbering, styles or layout. It never overwrites
an input or existing recovery file. Recovering text from an invalid DOCX
still returns failure. Containers are inspected without extracting ZIP members.

Use a complete caller loop: generate the script, run it, check every requested
output, return the checks as tool results, and repair failures. Review the
rendered document and its content separately before delivery. The native
service does not access the caller's filesystem or validate the document's
claims, citations or layout.

This helper is imported from Linux sibling release
[`v1.5.1-native-vl.9`](https://github.com/skyguan92/AIMA-AMD395-Qwen36-35B-Linux-Engine/releases/tag/v1.5.1-native-vl.9),
tag `a1a41b529811a5d65a6bc349d820be85d6b9e0c2`, under Apache-2.0.
The applicable protocol repairs are implemented in the Windows Rust service.
Their validation does not establish end-to-end agent task quality or replace
the Windows real-model numerical and performance gates.
