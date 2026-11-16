# Security policy

## Threat model

meshforge is a **local, single-user** tool: it runs on your machine, and the
Workbench binds `127.0.0.1` only. The copilot starts a Workbench on
`127.0.0.1` or `localhost` only. The one network call it makes is to the
Anthropic API, with the key you set in your own shell. It sends your
request, the CT's file path as you typed it, and the tools' results (numbers
and messages). The CT itself is never uploaded.

The user and their own command-line arguments are **trusted**. The real
trust boundary is the **files you open** (NIfTI/NRRD CT volumes, `.vtp`
surfaces, `.mfwork` work files, OpenFOAM cases given to `--flow-case`) and
the **output of external programs** (TotalSegmentator, CalculiX `.frd`/`.dat`,
OpenFOAM logs and results), all of which may be crafted or malformed.
Against that boundary the project's hard rule is **fail loudly, never fail
open**: a bad input must raise, not quietly produce a plausible-looking
result.

Out of scope: multi-tenant or networked deployment (the app refuses any
non-loopback host), privilege boundaries, and the `MESHFORGE_TOTALSEG` /
`MESHFORGE_CCX` / `MESHFORGE_OPENFOAM` executable-path overrides. Those are
documented, deliberately trusted settings, though the *output* of whatever
they point at is still treated as untrusted and cleaned before it reaches
your terminal.

## Reporting a vulnerability

Please report privately rather than opening a public issue: use
[GitHub's private vulnerability reporting](https://github.com/LeventS2005/meshforge/security/advisories/new)
on this repository. Include the crafted input (a *synthetic* one, never a
clinical file), the command or Workbench step that triggers it, and what
went wrong.

If that link is unavailable to you, open a public issue titled
`security contact request` containing **no details of the issue**, just a
request for a private channel, and you will be sent one.

This is a research project maintained by one person; expect a first
response within about two weeks, and no formal SLA beyond that.

Please do **not** send clinical or patient-derived data in a report under
any circumstances. A tiny synthetic fixture is enough to reproduce a
finding.

## Supported versions

Only the latest release on `main` receives fixes.
