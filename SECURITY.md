# Security Policy

## Reporting a vulnerability

Report privately, not as a public issue.

- **GitHub Security Advisories** — [open a draft advisory](https://github.com/tuiml/tuiml/security/advisories/new). Preferred: it keeps the report private until a fix ships and gives us a place to work with you on it.
- **Email** — nilesh.verma@waikato.ac.nz

Please include what you were running (version, OS, Python), what happens, and the smallest thing that reproduces it. A proof of concept helps but is not required to report.

We aim to acknowledge within a week. TuiML is maintained by a small academic group, so please allow reasonable time before disclosing publicly. Tell us if you have a disclosure deadline and we will work to it.

## Supported versions

TuiML is pre-1.0 and alpha. Only the latest release gets fixes; there are no backports to earlier ones.

| Version | Supported |
|---------|-----------|
| 0.1.x (latest) | Yes |
| anything older | No — upgrade |

## What is in scope

The parts of TuiML that take input from somewhere other than the person running it:

- **The MCP server** (`tuiml-mcp`, `tuiml.agent`). The threat model is an LLM agent driving the tools, possibly steered by untrusted content in the data it was asked to analyse. A tool that escapes its directory, executes attacker-chosen code, or reaches the network unprompted is a vulnerability.
- **The serving API** (`tuiml.serving`). Authentication bypass, unbounded deserialisation, or anything that lets a request reach a file outside `models_dir`.
- **The installers** (`website/static/install.sh`, `install.ps1`), which run with elevated privileges.
- **Deserialisation** of models in `tuiml.utils.serialization` beyond the documented behaviour below.

## What is not

Some of these look like vulnerabilities and are documented behaviour. Reporting them is not useful, but arguing that the documentation is wrong is welcome as a normal issue.

**Loading a model executes code.** Models are pickles. `tuiml.utils.serialization.load_model()`, `tuiml_predict` with a `model_path`, and `POST /models` all deserialise, and a pickle can run anything on unpickling. Load models you produced or trust, exactly as you would with joblib or `pickle` directly. The bound on `POST /models` — a `models_dir` you configure, with no HTTP loading at all until you do — exists because of this, not in spite of it.

**Agent-authored algorithms are not sandboxed.** `tuiml_create_algorithm` writes Python that then runs in-process. A static AST check refuses imports outside a numerical allowlist and re-runs before every load, but it is a source check, not a sandbox: it raises the cost of getting code to run and does not make execution safe. Do not point an untrusted agent at a machine you care about. Running these in a subprocess under resource limits is planned.

**`auth_token=False` or `--no-auth` disables authentication.** That is the point of the flag. It is for deployment behind a proxy that authenticates instead.

**Binding a non-loopback host exposes the server.** `serve(host="0.0.0.0")` warns, and refuses outright when authentication is off. If you pass both deliberately, the exposure is yours to manage — and the traffic is unencrypted, so put TLS in front of it.

**A model can be adversarially manipulated** through poisoned training data, and predictions can leak properties of the training set. These are real concerns and belong in the ML-security literature; they are not TuiML implementation bugs.

## What TuiML does not do

There is no telemetry, analytics, crash reporting, or phone-home anywhere in the codebase. The single outbound request is a PyPI version check that sends the package name and nothing else. No dataset, model, or query leaves the machine unless you explicitly serve it.

Model weights for `tuiml[foundation]` are downloaded by the upstream package from its own hub — a transaction between you and that publisher, under their licence. TuiML never mirrors or ships weights.
