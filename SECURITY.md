# Security policy

This is a Python library, not an authenticated multi-tenant server. Do not expose
it directly to the public internet. A deployment must supply authentication,
request limits, concurrency/queue limits and authorization for downstream actions.

## Implemented boundaries

- Local inference is offline; loading uses safetensors, strict state dictionaries
  and `trust_remote_code=False`. Only the supported ModernBERT architecture loads.
- Hub downloads require an explicit `from_hub` call. Python and pickle checkpoint
  files are excluded; no downloaded Python is executed. Use trusted repositories.
- A checkpoint folder's `config.json` and tokenizer files are parsed by Transformers, so load only
  checkpoints you trust. Transformers 4.x has published advisories (for example CVE-2026-4372, a
  malicious `config.json`) that are fixed in 5.x, which is why the package allows Transformers 5 and a
  fresh install uses it; the Transformers features those advisories concern (`Trainer`, checkpoint
  conversion, `save_pretrained`, remote code) are not used by this package.
- Provider use is explicit. Full request context and option text are sent to the
  selected provider. No automatic local-to-cloud fallback exists.
- Provider SDKs use explicit official HTTPS endpoints, finite request timeouts,
  bounded retries, no tools, and no dynamic code execution. OpenAI requests set
  `store=False`; this does not promise zero provider retention.
- Public provider errors use sanitized codes, not upstream bodies or messages. They are raised
  without a chained cause or context, so the original SDK exception (which holds the HTTP request,
  including the API key header and the request text) is not kept alive behind them. The same
  holds for JSON parse errors, which would otherwise keep the whole document in `.doc`.
  Do not enable verbose upstream HTTP logging with private data or credentials.
- Provider output is locally checked for an exact integer option index, no extra
  keys and no duplicate keys. Refusals/incomplete outputs never become decisions.
- Model text is untrusted. Prompt separation and JSON validation restrict output
  form, but cannot guarantee resistance to prompt injection or correct decisions.
- The package does not log request context or API keys. Caller-owned clients,
  debuggers, exception-local capture and provider infrastructure are outside this
  guarantee. In particular, tools that record the *local variables* of every frame a
  traceback passes through (`show_locals`, some crash reporters) can still reach objects
  held by those frames, such as your own `Request` and `ProviderDecision._client`, which
  holds the API key. Turn local-variable capture off in production or scrub `api_key`
  fields. Do not publish crash dumps or environment dumps.
- Every exception the package defines can be pickled and copied, so errors raised in
  worker processes (`ProcessPoolExecutor`, `multiprocessing`, Celery, Ray, Dask) reach the
  parent intact.

## SystemOne adapter and reference server

`SystemOne` validates every request before inference, rejects images, unknown fields and over-long
input instead of dropping or truncating them, bounds the work per request (`max_questions`,
`max_state_chars`), and never puts request text into error messages. `examples/systemone_server.py`
is a reference, not a hardened server: it binds to loopback, refuses a non-loopback address without an
API key, caps the body size, never logs bodies, and returns JSON errors only, but TLS, rate limiting,
quotas and multi-tenant isolation remain the deployer's responsibility.

## Repository settings to enable after creating the repository

Enable branch protection, required Checks, secret scanning/push protection where
available, and private vulnerability reporting. Workflow tokens are read-only;
PR workflows do not use `pull_request_target` or deployment credentials. Enable
Dependabot. Review/pin action revisions before production release according to
your organization's supply-chain policy. Never put keys, private datasets or
checkpoint archives in Git history. `.gitignore` does not remove tracked secrets.

Report vulnerabilities through the repository's private vulnerability reporting
once configured. Do not file public issues containing credentials or private inputs.
Until a repository exists, the distribution has no configured reporting endpoint.

## Verification limits

No penetration test or blanket security certification is claimed. Provider tests
are offline contract mocks, not live API acceptance. CI configuration is supplied;
its dependency audit and Bandit results must pass in the actual repository.
Runtime GPU dependency auditing is separate from the lightweight provider CI job.
