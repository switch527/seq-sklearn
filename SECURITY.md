# Security policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.x (pre-release) | Latest only |

Security patches will be applied to the latest released MINOR. Once v1
ships, the policy will widen to cover the prior MINOR for a documented
window.

## Reporting a vulnerability

Please report security vulnerabilities privately rather than opening
public issues. Email the maintainer at the address listed in
`pyproject.toml` `[project.authors]` or use GitHub's private
vulnerability reporting form on the repository.

We aim to acknowledge reports within 5 business days and to coordinate
disclosure timing once a fix is available.

## Notes on safe deserialization

`seq-sklearn` uses safetensors plus JSON for the save / load artifact
format (see `docs/architecture.md` A17). No `torch.load` is invoked on
user-supplied paths, and there is no `trust=True` /
`weights_only=False` escape hatch in v1. A future release that
requires pickled state would be a MINOR bump with an explicit security
note.
