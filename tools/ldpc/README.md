# Vendored LDPC Helpers

These are the three command-line helpers used by `src/LDPC.jl`. Their source
and random-number table are vendored under `src/` from Radford M. Neal's 2012
LDPC tools; see `src/COPYRIGHT` for the upstream terms.

The upstream build embeds an absolute path to `randfile`. The tracked helpers
use the repository-relative `src/randfile`, and `LDPC.jl` launches them from
this directory. This makes a given valid seed reproducible without the original
developer-home dependency.

Reference source is included for provenance and can rebuild the Linux helpers:

```bash
make -C tools/ldpc/src clean all
```

The legacy C random-number implementation is compiler-sensitive. A rebuild can
change generated parity matrices, so it must be followed by the LDPC
determinism/parity tests and a deliberate regeneration of affected performance
reports.
