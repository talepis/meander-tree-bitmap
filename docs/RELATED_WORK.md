# Related work and priority

- [OEIS A005315](https://oeis.org/A005315) is the established closed-meander sequence. Its public b-file was checked through n=34 when this candidate was assembled.
- [Efthimios Alepis, earlier checked C5/V2 work](https://doi.org/10.5281/zenodo.23687622) and [source](https://github.com/talepis/meander-checked-c5-v2) describe a separate implementation lineage.
- [Vladimir Osipov, Exact enumeration of closed and open meanders in O*(2^n) time and space via a first-crossing partition of matching pairs](https://doi.org/10.5281/zenodo.22773575), 2026, reported M29 and M30 earlier. The A005315 entry already credits him. We claim no priority for those two terms.

Measured time and RAM do not imply a proven new asymptotic complexity. The M35-M37 modular passes reuse important structure of their exact runs. M38 is currently supported by one exact four-family run and internal checks, with no CRT.
