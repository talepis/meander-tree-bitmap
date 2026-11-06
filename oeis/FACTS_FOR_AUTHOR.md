# Factual notes for the A005315 contributor

New consecutive terms:

| n | M_n | Evidence |
|---:|---:|---|
| 35 | 211433268000094602372146238438566 | exact run and two matching shared-geometry modular passes |
| 36 | 2358261838751729019212344104926012 | exact run and two matching shared-geometry modular passes |
| 37 | 26370652969449510879209096913242792 | exact run and 11 matching shared-geometry family modular passes |
| 38 | 295598081822804246724487760690994952 | exact four-family run, 272 stages; no CRT yet |

Public source: https://github.com/talepis/meander-tree-bitmap (verify after publication). Historical related source: https://github.com/talepis/meander-checked-c5-v2. Earlier related DOI: https://doi.org/10.5281/zenodo.23687622. Vladimir Osipov's earlier M29/M30 work: https://doi.org/10.5281/zenodo.22773575. A new Zenodo DOI for this tree release has not been assigned here.

The modular checks share structural code with the exact computation and are not fully independent. M38 has no modular check. No claim of a formal complexity improvement or fully proved software stack is made. The author should decide the exact scientific wording for OEIS.
