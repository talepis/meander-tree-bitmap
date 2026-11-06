# Tree and bitmap geometry for exact closed meander counts

Public research source and computational evidence, 6 November 2026. Author: **Efthimios Alepis** ([ORCID 0000-0002-3862-7218](https://orcid.org/0000-0002-3862-7218)). The author's code and documentation use the MIT license; dependencies keep their own licenses.

This is a separate implementation lineage from the [earlier C5/V2 project](https://github.com/talepis/meander-checked-c5-v2). The method counts pairs of noncrossing matchings whose union is one cycle. It uses weighted tree/centroid states, sparse frontier geometry, cropped bitmap supports, exact integer coefficient transfers, and an eight-crossing terminal operator. The M38 run partitions the lower-matching families into four groups with rank-64 geometry. No GPU or disk-backed coefficient arrays were used.

## New values and measured resources

| n | M_n | Revision | Wall seconds | Peak sampled RSS bytes | Matrix read/write bytes | Modular check |
|---:|---:|---|---:|---:|---:|---|
| 35 | 211433268000094602372146238438566 | tree/centroid | 31603.365278962 | 828465958912 | 0/0 | two modular passes, matching |
| 36 | 2358261838751729019212344104926012 | parallel geometry | 2191.353680423 | 945597001728 | 0/0 | two modular passes, matching |
| 37 | 26370652969449510879209096913242792 | bitmap + memory release | 5689.257196069 | 1771293687808 | 0/0 | 11 family modular passes, matching |
| 38 | 295598081822804246724487760690994952 | rank-64 four-family | 12387.379102375358 | 1599349862400 | 0/0 | not yet done |

These are distinct revisions, not a controlled same-order speed benchmark. RSS is measured physical process memory, not the coefficient-array accounting. The M37 modular passes share the tree grammar and bitmap geometry with the exact algorithm, so they are **not a fully independent second algorithm**. M38 has no CRT. The checks establish reproducible numerical agreement within stated bounds; they do not constitute a formal proof of the entire program. See `docs/VALIDATION.md` and `docs/OEIS_EVIDENCE_M35_M38_EN.pdf`.

## Reproduce and inspect

Run `python3 verify_release.py` to check every file's SHA-256 digest. See `docs/METHOD.md`, `docs/RESULTS.md`, and `docs/VALIDATION.md`. The exact executed source packages, untouched within this release, are in `source_frozen/` (M37), `source_frozen_m38/`, and `source_crt_m37/`. Each has its original source manifest. The M38 launch procedure is in `source_frozen_m38/START_M38.sh`; the CRT source has its own verifier. `evidence/` contains original final and per-stage records. The candidate replacement OEIS b-file is `oeis/b005315.txt`; the public OEIS entry may still end at n=34 until editorial approval.

The full M37 and M38 runs required a Linux host with approximately 1.8 TB and 1.6 TB sampled peak RSS, respectively, and 16 OpenMP threads. These resource observations are not minimum requirements. Memory guards and preflight checks may stop a run safely on smaller hardware.

## Credit and use in OEIS

The prior M29/M30 contribution by Vladimir Osipov is acknowledged in `docs/RELATED_WORK.md` and the public OEIS entry. No priority for those terms is claimed. `oeis/UPDATE_CHECKLIST_EN.md` explains the author review and upload steps. Publication of this repository does not by itself submit the four new terms to OEIS. Any new Zenodo DOI should be cited only after an actual record exists. Documents were assembled with AI assistance and require the author's scientific review before OEIS submission.
