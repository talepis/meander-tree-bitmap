# Results and measured resources

| n | M_n | Measured wall seconds | Peak sampled RSS bytes | Matrix I/O bytes | Source |
|---:|---:|---:|---:|---:|---|
| 35 | 211433268000094602372146238438566 | 31603.365278962 | 828465958912 | 0/0 | `evidence/m35/` |
| 36 | 2358261838751729019212344104926012 | 2191.353680423 | 945597001728 | 0/0 | `evidence/m36/` |
| 37 | 26370652969449510879209096913242792 | 5689.257196069 | 1771293687808 | 0/0 | `evidence/m37/` |
| 38 | 295598081822804246724487760690994952 | 12387.379102375358 | 1599349862400 | 0/0 | `evidence/m38/` |

M38: 4 consecutive family groups, 68 transfer stages each (272 total). Group contributions were 53022188056440864580946848895551542, 77371847600821090535942536771124028, 128106091870808593177303427240780026, and 37097954294733698430294947783539356. Their exact sum is M38. The controller session time is the reported wall time; the numerical worker time was 12381.828537923284 s. Peak source-plus-target coefficient arena accounting was 1247591018336 bytes, distinct from RSS.

M37: 66 exact stages. Eleven fresh modular family passes reconstructed all four contributions, whose sum matches the exact M37. Each pass has 66 stages. Per-pass times and RSS are in `evidence/NEW_NUMERIC_AUDIT.json`; full raw stage files are in `evidence/crt_m37/raw/`. M35 and M36 have separate modular reconstructions under `evidence/crt_m35_m36/`.

Each run used 16 OpenMP numerical threads. The figures span different code revisions and orders; they should not be compared as a controlled speed benchmark. Zero matrix I/O excludes ordinary logs and JIT cache traffic.

