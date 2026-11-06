# Method and computational model

A closed meander of order n can be represented by an upper and a lower noncrossing perfect matching on 2n linearly ordered crossings whose union is one cycle. The output is the number of such closed configurations.

The frozen implementation constructs a deterministic weighted grammar for lower paths. It distinguishes the first completed branch and uses weighted normalization to avoid enumerating equivalent unrestricted suffixes repeatedly. Upper/lower frontier support is represented sparsely. The selected M37 geometry engine constructs cropped supports with bitmaps and translates them into coefficient blocks. Parallel OpenMP/Numba kernels perform the integer transfers; exact source maxima determine block capacities before numerical writes.

The first 2n-8 crossings are processed as numbered transfer stages. An exact finite recurrence constructs a universal table for the final eight crossings, which is contracted with the surviving frontier counts. Thus these crossings are computed, not omitted or replaced by an unproved closed formula. The eight-crossing horizon is an implementation parameter; changing it does not change the combinatorial definition.

The memory policy uses Linux `madvise(MADV_DONTNEED)` on dead private buffer pages and shortens the lifetime of obsolete translation indices and source supports. It preserves active values and does not alter the numerical kernels or the proven capacity checks. It is not a proof of lower asymptotic complexity. The M37 computation used 16 OpenMP threads, a 1,900 GiB cgroup hard limit and a 16 GiB reserve. Matrix spill was forbidden.

The source package includes experimental and historical modules as well as the selected path. The actual M37 command and selected flags are recorded in `source_frozen/START_M37.sh`, `evidence/m37/PREFLIGHT.json` and `evidence/m37/RESULT.json`. The exact input sources are identified by the frozen SHA-256 manifest. Historical M35/M36 measurements are from related revisions and are kept as separate evidence.

## M38 extension

The M38 revision divides lower-matching families into four disjoint groups. A rank-64 indexed bitmap geometry and checked exact transfers process 68 stages per group; the four terminal contributions are summed. The M37 modular revision evaluates the same family partition in coprime modular rings. This shared combinatorial grammar is the reason the modular check is numerical confirmation, not an independent structural proof. See the untouched code and source manifests in `source_frozen_m38/` and `source_crt_m37/`.
