# V8 runtime measurement

This separate engineering check measures the portable decoder selected by the
V8 development study. It does not change the scientific producer, selection,
fresh cohort, or accuracy gates. The [runtime plan](runtime-plan.json) and
[implementation](runtime.py) are committed before timing begins.

The representative is the first T2 session in the factorial development split,
archive rank 16, using the selected primary model already exported by that run.
Two 60-second replays process the first 3,000 observations at 50 Hz. Each replay
warms the interpreter with 500 steps, resets filter state, and preserves the
recorded acquisition-block resets. The benchmark retains every prediction and
raw release/start/completion/deadline clock locally, including deadline misses.
No labels or behavioral context enter runtime prediction.

The model was fitted using labels through 150 seconds, which are later than these
replay inputs. Consequently, parity over this interval is a numerical check,
not causal accuracy evidence. The service timing includes the feature-row
dictionary, block reset, input validation, transform, filters, normalization,
linear head and unit-disk projection. Release-to-completion timing additionally
includes ordinary operating-system scheduling delay. Input acquisition,
upstream feature extraction, transport, file loading and fitting are excluded.
The two finite runs cannot establish a hard realtime guarantee.

The report will give both runs separately, process CPU consumption, exact model
and filter numeric storage, and model artifact size. Numeric component storage
excludes Python overhead, scratch arrays and the input cache. Public output will
contain aggregate timing and identities only, after independent verification.

No timing outcome has been recorded in this document yet.
