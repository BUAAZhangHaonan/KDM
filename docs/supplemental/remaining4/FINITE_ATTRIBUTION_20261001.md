# Supplemental finite attribution, 2026-10-01

The supplemental comparison uses the newly generated native VCD baseline against
registered instruction-preserving VCD. Existing five-model guided-baseline data
and its registration remain frozen.

`prepare_registered_detail_sources.py` prepares Qwen3-VL inputs from the original
six named diagnostic strata. It joins the complete 2,424-input native/IP panel
to resolved Food references, selects at most eight inputs per stratum, preserves
empty strata, and reads only the selected original raw lines. Actual saved token
paths determine the first shared-prefix divergence; no termination token is
synthesized. The earlier finite source cache and cumulative budget are reused.

`measure_details.py` measures bounded diagnostic positions and complete real
reply paths through the existing four-view implementation and admitted model
runtime. It checks sample, seed, marker, reference, source identity and token
prefixes, reuses verified first-position events, and enforces the recorded budget.

Actual received verification:
`outputs/supplemental/remaining4/dispatch_20261001_1600/qwen3_details_immutable_event_20261001_1840/received_verification.json`.
It verifies 21 source files, 32 unique diagnostic inputs and three complete cases.
The maximum decomposition/IP closure residual is 3.552713678800501e-15, with
212.29530226089992 GPU seconds used against a 2,880-second model cap. The other
three supplemental models' detailed-case coverage remains incomplete.

The two execution sources match their actual immutable provenance hashes:
`946fd4dfc171920aaad05c1b9c925097890d5147b67883fdabfb84f31c2c48c7`
and `beb38310e22517026651cffffeac18be0812cec6ddd0d0847bfe7c4c65b6a2c3`,
respectively for measurement and named-strata preparation. Validation comprises
actual received measurement/source checks and Python compilation; no additional
GPU generation was launched for this commit.
