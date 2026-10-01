# Immutable received-score union

`merge_received_scores.py` accepts explicit passed CPU score cohorts and verifies
their recorded file SHA, row count, unique generation keys and registered datasets.
It copies each complete score object unchanged into a new exclusive output directory.
It does not reopen raw predictions, rejoin or replace references, rescore, or use GPUs.

Actual validation: `outputs/supplemental/remaining4/received139185_20261001_2140_cpu/receipt.json`
passed for 139185 unique keys: Food-101 72776 and VizWiz 66409. The union consists
of the accepted 123065-row cohort and 16120 additional immutable K100 replies.
This is a received CPU cohort count, not completion of all registered experiments.
