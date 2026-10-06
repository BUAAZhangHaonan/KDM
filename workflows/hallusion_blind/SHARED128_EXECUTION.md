# K100 128-token shared admission

The user authorized sharing a GPU when capacity allows. The Hall128 exclusive admission rejected this supported wrapper mode, so generate_shared128.py is a separate scheduling entry. It keeps all fifteen frozen generation/model sources, model precision, batch size, device maps, prompts and decoding settings unchanged.

The entry checks a shared main flock and an exclusive per-worker slot, the physical GPU UUID, current free memory and the actual registered peer process. It repeats the original checkpoint, environment and InternVL single-device gate checks, adds the original thirteen package states and frozen source hashes, and records its actual entry SHA and shared registry in owner.admission. It does not disguise a two-worker registry as exclusive execution.

InternVL owns slot 0 and Qwen3.5 owns slot 1. An original exclusive worker must finish and seal its current chunk before releasing its lock. Each recovered shard contains only verified missing keys in an already supported audit namespace. Original failed claims and receipts remain intact. A priority key changes task ordering only and must already belong to the assigned formal shard.
