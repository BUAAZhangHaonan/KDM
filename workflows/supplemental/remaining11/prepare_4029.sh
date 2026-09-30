#!/usr/bin/env bash
set -euo pipefail
CENTRAL=/home/g203-4028/projects/knowledge-deficit-mitigation
REMOTE=zhanghaonan@172.17.43.172
TARGET=/home/hdd3/zhanghaonan/projects/knowledge-deficit-mitigation/supplemental/remaining11/runtime_20260930
SSH_OPTIONS='ssh -o BatchMode=yes -o ConnectTimeout=12'
ssh -o BatchMode=yes "$REMOTE" "mkdir -p '$TARGET/.environments' '$TARGET/cache/models' '$TARGET/data/current' '$TARGET/data/provenance' '$TARGET/outputs/records' '$TARGET/workflows/supplemental/remaining11'; /home/hdd3/zhanghaonan/projects/knowledge-deficit-mitigation/.environments/base311/bin/python -m venv '$TARGET/.environments/tf553'"
rsync -aL -e "$SSH_OPTIONS" "$CENTRAL/.environments/mprisk-tf553/lib/python3.11/site-packages/" "$REMOTE:$TARGET/.environments/tf553/lib/python3.11/site-packages/"
rsync -aL -e "$SSH_OPTIONS" "$CENTRAL/cache/models/llava-1.5-7b-hf/" "$REMOTE:$TARGET/cache/models/llava-1.5-7b-hf/"
rsync -aL -e "$SSH_OPTIONS" "$CENTRAL/cache/models/Phi-3.5-vision-instruct/" "$REMOTE:$TARGET/cache/models/Phi-3.5-vision-instruct/"
rsync -a -e "$SSH_OPTIONS" "$CENTRAL/src" "$CENTRAL/configs" "$CENTRAL/workflows" "$REMOTE:$TARGET/"
rsync -a -e "$SSH_OPTIONS" "$CENTRAL/data/current/" "$REMOTE:$TARGET/data/current/"
rsync -a -e "$SSH_OPTIONS" "$CENTRAL/data/provenance/frozen_contract" "$REMOTE:$TARGET/data/provenance/"
rsync -a -e "$SSH_OPTIONS" "$CENTRAL/data/images" "$REMOTE:$TARGET/data/"
rsync -a -e "$SSH_OPTIONS" "$CENTRAL/outputs/records/image_content_catalog.json" "$REMOTE:$TARGET/outputs/records/"
ssh -o BatchMode=yes "$REMOTE" "$TARGET/.environments/tf553/bin/python '$TARGET/workflows/supplemental/remaining11/inspect_environment.py'; printf 'PREPARE_4029_COMPLETE\n'"
