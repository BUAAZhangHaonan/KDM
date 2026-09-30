#!/usr/bin/env bash
set -euo pipefail
CENTRAL=/home/g203-4028/projects/knowledge-deficit-mitigation
REMOTE=zhanghaonan@172.17.43.172
TARGET=/home/hdd3/zhanghaonan/projects/knowledge-deficit-mitigation/supplemental/remaining11/runtime_20260930
ssh -o BatchMode=yes "$REMOTE" "/home/hdd3/zhanghaonan/projects/knowledge-deficit-mitigation/.environments/base311/bin/python -m venv '$TARGET/.environments/phi443'"
rsync -aL -e 'ssh -o BatchMode=yes -o ConnectTimeout=12' "$CENTRAL/.environments/phi443/lib/python3.11/site-packages/" "$REMOTE:$TARGET/.environments/phi443/lib/python3.11/site-packages/"
rsync -a -e 'ssh -o BatchMode=yes -o ConnectTimeout=12' "$CENTRAL/workflows/supplemental/remaining11/host_registry.json" "$REMOTE:$TARGET/workflows/supplemental/remaining11/"
ssh -o BatchMode=yes "$REMOTE" "$TARGET/.environments/phi443/bin/python '$TARGET/workflows/supplemental/remaining11/inspect_environment.py'; printf 'PREPARE_4029_PHI_COMPLETE\n'"
