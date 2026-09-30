"""Print a compact, read-only environment and checkpoint path inventory."""
import importlib.metadata
import json
from pathlib import Path
import sys

NAMES = ('torch', 'torchvision', 'transformers', 'tokenizers', 'accelerate',
         'numpy', 'Pillow', 'sentencepiece', 'timm', 'safetensors', 'einops',
         'scipy', 'requests')
installed = {d.metadata['Name'].lower(): d.version
             for d in importlib.metadata.distributions() if d.metadata['Name']}
print(json.dumps({'python': sys.executable,
                  'versions': {n: installed.get(n.lower()) for n in NAMES},
                  'paths': {p: {'exists': Path(p).exists(),
                                'resolved': str(Path(p).resolve())}
                            for p in sys.argv[1:]}}, indent=2))
