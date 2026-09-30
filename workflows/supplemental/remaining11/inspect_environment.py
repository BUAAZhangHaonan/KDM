"""Print a compact, read-only environment and checkpoint path inventory."""
import importlib.metadata
import json
from pathlib import Path
import sys
import torch
import transformers

NAMES = ('torch', 'torchvision', 'transformers', 'tokenizers', 'accelerate',
         'numpy', 'Pillow', 'sentencepiece', 'timm', 'safetensors', 'einops',
         'scipy', 'requests')
installed = {d.metadata['Name'].lower() for d in importlib.metadata.distributions()
             if d.metadata['Name']}
print(json.dumps({'python': sys.executable,
                  'loaded_modules': {'torch': {'version': torch.__version__,
                                                'path': torch.__file__},
                                     'transformers': {'version': transformers.__version__,
                                                      'path': transformers.__file__}},
                  'versions': {n: importlib.metadata.version(n) if n.lower() in installed else None
                               for n in NAMES},
                  'paths': {p: {'exists': Path(p).exists(),
                                'resolved': str(Path(p).resolve())}
                            for p in sys.argv[1:]}}, indent=2))
