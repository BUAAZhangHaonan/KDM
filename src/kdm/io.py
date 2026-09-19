"""Content-addressed manifests and transactional append-only records."""
from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path


def stable_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
                       separators=(",",":"),allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def within(root,path):
    root=Path(root).resolve(); path=Path(path)
    path=(root/path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root): raise ValueError(f"Output outside project: {path}")
    return path


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    fd,name=tempfile.mkstemp(prefix=".write-",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf8") as f:
            f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if Path(name).exists(): Path(name).unlink()


def read_jsonl(path):
    with open(path,encoding="utf8") as f:
        for i,line in enumerate(f,1):
            if not line.strip(): raise ValueError(f"Blank JSONL record {path}:{i}")
            try: value=json.loads(line)
            except json.JSONDecodeError as e: raise ValueError(f"Invalid JSONL {path}:{i}") from e
            if not isinstance(value,dict): raise ValueError(f"Object expected at {path}:{i}")
            yield value


class Ledger:
    """One writer per shard. A changed configuration cannot reuse a shard."""
    def __init__(self,path,identity):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.identity=stable_hash(identity);self.meta=self.path.with_suffix(".identity.json")
        if self.path.exists() and not self.meta.exists():
            raise ValueError("Existing ledger has no identity")
        if self.meta.exists():
            old=json.loads(self.meta.read_text())
            if old['identity']!=self.identity: raise ValueError("Run identity mismatch")
        else: atomic_json(self.meta,{"identity":self.identity,"definition":identity})
        self.keys=set()
        if self.path.exists():
            for r in read_jsonl(self.path):
                if r['key'] in self.keys: raise ValueError("Duplicate ledger key")
                if r['identity']!=self.identity: raise ValueError("Corrupt record identity")
                self.keys.add(r['key'])
    def add(self,key,record):
        if key in self.keys: raise ValueError("Duplicate record")
        if record.get('status','ok')!='ok': raise ValueError("Failures belong in a separate error log")
        row={**record,"key":key,"identity":self.identity}
        with open(self.path,"a",encoding="utf8") as f:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
            f.flush();os.fsync(f.fileno())
        self.keys.add(key)


def stable_seed(*parts):
    return int(stable_hash(list(parts))[:16],16) % (2**31)
