#!/usr/bin/env python3
"""Offline, standard-library-only integrity check of this public export."""
from pathlib import Path
import ast,csv,hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b""): h.update(chunk)
 return h.hexdigest()
def numeric(value,path=()):
 if isinstance(value,dict):
  for key in sorted(value): yield from numeric(value[key],path+(key,))
 elif isinstance(value,list):
  for index,child in enumerate(value): yield from numeric(child,path+(index,))
 elif isinstance(value,(int,float)) and not isinstance(value,bool): yield path,value
errors=[]
manifest=[]
for line in (ROOT/"SHA256SUMS").read_text(encoding="utf-8").splitlines():
 expected,relative=line.split("  ",1); path=ROOT/relative; manifest.append(relative)
 if not path.is_file() or digest(path)!=expected: errors.append("hash mismatch or missing: "+relative)
actual={str(p.relative_to(ROOT)).replace(chr(92),"/") for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.relative_to(ROOT).parts and "__pycache__" not in p.relative_to(ROOT).parts}
if actual != set(manifest)|{"SHA256SUMS"}: errors.append("file inventory differs from manifest")
exports=json.loads((ROOT/"EXPORTS.json").read_text(encoding="utf-8"))
for entry in exports["json_exports"]:
 value=json.loads((ROOT/entry["path"]).read_text(encoding="utf-8"))
 rows=[]
 for path,number in numeric(value):
  encoded="NaN" if isinstance(number,float) and math.isnan(number) else repr(number)
  rows.append([list(path),encoded])
 fingerprint=hashlib.sha256(json.dumps(rows,ensure_ascii=True,separators=(",",":")).encode()).hexdigest()
 if len(rows)!=entry["numeric_leaf_count"] or fingerprint!=entry["numeric_sha256"]: errors.append("numeric fingerprint mismatch: "+entry["path"])
for entry in exports["synthetic_csv_exports"]:
 with (ROOT/entry["path"]).open(newline="") as stream:
  reader=csv.reader(stream); header=next(reader); count=sum(1 for _ in reader)
 if header!=entry["columns"] or count!=entry["rows"]: errors.append("synthetic CSV shape mismatch: "+entry["path"])
for path in ROOT.rglob("*.py"):
 if ".git" not in path.relative_to(ROOT).parts: ast.parse(path.read_text(encoding="utf-8"),filename=str(path.relative_to(ROOT)))
print(json.dumps({"verified":not errors,"manifest_files":len(manifest),"aggregate_and_synthetic_json":len(exports["json_exports"]),"synthetic_csv":len(exports["synthetic_csv_exports"]),"errors":errors},indent=2))
sys.exit(1 if errors else 0)
