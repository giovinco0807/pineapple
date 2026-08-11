#!/usr/bin/env bash
# Label-generation worker startup for the learned-evaluator program.
#
# The fetch half is the proven skeleton from the dataset transport: content
# bindings arrive as base64 canonical JSON in instance metadata, every object
# is downloaded by generation and verified by size and sha256 before use, and
# writes are create-only. The execute half runs the portable label worker in
# bounded increments, publishing create-only position files and heartbeats.
# Replacements restore generation-pinned objects locally, and one fixed
# complete checkpoint is published only after the full file inventory.
set -euo pipefail
umask 077
ROOT=/var/lib/ofc-m31-labelgen-v1
STAGING=$ROOT/staging
LOG=/var/log/ofc-m31-labelgen-v1
META='http://metadata.google.internal/computeMetadata/v1/instance/attributes'
HEADER='Metadata-Flavor: Google'
mkdir -p "$STAGING" "$LOG"
chmod 0700 "$ROOT" "$STAGING" "$LOG"
# Mirror everything to the serial console: the first fleet died in nine
# seconds with its only explanation locked inside an unreachable disk.
exec > >(tee -a "$LOG/startup.log" >/dev/console) 2>&1
meta() { curl -fsS -H "$HEADER" "$META/$1"; }
BUCKET="$(meta lg-bucket)"
SHARD_ID="$(meta lg-shard-id)"
ATTEMPT_ID="$(meta lg-attempt-id)"
PREFIX="$(meta lg-object-prefix)"
PLAN_SHA="$(meta lg-plan-sha256)"
BINDINGS_B64="$(meta lg-content-bindings-b64)"
WATCHDOG="$(meta lg-watchdog-seconds)"
WORKERS="$(meta lg-worker-count)"
( sleep "$WATCHDOG"; shutdown -h now ) &
WATCHDOG_PID=$!
trap 'kill "$WATCHDOG_PID" >/dev/null 2>&1 || true' EXIT

python3 - "$STAGING" "$BUCKET" "$BINDINGS_B64" <<'PY'
import base64,hashlib,json,os,pathlib,sys,urllib.parse,urllib.request
root=pathlib.Path(sys.argv[1]).resolve(); bucket=sys.argv[2]
raw=base64.b64decode(sys.argv[3],validate=True); rows=json.loads(raw)
canon=lambda v: json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
if raw!=canon(rows) or not isinstance(rows,list): raise SystemExit("bad bindings")
token=json.load(urllib.request.urlopen(urllib.request.Request(
 "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
 headers={"Metadata-Flavor":"Google"})))["access_token"]
for row in rows:
 if set(row)!={"relative","object","generation","sha256","bytes"}: raise SystemExit("binding fields")
 rel=pathlib.PurePosixPath(row["relative"])
 if rel.is_absolute() or ".." in rel.parts: raise SystemExit("unsafe binding")
 url=("https://storage.googleapis.com/download/storage/v1/b/"+urllib.parse.quote(bucket,safe="")+
      "/o/"+urllib.parse.quote(row["object"],safe="")+"?alt=media&generation="+row["generation"])
 data=urllib.request.urlopen(urllib.request.Request(url,headers={"Authorization":"Bearer "+token}),timeout=600).read()
 if len(data)!=row["bytes"] or hashlib.sha256(data).hexdigest()!=row["sha256"]: raise SystemExit("content drift")
 path=root.joinpath(*rel.parts); path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("xb") as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
PY

python3 - "$STAGING" "$PLAN_SHA" <<'PY'
import hashlib,pathlib,shutil,sys,tarfile,zipfile
root=pathlib.Path(sys.argv[1]); plan_sha=sys.argv[2]
def untar(source,target,top):
 with tarfile.open(source,"r:*") as a:
  for m in a.getmembers():
   p=pathlib.PurePosixPath(m.name)
   if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0]!=top or m.issym() or m.islnk(): raise SystemExit("unsafe tar")
  a.extractall(target)
def unzip(source,target,top):
 with zipfile.ZipFile(source) as a:
  for n in a.namelist():
   p=pathlib.PurePosixPath(n)
   if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0]!=top: raise SystemExit("unsafe zip")
  a.extractall(target)
untar(next((root/"static/runtime_archive").iterdir()),root,"runtime")
unzip(next((root/"static/wheelhouse_archive").iterdir()),root,"wheelhouse")
plan=next((root/"static/plan").iterdir())
if hashlib.sha256(plan.read_bytes()).hexdigest()!=plan_sha: raise SystemExit("plan drift")
shard=root/"shard"; shard.mkdir()
resume=root/"resume"
if resume.exists():
 for p in resume.rglob("*"):
  if p.is_file():
   q=shard/p.relative_to(resume); q.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,q)
PY

# No venv: the debian-12 image ships python3 without python3-venv (the second
# fleet died there), and pip is unnecessary anyway -- wheels are archives, and
# extracting them preserves the RPATH-relative native libraries these
# particular wheels rely on.
mkdir -p "$ROOT/pylib"
python3 - "$STAGING" "$ROOT" <<'PY'
import pathlib,sys,zipfile
staging,root=map(pathlib.Path,sys.argv[1:3])
for wheel in sorted((staging/"wheelhouse").glob("*.whl")):
 with zipfile.ZipFile(wheel) as archive: archive.extractall(root/"pylib")
print(f"extracted {len(list((staging/'wheelhouse').glob('*.whl')))} wheels")
PY
export PYTHONPATH="$ROOT/pylib:$STAGING/runtime/src"
cd "$STAGING/runtime"
PLAN="$(find "$STAGING/static/plan" -type f -maxdepth 1)"

# --- what the previous attempt already published -----------------------------
# A recreated Spot VM has an empty disk.  Names alone are not a resume: the
# final checkpoint inventories local bytes, and a manifest-only skip leaves
# those prior positions out of that inventory forever.  The Python parent below
# generation-pins, downloads, size/SHA/provenance-verifies, and restores every
# existing position into the local shard before it starts a worker.
MANIFEST="$ROOT/existing_positions.txt"
: >"$MANIFEST"
CARRY="$STAGING/static/manifest/carry_manifest.txt"
if [ -f "$CARRY" ]; then
  echo "FATAL: name-only carry manifests cannot satisfy the complete-checkpoint inventory; bind generation/bytes/SHA objects in a fresh run"
  exit 1
fi

python3 - "$STAGING" "$ROOT" "$BUCKET" "$PREFIX" "$PLAN_SHA" "$SHARD_ID" "$ATTEMPT_ID" "$PLAN" "$WORKERS" "$MANIFEST" <<'PY'
import datetime,hashlib,json,os,pathlib,subprocess,sys,threading,time,urllib.error,urllib.parse,urllib.request,uuid
staging,root=map(pathlib.Path,sys.argv[1:3]); bucket,prefix,plan_sha,shard,attempt=sys.argv[3:8]
plan=pathlib.Path(sys.argv[8]); workers=int(sys.argv[9]); out=staging/"shard"
manifest=pathlib.Path(sys.argv[10])
canon=lambda v: json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def expected_position_schema(value):
 kind=value.get("plan_kind")
 schemas={None:"hu_m31_label_gen_position_v1","t3_vs_fl":"hu_m31_label_gen_t3_vs_fl_position_v1",
  "t2_vs_fl":"hu_m31_label_gen_t2_vs_fl_position_v1","t1_vs_fl":"hu_m31_label_gen_t1_vs_fl_position_v1",
  "t0_first_vs_fl":"hu_m31_label_gen_t0_first_vs_fl_position_v1"}
 if kind not in schemas: raise SystemExit(f"unsupported cached-position plan_kind {kind!r}")
 return schemas[kind]
def restore_cached_position_object(destination,raw,*,object_name,expected_object_name,generation,
 expected_bytes,expected_sha256,plan_sha256,offset,position_schema):
 if object_name!=expected_object_name: raise ValueError("cached object name drifted")
 if not isinstance(generation,str) or not generation.isdigit() or int(generation)<=0:
  raise ValueError("cached object generation drifted")
 if type(expected_bytes) is not int or expected_bytes<=0 or len(raw)!=expected_bytes:
  raise ValueError("cached object byte count drifted")
 if (not isinstance(expected_sha256,str) or len(expected_sha256)!=64 or
     any(ch not in "0123456789abcdef" for ch in expected_sha256)):
  raise ValueError("cached object has no canonical sha256 metadata")
 actual=hashlib.sha256(raw).hexdigest()
 if actual!=expected_sha256: raise ValueError("cached object sha256 drifted")
 try: payload=json.loads(raw)
 except (UnicodeDecodeError,json.JSONDecodeError) as error: raise ValueError("cached position is not JSON") from error
 if raw!=canon(payload): raise ValueError("cached position is not canonical JSON")
 if (not isinstance(payload,dict) or payload.get("schema")!=position_schema or
     payload.get("plan_sha256")!=plan_sha256 or type(payload.get("offset")) is not int or
     payload.get("offset")!=offset): raise ValueError("cached position provenance drifted")
 if destination.exists():
  if destination.read_bytes()!=raw: raise ValueError("local resume copy disagrees with GCS")
 else:
  with destination.open("xb") as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
 return actual
def complete_checkpoint_object_name(value): return value+"/checkpoints/complete.json"
def build_complete_checkpoint(*,plan_sha256,shard_id,attempt_id,shard_start,shard_count,files):
 rows=[dict(row) for row in files]
 expected={"SHARD_DONE.json"}|{f"position_{offset:08d}.json" for offset in range(shard_start,shard_start+shard_count)}
 relatives=[row.get("relative_path") for row in rows]
 if len(relatives)!=len(set(relatives)) or set(relatives)!=expected:
  raise ValueError("complete checkpoint does not inventory the whole shard")
 payload={"schema":"hu_m31_label_gen_complete_checkpoint_v1","checkpoint_kind":"complete",
  "plan_sha256":plan_sha256,"shard_id":shard_id,"attempt_id":attempt_id,
  "completed_position_count":shard_count,"complete":True,"files":rows,
  "checkpoint_published_after_files":True,"create_only":True}
 payload["checkpoint_sha256"]=hashlib.sha256(canon(payload)).hexdigest(); return payload
plan_payload=json.loads(plan.read_bytes())
matches=[entry for entry in plan_payload["shards"] if entry.get("shard_id")==shard]
if len(matches)!=1: raise SystemExit(f"plan has {len(matches)} entries for shard {shard!r}")
shard_spec=matches[0]; shard_start=int(shard_spec["start"]); shard_count=int(shard_spec["count"])
position_schema=expected_position_schema(plan_payload)
def token():
 return json.load(urllib.request.urlopen(urllib.request.Request(
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
  headers={"Metadata-Flavor":"Google"}),timeout=60))["access_token"]
def get_bytes(url):
 for retry in range(3):
  try:
   req=urllib.request.Request(url,headers={"Authorization":"Bearer "+token()})
   return urllib.request.urlopen(req,timeout=600).read()
  except Exception:
   if retry==2: raise
   time.sleep(5*(retry+1))
def object_bytes(obj,generation=None):
 query={"alt":"media"}
 if generation is not None: query["generation"]=generation
 return get_bytes("https://storage.googleapis.com/download/storage/v1/b/"+
  urllib.parse.quote(bucket,safe="")+"/o/"+urllib.parse.quote(obj,safe="")+"?"+
  urllib.parse.urlencode(query))
def put(obj,data):
 # Bind SHA-256 metadata and bytes to the same create-only generation. A
 # replacement VM will not trust a cached position without this metadata.
 digest=hashlib.sha256(data).hexdigest(); boundary="ofc-"+uuid.uuid4().hex
 metadata=canon({"name":obj,"metadata":{"sha256":digest}})
 body=(b"--"+boundary.encode()+b"\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n"+
       metadata+b"\r\n--"+boundary.encode()+b"\r\nContent-Type: application/octet-stream\r\n\r\n"+
       data+b"\r\n--"+boundary.encode()+b"--\r\n")
 url=("https://storage.googleapis.com/upload/storage/v1/b/"+urllib.parse.quote(bucket,safe="")+
      "/o?uploadType=multipart&ifGenerationMatch=0")
 for retry in range(3):
  req=urllib.request.Request(url,data=body,method="POST",headers={"Authorization":"Bearer "+token(),
   "Content-Type":"multipart/related; boundary="+boundary})
  try: return json.load(urllib.request.urlopen(req,timeout=600))
  except urllib.error.HTTPError as error:
   if error.code==412:
    if object_bytes(obj)!=data:
     raise SystemExit(f"create-only collision has different bytes: gs://{bucket}/{obj}")
    return {"generation":"preexisting-identical"}
   if retry==2: raise
  except Exception:
   if retry==2: raise
  time.sleep(5*(retry+1))
def list_position_objects():
 result=[]; page=None; files_prefix=prefix+"/files/position_"
 while True:
  query={"prefix":files_prefix,"fields":"items(name,generation,size,metadata),nextPageToken"}
  if page: query["pageToken"]=page
  url=("https://storage.googleapis.com/storage/v1/b/"+urllib.parse.quote(bucket,safe="")+
       "/o?"+urllib.parse.urlencode(query))
  payload=json.loads(get_bytes(url)); items=payload.get("items",[])
  if not isinstance(items,list): raise SystemExit("GCS position listing has invalid items")
  result.extend(items); page=payload.get("nextPageToken")
  if not page: return result
uploaded={}
heartbeat_tick=0
def position_count():
 return len(list(out.glob("position_*.json")))
def heartbeat(completed):
 global heartbeat_tick
 tick=heartbeat_tick; heartbeat_tick+=1
 hb={"schema":"hu_m31_label_gen_heartbeat_v1","plan_sha256":plan_sha,"shard_id":shard,
     "attempt_id":attempt,"sequence":tick,"completed_position_count":completed,"create_only":True,
     "observed_at_utc":datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
 hb["heartbeat_sha256"]=hashlib.sha256(canon(hb)).hexdigest()
 put(prefix+f"/heartbeats/{attempt}-{tick:06d}.json",canon(hb))
heartbeat_stop=threading.Event(); heartbeat_errors=[]
def heartbeat_loop():
 while not heartbeat_stop.is_set():
  try: heartbeat(position_count())
  except BaseException as error:
   heartbeat_errors.append(error); return
  heartbeat_stop.wait(120)
heartbeat_thread=threading.Thread(target=heartbeat_loop,name="labelgen-heartbeat",daemon=True)
heartbeat_thread.start()
# Restore before spawning workers, but after the independent heartbeat starts.
# Every listed object is generation-bound and must carry the SHA metadata that
# the create-only multipart publisher wrote with its bytes.
restored=set()
for item in list_position_objects():
 if not isinstance(item,dict): raise SystemExit("GCS position listing row is not an object")
 obj=item.get("name"); generation=item.get("generation"); size=item.get("size")
 if not isinstance(obj,str) or not isinstance(size,str) or not size.isdigit():
  raise SystemExit("GCS position listing row has invalid name/size")
 name=obj.rsplit("/",1)[-1]
 if not (name.startswith("position_") and name.endswith(".json") and name[9:-5].isdigit()):
  raise SystemExit(f"unexpected object in position listing: {obj!r}")
 offset=int(name[9:-5]); expected_name=f"position_{offset:08d}.json"
 if name!=expected_name or not shard_start<=offset<shard_start+shard_count:
  raise SystemExit(f"cached position lies outside shard contract: {obj!r}")
 if name in restored: raise SystemExit(f"duplicate cached position listing: {obj!r}")
 metadata=item.get("metadata"); expected_sha=(metadata.get("sha256") if isinstance(metadata,dict) else None)
 raw=object_bytes(obj,generation)
 try:
  restore_cached_position_object(out/name,raw,object_name=obj,expected_object_name=prefix+"/files/"+name,
   generation=generation,expected_bytes=int(size),expected_sha256=expected_sha,
   plan_sha256=plan_sha,offset=offset,position_schema=position_schema)
 except ValueError as error: raise SystemExit(f"refusing cached position {obj}: {error}") from error
 uploaded[name]=generation; restored.add(name)
print(f"restored {len(restored)} generation/bytes/SHA-verified positions",flush=True)
manifest.write_text("".join(name+"\n" for name in sorted(restored)),encoding="utf-8")
worker_root=root/"worker-shards"; worker_root.mkdir(parents=True,exist_ok=True)
def promote_worker_positions(strict=False):
 promoted=0
 for path in sorted(worker_root.glob("*/position_*.json")):
  name=path.name
  if not (name.startswith("position_") and name.endswith(".json") and name[9:-5].isdigit()):
   if strict: raise SystemExit(f"worker emitted unexpected position path {path}")
   continue
  offset=int(name[9:-5])
  if name!=f"position_{offset:08d}.json" or not shard_start<=offset<shard_start+shard_count:
   raise SystemExit(f"worker emitted out-of-contract position path {path}")
  raw=path.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
  try:
   restore_cached_position_object(out/name,raw,object_name=prefix+"/files/"+name,
    expected_object_name=prefix+"/files/"+name,generation="1",expected_bytes=len(raw),
    expected_sha256=digest,plan_sha256=plan_sha,offset=offset,position_schema=position_schema)
  except ValueError as error:
   if strict: raise SystemExit(f"worker position {path} is incomplete or invalid: {error}") from error
   continue
  promoted+=1
 return promoted
def finalize_done_marker():
 expected={f"position_{offset:08d}.json" for offset in range(shard_start,shard_start+shard_count)}
 observed={path.name for path in out.glob("position_*.json")}
 if observed!=expected:
  raise SystemExit(f"workers exited with incomplete root inventory {len(observed)}/{len(expected)}")
 marker=out/"SHARD_DONE.json"
 payload=canon({"schema":"hu_m31_label_gen_shard_done_v1","plan_sha256":plan_sha,
  "shard_id":shard,"positions":shard_count})
 if marker.exists():
  if marker.read_bytes()!=payload: raise SystemExit("local SHARD_DONE provenance drifted")
 else:
  with marker.open("xb") as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())
complete_checkpoint_published=[False]
def publish():
 if heartbeat_errors: raise SystemExit(f"heartbeat publisher failed: {heartbeat_errors[0]}")
 files=[]
 # SHARD_DONE is a promise that every position is durable.  Lexicographic
 # order puts it before position_*.json, which lets a supervisor observe DONE
 # and delete this VM while final positions are still uploading.  Publish all
 # ordinary files first and the marker explicitly last; the complete
 # checkpoint below is a second, independent after-all-files witness.
 paths=sorted(
  (p for p in out.rglob("*") if p.is_file()),
  key=lambda p:(p.name=="SHARD_DONE.json",p.relative_to(out).as_posix()))
 for path in paths:
  rel=path.relative_to(out).as_posix(); data=path.read_bytes(); obj=prefix+"/files/"+rel
  if rel not in uploaded:
   uploaded[rel]=put(obj,data)["generation"]
  files.append({"relative_path":rel,"object_name":obj,"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)})
 count=position_count(); complete=(out/"SHARD_DONE.json").is_file()
 # Progress lives in heartbeats plus immutable positions. Only a true,
 # full-inventory completion may claim the fixed complete checkpoint name.
 if not complete: return count,complete
 if count!=shard_count: raise SystemExit(f"SHARD_DONE with local count {count}/{shard_count}")
 if not complete_checkpoint_published[0]:
  cp=build_complete_checkpoint(plan_sha256=plan_sha,shard_id=shard,attempt_id=attempt,
   shard_start=shard_start,shard_count=shard_count,files=files)
  put(complete_checkpoint_object_name(prefix),canon(cp)); complete_checkpoint_published[0]=True
 return count,complete
# One isolated directory per stride prevents a finishing worker's whole-shard
# scan from opening or unlinking a sibling's in-progress JSON. The parent only
# promotes canonical, provenance-valid bytes into the publish root.
procs=[]
for index in range(workers):
 worker_out=worker_root/f"stride-{index:02d}"
 procs.append(subprocess.Popen([
  "python3","-m","ofc_regular.hu_m31_label_gen_worker_v1","run",
  "--plan",str(plan),"--shard-id",shard,"--shard-directory",str(worker_out),
  "--runtime-root",str(staging/"runtime"),
  "--existing-manifest",str(manifest),
  "--stride-index",str(index),"--stride-count",str(workers),
  # One position per solver invocation. The default chunk is 32, which for the
  # streets this fleet was built for is a few minutes of work; at T0 a single
  # position costs about 1,531 core-seconds, so a stride of ten would compute
  # for four hours and write nothing until the end -- invisible to any progress
  # check, and lost in full to one Spot preemption. A plan may raise it.
  "--solver-chunk",str(plan_payload.get("solver_chunk",1))]))
while True:
 promote_worker_positions()
 publish()
 if all(p.poll() is not None for p in procs): break
 time.sleep(120)
for p in procs:
 if p.returncode!=0: raise SystemExit(f"worker exited {p.returncode}")
promote_worker_positions(strict=True)
finalize_done_marker()
count,complete=publish()
if not complete: raise SystemExit("all workers exited but the shard is not complete")
heartbeat_stop.set()
PY
kill "$WATCHDOG_PID" >/dev/null 2>&1 || true
shutdown -h now
