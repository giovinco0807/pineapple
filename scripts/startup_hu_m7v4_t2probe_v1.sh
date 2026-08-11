#!/usr/bin/env bash
# T2 particle-probe startup, derived from the label-generation one by
# scripts-level substitution (m7v4_t2probe_startup_make.py). The fetch
# half, the create-only publishing, the heartbeat/checkpoint discipline and
# the preemption handling are the label worker's, unchanged; what differs is
# that the unit of work is a probe ROOT, the process spawned is the probe
# module, and the parent writes the completion marker the probe does not.
#
# The fetch half is the proven skeleton from the dataset transport: content
# bindings arrive as base64 canonical JSON in instance metadata, every object
# is downloaded by generation and verified by size and sha256 before use, and
# writes are create-only. The execute half runs the portable label worker in
# bounded increments, publishing create-only position files and numbered
# checkpoint manifests between increments, so a Spot preemption costs at most
# one increment and resume is a file copy.
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
# A preempted Spot instance that is RECREATED rather than restarted boots with an
# empty disk, so the workers' local resume scan sees nothing and regenerates the
# whole shard, while the parent's create-only uploads collide with the previous
# attempt's objects. Listing the shard's files/ prefix once, here, gives the
# workers the same picture the object store has.
# All three parsers are total: an empty result is a legitimate answer (no token
# yet, nothing published yet, no further pages), and under `set -e -o pipefail` a
# bare grep that matches nothing would otherwise abort the boot.
gcs_token() {
  { curl -fsS -H "$HEADER" \
      'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
    || true; } \
    | tr ',' '\n' | sed -n 's/.*"access_token"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p'
}
# Pull the position file names out of one listing page. Object names come back
# fully qualified; the workers compare bare file names, and only position files
# are theirs to skip.
listing_names() {
  local strip="$1"
  { grep -o '"name"[[:space:]]*:[[:space:]]*"[^"]*"' || true; } \
    | sed 's/.*:[[:space:]]*"//; s/"$//' \
    | awk -v strip="$strip" '
        index($0, strip) == 1 {
          name = substr($0, length(strip) + 1)
          if (name ~ /^root_[0-9]+\.json$/) print name
        }'
}
listing_page_token() {
  { grep -o '"nextPageToken"[[:space:]]*:[[:space:]]*"[^"]*"' || true; } \
    | sed 's/.*:[[:space:]]*"//; s/"$//' | head -n 1
}
MANIFEST="$ROOT/existing_roots.txt"
: >"$MANIFEST"
FILES_PREFIX="$PREFIX/files/"
LIST_TOKEN="$(gcs_token)"
PAGE=''
ATTEMPTS=0
while : ; do
  PAGE_BODY=''
  # A listing that cannot be read is fatal: proceeding would mean regenerating a
  # shard that may already be finished, which is the defect this block exists to
  # prevent. Transient failures get three tries first.
  for try in 1 2 3; do
    if PAGE_BODY="$(curl -fsS --get \
        -H "Authorization: Bearer $LIST_TOKEN" \
        --data-urlencode "prefix=$FILES_PREFIX" \
        --data-urlencode 'fields=items(name),nextPageToken' \
        --data-urlencode "pageToken=$PAGE" \
        "https://storage.googleapis.com/storage/v1/b/$BUCKET/o")"; then
      break
    fi
    PAGE_BODY=''
    sleep $(( try * 5 ))
    LIST_TOKEN="$(gcs_token)"
  done
  if [ -z "$PAGE_BODY" ]; then
    echo "FATAL: cannot list gs://$BUCKET/$FILES_PREFIX; refusing to regenerate a shard whose state is unknown"
    exit 1
  fi
  printf '%s' "$PAGE_BODY" | listing_names "$FILES_PREFIX" >>"$MANIFEST"
  PAGE="$(printf '%s' "$PAGE_BODY" | listing_page_token)"
  ATTEMPTS=$(( ATTEMPTS + 1 ))
  [ -n "$PAGE" ] || break
done
echo "existing manifest: $(wc -l <"$MANIFEST") roots already in gs://$BUCKET/$FILES_PREFIX ($ATTEMPTS listing page(s))"

# A run re-sharded into a new job (say, to spread across regions) publishes
# under a new prefix, so the listing above is empty even though most of the
# work may already be durable under the previous run's prefix. When the plan
# binds a carry manifest, the fetch loop above has already downloaded and
# digest-verified it; its names are position files exactly like the listed
# ones. Position file names encode the global offset, so a name is unique
# across every shard of every run and a worker can only ever match its own.
CARRY="$STAGING/static/manifest/carry_manifest.txt"
if [ -f "$CARRY" ]; then
  cat "$CARRY" >>"$MANIFEST"
  echo "carry manifest: $(wc -l <"$CARRY") positions carried from earlier runs"
else
  echo "carry manifest: none bound"
fi
echo "manifest total: $(sort -u "$MANIFEST" | wc -l) distinct positions to skip"

python3 - "$STAGING" "$ROOT" "$BUCKET" "$PREFIX" "$PLAN_SHA" "$SHARD_ID" "$ATTEMPT_ID" "$PLAN" "$WORKERS" "$MANIFEST" <<'PY'
import datetime,hashlib,json,pathlib,subprocess,sys,threading,time,urllib.parse,urllib.request
staging,root=map(pathlib.Path,sys.argv[1:3]); bucket,prefix,plan_sha,shard,attempt=sys.argv[3:8]
plan=pathlib.Path(sys.argv[8]); workers=int(sys.argv[9]); out=staging/"shard"
manifest=pathlib.Path(sys.argv[10])
canon=lambda v: json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")
def token():
 return json.load(urllib.request.urlopen(urllib.request.Request(
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
  headers={"Metadata-Flavor":"Google"})))["access_token"]
def put(obj,data):
 url=("https://storage.googleapis.com/upload/storage/v1/b/"+urllib.parse.quote(bucket,safe="")+
      "/o?uploadType=media&ifGenerationMatch=0&name="+urllib.parse.quote(obj,safe=""))
 req=urllib.request.Request(url,data=data,method="POST",headers={"Authorization":"Bearer "+token(),"Content-Type":"application/octet-stream"})
 return json.load(urllib.request.urlopen(req,timeout=600))
uploaded={}
heartbeat_tick=0
# Named for the field the checkpoint schema already carries; the unit here
# is a probe root.
def position_count():
 return len(list(out.glob("root_*.json")))
def heartbeat(completed):
 global heartbeat_tick
 tick=heartbeat_tick; heartbeat_tick+=1
 hb={"schema":"hu_m31_label_gen_heartbeat_v1","plan_sha256":plan_sha,"shard_id":shard,
     "attempt_id":attempt,"sequence":tick,"completed_position_count":completed,"create_only":True,
     "observed_at_utc":datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
 hb["heartbeat_sha256"]=hashlib.sha256(canon(hb)).hexdigest()
 # Named by attempt as well as tick. A recreated instance restarts its tick
 # counter, so a bare tick name collides with the previous attempt's object and
 # the create-only PUT returns 412 -- which killed the parent loop and left its
 # workers generating into a disk nobody was uploading. The 412 is also caught,
 # so a repeated attempt id could never take the loop down again.
 try: put(prefix+f"/heartbeats/{attempt}-{tick:06d}.json",canon(hb))
 except urllib.error.HTTPError as error:
  if error.code!=412: raise
  print(f"benign 412: heartbeat {attempt}-{tick:06d} already present",flush=True)
last_state=[None]
def publish():
 files=[]
 for path in sorted(p for p in out.rglob("*") if p.is_file()):
  rel=path.relative_to(out).as_posix(); data=path.read_bytes(); obj=prefix+"/files/"+rel
  if rel not in uploaded:
   # 412 means the object is already there: a previous attempt of this shard
   # published it and this instance was recreated with a disk that had to
   # regenerate it, or a resume copy re-presented it. Create-only is intact
   # either way and the object we would have written is the one that exists.
   try: uploaded[rel]=put(obj,data)["generation"]
   except urllib.error.HTTPError as error:
    if error.code!=412: raise
    print(f"benign 412: {obj} already published by an earlier attempt",flush=True)
    uploaded[rel]="preexisting"
  files.append({"relative_path":rel,"object_name":obj,"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)})
 count=position_count(); complete=(out/"SHARD_DONE.json").is_file()
 # Checkpoints are create-only and named by count; re-publishing an unchanged
 # state would collide with our own earlier object (HTTP 412 -- the defect
 # that killed the first referee fleet on every slow shard).
 if last_state[0]==(count,complete): return count,complete
 cp={"schema":"hu_m31_label_gen_checkpoint_v1","plan_sha256":plan_sha,"shard_id":shard,
     "attempt_id":attempt,"sequence":count,"completed_position_count":count,"complete":complete,
     "files":files,"checkpoint_published_after_files":True,"create_only":True}
 cp["checkpoint_sha256"]=hashlib.sha256(canon(cp)).hexdigest()
 try: put(prefix+f"/checkpoints/{count:06d}.json",canon(cp))
 except urllib.error.HTTPError as error:
  if error.code!=412: raise  # our own earlier object; create-only is intact
 last_state[0]=(count,complete); return count,complete
# One process per stride: the split is by offset, so no two processes ever
# touch the same position. The parent heartbeats and publishes completed
# position files every two minutes; files are complete on create, so partial
# uploads cannot exist.
# Everything the probe needs comes off the plan the fleet was given, so the
# command line cannot disagree with the plan the run is pinned to.
spec=json.loads(plan.read_text())
entry=[s for s in spec["shards"] if s["shard_id"]==shard]
if len(entry)!=1: raise SystemExit(f"plan has {len(entry)} entries for shard {shard}")
entry=entry[0]
probe=spec["probe"]
expected=int(entry["count"])
argv=["python3","-m","ofc_regular.hu_m7v4_t2_noise_probe_v1",
 "--runtime-root",str(staging/"runtime"),
 "--engine-library",spec["engine_library"],
 "--feature-encoder",spec["feature_encoder_library"],
 "--t4-model",spec["t4_model"],
 "--t3-second-model",spec["t3_second_model"],
 "--t3-first-model",spec["t3_first_model"],
 "--roots",str(expected),"--root-start",str(entry["start"]),
 "--samples",",".join(str(r) for r in probe["rungs"]),
 "--reference-samples",str(probe["reference_samples"]),
 "--seats",spec["seat"],
 "--rung-seed-base",str(probe["rung_seed_base"]),
 "--reference-seed-base",str(probe["reference_seed_base"]),
 "--hand-seed-base",str(spec["hand_seed_base"]),
 "--behavior-seed-offset",str(spec["behavior_seed_offset"]),
 "--out",str(out),"--tag",spec["job_id"]]
if "t2_second_model" in spec:
 argv+=["--t2-second-model",spec["t2_second_model"]]
# Roots this run already published, pulled back onto the disk: the probe's
# resume is "root_NNNNN.json exists, skip it", which a recreated instance's
# empty disk would otherwise defeat.
def get(obj):
 url=("https://storage.googleapis.com/storage/v1/b/"+urllib.parse.quote(bucket,safe="")+
      "/o/"+urllib.parse.quote(obj,safe="")+"?alt=media")
 req=urllib.request.Request(url,headers={"Authorization":"Bearer "+token()})
 return urllib.request.urlopen(req,timeout=600).read()
recovered=0
for name in sorted(set(x.strip() for x in manifest.read_text().splitlines() if x.strip())):
 target=out/name
 if target.exists(): continue
 target.write_bytes(get(prefix+"/files/"+name)); recovered+=1
print(f"recovered {recovered} already-published roots onto the disk",flush=True)
procs=[]
for index in range(workers):
 procs.append(subprocess.Popen(argv+[
  "--stride-index",str(index),"--stride-count",str(workers)]))
while True:
 heartbeat(position_count()); publish()
 if all(p.poll() is not None for p in procs): break
 time.sleep(120)
for p in procs:
 if p.returncode!=0: raise SystemExit(f"worker exited {p.returncode}")
# The probe writes roots, not a completion marker, so the parent writes one --
# and only against the count this shard's plan entry asks for, so a shard that
# stopped early is never marked finished and the chain watcher keeps waiting.
found=position_count()
if found!=expected: raise SystemExit(f"all workers exited with {found}/{expected} roots")
done={"schema":"hu_m7v4_t2probe_shard_done_v1","job_id":spec["job_id"],
 "plan_sha256":plan_sha,"shard_id":shard,"attempt_id":attempt,
 "root_start":int(entry["start"]),"root_count":expected,
 "completed_at_utc":datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
(out/"SHARD_DONE.json").write_bytes(canon(done))
count,complete=publish()
if not complete: raise SystemExit("all workers exited but the shard is not complete")
PY
kill "$WATCHDOG_PID" >/dev/null 2>&1 || true
shutdown -h now
