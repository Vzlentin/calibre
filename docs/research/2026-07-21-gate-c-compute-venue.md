# Gate C compute venue selection

## Executive recommendation

Select **Amazon EC2 `c7a.8xlarge` in `us-east-1` (US East, N. Virginia),
on-demand shared tenancy** for the Gate C trial and the later U17 acceptance
run.

Pin this environment:

- **Compute:** `c7a.8xlarge`, which AWS specifies as 64 GiB RAM, 32 vCPUs,
  **32 CPU cores, and one thread per core** on AMD EPYC 9R14. This is 32
  physical cores, not 32 SMT threads ([AWS compute-optimized instance
  specifications][aws-co]).
- **Image:** Canonical Ubuntu Server 24.04 LTS (Noble), release serial
  `20260714`, AMD64, HVM, EBS gp3, AMI `ami-052355af2a014bd2c`. Canonical's
  official locator listed this exact `us-east-1` image when this research was
  done ([Ubuntu EC2 image locator][ubuntu-locator]). Issue 436 must verify the
  owner is Canonical account `099720109477` before launch; Canonical documents
  both that owner ID and the image-name query ([Canonical image discovery
  guide][ubuntu-images]).
- **Storage:** one encrypted **300 GiB gp3** root volume, 3,000 IOPS, 125 MiB/s,
  `DeleteOnTermination=true`. gp3 includes this IOPS and throughput baseline
  and supports 1 GiB through 64 TiB ([AWS gp3 documentation][aws-gp3]).
- **Runtime:** managed CPython **3.12.13**, uv **0.11.30**, and the exact
  candidate `newcalibre/uv.lock` installed with `uv sync --locked`. Pinning an
  exact Python patch and uv release is supported by uv's first-party install
  and Python-version interfaces ([uv installation][uv-install], [uv Python
  versions][uv-python]).
- **Thread policy:** `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `VECLIB_MAXIMUM_THREADS=1`, and `BLIS_NUM_THREADS=1`, inherited by every Ray
  worker. U16/issue 439 may choose a worker count from its scaling curve, but
  it must not change this one-thread-per-worker numeric policy.
- **Provisioning and access:** one AWS CloudFormation stack that owns the VPC,
  public subnet, internet route, no-ingress security group, instance role,
  instance profile, EC2 instance, and volume. Access it only with AWS Systems
  Manager Session Manager. Session Manager supports CLI access without an
  inbound port, bastion, or SSH key ([AWS Session Manager][aws-session]).
  Delete the stack after each run.

This is a venue selection, not a Gate C result. It does not assert that the
workload passes. The unchanged U17 limits remain **full M5 <= 15 minutes wall
clock, pre-origin overhead <= 60 seconds, and peak RSS <= 32 GB**
([`[PRF-1]`, `[PRF-2]`, and `[PRF-20]` at revision `e5cb11b`][project-perf]).
No trial observation may lower or reinterpret them.

### Fallback

Use **Amazon EC2 `m7a.4xlarge` in `us-east-1` with the same AMI, storage,
thread policy, CloudFormation topology, and Session Manager access** only if
`c7a.8xlarge` cannot pass issue 436 because of account quota or current
capacity. AWS specifies `m7a.4xlarge` as 64 GiB RAM, 16 vCPUs, **16 CPU cores,
and one thread per core**, also on AMD EPYC 9R14 ([AWS general-purpose
instance specifications][aws-gp]). It satisfies the 16-physical-core floor
but has half the process-parallel capacity of the recommendation.

The fallback is not a silent U17 substitution. It must pass the same issue-436
preflight, and the plan and environment manifest must name the one selected
shape before any acceptance result is read.

## Decision boundary

ADR 0001 requires one workstation-class x86_64 Linux headline environment with
at least 16 physical cores, 64 GB RAM, Python 3.12, a locked toolchain, and a
declared thread policy. It leaves the concrete instance pending
([ADR 0001, lines 65-94 at `e5cb11b`][project-adr]). The successor currently
requires Python 3.12 and has a separate lockfile
([`newcalibre/pyproject.toml`][project-toolchain],
[`newcalibre/uv.lock`][project-lock]). The lockfile at this research revision
has SHA-256
`1de477cd122b6763955df2f2b59c12cbda40fe56816f3046083c2ae67ea63336`.
U16 will add the final Ray-bearing dependency set ([issue 439][issue-439]), so
the U17 manifest must record the later candidate lockfile digest, not copy
this research-time digest.

The M5 workload has 30,490 bottom series, 33,563 lattice nodes, 64 daily
origins in the reference configuration, and a 28-day horizon. The prior
profile wrote about 2.6 GB of ledger files. The performance design requires
streaming ledger I/O and sparse reconciliation
([M5 protocol][project-m5], [performance storage requirements][project-storage]).
A 300 GiB volume is therefore a conservative capacity allocation, not a claim
about runtime. Issue 436 still measures the installed environment, downloaded
data, free space, and scratch output.

Ray must run as a normal single-machine local runtime with worker processes,
not `local_mode=True`. Ray documents that `ray.init()` starts a local instance
whose machine is the head node and that `ray.shutdown()` terminates the local
processes ([Ray single-machine runtime][ray-single]). Calibre separately
requires batch-placement invariance, a serial-order commit, and explicit
numeric thread budgets ([`[DET-3]`-`[DET-5]` at `e5cb11b`][project-determinism]).
The venue can support those checks, but only U16's same-engine tests can prove
them.

## Requirements matrix

| Requirement | Selected environment | Status before issue 436 | Issue-436 binding observation |
| --- | --- | --- | --- |
| x86_64 Linux | AMD64 Ubuntu 24.04 AMI on AMD EPYC | Provider/image contract | `uname -m=x86_64`; IMDS AMI/region/type; `/etc/os-release` is Ubuntu 24.04 |
| At least 16 physical cores | AWS table: 32 cores, one thread/core | **Meets with 2x headroom** | AWS API and `lscpu`: 32 CPUs, 32 cores, one thread/core; no cpuset restriction |
| At least 64 GiB RAM | AWS table: 64 GiB | **Meets** | AWS API reports 65,536 MiB; guest memory is consistent after kernel reserve |
| Python 3.12 and locked uv environment | CPython 3.12.13; uv 0.11.30; candidate lock | Exact versions selected | `uv --version`; Python version; `uv sync --locked`; no lockfile change; lock SHA recorded |
| Numeric-library provenance | Wheels selected by the lock on Linux x86_64 | Must be observed | NumPy/SciPy versions and build configuration, shared-library mapping, and `threadpoolctl` output recorded |
| Storage for input, environment, artifacts, profiles | Encrypted 300 GiB gp3 | Capacity is far above known ledger size | Control plane reports 300 GiB/3,000 IOPS/125 MiB/s; >= 200 GiB free after sync and M5 download |
| Process/thread observability | Ubuntu `/proc`, cgroup v2, `ps`, `pidstat`, GNU `time` | Standard OS facilities | Ray worker PIDs/threads visible; `pidstat` samples; `/usr/bin/time -v` reports maximum RSS; cgroup memory files readable |
| Deterministic single-node Ray support | 32 non-SMT cores; one local Ray node | Plausible, not proven | One live Ray node, multiple worker PIDs, 32 advertised CPUs, worker thread limits all one, repeated probe bytes equal |
| Reproducible provisioning/teardown | One declarative CloudFormation stack | Method selected | Stack creation reaches `CREATE_COMPLETE`; exact AMI/template hashes recorded; deletion removes volume, ENIs, and VPC |
| Practical one-off cost | About USD 1.680/hour including selected compute, 300 GiB gp3 planning allocation, and one public IPv4 address | Current estimate only | Current rate query, launch/termination timestamps, estimate, and later bill reference recorded |
| Gate C budgets | 15 min / 60 s / 32 GB | **Unchanged and untested here** | U17 only; issue 436 must not run or claim the full acceptance verdict |

## Current option comparison

### Price timestamp and method

Prices are public list prices in USD, without tax, discount, support, data
egress, or attached-disk charges unless stated. They are dynamic and
region-specific.

- **AWS:** official EC2 bulk-price snapshot version `20260721012550`, published
  `2026-07-21T01:25:50Z`, region `us-east-1`, Linux, shared tenancy,
  on-demand, no preinstalled software ([versioned AWS EC2 price
  snapshot][aws-ec2-price]). The listed terms are effective
  `2026-07-01T00:00:00Z`.
- **Azure:** official Retail Prices API queried at `2026-07-21T01:28:15Z`,
  billing currency USD, region `westeurope`, consumption meter only
  ([Azure Retail Prices API query][azure-price]).

| Option | Physical-core evidence | RAM and storage | Compute list price | Assessment |
| --- | --- | --- | ---: | --- |
| **AWS `c7a.8xlarge`, `us-east-1`** | AWS: 32 vCPU, **32 cores, 1 thread/core**, AMD EPYC 9R14 | 64 GiB; EBS only | **$1.64224/h**; SKU `6MV5XKZYPVXSQCN4`, rate `6MV5XKZYPVXSQCN4.JRTCKXETXF.6YS6EN2CT7` | **Selected.** Most physical-core headroom, no SMT ambiguity, fixed documented processor family, and low absolute trial cost. |
| AWS `m7a.4xlarge`, `us-east-1` | AWS: 16 vCPU, **16 cores, 1 thread/core**, AMD EPYC 9R14 | 64 GiB; EBS only | **$0.92736/h**; SKU `TTPYSE2GSHDGSMTX`, rate `TTPYSE2GSHDGSMTX.JRTCKXETXF.6YS6EN2CT7` | **Fallback.** Exact minimum and same processor family, but half the Ray process slots. |
| AWS `c7i.8xlarge`, `us-east-1` | AWS: 32 vCPU, **16 cores, 2 threads/core**, Intel Xeon Sapphire Rapids | 64 GiB; EBS only | **$1.42800/h**; SKU `7RFUEP9XE7QPFUGE`, rate `7RFUEP9XE7QPFUGE.JRTCKXETXF.6YS6EN2CT7` | Meets the class. It is cheaper than C7a but supplies half as many physical cores and exposes SMT siblings. |
| Azure `Standard_D32s_v5`, West Europe | Microsoft: 32 vCPU in a hyper-threaded configuration. The cited size page does not publish a physical-core count. Treat 16 cores as an expectation only and require guest preflight. Host CPU may be Emerald Rapids, Sapphire Rapids, or Ice Lake. | 128 GB; no local temp disk, so attach a managed disk | **$1.84000/h**; meter `4c9105f0-0fae-5bba-88c9-f2c736f78eb9` | Credible second provider, but a higher price, variable documented host CPU, attached-disk work, and no provider-published core count in the size table make it a weaker reference pin. |

AWS core, thread, memory, processor, network, and EBS-only facts come from the
provider's instance specification tables ([compute optimized][aws-co],
[general purpose][aws-gp]). Azure facts come from Microsoft's Dsv5 page, which
states the processor roster, hyper-threaded configuration, 32 vCPU/128 GB size,
and lack of local temporary disk ([Azure Dsv5 specifications][azure-dsv5]).
No physical-core claim is made for Azure because the provider page does not
state that count.

### Why the selected shape is worth the small premium

The prior profile used essentially one of 14 laptop cores, while the successor
architecture requires parallel work across permitted task/series axes. The
performance chapter identifies idle-core parallelism as the largest untapped
lever ([`[PRF-14]` at `e5cb11b`][project-parallel]). C7a gives U16 up to 32
independent physical worker slots without SMT siblings. Compared with C7i, its
compute premium is about $0.214/hour, which is immaterial for two short-lived
runs but preserves twice the physical-core ceiling. Compared with the M7a
fallback, the premium buys twice the physical cores while retaining the same
RAM and documented CPU model.

This reasoning does not assume linear speedup. U16 must report worker-count
scaling and parallel efficiency, and U17 must use the preregistered worker
count. The selected machine only provides the stronger ceiling.

## Exact proposed environment

| Field | Pin |
| --- | --- |
| Provider / region | AWS EC2 / `us-east-1` |
| Purchase / tenancy | On-demand Linux / shared tenancy; no Spot, reservation, or Savings Plan |
| Instance | `c7a.8xlarge` |
| Provider CPU contract | AMD EPYC 9R14; 32 vCPUs; 32 cores; one thread/core |
| Provider memory contract | 64 GiB |
| Image | Canonical Ubuntu Server 24.04 LTS Noble, serial `20260714`, AMD64 HVM EBS gp3, `ami-052355af2a014bd2c` |
| Python | uv-managed CPython `3.12.13` |
| uv | `0.11.30`; installer SHA-256 `f633daff5c2a1b5e550d5dab074f21ab2d5fda2d147babf4525844ff1276e57e` |
| Project install | Exact candidate SHA; `uv sync --project newcalibre --locked --group dev --python 3.12.13`; candidate `newcalibre/uv.lock` SHA-256 in manifest |
| Thread policy | Six variables listed above set to `1` before sync, import, Ray start, and benchmark |
| Ray | Candidate-locked version; one local node; `local_mode=False`; dashboard disabled; no external Ray address |
| Root storage | 300 GiB encrypted gp3; 3,000 IOPS; 125 MiB/s; delete on termination |
| Network | Stack-owned VPC and public subnet; one temporary public IPv4; outbound HTTP/HTTPS for packages and public data; security group has no ingress |
| Access | Session Manager through a least-privilege instance profile; no SSH key and no port 22 |
| Provision / destroy | AWS CloudFormation create/delete from one reviewed template; stack tags include ticket, candidate SHA, and expiry time |
| Evidence clock | UTC timestamps from the guest and AWS API |

### CloudFormation resource contract

Issue 436 should make one temporary template with these resources and no
others:

1. `AWS::EC2::VPC`, one `AWS::EC2::Subnet`, internet gateway attachment,
   route table, and default route.
2. A security group with **no ingress** and only required outbound HTTP/HTTPS
   and DNS. The instance receives a temporary public IPv4 for low-friction
   package and M5 acquisition.
3. An IAM role and instance profile with `AmazonSSMManagedInstanceCore`. Do not
   place credentials in user data, tags, outputs, or the artifact.
4. One `AWS::EC2::Instance` with the exact AMI and shape above. Its block
   mapping sets encrypted gp3, 300 GiB, 3,000 IOPS, 125 MiB/s, and delete on
   termination.
5. Outputs for instance ID, security-group ID, subnet ID, and VPC ID. Derive
   the root-volume and primary-interface IDs from `describe-instances` after
   launch. These IDs make teardown verification mechanical.

CloudFormation treats a template as the description of the resources to
provision and manages them as one stack ([AWS CloudFormation overview][aws-cfn]).
Store the template hash with the issue-436 evidence. Do not commit that
temporary infrastructure from this research ticket.

### Cost estimate

The current selected compute rate is $1.64224/hour. The AWS price snapshot
lists gp3 in `us-east-1` at $0.08/GB-month (SKU `JG3KUJMBRGHV3N8G`, rate
`JG3KUJMBRGHV3N8G.JRTCKXETXF.6YS6EN2CT7`), and the versioned VPC snapshot
lists one in-use public IPv4 at $0.005/hour (SKU `4GQUNXTFWVSGPUZK`, rate
`4GQUNXTFWVSGPUZK.JRTCKXETXF.6YS6EN2CT7`) ([AWS EC2 price
snapshot][aws-ec2-price], [AWS VPC price snapshot][aws-vpc-price]). Using 730
hours/month only as a planning conversion:

```text
compute                         1.64224 / hour
300 GiB gp3 planning allocation 0.03288 / hour
one public IPv4                 0.00500 / hour
estimated running total         1.68012 / hour
```

Cap the trial at two hours and the acceptance session at two hours. The
planning estimate is about **$6.72 total** for both sessions. It excludes tax,
data egress, and any resource that stack deletion misses. Linux on-demand
compute is billed per second with a 60-second minimum ([AWS on-demand
pricing][aws-on-demand]); the issue must record actual timestamps and the
post-run provider charge when available.

## Issue 436 short-lived trial preflight

### Result rule

Issue 436 passes only when every item below has its raw output attached or
linked, every stated pass condition holds, and teardown is verified. A setup
failure, quota failure, topology mismatch, package-resolution change, missing
observability signal, or teardown leak is a **venue preflight failure**, not a
Gate C verdict.

Use current `origin/main` revision
`e5cb11bd5b4487724701d4cf5a4626c91031c0e1` for an immediate infrastructure
trial. That revision does not yet contain U16's successor Ray dependency. The
Ray probe below therefore uses the repository root's locked Ray environment
for infrastructure validation only. Before U17, repeat the toolchain,
thread-policy, Ray, and observability sections at the exact U16/U17 candidate
SHA from `newcalibre/uv.lock`. A current-root Ray pass cannot certify the
future successor backend.

### 1. Verify quota, offering, image, and provider contract before launch

Run from the operator machine with AWS CLI credentials. The standard
on-demand quota is regional and defaults to five vCPUs in a new account, while
the selected shape needs 32; AWS identifies the adjustable standard-instance
quota as `L-1216C47A` ([AWS EC2 quotas][aws-quotas]).

```bash
set -euo pipefail
export AWS_REGION=us-east-1
export INSTANCE_TYPE=c7a.8xlarge
export AMI_ID=ami-052355af2a014bd2c
export CANONICAL_OWNER=099720109477

aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-1216C47A \
  --region "$AWS_REGION" \
  --output json > quota.json
jq -e '.Quota.Value >= 32' quota.json

aws ec2 describe-instance-type-offerings \
  --region "$AWS_REGION" \
  --location-type availability-zone \
  --filters "Name=instance-type,Values=$INSTANCE_TYPE" \
  --output json > offerings.json
jq -e '.InstanceTypeOfferings | length > 0' offerings.json

aws ec2 describe-instance-types \
  --region "$AWS_REGION" \
  --instance-types "$INSTANCE_TYPE" \
  --output json > instance-type.json
jq -e '.InstanceTypes[0] |
  .ProcessorInfo.SupportedArchitectures == ["x86_64"] and
  .VCpuInfo.DefaultVCpus == 32 and
  .VCpuInfo.DefaultCores == 32 and
  .VCpuInfo.DefaultThreadsPerCore == 1 and
  .MemoryInfo.SizeInMiB == 65536 and
  .InstanceStorageSupported == false' instance-type.json

aws ec2 describe-images \
  --region "$AWS_REGION" \
  --owners "$CANONICAL_OWNER" \
  --image-ids "$AMI_ID" \
  --output json > image.json
jq -e --arg owner "$CANONICAL_OWNER" --arg image "$AMI_ID" '
  .Images | length == 1 and
  .[0].ImageId == $image and
  .[0].OwnerId == $owner and
  .[0].Name == "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-20260714" and
  .[0].Architecture == "x86_64" and
  .[0].VirtualizationType == "hvm" and
  .[0].State == "available"' image.json
```

**Pass:** quota is at least 32; at least one AZ currently offers the shape; the
API contract is 32/32/1 and 65,536 MiB; the exact AMI is available and owned by
Canonical. AWS documents `describe-instance-type-offerings` as the mechanism
to check an instance type by Availability Zone ([AWS instance discovery][aws-discovery]).
If quota is below 32, request the increase before scheduling the paid trial.
Do not substitute a shape inside the command.

### 2. Create the stack and verify access isolation

```bash
set -euo pipefail
export STACK=calibre-gate-c-436
export TEMPLATE=gate-c-instance.yaml
export STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export EXPIRES_AT="$(date -u -d '+2 hours' +%Y-%m-%dT%H:%M:%SZ)"
sha256sum "$TEMPLATE" > template.sha256
aws cloudformation validate-template \
  --region "$AWS_REGION" \
  --template-body "file://$TEMPLATE" > template-validation.json
aws cloudformation deploy \
  --region "$AWS_REGION" \
  --stack-name "$STACK" \
  --template-file "$TEMPLATE" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    InstanceType="$INSTANCE_TYPE" \
    ImageId="$AMI_ID" \
    RootVolumeGiB=300 \
    CandidateSha=e5cb11bd5b4487724701d4cf5a4626c91031c0e1 \
  --tags \
    Key=Ticket,Value=434 \
    Key=CandidateSha,Value=e5cb11bd5b4487724701d4cf5a4626c91031c0e1 \
    Key=ExpiresAt,Value="$EXPIRES_AT"

aws cloudformation describe-stacks \
  --region "$AWS_REGION" \
  --stack-name "$STACK" \
  --query 'Stacks[0].Outputs' \
  --output json > stack-outputs.json
export INSTANCE_ID="$(jq -r '.[] | select(.OutputKey=="InstanceId").OutputValue' stack-outputs.json)"
export SECURITY_GROUP_ID="$(jq -r '.[] | select(.OutputKey=="SecurityGroupId").OutputValue' stack-outputs.json)"
aws ec2 describe-instances \
  --region "$AWS_REGION" \
  --instance-ids "$INSTANCE_ID" \
  --output json > instance.json
export VOLUME_ID="$(jq -r '.Reservations[0].Instances[0].BlockDeviceMappings[0].Ebs.VolumeId' instance.json)"
export NETWORK_INTERFACE_ID="$(jq -r '.Reservations[0].Instances[0].NetworkInterfaces[0].NetworkInterfaceId' instance.json)"
test -n "$VOLUME_ID"
test -n "$NETWORK_INTERFACE_ID"

aws ec2 describe-security-groups \
  --region "$AWS_REGION" \
  --group-ids "$SECURITY_GROUP_ID" \
  --output json > security-group.json
jq -e '.SecurityGroups[0].IpPermissions | length == 0' security-group.json

until test "$(aws ssm describe-instance-information \
  --region "$AWS_REGION" \
  --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
  --query 'InstanceInformationList[0].PingStatus' \
  --output text)" = Online; do sleep 10; done

aws ssm start-session --region "$AWS_REGION" --target "$INSTANCE_ID"
```

**Pass:** stack is `CREATE_COMPLETE`; all outputs are nonempty; the security
group has no ingress; SSM reports `Online`; access works without an SSH key or
inbound rule.

### 3. Pin the bootstrap and repository

Run inside the Session Manager shell. Do not run a distribution upgrade.

```bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y curl git jq numactl procps sysstat time

curl -LsSf https://astral.sh/uv/0.11.30/install.sh -o /tmp/uv-install.sh
echo 'f633daff5c2a1b5e550d5dab074f21ab2d5fda2d147babf4525844ff1276e57e  /tmp/uv-install.sh' \
  | sha256sum -c -
sh /tmp/uv-install.sh
export PATH="$HOME/.local/bin:$PATH"
test "$(uv --version | awk '{print $1, $2}')" = 'uv 0.11.30'
uv python install 3.12.13

export SOURCE_SHA=e5cb11bd5b4487724701d4cf5a4626c91031c0e1
git clone https://github.com/Vzlentin/calibre.git
cd calibre
git checkout --detach "$SOURCE_SHA"
test "$(git rev-parse HEAD)" = "$SOURCE_SHA"
test -z "$(git status --short)"
sha256sum newcalibre/uv.lock uv.lock | tee lockfiles.sha256

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

uv sync --project newcalibre --locked --group dev --python 3.12.13
uv sync --locked --extra dev --extra benchmarks --extra hierarchy --python 3.12.13
git diff --exit-code -- newcalibre/uv.lock uv.lock
```

**Pass:** uv and Python install exactly; both locked syncs succeed; `git status`
was empty before local evidence files; lockfiles do not change. For the later
candidate repetition, replace `SOURCE_SHA`, omit the root environment if
successor Ray is locked, and record the new successor lock digest.

### 4. Record instance, architecture, sockets, cores, threads, RAM, and OS

```bash
set -euo pipefail
export LC_ALL=C
mkdir -p preflight
TOKEN="$(curl -fsS -X PUT \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' \
  http://169.254.169.254/latest/api/token)"
for key in ami-id instance-id instance-type placement/region; do
  curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" \
    "http://169.254.169.254/latest/meta-data/$key"
  printf '\n'
done | tee preflight/imds.txt

uname -a | tee preflight/uname.txt
test "$(uname -m)" = x86_64
cat /etc/os-release | tee preflight/os-release.txt
. /etc/os-release
test "$ID" = ubuntu
test "$VERSION_ID" = 24.04

lscpu | tee preflight/lscpu.txt
lscpu --json > preflight/lscpu.json
lscpu --parse=CPU,CORE,SOCKET,NODE > preflight/lscpu-topology.csv
test "$(lscpu -p=CPU | grep -vc '^#')" -eq 32
test "$(lscpu -p=SOCKET,CORE | grep -v '^#' | sort -u | wc -l)" -eq 32
test "$(lscpu | awk -F: '/Thread.s. per core/{gsub(/ /,"",$2); print $2}')" -eq 1
test "$(nproc --all)" -eq 32
test "$(nproc)" -eq 32
numactl --hardware | tee preflight/numa.txt

free -b | tee preflight/free-bytes.txt
awk '/MemTotal/{printf "%.0f\n", $2 * 1024}' /proc/meminfo \
  | tee preflight/memtotal-bytes.txt
test "$(awk '/MemTotal/{printf "%.0f\n", $2 * 1024}' /proc/meminfo)" \
  -ge 64000000000
```

**Pass:** IMDS reports the selected AMI, type, and region; OS and architecture
match; all 32 CPUs are online and available to the process; there are 32 unique
core IDs and one thread/core; socket and NUMA topology are recorded; AWS still
reports 65,536 MiB and guest `MemTotal` is at least 64,000,000,000 bytes after
kernel reservation. `lscpu` is the Ubuntu utility that reports CPU, core,
socket, and NUMA topology ([Ubuntu `lscpu` manual][ubuntu-lscpu]).

### 5. Record Python and numeric-library provenance

```bash
uv run --project newcalibre --locked --no-sync python - <<'PY' \
  | tee preflight/python-numeric.txt
import importlib.metadata
import json
import platform
import sys

import numpy
import pandas
import pyarrow
import scipy
from threadpoolctl import threadpool_info

assert sys.version_info[:3] == (3, 12, 13)
print(json.dumps({
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "pandas": pandas.__version__,
    "pyarrow": pyarrow.__version__,
    "hierarchicalforecast": importlib.metadata.version("hierarchicalforecast"),
    "threadpools": threadpool_info(),
}, sort_keys=True, indent=2))
print("NUMPY_CONFIG")
numpy.show_config()
print("SCIPY_CONFIG")
scipy.show_config()
PY

uv run --project newcalibre --locked --no-sync python - <<'PY' \
  | tee preflight/shared-libraries.txt
from pathlib import Path
import numpy
import scipy
print(Path(numpy.__file__).resolve())
print(Path(scipy.__file__).resolve())
PY

for var in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS \
  NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS; do
  test "$(printenv "$var")" = 1
done
env | grep -E '^(OMP|OPENBLAS|MKL|NUMEXPR|VECLIB|BLIS)_.*THREAD' \
  | sort | tee preflight/thread-policy.txt
```

**Pass:** Python is 3.12.13; all package versions agree with the candidate
lock; NumPy and SciPy report their actual BLAS/LAPACK build; no source build or
unexpected resolver change occurred; all thread-policy variables equal one.
The manifest must store the complete configuration output, not only package
version strings.

### 6. Verify single-node Ray processes and inherited thread policy

For the immediate `e5cb11b` trial, the root lock provides Ray 2.54.1
([locked package entry][project-root-ray]). Write and run this infrastructure
probe:

```bash
cat > /tmp/ray_preflight.py <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import ray
from threadpoolctl import threadpool_info

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

@ray.remote(num_cpus=1)
def probe(index: int) -> dict[str, object]:
    values = np.arange(100_000, dtype=np.int64) + index
    time.sleep(2)
    return {
        "index": index,
        "pid": os.getpid(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "threads": {name: os.environ.get(name) for name in THREAD_VARS},
        "threadpools": threadpool_info(),
        "digest": hashlib.sha256(values.tobytes()).hexdigest(),
    }

context = ray.init(num_cpus=32, include_dashboard=False, local_mode=False)
assert len([node for node in ray.nodes() if node["Alive"]]) == 1
assert int(ray.cluster_resources()["CPU"]) == 32

def run() -> list[dict[str, object]]:
    return sorted(ray.get([probe.remote(i) for i in range(32)]), key=lambda row: row["index"])

first = run()
second = run()
assert [row["digest"] for row in first] == [row["digest"] for row in second]
driver_pid = os.getpid()
worker_pids = {int(row["pid"]) for row in first}
assert driver_pid not in worker_pids
assert len(worker_pids) >= 16
for row in first + second:
    assert all(row["threads"][name] == "1" for name in THREAD_VARS)
    assert all(int(pool["num_threads"]) == 1 for pool in row["threadpools"])

print(json.dumps({
    "ray_version": ray.__version__,
    "address": context.address_info,
    "resources": ray.cluster_resources(),
    "alive_nodes": len([node for node in ray.nodes() if node["Alive"]]),
    "driver_pid": driver_pid,
    "worker_pids": sorted(worker_pids),
    "first": first,
}, sort_keys=True, indent=2, default=str))
ray.shutdown()
PY

pidstat -h -r -u -t -p ALL 1 8 > preflight/pidstat-ray.txt &
PIDSTAT_PID=$!
uv run --locked --no-sync python /tmp/ray_preflight.py \
  > preflight/ray-process-probe.json &
RAY_PROBE_PID=$!
sleep 3
ps -eLo pid,ppid,lwp,psr,pcpu,pmem,rss,comm,args --sort=pid \
  > preflight/ps-ray.txt
wait "$RAY_PROBE_PID"
wait "$PIDSTAT_PID"
uv run --locked --no-sync ray stop --force
! pgrep -fa 'raylet|gcs_server'
```

**Pass:** one Ray node advertises 32 CPUs; at least 16 distinct worker
processes execute concurrent tasks; worker PIDs differ from the driver; every
worker sees all six thread limits as one; loaded numeric pools report one
thread; two probe rounds have identical ordered digests; `pidstat` and `ps`
show worker processes and threads; shutdown leaves no Ray control process.

This probe proves process launch, inspection, environment inheritance, and a
small deterministic computation. It does **not** prove Calibre's distributed
ledger identity. Repeat it from the successor candidate lock after U16, then
run U16's tier-2 distributed-equals-sequential test. That same-engine identity
is a permanent requirement ([test strategy at `e5cb11b`][project-test]).

### 7. Verify storage, scratch space, and M5 acquisition

First verify the control-plane volume:

```bash
aws ec2 describe-volumes \
  --region "$AWS_REGION" \
  --volume-ids "$VOLUME_ID" \
  --output json > volume.json
jq -e '.Volumes[0] |
  .Encrypted == true and
  .VolumeType == "gp3" and
  .Size == 300 and
  .Iops == 3000 and
  .Throughput == 125 and
  .State == "in-use"' volume.json
aws ec2 describe-instance-attribute \
  --region "$AWS_REGION" \
  --instance-id "$INSTANCE_ID" \
  --attribute blockDeviceMapping \
  --output json > block-device-mapping.json
jq -e '.BlockDeviceMappings[0].Ebs.VolumeId == env.VOLUME_ID and
  .BlockDeviceMappings[0].Ebs.DeleteOnTermination == true' \
  block-device-mapping.json
```

Then run on the instance. The immediate-trial download command is the
repository's current M5 downloader at this exact revision
([source][project-m5-downloader]):

```bash
lsblk -b -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS,MODEL \
  | tee preflight/lsblk.txt
findmnt --bytes | tee preflight/findmnt.txt
df -B1 / /dev/shm | tee preflight/df-bytes.txt
test "$(df -B1 --output=avail / | tail -1)" -ge 200000000000
test -r /sys/fs/cgroup/cgroup.controllers
grep -qw memory /sys/fs/cgroup/cgroup.controllers
cat /sys/fs/cgroup/memory.current | tee preflight/cgroup-memory-current.txt

mkdir -p scratch
/usr/bin/time -v -o preflight/storage-write-time.txt \
  dd if=/dev/zero of=scratch/io-probe.bin bs=8M count=512 oflag=direct status=progress
sync
sha256sum scratch/io-probe.bin | tee preflight/storage-probe.sha256
rm scratch/io-probe.bin

curl -fsSIL https://github.com/Vzlentin/calibre \
  | tee preflight/github-head.txt
curl -fsSIL https://pypi.org/simple/ \
  | tee preflight/pypi-head.txt
/usr/bin/time -v -o preflight/m5-download-time.txt \
  uv run --locked --no-sync python benchmarks/m5/download_m5_data.py \
    --target data/m5
sha256sum data/m5/sales_train_evaluation.csv data/m5/calendar.csv \
  data/m5/sell_prices.csv | tee preflight/m5-input-sha256.txt

uv run --locked --no-sync python - <<'PY' | tee preflight/m5-shape.txt
from pathlib import Path
import pandas as pd
root = Path("data/m5")
sales = root / "sales_train_evaluation.csv"
header = pd.read_csv(sales, nrows=0)
calendar = pd.read_csv(root / "calendar.csv")
days = [name for name in header if name.startswith("d_")]
with sales.open(encoding="utf-8") as stream:
    rows = sum(1 for _ in stream) - 1
dates = pd.to_datetime(calendar.loc[calendar["d"].isin(days), "date"])
print(rows, len(days), days[0], days[-1], dates.min().date(), dates.max().date())
assert rows == 30_490
assert len(days) == 1_941
assert days == [f"d_{index}" for index in range(1, 1_942)]
assert str(dates.min().date()) == "2011-01-29"
assert str(dates.max().date()) == "2016-05-22"
PY

du -sh .venv newcalibre/.venv data/m5 preflight | tee preflight/disk-use.txt
df -B1 / | tee -a preflight/df-bytes.txt
test "$(df -B1 --output=avail / | tail -1)" -ge 200000000000
```

**Pass:** provider volume facts match; the root and `/dev/shm` facts are
recorded; cgroup v2 exposes memory accounting; the direct-write probe completes
without error; HTTPS access works; M5 download completes; file digests are
recorded; evaluation data has exactly 30,490 rows, `d_1..d_1941`, and the
canonical date range; at least 200,000,000,000 bytes remain free. The M5 shape
and evaluation phase are fixed by the public protocol
([`[M5-D1]`-`[M5-D3]` at `e5cb11b`][project-m5-data]).

When U15 replaces the predecessor downloader, rerun this section with U15's
successor-owned acquisition and digest command. Do not treat successful public
network access as dataset integrity.

### 8. Verify timing and RSS tools

```bash
/usr/bin/time -v -o preflight/rss-probe-time.txt \
  uv run --project newcalibre --locked --no-sync python - <<'PY'
import numpy as np
value = np.ones(128 * 1024 * 1024, dtype=np.uint8)
print(value.nbytes, int(value.sum()))
PY

grep -E 'Elapsed|Maximum resident set size' preflight/rss-probe-time.txt
pidstat -V | tee preflight/pidstat-version.txt
/usr/bin/time --version | head -1 | tee preflight/time-version.txt
```

**Pass:** GNU `time` reports elapsed time and a nonzero maximum resident set
for a known 128 MiB allocation; `pidstat` captured Ray process/thread CPU and
RSS samples; cgroup memory accounting is readable. Ubuntu supplies documented
`time` and `pidstat` interfaces ([Ubuntu GNU `time` manual][ubuntu-time],
[Ubuntu `pidstat` manual][ubuntu-pidstat]).

These tools are only independent preflight witnesses. U16's harness-owned
stage timing and peak-RSS telemetry remains the acceptance source and must
reconcile at least 99% of wall time as chapter 30 requires
([`[PRF-30]`-`[PRF-33]` at `e5cb11b`][project-profile]).

### 9. Estimate cost and enforce the trial time cap

From the operator machine, record the API launch time and a UTC stop time:

```bash
LAUNCHED_AT="$(aws ec2 describe-instances \
  --region "$AWS_REGION" \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].LaunchTime' \
  --output text)"
ENDED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_SECONDS="$(date -u -d "$LAUNCHED_AT" +%s)"
END_SECONDS="$(date -u -d "$ENDED_AT" +%s)"
BILLED_SECONDS="$((END_SECONDS - START_SECONDS))"
test "$BILLED_SECONDS" -le 7200
awk -v seconds="$BILLED_SECONDS" 'BEGIN {
  hours = seconds / 3600
  compute = hours * 1.64224
  storage = hours * (300 * 0.08 / 730)
  ipv4 = hours * 0.005
  printf "seconds=%d hours=%.6f compute=%.4f storage=%.4f ipv4=%.4f estimate=%.4f\n",
         seconds, hours, compute, storage, ipv4, compute + storage + ipv4
}' | tee cost-estimate.txt
```

**Pass:** paid instance time is no more than two hours and the estimate uses
current recorded regional rates. Attach the later billing reference when it
becomes available; label this calculation as an estimate.

### 10. Tear down and prove deletion

Capture output IDs before deletion, exit the Session Manager shell, then run:

```bash
set -euo pipefail
aws cloudformation delete-stack \
  --region "$AWS_REGION" \
  --stack-name "$STACK"
aws cloudformation wait stack-delete-complete \
  --region "$AWS_REGION" \
  --stack-name "$STACK"

STATE="$(aws ec2 describe-instances \
  --region "$AWS_REGION" \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text)"
test "$STATE" = terminated

if aws ec2 describe-volumes \
  --region "$AWS_REGION" \
  --volume-ids "$VOLUME_ID" > volume-after-delete.json 2> volume-delete-error.txt; then
  echo 'root volume still exists' >&2
  exit 1
fi
grep -q 'InvalidVolume.NotFound' volume-delete-error.txt

if aws ec2 describe-network-interfaces \
  --region "$AWS_REGION" \
  --network-interface-ids "$NETWORK_INTERFACE_ID" \
  > interface-after-delete.json 2> interface-delete-error.txt; then
  echo 'primary network interface still exists' >&2
  exit 1
fi
grep -q 'InvalidNetworkInterfaceID.NotFound' interface-delete-error.txt

if aws cloudformation describe-stacks \
  --region "$AWS_REGION" \
  --stack-name "$STACK" >/dev/null 2>&1; then
  echo 'stack still exists' >&2
  exit 1
fi
```

**Pass:** the stack no longer exists, the instance is terminated, and the
captured root-volume and primary-interface IDs no longer exist. Record the
deletion-complete UTC time. A failed deletion is a blocker until the leaked
resource is removed and verified.

## Risks and remaining uncertainty

1. **Account quota is the first operational uncertainty.** AWS documents a
   default five-vCPU standard on-demand quota, below both the 32-vCPU primary
   and 16-vCPU fallback. Existing account history can raise it automatically,
   but issue 436 must query the actual account and request any increase
   ([AWS EC2 quotas][aws-quotas]).
2. **On-demand offering is not capacity assurance.** The offering API can show
   a shape in an AZ while a later launch still lacks capacity. Record the AZ
   used. Retry another offering AZ before invoking the fallback; do not change
   region or shape without a recorded decision.
3. **The final successor Ray lock does not exist at this research revision.**
   The current trial can prove machine and root-locked Ray operation. It cannot
   prove U16's final package set or Calibre dispatch invariance. Repeat the
   relevant sections at the exact candidate SHA before U17.
4. **64 GiB is the provider floor, not unlimited headroom.** The process RSS
   gate is 32 GB, while Ray object-store memory, page cache, the OS, and
   profiler also consume RAM. Issue 439 must set and report Ray object-store
   and worker limits. U17 fails rather than moving the 32 GB gate if total
   behavior causes pressure.
5. **A shared on-demand host can vary in contention and turbo behavior.** Pin
   the instance type, CPU model observed in the guest, AMI, lock, and thread
   policy. Record the one acceptance run honestly; do not select a favorable
   rerun after seeing a binding result.
6. **Price and image catalogs change.** The exact price snapshot is versioned
   here. Re-query the live rate before launch. Verify the pinned AMI is still
   available and Canonical-owned; any AMI replacement changes the environment
   manifest and needs an explicit record.
7. **The trial is not performance acceptance.** Small Ray, disk, and memory
   probes only show that required facilities operate. The full-M5 harness is
   the sole source of the 15-minute, 60-second, and 32-GB verdict.

### Primary issue-436 uncertainty

The main question that paper research cannot settle is whether the actual AWS
account can launch the selected C7a allocation and whether that allocation,
with the exact locked candidate, exposes the documented 32/32/1 topology while
Ray workers inherit one-thread numeric limits and remain fully observable.
Issue 436 must settle that with the raw control-plane and guest observations
above. Full-M5 speed remains deliberately deferred to U17.

## Primary sources

### Calibre authorities at `e5cb11bd5b4487724701d4cf5a4626c91031c0e1`

- [ADR 0001 reference-environment decision][project-adr]
- [Chapter 30 performance budgets and deliverables][project-perf]
- [Chapter 21 M5 protocol][project-m5]
- [Chapter 03 determinism and thread policy][project-determinism]
- [Chapter 50 same-engine distributed identity][project-test]
- [Successor Python and dependency declaration][project-toolchain]
- [Successor lockfile][project-lock]
- [Issue 434: Select the Gate C compute venue](https://github.com/Vzlentin/calibre/issues/434)
- [Issue 436: Verify the selected Gate C instance](https://github.com/Vzlentin/calibre/issues/436)

### Provider, OS, toolchain, and runtime authorities

- [AWS compute-optimized instance specifications][aws-co]
- [AWS general-purpose instance specifications][aws-gp]
- [Versioned AWS EC2 price snapshot][aws-ec2-price]
- [Versioned AWS VPC price snapshot][aws-vpc-price]
- [AWS gp3 volume specification][aws-gp3]
- [AWS EC2 on-demand billing][aws-on-demand]
- [AWS EC2 instance quotas][aws-quotas]
- [AWS instance offering discovery][aws-discovery]
- [AWS Systems Manager Session Manager][aws-session]
- [AWS CloudFormation overview][aws-cfn]
- [Canonical Ubuntu EC2 image locator][ubuntu-locator]
- [Canonical Ubuntu image discovery and owner verification][ubuntu-images]
- [Microsoft Azure Dsv5 size specification][azure-dsv5]
- [Microsoft Azure Retail Prices API][azure-price]
- [uv installation and version pinning][uv-install]
- [uv Python version pinning][uv-python]
- [uv locked synchronization][uv-sync]
- [Ray single-machine startup and shutdown][ray-single]
- [Ubuntu `lscpu`][ubuntu-lscpu], [GNU `time`][ubuntu-time], and
  [`pidstat`][ubuntu-pidstat] manuals

[project-adr]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/docs/spec/adr/0001-reference-environment.md#L65-L94
[project-perf]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/docs/spec/30-performance.md#L109-L123
[project-storage]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/docs/spec/30-performance.md#L177-L191
[project-profile]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/docs/spec/30-performance.md#L197-L221
[project-parallel]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/docs/spec/30-performance.md#L163-L174
[project-m5]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/docs/spec/21-protocol-m5.md#L68-L147
[project-m5-data]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/docs/spec/21-protocol-m5.md#L31-L60
[project-determinism]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/docs/spec/03-engine-core.md#L153-L177
[project-test]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/docs/spec/50-test-and-oracle-strategy.md#L136-L153
[project-toolchain]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/newcalibre/pyproject.toml#L1-L17
[project-lock]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/newcalibre/uv.lock#L1-L3
[project-root-ray]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/uv.lock#L3959-L3961
[project-m5-downloader]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/benchmarks/m5/download_m5_data.py
[issue-439]: https://github.com/Vzlentin/calibre/issues/439
[aws-co]: https://docs.aws.amazon.com/ec2/latest/instancetypes/co.html
[aws-gp]: https://docs.aws.amazon.com/ec2/latest/instancetypes/gp.html
[aws-ec2-price]: https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/20260721012550/us-east-1/index.json
[aws-vpc-price]: https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonVPC/20260713171339/us-east-1/index.json
[aws-gp3]: https://docs.aws.amazon.com/ebs/latest/userguide/general-purpose.html#gp3-ebs-volume-type
[aws-on-demand]: https://aws.amazon.com/ec2/pricing/on-demand/
[aws-quotas]: https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html#on-demand-instance-quotas
[aws-discovery]: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-discovery.html#instance-discovery-cli
[aws-session]: https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html
[aws-cfn]: https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/Welcome.html
[ubuntu-locator]: https://cloud-images.ubuntu.com/locator/ec2/releasesTable
[ubuntu-images]: https://documentation.ubuntu.com/aws/aws-how-to/instances/find-ubuntu-images/
[azure-dsv5]: https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/general-purpose/dsv5-series
[azure-price]: https://prices.azure.com/api/retail/prices?$filter=armRegionName%20eq%20%27westeurope%27%20and%20armSkuName%20eq%20%27Standard_D32s_v5%27%20and%20priceType%20eq%20%27Consumption%27
[uv-install]: https://docs.astral.sh/uv/getting-started/installation/#standalone-installer
[uv-python]: https://docs.astral.sh/uv/concepts/python-versions/#installing-a-python-version
[uv-sync]: https://docs.astral.sh/uv/concepts/projects/sync/#automatic-lock-and-sync
[ray-single]: https://docs.ray.io/en/latest/ray-core/starting-ray.html#starting-ray-on-a-single-machine
[ubuntu-lscpu]: https://manpages.ubuntu.com/manpages/noble/en/man1/lscpu.1.html
[ubuntu-time]: https://manpages.ubuntu.com/manpages/noble/en/man1/time.1.html
[ubuntu-pidstat]: https://manpages.ubuntu.com/manpages/noble/en/man1/pidstat.1.html
