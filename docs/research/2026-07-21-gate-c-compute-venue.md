# Gate C compute venue selection

## Supersession notice

**The earlier AWS recommendation in commit `66c92ea` is superseded and must not
be used.** The owner requires Microsoft Azure in Europe
([owner correction on issue 434][issue-434-correction]). AWS is now a rejected
alternative because it violates that product constraint. All executable
selection, provisioning, access, cost, preflight, and teardown instructions
below are Azure-only.

## Executive recommendation

Select **Microsoft Azure `Standard_NC16as_T4_v3` in West Europe
(`westeurope`)**, pay-as-you-go Linux, for the issue-436 trial and the later
U17 acceptance run.

Pin this environment:

- **Compute:** `Standard_NC16as_T4_v3`. Microsoft specifies this size as 16
  vCPUs and 110 GB memory, within a series whose AMD EPYC 7V12 CPU cores are
  explicitly **non-multithreaded**. Therefore, its 16 visible vCPUs are 16
  physical CPU cores rather than 16 SMT threads. The size also contains one
  NVIDIA T4 GPU, which Calibre will not configure or use
  ([Azure NCasT4 v3 specification][azure-nc]).
- **Region:** `westeurope`. The Azure Retail Prices API has a current West
  Europe pay-as-you-go Linux meter for this exact size. No current primary
  evidence makes another European region necessary. Subscription availability,
  restrictions, and capacity remain issue-436 checks.
- **Image rule:** Canonical publisher `Canonical`, offer
  `ubuntu-24_04-lts`, SKU `server`, AMD64 Hyper-V Generation 2. Resolve the
  newest West Europe version before deployment, record its exact four-part
  URN, and pass that version—not `latest`—to Bicep. Canonical identifies
  `Canonical:ubuntu-24_04-lts:server:latest` as the Ubuntu 24.04 LTS AMD64
  Gen2 stream ([Canonical Azure image catalog][ubuntu-azure-image]).
- **Storage:** one **512 GiB Premium SSD P20 (`Premium_LRS`) OS disk** with
  2,300 provisioned IOPS and 150 MB/s throughput, read/write host caching,
  storage-service encryption with a platform-managed key, and
  `deleteOption: Delete`. Azure documents the P20 capacity and performance
  contract, hourly-prorated managed-disk billing, and server-side encryption
  at rest ([Azure managed disks][azure-disks], [Azure disk
  encryption][azure-disk-encryption]).
- **Temporary disk:** do not use it. The selected size exposes a 352 GiB local
  temporary disk, but no environment, M5 input, result, or profile path may
  resolve to it. It is not durable across VM lifecycle operations
  ([Azure NCasT4 v3 specification][azure-nc]).
- **Runtime:** uv-managed CPython **3.12.13**, uv **0.11.30**, and the exact
  candidate `newcalibre/uv.lock` installed with `uv sync --locked`. Record the
  candidate commit and lock SHA-256.
- **Thread policy:** set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and
  `BLIS_NUM_THREADS` to `1` before any numeric import or Ray start. Set
  `CUDA_VISIBLE_DEVICES` to the empty string; Ray starts with `num_gpus=0`.
- **Provisioning and teardown:** one Bicep deployment in one dedicated resource
  group. The resource group owns the VNet, subnet, no-inbound NSG, Standard
  static public IPv4, NIC, VM, and generated OS disk. Delete the resource group
  after each session.
- **Access:** Azure Run Command through Azure RBAC and the Azure VM Agent. The
  NSG has an explicit deny-all inbound rule; no SSH/RDP rule, password, private
  key, or embedded credential is used. Run Command is an Azure control-plane
  facility that uses the VM Agent to run Linux scripts
  ([Azure Run Command][azure-run-command]).

This selects a venue; it does not assert a Gate C pass. The binding limits
remain unchanged: **full M5 <= 15 minutes total wall time, pre-origin overhead
<= 60 seconds, and peak RSS <= 32 GB**
([`[PRF-1]`, `[PRF-2]`, and `[PRF-20]` at revision `e5cb11b`][project-perf]).
No Azure trial result may lower or reinterpret them.

### Fallback

Use **Microsoft Azure `Standard_F48s_v2` in `westeurope`** with the same image
rule, P20 OS disk, Bicep topology, Run Command access, Python/uv pins, and
thread policy only if the selected NC size cannot pass issue 436 because of
regional restrictions, family quota, or current capacity.

Microsoft specifies `Standard_F48s_v2` as 48 vCPUs and 96 GB memory and states
that Fsv2 uses Intel Hyper-Threading. The public size page does not publish a
physical-core count. The expected topology is **24 physical cores x 2 threads
= 48 vCPUs**, but this remains inference until both the Azure SKU capability
`vCPUsPerCore=2` and guest `lscpu` confirm it
([Azure Fsv2 specification][azure-fsv2]). The fallback cannot be used if the
control plane or guest exposes fewer than 16 physical cores.

The fallback is not a silent substitution. It must pass the same issue-436
checks with expected values changed to 48 logical CPUs, 24 physical cores, two
threads/core, and at least 96 GB provider memory. The plan and environment
manifest must name one selected shape before any acceptance result is read.

## Decision boundary

ADR 0001 requires a workstation-class x86_64 Linux environment with at least
16 physical cores, at least 64 GB RAM, Python 3.12, a locked toolchain, and an
explicit thread policy. It leaves the concrete instance pending
([ADR 0001 at `e5cb11b`][project-adr]). The successor pins Python 3.12 in its
project and lockfile ([successor project][project-toolchain], [successor
lockfile][project-lock]). U16 will add the final Ray-bearing lock
([issue 439][issue-439]); the U17 manifest must use that later candidate digest,
not the research-time digest.

The selected Azure series gives a first-party physical-core statement without
relying on vCPU arithmetic. The unused GPU is a deliberate trade: the NC size
has a lower current West Europe list price than the ordinary D/F candidates,
meets the RAM floor, and removes physical-core ambiguity. Issue 436 must still
prove that the subscription can deploy it and that the guest exposes the
published topology. If special-family quota or capacity makes it impractical,
the preregistered Fsv2 fallback uses a normal compute family but binds physical
topology to the trial.

The M5 workload has 30,490 bottom series, 33,563 lattice nodes, a 28-day
horizon, and 64 daily origins in the reference configuration
([M5 protocol][project-m5]). The prior profile wrote about 2.6 GB of ledger
files, while chapter 30 requires sparse reconciliation and streaming ledger
I/O ([performance storage requirements][project-storage]). A 512 GiB P20 disk
provides substantial capacity headroom. This is not a runtime claim; issue 436
must still measure installed environments, data, scratch output, and free
space.

Ray must run as a normal single-machine local runtime with worker processes,
not `local_mode=True`. Ray documents that `ray.init()` starts a local instance
and that `ray.shutdown()` terminates its local processes
([Ray single-machine runtime][ray-single]). Calibre separately requires
batch-placement invariance, serial-order commit, and explicit numeric thread
budgets ([`[DET-3]`-`[DET-5]` at `e5cb11b`][project-determinism]). The venue
preflight proves platform support only. U16's tier-2 tests must prove Calibre's
same-engine distributed identity.

## Requirements matrix

| Requirement | Selected Azure environment | Status before issue 436 | Binding issue-436 observation |
| --- | --- | --- | --- |
| Azure in Europe | Azure West Europe (`westeurope`) | Owner constraint met; current retail meter exists | Subscription SKU catalog lists the exact size in West Europe without restrictions |
| x86_64 Linux | AMD EPYC x86-64; Canonical Ubuntu 24.04 AMD64 Gen2 | Provider/image contract | Azure metadata, `uname -m=x86_64`, and Ubuntu `VERSION_ID=24.04` |
| At least 16 physical cores | NC series explicitly non-multithreaded; selected size has 16 vCPUs | **Meets exactly** | SKU says 16 vCPUs and one vCPU/core; guest has 16 logical CPUs, 16 socket/core pairs, one thread/core |
| At least 64 GiB RAM | Size table: 110 GB | **Meets** | SKU memory capability and guest `MemTotal` are recorded; guest has at least 100,000,000,000 bytes |
| Python 3.12 and locked uv environment | CPython 3.12.13; uv 0.11.30; candidate lock | Versions selected | Exact versions, successful `uv sync --locked`, unchanged lock, commit and SHA-256 recorded |
| Numeric provenance and thread policy | Locked Linux x86_64 wheels; six thread variables set to one | Must be observed | NumPy/SciPy build configuration and `threadpoolctl`; every Ray worker sees one-thread policy |
| Storage | 512 GiB P20 Premium SSD; 2,300 IOPS; 150 MB/s | Provider contract | Azure disk object confirms size, SKU, performance, platform-key encryption, attachment, and deletion option |
| Temporary storage | 352 GiB local disk exists but is forbidden for Calibre | Explicit non-use | All repo/data/result/profile paths resolve to the OS disk; temp mount is recorded but unused |
| Process/RSS observability | Ubuntu `/proc`, cgroup v2, `ps`, `pidstat`, GNU `time` | Standard OS facilities | Ray PIDs/threads visible; `pidstat` samples; GNU `time` RSS; cgroup memory files readable |
| Deterministic single-node Ray | 16 non-SMT CPU cores; GPU disabled | Plausible, not proven | One local Ray node, 16 CPU resources, multiple worker PIDs, repeated bytes equal, clean shutdown |
| Reproducible provisioning | Bicep and exact image version in one resource group | Method selected | Bicep build/what-if/deploy outputs and file hashes recorded |
| Secure command path | Azure Run Command; explicit deny-all inbound | Provider control-plane facility | NSG has no allow-inbound rule; Run Command succeeds through Azure RBAC/VM Agent |
| Practical one-off cost | About USD 1.620/hour with VM, P20 disk allocation, and Standard IPv4 | Current estimate only | Live retail rates, deployment/deletion timestamps, estimate, and later bill reference recorded |
| Gate C limits | 15 min / 60 s / 32 GB | **Unchanged and untested here** | U17 only; issue 436 cannot claim the full acceptance verdict |

## Azure option comparison

### Price timestamp and method

The Azure Retail Prices API was queried at **2026-07-21T02:17:21Z** for
`armRegionName=westeurope`, `priceType=Consumption`, USD retail billing. VM
prices below are pay-as-you-go Linux meters. They exclude tax, support,
outbound transfer, disk, and public IP. Prices are dynamic and region-specific.
Issue 436 must query them again before launch.

| Azure West Europe size | Physical-core evidence | RAM / local disk | Linux PAYG price | Assessment |
| --- | --- | --- | ---: | --- |
| **`Standard_NC16as_T4_v3`** | Microsoft: 16 vCPUs in a series with **non-multithreaded AMD EPYC cores** -> 16 physical cores | 110 GB / 352 GiB temp; one unused T4 GPU | **$1.505/h**; meter `e50945e6-2e4e-595e-9f17-5315266b8b27` | **Selected.** Explicit physical contract, enough RAM, lowest current compute price in this set. Special-family quota/capacity is the main risk. |
| **`Standard_F48s_v2`** | 48 vCPUs with Intel Hyper-Threading -> expected 24 physical cores; provider page does not state the count | 96 GB / 384 GiB temp | **$2.328/h**; meter `dd086213-b92e-4f4a-935f-9ef09dd10353` | **Fallback.** More CPU margin and ordinary compute role, but topology and exact processor are trial-bound. |
| `Standard_HB120-16rs_v3` | Microsoft: 16 active vCPUs and **no simultaneous multithreading** -> 16 physical cores | 448 GB / 480 GiB temp plus local NVMe | **$4.680/h**; meter `587efe50-7630-544e-8861-b3df5a8a09f1` | Explicit topology, but HPC quota, irrelevant InfiniBand/memory, and 3x selected compute price add friction. |
| `Standard_D32s_v5` | 32 vCPUs in a hyper-threaded configuration -> expected 16 physical cores; not provider-published as a core count | 128 GB / no temp disk | **$1.840/h**; meter `4c9105f0-0fae-5bba-88c9-f2c736f78eb9` | Meets on expected topology only, has no physical-core margin, and may use one of several processor generations. |

Primary specification sources are the Azure size pages for
[NCasT4 v3][azure-nc], [Fsv2][azure-fsv2], [HBv3][azure-hbv3], and
[Dsv5][azure-dsv5]. The HB constrained-size list confirms 16 active vCPUs and
states that disabled vCPUs are unavailable
([Azure constrained vCPU sizes][azure-constrained]). Current price evidence is
from exact Azure Retail Prices API queries
([NC price][azure-price-nc], [F48 price][azure-price-f48], [HB price][azure-price-hb],
[D32 price][azure-price-d32]).

### Why West Europe remains the region

Every compared shape has a West Europe Linux consumption meter. The owner
specified Azure in Europe and the prior research already used West Europe.
There is no current primary evidence that another European region is required
to obtain the selected size, Ubuntu image stream, or managed-disk type. A
retail meter is not a capacity promise, so issue 436 must use the subscription's
live `az vm list-skus` result. A West Europe restriction or quota failure
invokes the West Europe fallback; it does not silently move the reference
region.

### Rejected alternatives

- **AWS:** superseded by the binding Azure-only owner decision, regardless of
  technical fit or price.
- **HBv3:** valid explicit-core alternative, but it adds HPC-family quota,
  InfiniBand, 448 GB RAM, and a much higher rate without improving the minimum
  reference contract.
- **D32s v5:** inexpensive and ordinary, but its exact 16-physical-core result
  remains inferred and has no margin if topology differs.
- **Spot/low-priority Azure VMs:** excluded. Interruption would turn the one
  acceptance run into an avoidable operational non-verdict.

## Exact proposed environment

| Field | Pin |
| --- | --- |
| Provider / region | Microsoft Azure / `westeurope` |
| Purchase model | Pay-as-you-go Linux, `priority: Regular`; no Spot, reservation, or savings assumption |
| VM size | `Standard_NC16as_T4_v3` |
| CPU contract | AMD EPYC 7V12; 16 visible non-multithreaded vCPUs -> 16 physical cores |
| Memory contract | 110 GB |
| GPU | One T4 physically present; no driver, CUDA package, Ray GPU resource, or workload use |
| Image stream | `Canonical:ubuntu-24_04-lts:server:<resolved-version>`; AMD64, Gen2; exact version resolved and recorded before deployment |
| Python | uv-managed CPython `3.12.13` |
| uv | `0.11.30`; installer SHA-256 `f633daff5c2a1b5e550d5dab074f21ab2d5fda2d147babf4525844ff1276e57e` |
| Project install | Exact candidate commit; `uv sync --project newcalibre --locked --group dev --python 3.12.13`; lock SHA-256 in manifest |
| Numeric thread policy | Six named variables set to `1`; `CUDA_VISIBLE_DEVICES` empty before import and Ray start |
| Ray | Candidate-locked version; one local node; `local_mode=False`; dashboard disabled; `num_gpus=0`; no external address |
| OS disk | 512 GiB Premium SSD P20 (`Premium_LRS`), 2,300 IOPS, 150 MB/s, read/write cache |
| Encryption / deletion | Storage-service encryption with platform-managed key; OS disk and NIC `deleteOption: Delete`; resource-group deletion owns final cleanup |
| Temporary storage | Present but never used for repo, environment, M5 data, results, or profiles |
| Network | Dedicated VNet/subnet, Standard static IPv4 for outbound acquisition, NSG custom deny-all inbound |
| Access | Azure Run Command authorized by Azure RBAC; no SSH/RDP rule and no embedded credential |
| Provision / destroy | Bicep group deployment; dedicated tagged resource group; `az group delete` and absence checks |
| Evidence clock | UTC operator and guest timestamps |

### Bicep resource contract

Issue 436 should create one temporary `gate-c.bicep` plus one credential-free
cloud-init/bootstrap file. Bicep is Azure's declarative language for Azure
resources ([Bicep overview][azure-bicep]). The deployment contains only:

1. One VNet, subnet, and NSG. A priority-100 custom rule denies all inbound
   traffic. Keep the NSG's default outbound allow rule; the NIC's Standard
   public IPv4 supplies explicit outbound connectivity. Do not add SSH or RDP
   rules.
2. One Standard, static IPv4 public IP and one NIC. The IP exists for outbound
   package/data traffic, not inbound administration.
3. One `Microsoft.Compute/virtualMachines` resource with `priority: Regular`,
   system-assigned identity, the exact image version parameter, and no VM
   credentials in source. If the API requires an administrator SSH public key,
   pass a disposable public key as a deployment parameter; no inbound SSH rule
   makes it unusable as an access path.
4. One generated OS disk from the image with `Premium_LRS`, 512 GiB,
   `ReadWrite` cache, and `deleteOption: Delete`. The NIC also uses
   `deleteOption: Delete`. Disable boot diagnostics unless issue 436 explicitly
   records its generated storage.
5. Tags on every resource: ticket `434`, candidate SHA, purpose
   `gate-c-preflight`, and UTC expiry.
6. Outputs for VM ID, VM name, NIC ID, public-IP ID, subnet ID, and NSG ID. The
   OS-disk ID is derived from `az vm show` after deployment.

The resource group is the teardown boundary. Microsoft documents that deleting
a resource group deletes its resources and that deletion can be verified
([Azure resource-group deletion][azure-rg-delete]). Do not commit temporary
infrastructure from this research task.

### Cost estimate

Current West Europe prices:

- selected VM: **$1.505/hour**;
- P20 LRS disk: **$80.54/month**, meter
  `56bef7bd-ecf5-4e06-9587-37d3fc7ab5f4`;
- Standard IPv4 static public IP: **$0.005/hour**, meter
  `9c150bf9-2bad-430e-a53c-c213804f49ef`.

The disk and IP values come from their exact West Europe Retail Prices API
meters ([P20 price][azure-price-p20], [Standard IPv4 price][azure-price-ip]).

Using 730 hours/month only as a planning conversion:

```text
NC16as_T4_v3 compute        1.50500 / hour
P20 LRS allocation          0.11033 / hour
Standard static IPv4        0.00500 / hour
estimated selected total    1.62033 / hour
```

Azure documents that managed-disk billing is prorated hourly from the monthly
price ([Azure managed-disk billing][azure-disk-billing]). Cap the trial at two
hours and the acceptance session at two hours: the selected planning total is
about **$6.48**. The F48 fallback estimate for the same four hours is about
**$9.77**. Both exclude tax, support, and outbound transfer. Record actual UTC
timestamps and attach the later Azure cost record when available.

## Issue 436 executable preflight

### Result rule

Issue 436 passes only when every section below records raw output and every
pass condition holds. A quota, restriction, allocation, topology, package,
observability, cost-cap, or teardown failure is a venue preflight failure—not a
Gate C verdict.

Use current `origin/main`
`e5cb11bd5b4487724701d4cf5a4626c91031c0e1` for an immediate infrastructure
trial. This revision does not yet contain U16's successor Ray dependency. The
immediate Ray probe therefore uses the repository root's locked Ray 2.54.1
([root lock entry][project-root-ray]) only to validate infrastructure. Repeat
the toolchain, thread, Ray, and observability sections from the final successor
lock before U17.

### 1. Pin Azure CLI context and refresh current prices

Run on the operator machine. Do not put subscription or tenant identifiers in
the public artifact.

```bash
set -euo pipefail
export LOCATION=westeurope
export VM_SIZE=Standard_NC16as_T4_v3
export FALLBACK_SIZE=Standard_F48s_v2
export RG=calibre-gate-c-436
export VM_NAME=calibre-gate-c
export SOURCE_SHA=e5cb11bd5b4487724701d4cf5a4626c91031c0e1
export STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export EXPIRES_AT="$(date -u -d '+2 hours' +%Y-%m-%dT%H:%M:%SZ)"

az version > azure-cli-version.json
az account show --output json > azure-account-context.private.json
az account show --query state --output tsv | grep -qx Enabled

PRICE_URL='https://prices.azure.com/api/retail/prices'
PRICE_FILTER="armRegionName eq '$LOCATION' and armSkuName eq '$VM_SIZE' and priceType eq 'Consumption'"
curl -G --fail --silent --show-error \
  --data-urlencode "\$filter=$PRICE_FILTER" \
  "$PRICE_URL" > selected-price.json
jq -e '.Items[] |
  select(.productName == "Virtual Machines NCasT4 v3 Series") |
  select(.meterName == "NC16asT4 v3") |
  .currencyCode == "USD" and .unitOfMeasure == "1 Hour" and .retailPrice > 0' \
  selected-price.json
```

**Pass:** Azure CLI is recorded; the selected subscription is enabled; the live
API returns one ordinary Linux consumption meter. Keep account context private
or redact identifiers before attaching evidence. Record the current rate and
query UTC time.

### 2. Verify regional SKU capabilities, restrictions, and quotas

Azure applies both total regional vCPU and VM-family vCPU quotas
([Azure VM quotas][azure-quotas]). Query the exact live subscription catalog
with the Azure CLI SKU and usage interfaces ([Azure VM CLI][azure-cli-vm]):

```bash
set -euo pipefail
az vm list-skus \
  --location "$LOCATION" \
  --size "$VM_SIZE" \
  --all \
  --output json > selected-sku.json

jq -e --arg location "$LOCATION" '
  length == 1 and
  .[0].resourceType == "virtualMachines" and
  (.[0].locations | index($location)) != null and
  (.[0].restrictions | length) == 0 and
  ((.[0].capabilities | map({key: .name, value: .value}) | from_entries) as $c |
    ($c.vCPUs | tonumber) == 16 and
    (($c.vCPUsAvailable // $c.vCPUs) | tonumber) == 16 and
    ($c.vCPUsPerCore | tonumber) == 1 and
    ($c.MemoryGB | tonumber) >= 100 and
    $c.PremiumIO == "True" and
    ($c.HyperVGenerations | contains("V2"))))' selected-sku.json

export SKU_FAMILY="$(jq -r '.[0].family' selected-sku.json)"
export REQUIRED_VCPUS="$(jq -r '.[0].capabilities |
  map({key: .name, value: .value}) | from_entries |
  (.vCPUsAvailable // .vCPUs)' selected-sku.json)"

az vm list-usage --location "$LOCATION" --output json > regional-usage.json
jq -e --arg family "$SKU_FAMILY" --argjson required "$REQUIRED_VCPUS" '
  ([.[] | select((.name.value | ascii_downcase) == ($family | ascii_downcase))][0]) as $family_usage |
  ([.[] | select(.name.value == "cores")][0]) as $total_usage |
  $family_usage != null and $total_usage != null and
  ($family_usage.limit - $family_usage.currentValue) >= $required and
  ($total_usage.limit - $total_usage.currentValue) >= $required' regional-usage.json
```

**Pass:** exactly one West Europe SKU entry has no restrictions; the live
catalog reports 16 visible vCPUs, one vCPU/core, at least 100 GB memory,
Premium I/O, and Gen2; both family and total regional quotas have 16 unused
vCPUs. If this fails due to quota, request the increase before paid launch. If
it fails due to restriction or capacity, run the same section for the
preregistered F48 fallback and require 48 vCPUs, 24 physical cores inferred
from `vCPUsPerCore=2`, and memory >= 90 GB.

### 3. Resolve and pin the Canonical image before launch

```bash
set -euo pipefail
export IMAGE_PUBLISHER=Canonical
export IMAGE_OFFER=ubuntu-24_04-lts
export IMAGE_SKU=server

az vm image list \
  --location "$LOCATION" \
  --publisher "$IMAGE_PUBLISHER" \
  --offer "$IMAGE_OFFER" \
  --sku "$IMAGE_SKU" \
  --architecture x64 \
  --all \
  --output json > ubuntu-images.json

jq -e 'length > 0' ubuntu-images.json
export IMAGE_VERSION="$(jq -r 'sort_by(.version) | last | .version' ubuntu-images.json)"
test -n "$IMAGE_VERSION"
test "$IMAGE_VERSION" != null
export IMAGE_URN="$IMAGE_PUBLISHER:$IMAGE_OFFER:$IMAGE_SKU:$IMAGE_VERSION"
az vm image show \
  --location "$LOCATION" \
  --urn "$IMAGE_URN" \
  --output json > selected-image.json

jq -e --arg version "$IMAGE_VERSION" '
  .name == $version and
  .location == "westeurope" and
  .hyperVGeneration == "V2" and
  .plan == null and
  .osDiskImage.operatingSystem == "Linux"' selected-image.json
printf '%s\n' "$IMAGE_URN" | tee selected-image-urn.txt
sha256sum selected-image.json ubuntu-images.json > image-catalog.sha256
```

**Pass:** the exact version is nonempty, Linux, Gen2, plan-free, and in West
Europe. Bicep receives `IMAGE_VERSION`; it never receives `latest`. Azure
identifies image publisher, offer, SKU, and version as the reproducible URN
fields and supports listing all regional versions
([Azure image discovery][azure-image-discovery]).

### 4. Build, review, and deploy the Bicep resource group

Create a disposable public key only to satisfy the Linux OS profile. There is
no SSH inbound rule and the private key is not copied to the VM or artifact.

```bash
set -euo pipefail
export TEMPLATE=gate-c.bicep
export CLOUD_INIT=cloud-init.yaml
ssh-keygen -q -t ed25519 -N '' -f /tmp/gate-c-unused-key
export ADMIN_PUBLIC_KEY="$(cat /tmp/gate-c-unused-key.pub)"

az bicep version | tee bicep-version.txt
az bicep build --file "$TEMPLATE" --outfile /tmp/gate-c.json
sha256sum "$TEMPLATE" "$CLOUD_INIT" /tmp/gate-c.json > provisioning.sha256

az group create \
  --name "$RG" \
  --location "$LOCATION" \
  --tags Ticket=434 Purpose=gate-c-preflight CandidateSha="$SOURCE_SHA" ExpiresAt="$EXPIRES_AT" \
  --output json > group-create.json

az deployment group what-if \
  --resource-group "$RG" \
  --template-file "$TEMPLATE" \
  --parameters \
    vmName="$VM_NAME" \
    vmSize="$VM_SIZE" \
    imageVersion="$IMAGE_VERSION" \
    adminPublicKey="$ADMIN_PUBLIC_KEY" \
    candidateSha="$SOURCE_SHA" \
    expiresAt="$EXPIRES_AT" \
  --output json > deployment-what-if.json

az deployment group create \
  --resource-group "$RG" \
  --name gate-c \
  --mode Complete \
  --template-file "$TEMPLATE" \
  --parameters \
    vmName="$VM_NAME" \
    vmSize="$VM_SIZE" \
    imageVersion="$IMAGE_VERSION" \
    adminPublicKey="$ADMIN_PUBLIC_KEY" \
    candidateSha="$SOURCE_SHA" \
    expiresAt="$EXPIRES_AT" \
  --output json > deployment.json

jq -e '.properties.provisioningState == "Succeeded"' deployment.json
```

**Pass:** Bicep builds; hashes and what-if output are recorded; only declared
resources appear; deployment succeeds. Delete the disposable key immediately
after the VM Agent reports ready.

### 5. Verify the Azure control-plane result and no-inbound access model

```bash
set -euo pipefail
az vm show \
  --resource-group "$RG" \
  --name "$VM_NAME" \
  --show-details \
  --output json > vm.json
jq -e --arg size "$VM_SIZE" --arg version "$IMAGE_VERSION" '
  .hardwareProfile.vmSize == $size and
  .provisioningState == "Succeeded" and
  .powerState == "VM running" and
  .storageProfile.imageReference.publisher == "Canonical" and
  .storageProfile.imageReference.offer == "ubuntu-24_04-lts" and
  .storageProfile.imageReference.sku == "server" and
  .storageProfile.imageReference.version == $version and
  .storageProfile.osDisk.diskSizeGb == 512 and
  .storageProfile.osDisk.managedDisk.storageAccountType == "Premium_LRS" and
  .storageProfile.osDisk.deleteOption == "Delete"' vm.json

export OS_DISK_ID="$(jq -r '.storageProfile.osDisk.managedDisk.id' vm.json)"
export NIC_ID="$(jq -r '.networkProfile.networkInterfaces[0].id' vm.json)"
export NSG_ID="$(az network nic show --ids "$NIC_ID" --query networkSecurityGroup.id --output tsv)"
export NSG_NAME="${NSG_ID##*/}"

az disk show --ids "$OS_DISK_ID" --output json > os-disk.json
jq -e '.diskSizeGb == 512 and
  .sku.name == "Premium_LRS" and
  .diskIOPSReadWrite == 2300 and
  .diskMBpsReadWrite == 150 and
  .encryption.type == "EncryptionAtRestWithPlatformKey" and
  .diskState == "Attached"' os-disk.json

az network nsg rule list \
  --resource-group "$RG" \
  --nsg-name "$NSG_NAME" \
  --output json > nsg-rules.json
jq -e 'any(.[];
  .name == "DenyAllInbound" and
  .priority == 100 and
  .access == "Deny" and
  .direction == "Inbound" and
  .sourceAddressPrefix == "*" and
  .destinationPortRange == "*")' nsg-rules.json

az vm run-command invoke \
  --resource-group "$RG" \
  --name "$VM_NAME" \
  --command-id RunShellScript \
  --scripts 'echo RUN_COMMAND_OK' \
  --output json > run-command-smoke.json
jq -e '.value[] | select(.message | contains("RUN_COMMAND_OK"))' run-command-smoke.json
rm -f /tmp/gate-c-unused-key /tmp/gate-c-unused-key.pub
```

**Pass:** exact VM/image/disk values match; disk encryption and performance are
reported; the priority-100 inbound deny exists; Run Command works; no SSH
private key remains. Do not add an inbound exception after this check.

### 6. Bootstrap the exact source and locked environments

Submit a credential-free script with Run Command. The script writes detailed
output under `/opt/calibre-evidence` on the OS disk.

```bash
az vm run-command invoke \
  --resource-group "$RG" \
  --name "$VM_NAME" \
  --command-id RunShellScript \
  --scripts @guest-bootstrap.sh \
  --output json > guest-bootstrap-command.json
```

`guest-bootstrap.sh` must execute:

```bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export LC_ALL=C
export PATH="/root/.local/bin:$PATH"
SOURCE_SHA=e5cb11bd5b4487724701d4cf5a4626c91031c0e1
mkdir -p /opt/calibre-evidence

apt-get update
apt-get install -y curl git jq numactl procps sysstat time
curl -LsSf https://astral.sh/uv/0.11.30/install.sh -o /tmp/uv-install.sh
echo 'f633daff5c2a1b5e550d5dab074f21ab2d5fda2d147babf4525844ff1276e57e  /tmp/uv-install.sh' \
  | sha256sum -c -
sh /tmp/uv-install.sh
test "$(uv --version | awk '{print $1, $2}')" = 'uv 0.11.30'
uv python install 3.12.13

cat > /etc/profile.d/calibre-thread-policy.sh <<'EOF'
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=
EOF
. /etc/profile.d/calibre-thread-policy.sh

git clone https://github.com/Vzlentin/calibre.git /opt/calibre
cd /opt/calibre
git checkout --detach "$SOURCE_SHA"
test "$(git rev-parse HEAD)" = "$SOURCE_SHA"
test -z "$(git status --short)"
sha256sum newcalibre/uv.lock uv.lock | tee /opt/calibre-evidence/lockfiles.sha256

uv sync --project newcalibre --locked --group dev --python 3.12.13
uv sync --locked --extra dev --extra benchmarks --extra hierarchy --python 3.12.13
git diff --exit-code -- newcalibre/uv.lock uv.lock
```

**Pass:** exact uv and Python install; source SHA matches; repository starts
clean; both locked syncs succeed without lock changes. For the later U16/U17
candidate, replace `SOURCE_SHA`, use successor Ray, and record the new lock
digest.

### 7. Record metadata, architecture, topology, memory, and OS

Run this guest section through Run Command:

```bash
set -euo pipefail
cd /opt/calibre
. /etc/profile.d/calibre-thread-policy.sh
E=/opt/calibre-evidence

curl -fsS -H Metadata:true \
  'http://169.254.169.254/metadata/instance?api-version=2021-02-01' \
  > "$E/azure-instance-metadata.json"
jq -e '.compute.location == "westeurope" and
  .compute.vmSize == "Standard_NC16as_T4_v3" and
  .compute.osType == "Linux" and
  .compute.storageProfile.imageReference.publisher == "Canonical" and
  .compute.storageProfile.imageReference.offer == "ubuntu-24_04-lts" and
  .compute.storageProfile.imageReference.sku == "server"' \
  "$E/azure-instance-metadata.json"

uname -a | tee "$E/uname.txt"
test "$(uname -m)" = x86_64
cat /etc/os-release | tee "$E/os-release.txt"
. /etc/os-release
test "$ID" = ubuntu
test "$VERSION_ID" = 24.04

lscpu | tee "$E/lscpu.txt"
lscpu --json > "$E/lscpu.json"
lscpu --parse=CPU,CORE,SOCKET,NODE > "$E/lscpu-topology.csv"
test "$(lscpu -p=CPU | grep -vc '^#')" -eq 16
test "$(lscpu -p=SOCKET,CORE | grep -v '^#' | sort -u | wc -l)" -eq 16
test "$(lscpu | awk -F: '/Thread.s. per core/{gsub(/ /,"",$2); print $2}')" -eq 1
test "$(nproc --all)" -eq 16
test "$(nproc)" -eq 16
numactl --hardware | tee "$E/numa.txt"

free -b | tee "$E/free-bytes.txt"
awk '/MemTotal/{printf "%.0f\n", $2 * 1024}' /proc/meminfo \
  | tee "$E/memtotal-bytes.txt"
test "$(awk '/MemTotal/{printf "%.0f\n", $2 * 1024}' /proc/meminfo)" \
  -ge 100000000000
```

**Pass:** metadata names the selected Azure environment; Ubuntu and x86_64
match; all 16 CPUs are available; there are 16 physical socket/core pairs and
one thread/core; socket/NUMA facts are recorded; guest memory is at least 100
billion bytes. The provider's SKU capability and guest topology must agree.

### 8. Record Python and numeric-library provenance

```bash
set -euo pipefail
cd /opt/calibre
. /etc/profile.d/calibre-thread-policy.sh
E=/opt/calibre-evidence

uv run --project newcalibre --locked --no-sync python - <<'PY' \
  | tee "$E/python-numeric.txt"
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

for var in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS \
  NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS; do
  test "$(printenv "$var")" = 1
done
test -z "${CUDA_VISIBLE_DEVICES:-}"
env | grep -E '^(OMP|OPENBLAS|MKL|NUMEXPR|VECLIB|BLIS|CUDA_VISIBLE)' \
  | sort | tee "$E/thread-policy.txt"
```

**Pass:** Python is 3.12.13; packages match the candidate lock; NumPy/SciPy
report actual BLAS/LAPACK provenance; every thread limit is one; GPU visibility
is empty. Store the complete configuration, not only version strings.

### 9. Verify local multi-process Ray and observability

For the immediate trial, write this root-lock infrastructure probe to
`/tmp/ray_preflight.py`:

```python
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

@ray.remote(num_cpus=1, num_gpus=0)
def probe(index: int) -> dict[str, object]:
    values = np.arange(100_000, dtype=np.int64) + index
    time.sleep(2)
    return {
        "index": index,
        "pid": os.getpid(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "threads": {name: os.environ.get(name) for name in THREAD_VARS},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "threadpools": threadpool_info(),
        "digest": hashlib.sha256(values.tobytes()).hexdigest(),
    }

context = ray.init(
    num_cpus=16,
    num_gpus=0,
    include_dashboard=False,
    local_mode=False,
)
assert len([node for node in ray.nodes() if node["Alive"]]) == 1
assert int(ray.cluster_resources()["CPU"]) == 16
assert ray.cluster_resources().get("GPU", 0) == 0

def run() -> list[dict[str, object]]:
    return sorted(
        ray.get([probe.remote(index) for index in range(16)]),
        key=lambda row: row["index"],
    )

first = run()
second = run()
assert [row["digest"] for row in first] == [row["digest"] for row in second]
driver_pid = os.getpid()
worker_pids = {int(row["pid"]) for row in first}
assert driver_pid not in worker_pids
assert len(worker_pids) == 16
for row in first + second:
    assert all(row["threads"][name] == "1" for name in THREAD_VARS)
    assert row["cuda_visible_devices"] == ""
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
```

Run it with process/thread sampling:

```bash
set -euo pipefail
cd /opt/calibre
. /etc/profile.d/calibre-thread-policy.sh
E=/opt/calibre-evidence

pidstat -h -r -u -t -p ALL 1 8 > "$E/pidstat-ray.txt" &
PIDSTAT_PID=$!
uv run --locked --no-sync python /tmp/ray_preflight.py \
  > "$E/ray-process-probe.json" &
RAY_PROBE_PID=$!
sleep 3
ps -eLo pid,ppid,lwp,psr,pcpu,pmem,rss,comm,args --sort=pid \
  > "$E/ps-ray.txt"
wait "$RAY_PROBE_PID"
wait "$PIDSTAT_PID"
uv run --locked --no-sync ray stop --force
! pgrep -fa 'raylet|gcs_server'
```

**Pass:** one Ray node advertises 16 CPU and zero GPU resources; exactly 16
concurrent worker PIDs differ from the driver; all workers inherit one-thread
numeric limits and an empty GPU view; repeated ordered digests match;
`pidstat`/`ps` expose processes and threads; shutdown leaves no Ray control
process. Repeat from the final successor lock and run U16's
serial-equals-distributed tier-2 test. This probe cannot certify Calibre ledger
identity ([test strategy][project-test]).

### 10. Verify disk placement, free space, M5 acquisition, timing, and RSS

```bash
set -euo pipefail
cd /opt/calibre
. /etc/profile.d/calibre-thread-policy.sh
E=/opt/calibre-evidence

lsblk -b -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS,MODEL | tee "$E/lsblk.txt"
findmnt --bytes | tee "$E/findmnt.txt"
df -B1 / /dev/shm | tee "$E/df-bytes.txt"
test "$(df -B1 --output=avail / | tail -1)" -ge 300000000000
for path in /opt/calibre /opt/calibre/.venv /opt/calibre/newcalibre/.venv "$E"; do
  test "$(findmnt -n -o TARGET -T "$path")" = /
done
findmnt /mnt > "$E/temporary-mount.txt" 2>&1 || true

test -r /sys/fs/cgroup/cgroup.controllers
grep -qw memory /sys/fs/cgroup/cgroup.controllers
cat /sys/fs/cgroup/memory.current | tee "$E/cgroup-memory-current.txt"

mkdir -p /opt/calibre-scratch
/usr/bin/time -v -o "$E/storage-write-time.txt" \
  dd if=/dev/zero of=/opt/calibre-scratch/io-probe.bin \
    bs=8M count=512 oflag=direct status=progress
sync
sha256sum /opt/calibre-scratch/io-probe.bin | tee "$E/storage-probe.sha256"
rm /opt/calibre-scratch/io-probe.bin

curl -fsSIL https://github.com/Vzlentin/calibre | tee "$E/github-head.txt"
curl -fsSIL https://pypi.org/simple/ | tee "$E/pypi-head.txt"
/usr/bin/time -v -o "$E/m5-download-time.txt" \
  uv run --locked --no-sync python benchmarks/m5/download_m5_data.py \
    --target data/m5
sha256sum data/m5/sales_train_evaluation.csv data/m5/calendar.csv \
  data/m5/sell_prices.csv | tee "$E/m5-input-sha256.txt"

uv run --locked --no-sync python - <<'PY' | tee "$E/m5-shape.txt"
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

/usr/bin/time -v -o "$E/rss-probe-time.txt" \
  uv run --project newcalibre --locked --no-sync python - <<'PY'
import numpy as np
value = np.ones(128 * 1024 * 1024, dtype=np.uint8)
print(value.nbytes, int(value.sum()))
PY

grep -E 'Elapsed|Maximum resident set size' "$E/rss-probe-time.txt"
pidstat -V | tee "$E/pidstat-version.txt"
/usr/bin/time --version | head -1 | tee "$E/time-version.txt"
du -sh .venv newcalibre/.venv data/m5 "$E" | tee "$E/disk-use.txt"
df -B1 / | tee -a "$E/df-bytes.txt"
test "$(df -B1 --output=avail / | tail -1)" -ge 300000000000
```

**Pass:** the root filesystem is the 512 GiB OS disk; all Calibre paths resolve
to `/`, not temporary storage; >= 300 billion bytes remain free after both
environments and M5 data; cgroup memory accounting works; direct I/O and HTTPS
succeed; M5 digests are recorded; evaluation data has 30,490 rows,
`d_1..d_1941`, and the canonical date range; GNU `time` reports elapsed and
maximum RSS; `pidstat` reports process/thread samples. M5 dimensions come from
`[M5-D1]`-`[M5-D3]` ([protocol source][project-m5-data]). When U15 provides its
successor acquisition command, rerun with that command before U17.

U16's harness-emitted per-stage timing and RSS remain the acceptance sources
and must reconcile at least 99% of wall time
([`[PRF-30]`-`[PRF-33]`][project-profile]).

### 11. Record evidence summary and estimate trial cost

Run a final guest summary through Run Command:

```bash
set -euo pipefail
E=/opt/calibre-evidence
sha256sum "$E"/* | sort > "$E/evidence-files.sha256"
tar -C /opt -czf /opt/calibre-evidence.tar.gz calibre-evidence
sha256sum /opt/calibre-evidence.tar.gz
printf 'PREFLIGHT_COMPLETE %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
```

Record operator-side elapsed time and estimate:

```bash
ENDED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LAUNCHED_AT="$(jq -r '.timeCreated' vm.json)"
test -n "$LAUNCHED_AT"
START_SECONDS="$(date -u -d "$LAUNCHED_AT" +%s)"
END_SECONDS="$(date -u -d "$ENDED_AT" +%s)"
ELAPSED_SECONDS="$((END_SECONDS - START_SECONDS))"
test "$ELAPSED_SECONDS" -le 7200

VM_RATE="$(jq -r '.Items[] |
  select(.productName == "Virtual Machines NCasT4 v3 Series") |
  select(.meterName == "NC16asT4 v3") | .retailPrice' selected-price.json)"
awk -v seconds="$ELAPSED_SECONDS" -v vm="$VM_RATE" 'BEGIN {
  hours = seconds / 3600
  disk = hours * (80.54 / 730)
  ipv4 = hours * 0.005
  printf "seconds=%d hours=%.6f vm=%.4f disk=%.4f ipv4=%.4f estimate=%.4f\n",
         seconds, hours, hours * vm, disk, ipv4, hours * vm + disk + ipv4
}' | tee cost-estimate.txt
```

**Pass:** evidence hashes and completion UTC exist; paid elapsed time is no more
than two hours; estimate uses the refreshed VM rate and recorded disk/IP rates.
Retrieve required evidence before teardown by the owner's approved Azure
control-plane method; do not place a storage credential in Run Command or the
artifact.

### 12. Delete the resource group and prove teardown

```bash
set -euo pipefail
az group delete --name "$RG" --yes --no-wait
az group wait --name "$RG" --deleted

test "$(az group exists --name "$RG")" = false
test "$(az resource list \
  --tag Ticket=434 \
  --query "[?tags.Purpose=='gate-c-preflight' && tags.CandidateSha=='$SOURCE_SHA'] | length(@)" \
  --output tsv)" -eq 0
printf 'DELETE_COMPLETE %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  | tee delete-complete.txt
```

**Pass:** the resource group no longer exists and no subscription resource with
the trial's ticket/purpose/candidate tags remains. A deletion failure blocks
completion until every leaked disk, IP, NIC, VM, and network resource is
removed and verified.

## Risks and remaining uncertainty

1. **Selected-family quota and availability are the main operational risks.**
   GPU-family quotas can be zero or restricted per subscription. A retail
   meter proves a regional price, not that this subscription can allocate the
   VM. Issue 436 must bind live SKU restrictions, family quota, total quota,
   and an actual allocation before the plan uses the venue.
2. **The unused GPU is intentional but nonrepresentative.** No GPU driver,
   CUDA dependency, or Ray GPU resource is allowed. The size is selected for
   its explicit non-SMT CPU contract, RAM, and current price. If the GPU family
   causes allocation friction, use only the preregistered F48 fallback after
   its physical topology passes.
3. **The fallback topology is inferred until provisioned.** Fsv2 documents
   Hyper-Threading but not physical-core count. Both `vCPUsPerCore=2` and guest
   `lscpu` must show 24 physical cores. A mismatch is a fallback failure.
4. **The final successor Ray lock does not exist at this research revision.**
   Current root-locked Ray can prove Azure process launch and observability,
   not U16's package set or dispatch identity. Repeat from the final lock.
5. **Temporary storage can create false evidence.** Azure local disks are not
   the artifact store. Every data/result/profile path must resolve to the P20
   OS disk, and the resource-group deletion remains the lifecycle boundary.
6. **Shared-cloud performance can vary.** Record CPU model, NUMA topology,
   image version, lock, and thread policy. Do not select a favorable rerun
   after reading a binding U17 result.
7. **Price and image catalogs change.** Resolve the exact image and refresh
   prices immediately before launch. Any image or region change alters the
   environment manifest and requires an explicit decision.
8. **The trial is not performance acceptance.** Small CPU, Ray, disk, network,
   and RSS probes only prove venue facilities. U17 alone decides the unchanged
   15-minute, 60-second, and 32-GB limits.

### Primary issue-436 uncertainty

Paper research cannot prove that the owner's Azure subscription can allocate
`Standard_NC16as_T4_v3` in West Europe. Issue 436 must settle the live regional
SKU restrictions and both quotas, then prove on the allocated guest that Azure
exposes **16 logical CPUs as 16 physical non-SMT cores** and that the exact
locked Ray environment inherits one-thread numeric limits. Full-M5 speed
remains deliberately deferred to U17.

## Conclusion

The Gate C reference venue is **Azure West Europe
`Standard_NC16as_T4_v3`**, pay-as-you-go Linux, on an exact resolved Canonical
Ubuntu 24.04 Gen2 image and a 512 GiB P20 Premium SSD OS disk, provisioned by
Bicep and operated through Azure Run Command with all inbound traffic denied.
The only fallback is **Azure West Europe `Standard_F48s_v2`**, contingent on a
verified 24-physical-core topology. The prior AWS recommendation is void.

## Primary sources

### Calibre authorities at `e5cb11bd5b4487724701d4cf5a4626c91031c0e1`

- [ADR 0001 reference environment][project-adr]
- [Chapter 30 budgets, storage, and profile requirements][project-perf]
- [Chapter 21 M5 protocol][project-m5]
- [Chapter 03 determinism and thread budgets][project-determinism]
- [Chapter 50 same-engine distributed identity][project-test]
- [Successor project and lockfile][project-toolchain]
- [Issue 434 owner correction][issue-434-correction]
- [Issue 436 verification ticket](https://github.com/Vzlentin/calibre/issues/436)

### Azure, Canonical, uv, and Ray authorities

- [NCasT4 v3 size specification][azure-nc]
- [Fsv2 size specification][azure-fsv2]
- [HBv3 size specification][azure-hbv3]
- [Dsv5 size specification][azure-dsv5]
- [Constrained vCPU semantics and HB active-vCPU roster][azure-constrained]
- [Azure Retail Prices API queries][azure-price-nc]
- [West Europe P20 price][azure-price-p20] and [Standard IPv4 price][azure-price-ip]
- [Azure Premium SSD P20 contract and billing][azure-disks]
- [Azure managed-disk encryption][azure-disk-encryption]
- [Canonical Ubuntu 24.04 Azure URN][ubuntu-azure-image]
- [Azure image discovery][azure-image-discovery]
- [Azure VM quotas][azure-quotas] and [Azure VM CLI][azure-cli-vm]
- [Bicep overview][azure-bicep]
- [Azure Run Command][azure-run-command]
- [Azure resource-group deletion][azure-rg-delete]
- [uv installation][uv-install], [Python pinning][uv-python], and
  [locked synchronization][uv-sync]
- [Ray single-machine runtime][ray-single]
- [Ubuntu `lscpu`][ubuntu-lscpu], [GNU `time`][ubuntu-time], and
  [`pidstat`][ubuntu-pidstat] manuals

[issue-434-correction]: https://github.com/Vzlentin/calibre/issues/434#issuecomment-5035142835
[issue-439]: https://github.com/Vzlentin/calibre/issues/439
[project-adr]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/docs/spec/adr/0001-reference-environment.md#L65-L94
[project-perf]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/docs/spec/30-performance.md#L109-L123
[project-storage]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/docs/spec/30-performance.md#L177-L191
[project-profile]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/docs/spec/30-performance.md#L197-L221
[project-m5]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/docs/spec/21-protocol-m5.md#L68-L147
[project-m5-data]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/docs/spec/21-protocol-m5.md#L31-L60
[project-determinism]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/docs/spec/03-engine-core.md#L153-L177
[project-test]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/docs/spec/50-test-and-oracle-strategy.md#L136-L153
[project-toolchain]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/newcalibre/pyproject.toml#L1-L17
[project-lock]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/newcalibre/uv.lock#L1-L3
[project-root-ray]: https://github.com/Vzlentin/calibre/blob/e5cb11bd5b4487724701d4cf5a4626c91031c0e1/uv.lock#L3959-L3961
[azure-nc]: https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ncast4v3-series
[azure-fsv2]: https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/compute-optimized/fsv2-series
[azure-hbv3]: https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/high-performance-compute/hbv3-series
[azure-dsv5]: https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/general-purpose/dsv5-series
[azure-constrained]: https://learn.microsoft.com/en-us/azure/virtual-machines/constrained-vcpu
[azure-price-nc]: https://prices.azure.com/api/retail/prices?$filter=armRegionName%20eq%20%27westeurope%27%20and%20armSkuName%20eq%20%27Standard_NC16as_T4_v3%27%20and%20priceType%20eq%20%27Consumption%27
[azure-price-f48]: https://prices.azure.com/api/retail/prices?$filter=armRegionName%20eq%20%27westeurope%27%20and%20armSkuName%20eq%20%27Standard_F48s_v2%27%20and%20priceType%20eq%20%27Consumption%27
[azure-price-hb]: https://prices.azure.com/api/retail/prices?$filter=armRegionName%20eq%20%27westeurope%27%20and%20armSkuName%20eq%20%27Standard_HB120-16rs_v3%27%20and%20priceType%20eq%20%27Consumption%27
[azure-price-d32]: https://prices.azure.com/api/retail/prices?$filter=armRegionName%20eq%20%27westeurope%27%20and%20armSkuName%20eq%20%27Standard_D32s_v5%27%20and%20priceType%20eq%20%27Consumption%27
[azure-price-p20]: https://prices.azure.com/api/retail/prices?$filter=armRegionName%20eq%20%27westeurope%27%20and%20meterId%20eq%20%2756bef7bd-ecf5-4e06-9587-37d3fc7ab5f4%27%20and%20priceType%20eq%20%27Consumption%27
[azure-price-ip]: https://prices.azure.com/api/retail/prices?$filter=armRegionName%20eq%20%27westeurope%27%20and%20meterId%20eq%20%279c150bf9-2bad-430e-a53c-c213804f49ef%27%20and%20priceType%20eq%20%27Consumption%27
[azure-disks]: https://learn.microsoft.com/en-us/azure/virtual-machines/disks-types#premium-ssd-size
[azure-disk-encryption]: https://learn.microsoft.com/en-us/azure/virtual-machines/disk-encryption
[azure-disk-billing]: https://learn.microsoft.com/en-us/azure/virtual-machines/disks-types#billing
[ubuntu-azure-image]: https://documentation.ubuntu.com/azure/azure-how-to/instances/find-ubuntu-images/#ubuntu-24-04-lts-noble-numbat
[azure-image-discovery]: https://learn.microsoft.com/en-us/azure/virtual-machines/linux/cli-ps-findimage#look-at-all-available-images
[azure-quotas]: https://learn.microsoft.com/en-us/azure/virtual-machines/quotas
[azure-cli-vm]: https://learn.microsoft.com/en-us/cli/azure/vm#az-vm-list-skus
[azure-bicep]: https://learn.microsoft.com/en-us/azure/azure-resource-manager/bicep/overview
[azure-run-command]: https://learn.microsoft.com/en-us/azure/virtual-machines/run-command-overview
[azure-rg-delete]: https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/delete-resource-group
[uv-install]: https://docs.astral.sh/uv/getting-started/installation/#standalone-installer
[uv-python]: https://docs.astral.sh/uv/concepts/python-versions/#installing-a-python-version
[uv-sync]: https://docs.astral.sh/uv/concepts/projects/sync/#automatic-lock-and-sync
[ray-single]: https://docs.ray.io/en/latest/ray-core/starting-ray.html#starting-ray-on-a-single-machine
[ubuntu-lscpu]: https://manpages.ubuntu.com/manpages/noble/en/man1/lscpu.1.html
[ubuntu-time]: https://manpages.ubuntu.com/manpages/noble/en/man1/time.1.html
[ubuntu-pidstat]: https://manpages.ubuntu.com/manpages/noble/en/man1/pidstat.1.html
