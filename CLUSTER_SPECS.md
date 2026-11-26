# Compute infrastructure (status 26.11.2026)

Collected on 26.11.2026 from Slurm (`sinfo`, `scontrol`, `sacctmgr`), from short probe jobs on one node of each type we use (`lscpu`, `nvidia-smi`, DMI, `lsblk`, `/sys/class/net`), and from the interactive host. Values marked † come from Slurm's configuration only and were not probed directly.

## Overview

- **Cluster:** a Slurm cluster at the i6 chair (RWTH Aachen) shared by several groups. Our account is **`log`**, and it can use only the 4 `log_gpu_*` partitions.
- **What our account can use:**
  - **13 GPU nodes with 48 GPUs and about 2.4 TB of GPU memory in total:** 4× A100 80GB, 8× A10, 32× L40S and 4× RTX PRO 6000 Blackwell.
  - Each node has 500 GB of RAM. The maximum job length is 7 days.
- **Storage:** one shared NFS volume, `/work/log1` (143 TB). There is no InfiniBand; nodes are connected by 1 Gbit/s Ethernet.
- **Interactive host:** `desk-01`, with 2× NVIDIA L4. `sbatch` works from here; `srun` does not.

## GPU nodes available to us (account `log`)

| partition | nodes | server | CPUs | threads | RAM | GPUs per node | GPU memory | GPU TDP |
|---|---|---|---|---|---|---|---|---|
| `log_gpu_80gb` | cn-301 | Dell PowerEdge R750xa | 2× Intel Xeon Gold 6326 (16 cores, 2.9 GHz) | 64 | 503 GB | 4× **NVIDIA A100 80GB PCIe**, with NVLink bridge | 80 GB | 300 W |
| `log_gpu_24gb` | cn-302, cn-303 | Dell PowerEdge R750xa | 2× Intel Xeon Gold 6330 (28 cores, 2.0 GHz) | 112 | 503 GB | 4× **NVIDIA A10** | 24 GB | 150 W |
| `log_gpu_48gb` | cn-304 … cn-311 (8 nodes) | Dell PowerEdge R760xa | 2× Intel Xeon Silver 4510 (12 cores) | 48 | 503 GB | 4× **NVIDIA L40S** | 48 GB | 350 W |
| `log_gpu_96gb` | cn-312, cn-313 | Dell PowerEdge R7725 | 2× AMD EPYC 9115 (16 cores, up to 2.6 GHz) | 64 | 503 GB | 2× **NVIDIA RTX PRO 6000 Blackwell Server Edition** | 96 GB | 600 W |

Probed nodes: cn-301, cn-302, cn-308 and cn-312. The other nodes in each partition have the same Slurm configuration (CPU count, RAM, GPUs) †.

### GPU details

| GPU | architecture (compute capability) | memory | max SM clock | PCIe | count for `log` |
|---|---|---|---|---|---|
| A100 80GB PCIe | Ampere (8.0) | 81,920 MiB | 1410 MHz | Gen4 x16 | 4 |
| A10 | Ampere (8.6) | 23,028 MiB | 1695 MHz | Gen4 x16 | 8 |
| L40S | Ada Lovelace (8.9) | 46,068 MiB | 2520 MHz | Gen4 x16 | 32 |
| RTX PRO 6000 Blackwell Server | Blackwell (12.0) | 97,887 MiB | 2430 MHz | Gen5 x16 | 4 |
| **total** | | **about 2.4 TB** | | | **48** |

- **NVLink:** only the A100 node has it; `nvidia-smi nvlink` shows NVLink at 25 GB/s per link. All other GPUs are connected through PCIe only.
- **GPU sharing:** Slurm offers shares of a GPU (`--gres=shard:N`) besides whole GPUs (`--gres=gpu:N`). There are about 1 shard per GB of GPU memory: A100 80 shards/GPU, A10 24, L40S 48, RTX PRO 6000 96.
- **Driver:** 580.126.09 (580.178.04 on cn-312), with CUDA 13.0 driver support.

### Slurm limits and accounting

| | value |
|---|---|
| Slurm version | 24.11.6 |
| maximum job length | 7 days on every `log_gpu_*` partition |
| QoS | `normal`; there is no per-user GPU limit for `log` (`GrpTRES` / `MaxTRES` empty) |
| billing weights (CPU / GB memory / GPU) | 80gb: 30 / 10 / 224 · 24gb: 16 / 4 / 64 · 48gb: 16 / 4 / 160 · 96gb: 16 / 4 / 320 |
| our own rule | use at most half of the `log` GPUs (≤ 24), and limit job arrays with `%M` |

## Interactive host (`desk-01`)

| | |
|---|---|
| server | Dell PowerEdge R6625 |
| CPU | 2× AMD EPYC 9124 (16 cores, up to 3.0 GHz): 32 cores / 64 threads, 128 MB L3 cache, 2 NUMA nodes, AVX-512 |
| RAM | 125 GB |
| GPUs | 2× **NVIDIA L4** (24 GB, 72 W, Ada Lovelace 8.9). A CUDA MPS server runs on them, and other users' jobs usually occupy most of their memory. |
| boot disk | 447 GB Dell BOSS-N1 NVMe |
| network | 1 Gbit/s Ethernet |
| use | editing, job submission (`sbatch`) and light work on the GPUs. `srun` is not available. |

## Storage

| mount | type | size | notes |
|---|---|---|---|
| `/work/log1` | NFS 4.2 from `log-raid-1:/volume/work` | 143 TB (16 TB used) | the group's shared volume, mounted on every node. It holds code, data, checkpoints and logs. |
| `/work/rleap1` | NFS 4.2 from `rleap-raid-1` | – | another group's volume |
| `/` on compute nodes | local ext4 | 27 GB | small. Node-local data such as `/var/tmp` is not visible from other nodes, so job output must go to `/work/log1`. |
| local disks on compute nodes | SSD / NVMe | 480 GB SATA SSD (cn-302, cn-308), 1 TB Samsung 870 SSD (cn-301), 3.2 TB Dell PM1745 NVMe (cn-312) | most of the disk is outside the root partition; whether it can be used as scratch space was not checked |
| swap | zram | about 1 TB per node | compressed swap in RAM |
| `$HOME` | read-only for jobs | | tools redirect their caches to `/work/log1/erik.mueller/...` (W&B, GRACE) |

## Network

- **Ethernet only:** 1 Gbit/s on cn-302, cn-308, cn-312 and desk-01. cn-301's active link negotiated only **100 Mbit/s** at the time of the probe.
- **No InfiniBand:** no `/sys/class/infiniband` devices were found on any probed node.
- **Consequence:** multi-node jobs that exchange a lot of data, such as LAMMPS spread across many nodes or multi-node training, would be limited by the network. The work so far has run on one GPU per job.

## Software

| | version |
|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-90-generic (all nodes) |
| NVIDIA driver / CUDA driver API | 580.126.09 / 13.0 |
| CUDA toolkits installed (`/usr/local`) | 10.1, 11.2, 11.7, 12, 12.6 |
| compiler | GCC 11.4.0 |
| environment modules | none; software comes from Python virtual environments |
| main environment (`code/.venv`) | Python 3.10.13, PyTorch 2.10.0 + CUDA 12.8, PyTorch Geometric 2.7.0 |
| GRACE environment (`grace-venv`) | TensorFlow 2.20.0, tensorpotential 0.6.2.1, potential GRACE-2L-UEA-OMAT-large |
| experiment tracking | Weights & Biases, project `yara-crystals` |

## Measured workload figures (from our runs)

| workload | hardware | figure |
|---|---|---|
| GRACE single point + relaxation step, 3,000 atoms | L40S | 9.4 GB GPU memory, 0.41 s per step |
| GRACE, 6,000 / 12,288 atoms | L40S | 18.5 / 35.9 GB, 0.78 / 1.6 s per step. Memory grows by about 3 GB per 1,000 atoms, so the limit is about 7k atoms on an A10, 14k on an L40S, 25k on an A100 or RTX PRO 6000. |
| Point-defect data generation (5 cells × 3,000 atoms per compound) | 1 GPU (A10 / L40S) | 25 min per compound (median), up to 8 h for compounds whose relaxations don't converge |
| Unified PaiNN prediction | GPU + CPU | about 15 ms model forward on GPU; about 2.4 s per structure end to end (graph construction on CPU dominates) |
| CGCNN / Graph Transformer energy training, 2000 epochs | 1 GPU, 2 runs in parallel | 2.0–2.8 h |

## Rest of the cluster (other groups' accounts; not usable by `log`)

| partition | account | nodes | hardware (from Slurm) |
|---|---|---|---|
| `cpu_modern` (default) | hlt | 59 | 96 threads, 186 GB RAM each |
| `gpu_11gb` | hlt | 54 | GTX 1080 (4 per node), RTX 2080 (3–8), Titan X Pascal; 230 GPUs in total |
| `gpu_24gb` | hlt | 6 | 5× (4× A10), 1× (3× RTX 3090) |
| `gpu_32gb` | hlt | 2 | 16× V100 each, 1.5 TB RAM |
| `gpu_48gb` | hlt | 4 | 4× L40S each |
| `gpu_test_24gb` | hlt | 1 | 1× RTX 3090, 1 h limit |
| `rleap_gpu_24gb` / `rleap_gpu_48gb` | rleap | 15 / 11 | 4× A10 / 4× L40S each |
| `rleap_cpu` / `rleap_cpu_modern` | rleap | 16 / 1 | 96 threads, 190 GB / 384 threads, 2.3 TB RAM |
| `rescale_amd` / `rescale_intel` | rescale_speed | 1 / 1 | 32 / 64 threads |

## Not determined

- the exact CPU and RAM configuration of the nodes that weren't probed (taken from Slurm †);
- memory type and speed, and RAID layout (needs root);
- whether the local disks can be used as job scratch space;
- the bandwidth of `/work/log1` (NFS over 1 GbE, so roughly ≤ 120 MB/s per node at best).
