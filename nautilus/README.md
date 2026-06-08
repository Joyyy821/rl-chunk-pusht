# Nautilus workflow

Note that this readme file is more of a personal note for my job deployment on Nautilus. 
Any other Nautilus user should change the pod/job/pvc naming and configuration before running `kubectl apply`.

These files run `rl-chunk-pusht` on Nautilus with the repo and outputs stored on a long-lived personal PVC. The default storage name follows the shared namespace convention:

```text
yjiao-west-vol
```

The project lives under `/workspace/rl-chunk-pusht` inside that personal volume, so this workflow does not claim the whole PVC root for this repo.

To use central storage instead, change `yjiao-west-vol` to `yjiao-central-vol` in the manifests and use `rook-cephfs-central` in `manifests/pvc.yaml`.

## Files

- `manifests/pvc.yaml`: 2Ti RWX `rook-cephfs` personal PVC named `yjiao-west-vol`.
- `manifests/dev-pod.yaml`: interactive one-GPU pod mounted at `/workspace`.
- `manifests/train-job-position-c8.yaml`: finite Job for `position_target`, `acfql`, chunk size 8.
- `manifests/train-job-velocity-c8.yaml`: finite Job for `velocity_kinematic`, `acfql`, chunk size 8.
- `scripts/bootstrap_dev_pod.sh`: clone or update the repo on the PVC and run setup.
- `scripts/setup_env.sh`: install `uv`, sync CUDA JAX dependencies, and verify imports.
- `scripts/smoke_train.sh`: short training run for checking the pod.
- `scripts/train.sh`: Job entrypoint.

## PVC permissions

The manifests run the main container as UID `1000` and GID `100`, matching common Nautilus/Jupyter images. They also set:

```yaml
securityContext:
  runAsUser: 1000
  runAsGroup: 100
  fsGroup: 100
  fsGroupChangePolicy: OnRootMismatch
```

Each pod also has a root `initContainer` that creates project-local directories:

```text
/workspace/rl-chunk-pusht/src
/workspace/rl-chunk-pusht/outputs
/workspace/rl-chunk-pusht/cache
/workspace/rl-chunk-pusht/home
/workspace/rl-chunk-pusht/tmp
```

and changes ownership to `1000:100`. It only touches `/workspace/rl-chunk-pusht`, not the whole personal PVC. This avoids the common failure mode where cloning into the PVC fails because the mounted directory is owned by root.

## First setup

From the repo root on your local machine, first check whether the personal PVC already exists:

```bash
kubectl get pvc yjiao-west-vol
```

If it does not exist, create it:

```bash
kubectl apply -f nautilus/manifests/pvc.yaml
```

Then create the dev pod:

```bash
kubectl apply -f nautilus/manifests/dev-pod.yaml
```

If you use a non-default namespace, set it once before the helper scripts:

```bash
export NAMESPACE=<your-namespace>
```

Bootstrap the dev pod:

```bash
bash nautilus/scripts/bootstrap_dev_pod.sh
```

The bootstrap script waits for `pod/yjiao-rl-chunk-pusht-dev`, checks that `/workspace/rl-chunk-pusht` is writable, clones or updates:

```text
/workspace/rl-chunk-pusht/src/rl-chunk-pusht
```

and runs:

```bash
uv sync --group dev --extra cuda12
```

## Manual checks inside the dev pod

Attach to the pod:

```bash
kubectl exec -it yjiao-rl-chunk-pusht-dev -- bash
```

Check identity and PVC writes:

```bash
id
touch /workspace/rl-chunk-pusht/write-test
```

If `uv` is not found in an already-running dev pod, export the PVC-backed install path in that shell:

```bash
export PROJECT_DIR=/workspace/rl-chunk-pusht
export PATH="$PROJECT_DIR/home/.local/bin:$PROJECT_DIR/src/rl-chunk-pusht/.venv/bin:$PATH"
```

Check JAX GPU:

```bash
cd /workspace/rl-chunk-pusht/src/rl-chunk-pusht
env -u LD_LIBRARY_PATH HWLOC_HIDE_ERRORS=2 uv run python -c "import jax; print(jax.devices()); print(jax.default_backend())"
```

Run a short smoke train:

```bash
cd /workspace/rl-chunk-pusht/src/rl-chunk-pusht
bash nautilus/scripts/smoke_train.sh
```

Smoke outputs go under:

```text
/workspace/rl-chunk-pusht/outputs/smoke
```

## Run training Jobs

The personal PVC is RWX, so it can be mounted by multiple pods. Still delete the dev pod before starting a serious training Job if you want to free the dev pod GPU:

```bash
kubectl delete pod yjiao-rl-chunk-pusht-dev
```

Start the position-target chunking run:

```bash
kubectl apply -f nautilus/manifests/train-job-position-c8.yaml
kubectl logs -f job/yjiao-rl-chunk-pusht-position-c8
```

Or start the velocity-command chunking run:

```bash
kubectl apply -f nautilus/manifests/train-job-velocity-c8.yaml
kubectl logs -f job/yjiao-rl-chunk-pusht-velocity-c8
```

Training outputs go under:

```text
/workspace/rl-chunk-pusht/outputs/train
```

To rerun a Job with the same name, delete it first:

```bash
kubectl delete job yjiao-rl-chunk-pusht-position-c8
```

## Tuning common settings

The Job manifests set environment variables consumed by `scripts/train.sh`. Edit these fields in the YAML for normal runs:

- `ENV_MODE`: `position_target` or `velocity_kinematic`.
- `SEED`: training seed.
- `ONLINE_STEPS`: default `1000000`.
- `WARMUP_STEPS`: default `5000`.
- `BATCH_SIZE`: default `256`.
- `WANDB_MODE`: default `disabled`.
- `EXP_NAME`: run name stored in the output directory.

Keep `OUTPUT_DIR` under `/workspace/rl-chunk-pusht` so logs and checkpoints persist after the pod exits without mixing this repo's artifacts with other projects on the same personal PVC.

## Local validation

Before applying changes:

```bash
bash -n nautilus/scripts/*.sh
kubectl apply --dry-run=client -f nautilus/manifests/pvc.yaml
kubectl apply --dry-run=client -f nautilus/manifests/dev-pod.yaml
kubectl apply --dry-run=client -f nautilus/manifests/train-job-position-c8.yaml
kubectl apply --dry-run=client -f nautilus/manifests/train-job-velocity-c8.yaml
```

## Image notes

The manifests use:

```text
gitlab-registry.nrp-nautilus.io/nrp/scientific-images/python:latest
```

This is intended as a public starting image for early Nautilus testing. If it does not include enough system libraries for Push-T rendering or OpenCV, switch the `image` fields to another public image that includes Python, git, CUDA-compatible runtime libraries, and common OpenGL libraries. The scripts install Python packages with `uv`; they only install apt packages when the container is running as root.
