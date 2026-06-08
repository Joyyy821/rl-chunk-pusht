# Nautilus workflow

These files run `rl-chunk-pusht` on Nautilus with the repo and outputs stored on a PVC. Use the dev pod for setup and smoke tests, then delete it before starting a training Job because the default PVC uses `ReadWriteOnce`.

## Files

- `manifests/pvc.yaml`: 50Gi `rook-ceph-block` PVC.
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

Each pod also has a root `initContainer` that creates:

```text
/workspace/src
/workspace/outputs
/workspace/cache
/workspace/home
/workspace/tmp
```

and changes ownership to `1000:100`. This avoids the common failure mode where cloning into the PVC fails because the mounted directory is owned by root.

## First setup

Run these commands from the repo root on your local machine:

```bash
kubectl apply -f nautilus/manifests/pvc.yaml
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

The bootstrap script waits for `pod/yjiao-rl-chunk-pusht-dev`, checks that `/workspace` is writable, clones or updates:

```text
/workspace/src/rl-chunk-pusht
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
touch /workspace/write-test
```

Check JAX GPU:

```bash
cd /workspace/src/rl-chunk-pusht
env -u LD_LIBRARY_PATH HWLOC_HIDE_ERRORS=2 uv run python -c "import jax; print(jax.devices()); print(jax.default_backend())"
```

Run a short smoke train:

```bash
cd /workspace/src/rl-chunk-pusht
bash nautilus/scripts/smoke_train.sh
```

Smoke outputs go under:

```text
/workspace/outputs/smoke
```

## Run training Jobs

Delete the dev pod before starting a Job so the `ReadWriteOnce` PVC can attach cleanly:

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
/workspace/outputs/train
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

Keep `OUTPUT_DIR` on `/workspace` so logs and checkpoints persist after the pod exits.

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
