# Calibre Deployment

Calibre runs as a stateless container. Inputs and outputs should be addressed by
URI (`s3://`, `gs://`, `abfs://`, or local paths) and persistent run state should
live outside the container.

## Container

Build:

```bash
docker build -t calibre:full .
docker build -f Dockerfile.slim -t calibre:slim .
```

Smoke:

```bash
docker run --rm calibre:full health
docker run --rm calibre:slim run --config /app/benchmarks/vn2/config/smoke.yaml
```

CI publishes same-repository PR and main-branch images to GHCR as
`ghcr.io/<owner>/<repo>:pr-<number>-full`, `:pr-<number>-slim`, and short-SHA
`:<sha>-full` / `:<sha>-slim` tags after both image smoke tests pass.

## Kubernetes Job

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: calibre-backtest
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: calibre
          image: calibre:full
          args: ["run", "--config", "s3://bucket/configs/winning.yaml"]
          env:
            - name: AWS_REGION
              value: eu-west-1
```

No PVC is required when configs, inputs, and outputs use object-store URIs.

## AWS Batch

Use the same image with command:

```bash
calibre run --config s3://bucket/configs/winning.yaml
```

Grant the task role read access to input/config prefixes and write access to
artifact prefixes.

The starter AWS Terraform root module is under `infra/aws/`. It provisions an
artifact bucket, ECR repository, small RDS Postgres instance, ECS Fargate API
service, CloudWatch logs, a database URL secret, and task IAM for artifact IO:

```bash
terraform -chdir=infra/aws init
terraform -chdir=infra/aws apply
```

Provide `container_subnet_ids`, `vpc_id`, `db_subnet_group_name`, and
`db_password` in a `.tfvars` file. Set `api_ingress_cidr_blocks` to the operator
or load-balancer CIDRs that may call the API; leave it empty to deploy the
service without public API ingress.

For initial bootstrap, apply once with `desired_count=0` to create ECR without
starting tasks, push the image, then apply again with the desired task count:

```bash
terraform -chdir=infra/aws apply -var desired_count=0
ECR_REPO="$(terraform -chdir=infra/aws output -raw ecr_repository_url)"
ECR_REGISTRY="${ECR_REPO%/*}"
aws ecr get-login-password --region eu-west-1 \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"
docker tag calibre:full "$ECR_REPO:latest"
docker push "$ECR_REPO:latest"
terraform -chdir=infra/aws apply -var desired_count=1
```

The ECS task gets `CALIBRE_DATABASE_URL` from Secrets Manager. When set,
`POST /backtests` persists rows in `runs`, conformal snapshots in
`conformal_state`, and output artifact pointers in `forecast_pointers`. Run the
Alembic migration under `calibre/storage/migrations/` before starting the API.

## Azure Container Instances

Use command override:

```bash
calibre run --config abfs://container/configs/winning.yaml
```

Provide storage credentials via managed identity or environment variables
supported by `adlfs`.

## Databricks

Calibre runs on Databricks via the Spark execution backend. The workflow is:

1. **Build and upload the wheel** from your local checkout:

   ```bash
   uv build --wheel
   dbfs cp dist/calibre-0.1.0-py3-none-any.whl dbfs:/mnt/calibre/
   ```

2. **Upload benchmark files** (configs + fixture data) to DBFS:

   ```bash
   dbfs cp -r benchmarks/vn2/config dbfs:/mnt/calibre/configs/
   dbfs cp -r benchmarks/vn2/fixture dbfs:/mnt/calibre/vn2-fixture/
   dbfs cp scripts/databricks_smoke.py dbfs:/mnt/calibre/scripts/
   ```

3. **Run the smoke test** via a notebook or job:

   *Notebook:* import `scripts/databricks_notebook.py` into a Databricks notebook,
   adjust `REPO_ROOT` to your Repos path, and run all cells.

   *Job:* create a job from `scripts/databricks_job.json` via the Databricks UI or API:

   ```bash
   databricks jobs create --json @scripts/databricks_job.json
   ```

4. **MLflow** — The repo `.env` hardcodes a home MLflow server. On Databricks,
   either unset `MLFLOW_TRACKING_URI` to use Databricks-managed MLflow, or set
   `CALIBRE_NO_MLFLOW=1` to disable tracking entirely:

   ```python
   import os
   os.environ.pop("MLFLOW_TRACKING_URI", None)   # use Databricks MLflow
   # os.environ["CALIBRE_NO_MLFLOW"] = "1"       # disable entirely
   ```

5. **Run the winning benchmark** after the smoke test passes:

   ```python
   from benchmarks.vn2.run_winning import run_winning
   summary = run_winning(
       data_dir="/dbfs/mnt/calibre/vn2-fixture",
       results_dir="/dbfs/mnt/calibre/results",
       verbose=True,
   )
   display(summary)
   ```
