# Calibre Deployment

Calibre runs as a stateless container. Inputs and outputs should be addressed by
URI (`s3://`, `gs://`, `abfs://`, or local paths) and persistent run state should
live outside the container.

## Container

Build:

```bash
docker build -t calibre:full .
```

Smoke:

```bash
docker run --rm calibre:full health
```

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
artifact bucket, ECR repository, and small RDS Postgres instance:

```bash
terraform -chdir=infra/aws init
terraform -chdir=infra/aws apply
```

## Azure Container Instances

Use command override:

```bash
calibre run --config abfs://container/configs/winning.yaml
```

Provide storage credentials via managed identity or environment variables
supported by `adlfs`.
