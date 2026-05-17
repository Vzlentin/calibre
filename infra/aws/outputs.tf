output "artifact_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.calibre.repository_url
}

output "postgres_endpoint" {
  value = aws_db_instance.postgres.endpoint
}
