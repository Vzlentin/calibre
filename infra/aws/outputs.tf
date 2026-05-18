output "artifact_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.calibre.repository_url
}

output "postgres_endpoint" {
  value = aws_db_instance.postgres.endpoint
}

output "database_url_secret_arn" {
  value     = aws_secretsmanager_secret.database_url.arn
  sensitive = true
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.calibre.name
}

output "ecs_service_name" {
  value = aws_ecs_service.calibre_api.name
}

output "task_execution_role_arn" {
  value = aws_iam_role.ecs_execution.arn
}

output "task_role_arn" {
  value = aws_iam_role.calibre_task.arn
}

output "api_security_group_id" {
  value = aws_security_group.calibre_service.id
}
