variable "name_prefix" {
  type        = string
  description = "Name prefix for Calibre MVP resources."
  default     = "calibre-mvp"
}

variable "region" {
  type        = string
  description = "AWS region."
  default     = "eu-west-1"
}

variable "vpc_id" {
  type        = string
  description = "VPC id for the database security group."
}

variable "db_subnet_group_name" {
  type        = string
  description = "Existing DB subnet group for RDS Postgres."
}

variable "container_cidr_blocks" {
  type        = list(string)
  description = "Additional CIDR blocks allowed to connect to Postgres."
  default     = []
}

variable "container_subnet_ids" {
  type        = list(string)
  description = "Subnets where ECS Fargate tasks run."
}

variable "api_ingress_cidr_blocks" {
  type        = list(string)
  description = "CIDR blocks allowed to call the Calibre API container port."
  default     = []
}

variable "assign_public_ip" {
  type        = bool
  description = "Whether ECS tasks receive public IPs."
  default     = true
}

variable "desired_count" {
  type        = number
  description = "Number of Calibre API tasks to run."
  default     = 1
}

variable "image_tag" {
  type        = string
  description = "Container image tag in the generated ECR repository."
  default     = "latest"
}

variable "container_cpu" {
  type        = number
  description = "Fargate task CPU units."
  default     = 1024
}

variable "container_memory" {
  type        = number
  description = "Fargate task memory in MiB."
  default     = 2048
}

variable "container_port" {
  type        = number
  description = "Calibre API container port."
  default     = 8000
}

variable "log_retention_days" {
  type        = number
  description = "CloudWatch log retention for Calibre API tasks."
  default     = 14
}

variable "db_username" {
  type        = string
  description = "Postgres admin username."
  default     = "calibre"
}

variable "db_password" {
  type        = string
  description = "Postgres admin password."
  sensitive   = true
}
