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
  description = "CIDR blocks allowed to connect to Postgres."
  default     = []
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
