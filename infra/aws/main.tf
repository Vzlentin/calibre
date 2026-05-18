provider "aws" {
  region = var.region
}

locals {
  container_image = "${aws_ecr_repository.calibre.repository_url}:${var.image_tag}"
  database_url = format(
    "postgresql+psycopg://%s:%s@%s:%s/%s",
    var.db_username,
    var.db_password,
    aws_db_instance.postgres.address,
    aws_db_instance.postgres.port,
    aws_db_instance.postgres.db_name,
  )
}

resource "aws_s3_bucket" "artifacts" {
  bucket_prefix = "${var.name_prefix}-artifacts-"
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_ecr_repository" "calibre" {
  name                 = var.name_prefix
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_security_group" "postgres" {
  name_prefix = "${var.name_prefix}-postgres-"
  description = "Postgres access for Calibre container compute"
  vpc_id      = var.vpc_id

  dynamic "ingress" {
    for_each = length(var.container_cidr_blocks) > 0 ? [1] : []
    content {
      description = "Postgres from configured CIDRs"
      from_port   = 5432
      to_port     = 5432
      protocol    = "tcp"
      cidr_blocks = var.container_cidr_blocks
    }
  }

  ingress {
    description     = "Postgres from Calibre API tasks"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.calibre_service.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "calibre_service" {
  name_prefix = "${var.name_prefix}-svc-"
  description = "Calibre API service"
  vpc_id      = var.vpc_id

  dynamic "ingress" {
    for_each = length(var.api_ingress_cidr_blocks) > 0 ? [1] : []
    content {
      description = "Calibre API"
      from_port   = var.container_port
      to_port     = var.container_port
      protocol    = "tcp"
      cidr_blocks = var.api_ingress_cidr_blocks
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_instance" "postgres" {
  identifier             = var.name_prefix
  engine                 = "postgres"
  engine_version         = "16"
  instance_class         = "db.t4g.micro"
  allocated_storage      = 20
  storage_encrypted      = true
  db_name                = "calibre"
  username               = var.db_username
  password               = var.db_password
  db_subnet_group_name   = var.db_subnet_group_name
  vpc_security_group_ids = [aws_security_group.postgres.id]
  skip_final_snapshot    = true
}

resource "aws_secretsmanager_secret" "database_url" {
  name_prefix = "${var.name_prefix}-database-url-"
}

resource "aws_secretsmanager_secret_version" "database_url" {
  secret_id     = aws_secretsmanager_secret.database_url.id
  secret_string = local.database_url
}

resource "aws_cloudwatch_log_group" "calibre" {
  name              = "/ecs/${var.name_prefix}"
  retention_in_days = var.log_retention_days
}

data "aws_iam_policy_document" "ecs_tasks_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_execution" {
  name_prefix        = "${var.name_prefix}-exec-"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume_role.json
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_execution_secrets" {
  name = "${var.name_prefix}-execution-secrets"
  role = aws_iam_role.ecs_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = aws_secretsmanager_secret.database_url.arn
      },
    ]
  })
}

resource "aws_iam_role" "calibre_task" {
  name_prefix        = "${var.name_prefix}-task-"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume_role.json
}

resource "aws_iam_role_policy" "calibre_artifacts" {
  name = "${var.name_prefix}-artifacts"
  role = aws_iam_role.calibre_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
        ]
        Resource = aws_s3_bucket.artifacts.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
        ]
        Resource = "${aws_s3_bucket.artifacts.arn}/*"
      },
    ]
  })
}

resource "aws_ecs_cluster" "calibre" {
  name = var.name_prefix
}

resource "aws_ecs_task_definition" "calibre_api" {
  family                   = "${var.name_prefix}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.container_cpu
  memory                   = var.container_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.calibre_task.arn

  container_definitions = jsonencode([
    {
      name       = "calibre-api"
      image      = local.container_image
      essential  = true
      entryPoint = ["uvicorn"]
      command = [
        "calibre.api.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        tostring(var.container_port),
      ]
      portMappings = [
        {
          containerPort = var.container_port
          hostPort      = var.container_port
          protocol      = "tcp"
        },
      ]
      environment = [
        {
          name  = "AWS_REGION"
          value = var.region
        },
        {
          name  = "CALIBRE_ARTIFACT_BUCKET"
          value = aws_s3_bucket.artifacts.bucket
        },
      ]
      secrets = [
        {
          name      = "CALIBRE_DATABASE_URL"
          valueFrom = aws_secretsmanager_secret.database_url.arn
        },
      ]
      healthCheck = {
        command     = ["CMD-SHELL", "calibre health >/dev/null || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 60
      }
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.calibre.name
          awslogs-region        = var.region
          awslogs-stream-prefix = "api"
        }
      }
    },
  ])
}

resource "aws_ecs_service" "calibre_api" {
  name            = "${var.name_prefix}-api"
  cluster         = aws_ecs_cluster.calibre.id
  task_definition = aws_ecs_task_definition.calibre_api.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.container_subnet_ids
    security_groups  = [aws_security_group.calibre_service.id]
    assign_public_ip = var.assign_public_ip
  }
}
