locals {
  key_env = {
    anthropic = "ANTHROPIC_API_KEY"
    gemini    = "GEMINI_API_KEY"
    zai       = "ZAI_API_KEY"
    deepseek  = "DEEPSEEK_API_KEY"
    groq      = "GROQ_API_KEY"
  }
}

resource "aws_ssm_parameter" "llm_key" {
  name  = "/${var.name}/llm_api_key"
  type  = "SecureString"
  value = var.llm_api_key
}

resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${var.name}"
  retention_in_days = 7
}

resource "aws_iam_role" "lambda" {
  name = "${var.name}-lambda"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "lambda" {
  role = aws_iam_role.lambda.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "${aws_cloudwatch_log_group.lambda.arn}:*"
      },
      {
        Effect   = "Allow"
        Action   = ["ssm:GetParameter"]
        Resource = aws_ssm_parameter.llm_key.arn
      }
    ]
  })
}

resource "aws_lambda_function" "api" {
  function_name                  = var.name
  role                           = aws_iam_role.lambda.arn
  runtime                        = "python3.13"
  architectures                  = ["arm64"] # cheaper per GB-second
  handler                        = "finops_agent.lambda_entry.handler"
  filename                       = "${path.module}/../lambda.zip"
  source_code_hash               = filebase64sha256("${path.module}/../lambda.zip")
  memory_size                    = 512
  timeout                        = 60
  reserved_concurrent_executions = var.max_concurrency

  environment {
    variables = {
      LLM_KEY_SSM_PARAM       = aws_ssm_parameter.llm_key.name
      LLM_KEY_ENV             = local.key_env[var.llm_provider]
      FINOPS_PROVIDER         = var.llm_provider
      FINOPS_MODEL            = var.model
      FINOPS_DAILY_BUDGET_USD = tostring(var.daily_budget_usd)
    }
  }

  depends_on = [aws_cloudwatch_log_group.lambda]
}

resource "aws_apigatewayv2_api" "http" {
  name          = var.name
  protocol_type = "HTTP"
  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST"]
    allow_headers = ["content-type"]
  }
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.http.id
  name        = "$default"
  auto_deploy = true
  default_route_settings {
    throttling_burst_limit = 5
    throttling_rate_limit  = 2
  }
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*"
}

# A FinOps project should practice FinOps: alert before the bill surprises you.
resource "aws_budgets_budget" "monthly" {
  name         = "${var.name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.aws_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  dynamic "notification" {
    for_each = [80, 100]
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value
      threshold_type             = "PERCENTAGE"
      notification_type          = "ACTUAL"
      subscriber_email_addresses = [var.alert_email]
    }
  }
}
