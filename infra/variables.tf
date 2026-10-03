variable "region" {
  type    = string
  default = "us-east-1"
}

variable "name" {
  type    = string
  default = "finops-agent"
}

variable "llm_provider" {
  type        = string
  default     = "gemini"
  description = "anthropic | gemini | zai | deepseek | groq. gemini/zai/groq have free tiers."
  validation {
    condition     = contains(["anthropic", "gemini", "zai", "deepseek", "groq"], var.llm_provider)
    error_message = "llm_provider must be one of anthropic, gemini, zai, deepseek, groq."
  }
}

variable "llm_api_key" {
  type        = string
  sensitive   = true
  description = "Provider API key. Stored as an SSM SecureString; read by the Lambda at cold start."
}

variable "model" {
  type        = string
  default     = ""
  description = "Optional model id override; empty uses the provider default."
}

variable "daily_budget_usd" {
  type        = number
  default     = 3
  description = "Hard daily cap on Claude spend enforced by the API itself."
}

variable "aws_budget_usd" {
  type        = number
  default     = 5
  description = "Monthly AWS budget; an alert is emailed at 80% and 100%."
}

variable "alert_email" {
  type        = string
  description = "Where AWS Budgets sends alerts."
}

variable "max_concurrency" {
  type        = number
  default     = 2
  description = "Reserved concurrency: caps parallel executions (and therefore cost)."
}
