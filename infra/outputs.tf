output "api_url" {
  value       = aws_apigatewayv2_api.http.api_endpoint
  description = "POST {api_url}/ask  {\"question\": \"...\"}"
}
