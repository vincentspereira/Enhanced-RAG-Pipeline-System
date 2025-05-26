# API Integration Hub

## Overview

The API Integration Hub provides a unified gateway for interacting with multiple external APIs through a single interface. It includes comprehensive features for authentication, rate limiting, usage analytics, and webhooks.

## Key Features

- **OAuth2 Authentication**: Integrated support for Google and GitHub OAuth2 providers
- **Rate Limiting**: Configurable rate limiting to prevent abuse
- **Usage Analytics**: Detailed analytics for API usage with visualization capabilities
- **Webhooks**: Event-based notifications for service changes and errors
- **Batch Requests**: Support for making multiple API requests in a single call
- **Service Health Monitoring**: Health checks and performance metrics for external APIs

## Endpoints

### Services

- `GET /api/v1/api-hub/services` - List available API services
- `GET /api/v1/api-hub/services/{service_id}` - Get API service details
- `POST /api/v1/api-hub/services` - Register a new API service
- `PUT /api/v1/api-hub/services/{service_id}` - Update API service
- `DELETE /api/v1/api-hub/services/{service_id}` - Delete API service
- `POST /api/v1/api-hub/services/{service_id}/health-check` - Check API service health
- `GET /api/v1/api-hub/services/{service_id}/metrics` - Get API service metrics

### API Requests

- `POST /api/v1/api-hub/request` - Make an API request
- `POST /api/v1/api-hub/batch` - Make multiple API requests in a batch

### OAuth2 Authentication

- `POST /api/v1/api-hub/oauth2/authorize` - Get OAuth2 authorization URL
- `POST /api/v1/api-hub/oauth2/token` - Exchange OAuth2 code for token
- `GET /api/v1/api-hub/oauth2/providers` - List OAuth2 providers

### Rate Limiting

- `GET /api/v1/api-hub/rate-limits` - Get rate limit status

### Webhooks

- `GET /api/v1/api-hub/webhooks` - List webhooks
- `POST /api/v1/api-hub/webhooks` - Register a webhook
- `GET /api/v1/api-hub/webhooks/{webhook_id}` - Get webhook details
- `PUT /api/v1/api-hub/webhooks/{webhook_id}` - Update webhook
- `DELETE /api/v1/api-hub/webhooks/{webhook_id}` - Delete webhook
- `POST /api/v1/api-hub/webhooks/test` - Test a webhook

### Analytics

- `GET /api/v1/api-hub/analytics/requests` - Get API request analytics
- `GET /api/v1/api-hub/analytics/rate-limits` - Get rate limit analytics
- `GET /api/v1/api-hub/analytics/oauth` - Get OAuth analytics
- `GET /api/v1/api-hub/analytics/dashboard` - Get analytics dashboard data
- `GET /api/v1/api-hub/analytics/visualizations` - Generate and retrieve visualization charts

## Usage Examples

### Registering an API Service

```json
POST /api/v1/api-hub/services

{
  "id": "github-api",
  "name": "GitHub API",
  "description": "GitHub REST API for accessing repositories, issues, and more",
  "base_url": "https://api.github.com",
  "auth_type": "oauth2",
  "oauth2_provider": "github",
  "rate_limits": [
    {
      "name": "authenticated",
      "limit": 5000,
      "window": 3600,
      "description": "5000 requests per hour for authenticated users"
    },
    {
      "name": "unauthenticated",
      "limit": 60,
      "window": 3600,
      "description": "60 requests per hour for unauthenticated users"
    }
  ],
  "endpoints": [
    {
      "path": "/user",
      "method": "GET",
      "description": "Get authenticated user information",
      "requires_auth": true
    },
    {
      "path": "/repos/{owner}/{repo}",
      "method": "GET",
      "description": "Get repository information",
      "requires_auth": false,
      "parameters": [
        {"name": "owner", "in": "path", "required": true},
        {"name": "repo", "in": "path", "required": true}
      ]
    }
  ]
}
```

### Making an API Request

```json
POST /api/v1/api-hub/request

{
  "service_id": "github-api",
  "endpoint": "/repos/{owner}/{repo}",
  "method": "GET",
  "params": {
    "owner": "microsoft",
    "repo": "vscode"
  }
}
```

### OAuth2 Authorization

```json
POST /api/v1/api-hub/oauth2/authorize

{
  "provider": "github",
  "redirect_uri": "https://example.com/oauth2/callback",
  "scope": "user repo",
  "state": "random-state-string"
}
```

### Setting Up a Webhook

```json
POST /api/v1/api-hub/webhooks

{
  "url": "https://example.com/webhook",
  "secret": "your-webhook-secret",
  "events": ["service.created", "service.updated", "request.error"],
  "description": "Notify my service about API hub events",
  "active": true
}
```

## Webhook Events

The following events can trigger webhook notifications:

- `service.created` - When a new API service is registered
- `service.updated` - When an API service is updated
- `service.deleted` - When an API service is deleted
- `webhook.created` - When a new webhook is registered
- `webhook.updated` - When a webhook is updated
- `webhook.deleted` - When a webhook is deleted
- `request.error` - When an API request results in an error

## Security Considerations

- OAuth2 tokens are securely handled and stored
- Webhook payloads can be signed using HMAC-SHA256 for verification
- Rate limiting protects against abuse
- API keys are securely stored and never exposed in responses

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200 OK` - Request successful
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

Error responses include a detailed error message to help diagnose issues.
