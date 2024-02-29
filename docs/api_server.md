# API Server

## Manage API server

Start api server locally:

```
llm-serve start apiserver
```

Stop api server locally:

```
llm-serve stop apiserver
```

## API list

### Start models serving

*Start Serving*

```http
POST /api/start_serving HTTP/1.1

Content-Type: application/json
Accept: application/json
user-name: string
```

> Body parameter

```json
[
  {
    "model_id": "string",
    "model_task": "string",
    "model_revision": "string",
    "is_oob": true,
    "scaling_config": {
      "num_workers": 0,
      "num_gpus_per_worker": 1,
      "num_cpus_per_worker": 1
    }
  }
]
```

#### Parameters

| Name | In | Type | Required | Description |
|:---|:---|:---|:---|:---|
| body | body | [Body_start_serving_start_serving_post](#schemabody_start_serving_start_serving_post) | true | none |

#### Responses

| Status | Meaning | Description | Schema |
|:---|:---|:---|:---|
| 200 | [OK](https://tools.ietf.org/html/rfc7231#section-6.3.1) | Successful Response | Inline |
| 422 | [Unprocessable Entity](https://tools.ietf.org/html/rfc2518#section-10.3) | Validation Error |[HTTPValidationError](#schemahttpvalidationerror)|

### List serving status and predict URL

*Serving URL*

```http
GET /api/list_serving HTTP/1.1

Content-Type: application/json
Accept: application/json
user-name: string
```

> Body parameter

```json
[
  {
    "model_id": "string",
    "model_revision": "main"
  }
]
```

#### Parameters

| Name | In | Type | Required | Description |
|:---|:---|:---|:---|:---|
| body | body | [Body_serving_url_list_serving_get](#schemabody_serving_url_list_serving_get) | true | none |

#### Responses

| Status | Meaning | Description | Schema |
|:---|:---|:---|:---|
| 200 | [OK](https://tools.ietf.org/html/rfc7231#section-6.3.1) | Successful Response | Inline |
| 422 | [Unprocessable Entity](https://tools.ietf.org/html/rfc2518#section-10.3) | Validation Error | [HTTPValidationError](#schemahttpvalidationerror) |

### Stop models serving

*Delete Serving*

```http
POST /api/stop_serving HTTP/1.1

Content-Type: application/json
Accept: application/json
user-name: string
```

> Body parameter

```json
[
  {
    "model_id": "string",
    "model_revision": "main"
  }
]
```

#### Parameters

| Name | In | Type | Required | Description |
|:---|:---|:---|:---|:---|
| body | body | [Body_delete_serving_delete_serving_post](#schemabody_delete_serving_delete_serving_post) | true | none |

#### Responses

| Status | Meaning | Description | Schema |
|:---|:---|:---|:---|
| 200 | [OK](https://tools.ietf.org/html/rfc7231#section-6.3.1) | Successful Response | Inline |
| 422 | [Unprocessable Entity](https://tools.ietf.org/html/rfc2518#section-10.3) | Validation Error | [HTTPValidationError](#schemahttpvalidationerror) |

## Detailed API Documents

See more [API documents](../llmserve/backend/docs/api.md) and [OpenAPI Specification](../llmserve/backend/docs/openapi.json).
