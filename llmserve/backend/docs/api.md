---
title: FastAPI v0.1.0
language_tabs:
  - shell: Shell
  - http: HTTP
  - javascript: JavaScript
  - ruby: Ruby
  - python: Python
  - php: PHP
  - java: Java
  - go: Go
toc_footers: []
includes: []
search: true
highlight_theme: darkula
headingLevel: 2

---

<!-- Generator: Widdershins v4.0.1 -->

<h1 id="fastapi">FastAPI v0.1.0</h1>

> Scroll down for code samples, example requests and responses. Select a language for code samples from the tabs above or the mobile navigation menu.

Base URLs:

* <a href="/api">/api</a>

<h1 id="fastapi-default">Default</h1>

## generate_text_stream_stream_post

<a id="opIdgenerate_text_stream_stream_post"></a>

> Code samples

```shell
# You can also use wget
curl -X POST /api/stream \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json'

```

```http
POST /api/stream HTTP/1.1

Content-Type: application/json
Accept: application/json

```

```javascript
const inputBody = '{
  "prompt": null,
  "use_prompt_format": true
}';
const headers = {
  'Content-Type':'application/json',
  'Accept':'application/json'
};

fetch('/api/stream',
{
  method: 'POST',
  body: inputBody,
  headers: headers
})
.then(function(res) {
    return res.json();
}).then(function(body) {
    console.log(body);
});

```

```ruby
require 'rest-client'
require 'json'

headers = {
  'Content-Type' => 'application/json',
  'Accept' => 'application/json'
}

result = RestClient.post '/api/stream',
  params: {
  'self' => 'any'
}, headers: headers

p JSON.parse(result)

```

```python
import requests
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}

r = requests.post('/api/stream', params={
  'self': null
}, headers = headers)

print(r.json())

```

```php
<?php

require 'vendor/autoload.php';

$headers = array(
    'Content-Type' => 'application/json',
    'Accept' => 'application/json',
);

$client = new \GuzzleHttp\Client();

// Define array of request body.
$request_body = array();

try {
    $response = $client->request('POST','/api/stream', array(
        'headers' => $headers,
        'json' => $request_body,
       )
    );
    print_r($response->getBody()->getContents());
 }
 catch (\GuzzleHttp\Exception\BadResponseException $e) {
    // handle exception or api errors.
    print_r($e->getMessage());
 }

 // ...

```

```java
URL obj = new URL("/api/stream");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("POST");
int responseCode = con.getResponseCode();
BufferedReader in = new BufferedReader(
    new InputStreamReader(con.getInputStream()));
String inputLine;
StringBuffer response = new StringBuffer();
while ((inputLine = in.readLine()) != null) {
    response.append(inputLine);
}
in.close();
System.out.println(response.toString());

```

```go
package main

import (
       "bytes"
       "net/http"
)

func main() {

    headers := map[string][]string{
        "Content-Type": []string{"application/json"},
        "Accept": []string{"application/json"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("POST", "/api/stream", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

`POST /stream`

*Generate Text Stream*

> Body parameter

```json
{
  "prompt": null,
  "use_prompt_format": true
}
```

<h3 id="generate_text_stream_stream_post-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|self|query|any|true|none|
|body|body|[Prompt](#schemaprompt)|true|none|

> Example responses

> 200 Response

```json
null
```

<h3 id="generate_text_stream_stream_post-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|Inline|
|422|[Unprocessable Entity](https://tools.ietf.org/html/rfc2518#section-10.3)|Validation Error|[HTTPValidationError](#schemahttpvalidationerror)|

<h3 id="generate_text_stream_stream_post-responseschema">Response Schema</h3>

<aside class="success">
This operation does not require authentication
</aside>

## start_experimental_start_experimental_post

<a id="opIdstart_experimental_start_experimental_post"></a>

> Code samples

```shell
# You can also use wget
curl -X POST /api/start_experimental \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json'

```

```http
POST /api/start_experimental HTTP/1.1

Content-Type: application/json
Accept: application/json

```

```javascript
const inputBody = '{
  "models": {
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
}';
const headers = {
  'Content-Type':'application/json',
  'Accept':'application/json'
};

fetch('/api/start_experimental',
{
  method: 'POST',
  body: inputBody,
  headers: headers
})
.then(function(res) {
    return res.json();
}).then(function(body) {
    console.log(body);
});

```

```ruby
require 'rest-client'
require 'json'

headers = {
  'Content-Type' => 'application/json',
  'Accept' => 'application/json'
}

result = RestClient.post '/api/start_experimental',
  params: {
  }, headers: headers

p JSON.parse(result)

```

```python
import requests
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}

r = requests.post('/api/start_experimental', headers = headers)

print(r.json())

```

```php
<?php

require 'vendor/autoload.php';

$headers = array(
    'Content-Type' => 'application/json',
    'Accept' => 'application/json',
);

$client = new \GuzzleHttp\Client();

// Define array of request body.
$request_body = array();

try {
    $response = $client->request('POST','/api/start_experimental', array(
        'headers' => $headers,
        'json' => $request_body,
       )
    );
    print_r($response->getBody()->getContents());
 }
 catch (\GuzzleHttp\Exception\BadResponseException $e) {
    // handle exception or api errors.
    print_r($e->getMessage());
 }

 // ...

```

```java
URL obj = new URL("/api/start_experimental");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("POST");
int responseCode = con.getResponseCode();
BufferedReader in = new BufferedReader(
    new InputStreamReader(con.getInputStream()));
String inputLine;
StringBuffer response = new StringBuffer();
while ((inputLine = in.readLine()) != null) {
    response.append(inputLine);
}
in.close();
System.out.println(response.toString());

```

```go
package main

import (
       "bytes"
       "net/http"
)

func main() {

    headers := map[string][]string{
        "Content-Type": []string{"application/json"},
        "Accept": []string{"application/json"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("POST", "/api/start_experimental", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

`POST /start_experimental`

*Start Experimental*

> Body parameter

```json
{
  "models": {
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
}
```

<h3 id="start_experimental_start_experimental_post-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|body|body|[Body_start_experimental_start_experimental_post](#schemabody_start_experimental_start_experimental_post)|true|none|

> Example responses

> 200 Response

```json
{}
```

<h3 id="start_experimental_start_experimental_post-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|Inline|
|422|[Unprocessable Entity](https://tools.ietf.org/html/rfc2518#section-10.3)|Validation Error|[HTTPValidationError](#schemahttpvalidationerror)|

<h3 id="start_experimental_start_experimental_post-responseschema">Response Schema</h3>

Status Code **200**

*Response Start Experimental Start Experimental Post*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|

<aside class="success">
This operation does not require authentication
</aside>

## start_serving_start_serving_post

<a id="opIdstart_serving_start_serving_post"></a>

> Code samples

```shell
# You can also use wget
curl -X POST /api/start_serving \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -H 'user-name: string'

```

```http
POST /api/start_serving HTTP/1.1

Content-Type: application/json
Accept: application/json
user-name: string

```

```javascript
const inputBody = '[
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
]';
const headers = {
  'Content-Type':'application/json',
  'Accept':'application/json',
  'user-name':'string'
};

fetch('/api/start_serving',
{
  method: 'POST',
  body: inputBody,
  headers: headers
})
.then(function(res) {
    return res.json();
}).then(function(body) {
    console.log(body);
});

```

```ruby
require 'rest-client'
require 'json'

headers = {
  'Content-Type' => 'application/json',
  'Accept' => 'application/json',
  'user-name' => 'string'
}

result = RestClient.post '/api/start_serving',
  params: {
  }, headers: headers

p JSON.parse(result)

```

```python
import requests
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  'user-name': 'string'
}

r = requests.post('/api/start_serving', headers = headers)

print(r.json())

```

```php
<?php

require 'vendor/autoload.php';

$headers = array(
    'Content-Type' => 'application/json',
    'Accept' => 'application/json',
    'user-name' => 'string',
);

$client = new \GuzzleHttp\Client();

// Define array of request body.
$request_body = array();

try {
    $response = $client->request('POST','/api/start_serving', array(
        'headers' => $headers,
        'json' => $request_body,
       )
    );
    print_r($response->getBody()->getContents());
 }
 catch (\GuzzleHttp\Exception\BadResponseException $e) {
    // handle exception or api errors.
    print_r($e->getMessage());
 }

 // ...

```

```java
URL obj = new URL("/api/start_serving");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("POST");
int responseCode = con.getResponseCode();
BufferedReader in = new BufferedReader(
    new InputStreamReader(con.getInputStream()));
String inputLine;
StringBuffer response = new StringBuffer();
while ((inputLine = in.readLine()) != null) {
    response.append(inputLine);
}
in.close();
System.out.println(response.toString());

```

```go
package main

import (
       "bytes"
       "net/http"
)

func main() {

    headers := map[string][]string{
        "Content-Type": []string{"application/json"},
        "Accept": []string{"application/json"},
        "user-name": []string{"string"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("POST", "/api/start_serving", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

`POST /start_serving`

*Start Serving*

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

<h3 id="start_serving_start_serving_post-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|user-name|header|string|true|none|
|body|body|any|true|none|

> Example responses

> 200 Response

```json
{}
```

<h3 id="start_serving_start_serving_post-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|Inline|
|422|[Unprocessable Entity](https://tools.ietf.org/html/rfc2518#section-10.3)|Validation Error|[HTTPValidationError](#schemahttpvalidationerror)|

<h3 id="start_serving_start_serving_post-responseschema">Response Schema</h3>

Status Code **200**

*Response Start Serving Start Serving Post*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|

<aside class="success">
This operation does not require authentication
</aside>

## list_serving_list_serving_get

<a id="opIdlist_serving_list_serving_get"></a>

> Code samples

```shell
# You can also use wget
curl -X GET /api/list_serving \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -H 'user-name: string'

```

```http
GET /api/list_serving HTTP/1.1

Content-Type: application/json
Accept: application/json
user-name: string

```

```javascript
const inputBody = '[
  {
    "model_id": "string",
    "model_revision": "main"
  }
]';
const headers = {
  'Content-Type':'application/json',
  'Accept':'application/json',
  'user-name':'string'
};

fetch('/api/list_serving',
{
  method: 'GET',
  body: inputBody,
  headers: headers
})
.then(function(res) {
    return res.json();
}).then(function(body) {
    console.log(body);
});

```

```ruby
require 'rest-client'
require 'json'

headers = {
  'Content-Type' => 'application/json',
  'Accept' => 'application/json',
  'user-name' => 'string'
}

result = RestClient.get '/api/list_serving',
  params: {
  }, headers: headers

p JSON.parse(result)

```

```python
import requests
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  'user-name': 'string'
}

r = requests.get('/api/list_serving', headers = headers)

print(r.json())

```

```php
<?php

require 'vendor/autoload.php';

$headers = array(
    'Content-Type' => 'application/json',
    'Accept' => 'application/json',
    'user-name' => 'string',
);

$client = new \GuzzleHttp\Client();

// Define array of request body.
$request_body = array();

try {
    $response = $client->request('GET','/api/list_serving', array(
        'headers' => $headers,
        'json' => $request_body,
       )
    );
    print_r($response->getBody()->getContents());
 }
 catch (\GuzzleHttp\Exception\BadResponseException $e) {
    // handle exception or api errors.
    print_r($e->getMessage());
 }

 // ...

```

```java
URL obj = new URL("/api/list_serving");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("GET");
int responseCode = con.getResponseCode();
BufferedReader in = new BufferedReader(
    new InputStreamReader(con.getInputStream()));
String inputLine;
StringBuffer response = new StringBuffer();
while ((inputLine = in.readLine()) != null) {
    response.append(inputLine);
}
in.close();
System.out.println(response.toString());

```

```go
package main

import (
       "bytes"
       "net/http"
)

func main() {

    headers := map[string][]string{
        "Content-Type": []string{"application/json"},
        "Accept": []string{"application/json"},
        "user-name": []string{"string"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("GET", "/api/list_serving", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

`GET /list_serving`

*List Serving*

> Body parameter

```json
[
  {
    "model_id": "string",
    "model_revision": "main"
  }
]
```

<h3 id="list_serving_list_serving_get-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|user-name|header|string|true|none|
|body|body|any|false|none|

> Example responses

> 200 Response

```json
{}
```

<h3 id="list_serving_list_serving_get-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|Inline|
|422|[Unprocessable Entity](https://tools.ietf.org/html/rfc2518#section-10.3)|Validation Error|[HTTPValidationError](#schemahttpvalidationerror)|

<h3 id="list_serving_list_serving_get-responseschema">Response Schema</h3>

Status Code **200**

*Response List Serving List Serving Get*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|

<aside class="success">
This operation does not require authentication
</aside>

## stop_serving_stop_serving_post

<a id="opIdstop_serving_stop_serving_post"></a>

> Code samples

```shell
# You can also use wget
curl -X POST /api/stop_serving \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -H 'user-name: string'

```

```http
POST /api/stop_serving HTTP/1.1

Content-Type: application/json
Accept: application/json
user-name: string

```

```javascript
const inputBody = '[
  {
    "model_id": "string",
    "model_revision": "main"
  }
]';
const headers = {
  'Content-Type':'application/json',
  'Accept':'application/json',
  'user-name':'string'
};

fetch('/api/stop_serving',
{
  method: 'POST',
  body: inputBody,
  headers: headers
})
.then(function(res) {
    return res.json();
}).then(function(body) {
    console.log(body);
});

```

```ruby
require 'rest-client'
require 'json'

headers = {
  'Content-Type' => 'application/json',
  'Accept' => 'application/json',
  'user-name' => 'string'
}

result = RestClient.post '/api/stop_serving',
  params: {
  }, headers: headers

p JSON.parse(result)

```

```python
import requests
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  'user-name': 'string'
}

r = requests.post('/api/stop_serving', headers = headers)

print(r.json())

```

```php
<?php

require 'vendor/autoload.php';

$headers = array(
    'Content-Type' => 'application/json',
    'Accept' => 'application/json',
    'user-name' => 'string',
);

$client = new \GuzzleHttp\Client();

// Define array of request body.
$request_body = array();

try {
    $response = $client->request('POST','/api/stop_serving', array(
        'headers' => $headers,
        'json' => $request_body,
       )
    );
    print_r($response->getBody()->getContents());
 }
 catch (\GuzzleHttp\Exception\BadResponseException $e) {
    // handle exception or api errors.
    print_r($e->getMessage());
 }

 // ...

```

```java
URL obj = new URL("/api/stop_serving");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("POST");
int responseCode = con.getResponseCode();
BufferedReader in = new BufferedReader(
    new InputStreamReader(con.getInputStream()));
String inputLine;
StringBuffer response = new StringBuffer();
while ((inputLine = in.readLine()) != null) {
    response.append(inputLine);
}
in.close();
System.out.println(response.toString());

```

```go
package main

import (
       "bytes"
       "net/http"
)

func main() {

    headers := map[string][]string{
        "Content-Type": []string{"application/json"},
        "Accept": []string{"application/json"},
        "user-name": []string{"string"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("POST", "/api/stop_serving", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

`POST /stop_serving`

*Stop Serving*

> Body parameter

```json
[
  {
    "model_id": "string",
    "model_revision": "main"
  }
]
```

<h3 id="stop_serving_stop_serving_post-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|user-name|header|string|true|none|
|body|body|any|true|none|

> Example responses

> 200 Response

```json
{}
```

<h3 id="stop_serving_stop_serving_post-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|Inline|
|422|[Unprocessable Entity](https://tools.ietf.org/html/rfc2518#section-10.3)|Validation Error|[HTTPValidationError](#schemahttpvalidationerror)|

<h3 id="stop_serving_stop_serving_post-responseschema">Response Schema</h3>

Status Code **200**

*Response Stop Serving Stop Serving Post*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|

<aside class="success">
This operation does not require authentication
</aside>

## list_deployments_list_deployments_get

<a id="opIdlist_deployments_list_deployments_get"></a>

> Code samples

```shell
# You can also use wget
curl -X GET /api/list_deployments \
  -H 'Accept: application/json'

```

```http
GET /api/list_deployments HTTP/1.1

Accept: application/json

```

```javascript

const headers = {
  'Accept':'application/json'
};

fetch('/api/list_deployments',
{
  method: 'GET',

  headers: headers
})
.then(function(res) {
    return res.json();
}).then(function(body) {
    console.log(body);
});

```

```ruby
require 'rest-client'
require 'json'

headers = {
  'Accept' => 'application/json'
}

result = RestClient.get '/api/list_deployments',
  params: {
  }, headers: headers

p JSON.parse(result)

```

```python
import requests
headers = {
  'Accept': 'application/json'
}

r = requests.get('/api/list_deployments', headers = headers)

print(r.json())

```

```php
<?php

require 'vendor/autoload.php';

$headers = array(
    'Accept' => 'application/json',
);

$client = new \GuzzleHttp\Client();

// Define array of request body.
$request_body = array();

try {
    $response = $client->request('GET','/api/list_deployments', array(
        'headers' => $headers,
        'json' => $request_body,
       )
    );
    print_r($response->getBody()->getContents());
 }
 catch (\GuzzleHttp\Exception\BadResponseException $e) {
    // handle exception or api errors.
    print_r($e->getMessage());
 }

 // ...

```

```java
URL obj = new URL("/api/list_deployments");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("GET");
int responseCode = con.getResponseCode();
BufferedReader in = new BufferedReader(
    new InputStreamReader(con.getInputStream()));
String inputLine;
StringBuffer response = new StringBuffer();
while ((inputLine = in.readLine()) != null) {
    response.append(inputLine);
}
in.close();
System.out.println(response.toString());

```

```go
package main

import (
       "bytes"
       "net/http"
)

func main() {

    headers := map[string][]string{
        "Accept": []string{"application/json"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("GET", "/api/list_deployments", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

`GET /list_deployments`

*List Deployments*

> Example responses

> 200 Response

```json
[
  null
]
```

<h3 id="list_deployments_list_deployments_get-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|Inline|

<h3 id="list_deployments_list_deployments_get-responseschema">Response Schema</h3>

Status Code **200**

*Response List Deployments List Deployments Get*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|Response List Deployments List Deployments Get|[any]|false|none|none|

<aside class="success">
This operation does not require authentication
</aside>

## list_apps_list_apps_get

<a id="opIdlist_apps_list_apps_get"></a>

> Code samples

```shell
# You can also use wget
curl -X GET /api/list_apps \
  -H 'Accept: application/json'

```

```http
GET /api/list_apps HTTP/1.1

Accept: application/json

```

```javascript

const headers = {
  'Accept':'application/json'
};

fetch('/api/list_apps',
{
  method: 'GET',

  headers: headers
})
.then(function(res) {
    return res.json();
}).then(function(body) {
    console.log(body);
});

```

```ruby
require 'rest-client'
require 'json'

headers = {
  'Accept' => 'application/json'
}

result = RestClient.get '/api/list_apps',
  params: {
  }, headers: headers

p JSON.parse(result)

```

```python
import requests
headers = {
  'Accept': 'application/json'
}

r = requests.get('/api/list_apps', headers = headers)

print(r.json())

```

```php
<?php

require 'vendor/autoload.php';

$headers = array(
    'Accept' => 'application/json',
);

$client = new \GuzzleHttp\Client();

// Define array of request body.
$request_body = array();

try {
    $response = $client->request('GET','/api/list_apps', array(
        'headers' => $headers,
        'json' => $request_body,
       )
    );
    print_r($response->getBody()->getContents());
 }
 catch (\GuzzleHttp\Exception\BadResponseException $e) {
    // handle exception or api errors.
    print_r($e->getMessage());
 }

 // ...

```

```java
URL obj = new URL("/api/list_apps");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("GET");
int responseCode = con.getResponseCode();
BufferedReader in = new BufferedReader(
    new InputStreamReader(con.getInputStream()));
String inputLine;
StringBuffer response = new StringBuffer();
while ((inputLine = in.readLine()) != null) {
    response.append(inputLine);
}
in.close();
System.out.println(response.toString());

```

```go
package main

import (
       "bytes"
       "net/http"
)

func main() {

    headers := map[string][]string{
        "Accept": []string{"application/json"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("GET", "/api/list_apps", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

`GET /list_apps`

*List Apps*

> Example responses

> 200 Response

```json
{}
```

<h3 id="list_apps_list_apps_get-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|Inline|

<h3 id="list_apps_list_apps_get-responseschema">Response Schema</h3>

Status Code **200**

*Response List Apps List Apps Get*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|

<aside class="success">
This operation does not require authentication
</aside>

## list_oob_models_oob_models_get

<a id="opIdlist_oob_models_oob_models_get"></a>

> Code samples

```shell
# You can also use wget
curl -X GET /api/oob_models \
  -H 'Accept: application/json'

```

```http
GET /api/oob_models HTTP/1.1

Accept: application/json

```

```javascript

const headers = {
  'Accept':'application/json'
};

fetch('/api/oob_models',
{
  method: 'GET',

  headers: headers
})
.then(function(res) {
    return res.json();
}).then(function(body) {
    console.log(body);
});

```

```ruby
require 'rest-client'
require 'json'

headers = {
  'Accept' => 'application/json'
}

result = RestClient.get '/api/oob_models',
  params: {
  }, headers: headers

p JSON.parse(result)

```

```python
import requests
headers = {
  'Accept': 'application/json'
}

r = requests.get('/api/oob_models', headers = headers)

print(r.json())

```

```php
<?php

require 'vendor/autoload.php';

$headers = array(
    'Accept' => 'application/json',
);

$client = new \GuzzleHttp\Client();

// Define array of request body.
$request_body = array();

try {
    $response = $client->request('GET','/api/oob_models', array(
        'headers' => $headers,
        'json' => $request_body,
       )
    );
    print_r($response->getBody()->getContents());
 }
 catch (\GuzzleHttp\Exception\BadResponseException $e) {
    // handle exception or api errors.
    print_r($e->getMessage());
 }

 // ...

```

```java
URL obj = new URL("/api/oob_models");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("GET");
int responseCode = con.getResponseCode();
BufferedReader in = new BufferedReader(
    new InputStreamReader(con.getInputStream()));
String inputLine;
StringBuffer response = new StringBuffer();
while ((inputLine = in.readLine()) != null) {
    response.append(inputLine);
}
in.close();
System.out.println(response.toString());

```

```go
package main

import (
       "bytes"
       "net/http"
)

func main() {

    headers := map[string][]string{
        "Accept": []string{"application/json"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("GET", "/api/oob_models", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

`GET /oob_models`

*List Oob Models*

> Example responses

> 200 Response

```json
{}
```

<h3 id="list_oob_models_oob_models_get-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|Inline|

<h3 id="list_oob_models_oob_models_get-responseschema">Response Schema</h3>

Status Code **200**

*Response List Oob Models Oob Models Get*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|

<aside class="success">
This operation does not require authentication
</aside>

## get_model_models_get

<a id="opIdget_model_models_get"></a>

> Code samples

```shell
# You can also use wget
curl -X GET /api/models \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json'

```

```http
GET /api/models HTTP/1.1

Content-Type: application/json
Accept: application/json

```

```javascript
const inputBody = '{
  "models": [
    "string"
  ]
}';
const headers = {
  'Content-Type':'application/json',
  'Accept':'application/json'
};

fetch('/api/models',
{
  method: 'GET',
  body: inputBody,
  headers: headers
})
.then(function(res) {
    return res.json();
}).then(function(body) {
    console.log(body);
});

```

```ruby
require 'rest-client'
require 'json'

headers = {
  'Content-Type' => 'application/json',
  'Accept' => 'application/json'
}

result = RestClient.get '/api/models',
  params: {
  }, headers: headers

p JSON.parse(result)

```

```python
import requests
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}

r = requests.get('/api/models', headers = headers)

print(r.json())

```

```php
<?php

require 'vendor/autoload.php';

$headers = array(
    'Content-Type' => 'application/json',
    'Accept' => 'application/json',
);

$client = new \GuzzleHttp\Client();

// Define array of request body.
$request_body = array();

try {
    $response = $client->request('GET','/api/models', array(
        'headers' => $headers,
        'json' => $request_body,
       )
    );
    print_r($response->getBody()->getContents());
 }
 catch (\GuzzleHttp\Exception\BadResponseException $e) {
    // handle exception or api errors.
    print_r($e->getMessage());
 }

 // ...

```

```java
URL obj = new URL("/api/models");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("GET");
int responseCode = con.getResponseCode();
BufferedReader in = new BufferedReader(
    new InputStreamReader(con.getInputStream()));
String inputLine;
StringBuffer response = new StringBuffer();
while ((inputLine = in.readLine()) != null) {
    response.append(inputLine);
}
in.close();
System.out.println(response.toString());

```

```go
package main

import (
       "bytes"
       "net/http"
)

func main() {

    headers := map[string][]string{
        "Content-Type": []string{"application/json"},
        "Accept": []string{"application/json"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("GET", "/api/models", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

`GET /models`

*Get Model*

> Body parameter

```json
{
  "models": [
    "string"
  ]
}
```

<h3 id="get_model_models_get-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|body|body|[Body_get_model_models_get](#schemabody_get_model_models_get)|true|none|

> Example responses

> 200 Response

```json
{}
```

<h3 id="get_model_models_get-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|Inline|
|422|[Unprocessable Entity](https://tools.ietf.org/html/rfc2518#section-10.3)|Validation Error|[HTTPValidationError](#schemahttpvalidationerror)|

<h3 id="get_model_models_get-responseschema">Response Schema</h3>

Status Code **200**

*Response Get Model Models Get*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|

<aside class="success">
This operation does not require authentication
</aside>

## update_model_update_serving_post

<a id="opIdupdate_model_update_serving_post"></a>

> Code samples

```shell
# You can also use wget
curl -X POST /api/update_serving \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json'

```

```http
POST /api/update_serving HTTP/1.1

Content-Type: application/json
Accept: application/json

```

```javascript
const inputBody = '{
  "model": {
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
}';
const headers = {
  'Content-Type':'application/json',
  'Accept':'application/json'
};

fetch('/api/update_serving',
{
  method: 'POST',
  body: inputBody,
  headers: headers
})
.then(function(res) {
    return res.json();
}).then(function(body) {
    console.log(body);
});

```

```ruby
require 'rest-client'
require 'json'

headers = {
  'Content-Type' => 'application/json',
  'Accept' => 'application/json'
}

result = RestClient.post '/api/update_serving',
  params: {
  }, headers: headers

p JSON.parse(result)

```

```python
import requests
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}

r = requests.post('/api/update_serving', headers = headers)

print(r.json())

```

```php
<?php

require 'vendor/autoload.php';

$headers = array(
    'Content-Type' => 'application/json',
    'Accept' => 'application/json',
);

$client = new \GuzzleHttp\Client();

// Define array of request body.
$request_body = array();

try {
    $response = $client->request('POST','/api/update_serving', array(
        'headers' => $headers,
        'json' => $request_body,
       )
    );
    print_r($response->getBody()->getContents());
 }
 catch (\GuzzleHttp\Exception\BadResponseException $e) {
    // handle exception or api errors.
    print_r($e->getMessage());
 }

 // ...

```

```java
URL obj = new URL("/api/update_serving");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("POST");
int responseCode = con.getResponseCode();
BufferedReader in = new BufferedReader(
    new InputStreamReader(con.getInputStream()));
String inputLine;
StringBuffer response = new StringBuffer();
while ((inputLine = in.readLine()) != null) {
    response.append(inputLine);
}
in.close();
System.out.println(response.toString());

```

```go
package main

import (
       "bytes"
       "net/http"
)

func main() {

    headers := map[string][]string{
        "Content-Type": []string{"application/json"},
        "Accept": []string{"application/json"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("POST", "/api/update_serving", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

`POST /update_serving`

*Update Model*

> Body parameter

```json
{
  "model": {
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
}
```

<h3 id="update_model_update_serving_post-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|body|body|[Body_update_model_update_serving_post](#schemabody_update_model_update_serving_post)|true|none|

> Example responses

> 200 Response

```json
{}
```

<h3 id="update_model_update_serving_post-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|Inline|
|422|[Unprocessable Entity](https://tools.ietf.org/html/rfc2518#section-10.3)|Validation Error|[HTTPValidationError](#schemahttpvalidationerror)|

<h3 id="update_model_update_serving_post-responseschema">Response Schema</h3>

Status Code **200**

*Response Update Model Update Serving Post*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|

<aside class="success">
This operation does not require authentication
</aside>

## launch_comparation_launch_comparation_post

<a id="opIdlaunch_comparation_launch_comparation_post"></a>

> Code samples

```shell
# You can also use wget
curl -X POST /api/launch_comparation \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json'

```

```http
POST /api/launch_comparation HTTP/1.1

Content-Type: application/json
Accept: application/json

```

```javascript
const inputBody = '{
  "models": [
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
  ],
  "user": "string"
}';
const headers = {
  'Content-Type':'application/json',
  'Accept':'application/json'
};

fetch('/api/launch_comparation',
{
  method: 'POST',
  body: inputBody,
  headers: headers
})
.then(function(res) {
    return res.json();
}).then(function(body) {
    console.log(body);
});

```

```ruby
require 'rest-client'
require 'json'

headers = {
  'Content-Type' => 'application/json',
  'Accept' => 'application/json'
}

result = RestClient.post '/api/launch_comparation',
  params: {
  }, headers: headers

p JSON.parse(result)

```

```python
import requests
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}

r = requests.post('/api/launch_comparation', headers = headers)

print(r.json())

```

```php
<?php

require 'vendor/autoload.php';

$headers = array(
    'Content-Type' => 'application/json',
    'Accept' => 'application/json',
);

$client = new \GuzzleHttp\Client();

// Define array of request body.
$request_body = array();

try {
    $response = $client->request('POST','/api/launch_comparation', array(
        'headers' => $headers,
        'json' => $request_body,
       )
    );
    print_r($response->getBody()->getContents());
 }
 catch (\GuzzleHttp\Exception\BadResponseException $e) {
    // handle exception or api errors.
    print_r($e->getMessage());
 }

 // ...

```

```java
URL obj = new URL("/api/launch_comparation");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("POST");
int responseCode = con.getResponseCode();
BufferedReader in = new BufferedReader(
    new InputStreamReader(con.getInputStream()));
String inputLine;
StringBuffer response = new StringBuffer();
while ((inputLine = in.readLine()) != null) {
    response.append(inputLine);
}
in.close();
System.out.println(response.toString());

```

```go
package main

import (
       "bytes"
       "net/http"
)

func main() {

    headers := map[string][]string{
        "Content-Type": []string{"application/json"},
        "Accept": []string{"application/json"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("POST", "/api/launch_comparation", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

`POST /launch_comparation`

*Launch Comparation*

> Body parameter

```json
{
  "models": [
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
  ],
  "user": "string"
}
```

<h3 id="launch_comparation_launch_comparation_post-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|body|body|[Body_launch_comparation_launch_comparation_post](#schemabody_launch_comparation_launch_comparation_post)|true|none|

> Example responses

> 200 Response

```json
{}
```

<h3 id="launch_comparation_launch_comparation_post-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|Inline|
|422|[Unprocessable Entity](https://tools.ietf.org/html/rfc2518#section-10.3)|Validation Error|[HTTPValidationError](#schemahttpvalidationerror)|

<h3 id="launch_comparation_launch_comparation_post-responseschema">Response Schema</h3>

Status Code **200**

*Response Launch Comparation Launch Comparation Post*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|

<aside class="success">
This operation does not require authentication
</aside>

## update_comparation_update_comparation_post

<a id="opIdupdate_comparation_update_comparation_post"></a>

> Code samples

```shell
# You can also use wget
curl -X POST /api/update_comparation \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json'

```

```http
POST /api/update_comparation HTTP/1.1

Content-Type: application/json
Accept: application/json

```

```javascript
const inputBody = '{
  "models": [
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
  ],
  "name": "string"
}';
const headers = {
  'Content-Type':'application/json',
  'Accept':'application/json'
};

fetch('/api/update_comparation',
{
  method: 'POST',
  body: inputBody,
  headers: headers
})
.then(function(res) {
    return res.json();
}).then(function(body) {
    console.log(body);
});

```

```ruby
require 'rest-client'
require 'json'

headers = {
  'Content-Type' => 'application/json',
  'Accept' => 'application/json'
}

result = RestClient.post '/api/update_comparation',
  params: {
  }, headers: headers

p JSON.parse(result)

```

```python
import requests
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}

r = requests.post('/api/update_comparation', headers = headers)

print(r.json())

```

```php
<?php

require 'vendor/autoload.php';

$headers = array(
    'Content-Type' => 'application/json',
    'Accept' => 'application/json',
);

$client = new \GuzzleHttp\Client();

// Define array of request body.
$request_body = array();

try {
    $response = $client->request('POST','/api/update_comparation', array(
        'headers' => $headers,
        'json' => $request_body,
       )
    );
    print_r($response->getBody()->getContents());
 }
 catch (\GuzzleHttp\Exception\BadResponseException $e) {
    // handle exception or api errors.
    print_r($e->getMessage());
 }

 // ...

```

```java
URL obj = new URL("/api/update_comparation");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("POST");
int responseCode = con.getResponseCode();
BufferedReader in = new BufferedReader(
    new InputStreamReader(con.getInputStream()));
String inputLine;
StringBuffer response = new StringBuffer();
while ((inputLine = in.readLine()) != null) {
    response.append(inputLine);
}
in.close();
System.out.println(response.toString());

```

```go
package main

import (
       "bytes"
       "net/http"
)

func main() {

    headers := map[string][]string{
        "Content-Type": []string{"application/json"},
        "Accept": []string{"application/json"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("POST", "/api/update_comparation", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

`POST /update_comparation`

*Update Comparation*

> Body parameter

```json
{
  "models": [
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
  ],
  "name": "string"
}
```

<h3 id="update_comparation_update_comparation_post-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|body|body|[Body_update_comparation_update_comparation_post](#schemabody_update_comparation_update_comparation_post)|true|none|

> Example responses

> 200 Response

```json
{}
```

<h3 id="update_comparation_update_comparation_post-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|Inline|
|422|[Unprocessable Entity](https://tools.ietf.org/html/rfc2518#section-10.3)|Validation Error|[HTTPValidationError](#schemahttpvalidationerror)|

<h3 id="update_comparation_update_comparation_post-responseschema">Response Schema</h3>

Status Code **200**

*Response Update Comparation Update Comparation Post*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|

<aside class="success">
This operation does not require authentication
</aside>

## models_comparation_models_comparation_get

<a id="opIdmodels_comparation_models_comparation_get"></a>

> Code samples

```shell
# You can also use wget
curl -X GET /api/models_comparation \
  -H 'Accept: application/json'

```

```http
GET /api/models_comparation HTTP/1.1

Accept: application/json

```

```javascript

const headers = {
  'Accept':'application/json'
};

fetch('/api/models_comparation',
{
  method: 'GET',

  headers: headers
})
.then(function(res) {
    return res.json();
}).then(function(body) {
    console.log(body);
});

```

```ruby
require 'rest-client'
require 'json'

headers = {
  'Accept' => 'application/json'
}

result = RestClient.get '/api/models_comparation',
  params: {
  }, headers: headers

p JSON.parse(result)

```

```python
import requests
headers = {
  'Accept': 'application/json'
}

r = requests.get('/api/models_comparation', headers = headers)

print(r.json())

```

```php
<?php

require 'vendor/autoload.php';

$headers = array(
    'Accept' => 'application/json',
);

$client = new \GuzzleHttp\Client();

// Define array of request body.
$request_body = array();

try {
    $response = $client->request('GET','/api/models_comparation', array(
        'headers' => $headers,
        'json' => $request_body,
       )
    );
    print_r($response->getBody()->getContents());
 }
 catch (\GuzzleHttp\Exception\BadResponseException $e) {
    // handle exception or api errors.
    print_r($e->getMessage());
 }

 // ...

```

```java
URL obj = new URL("/api/models_comparation");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("GET");
int responseCode = con.getResponseCode();
BufferedReader in = new BufferedReader(
    new InputStreamReader(con.getInputStream()));
String inputLine;
StringBuffer response = new StringBuffer();
while ((inputLine = in.readLine()) != null) {
    response.append(inputLine);
}
in.close();
System.out.println(response.toString());

```

```go
package main

import (
       "bytes"
       "net/http"
)

func main() {

    headers := map[string][]string{
        "Accept": []string{"application/json"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("GET", "/api/models_comparation", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

`GET /models_comparation`

*Models Comparation*

> Example responses

> 200 Response

```json
{}
```

<h3 id="models_comparation_models_comparation_get-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|Inline|

<h3 id="models_comparation_models_comparation_get-responseschema">Response Schema</h3>

Status Code **200**

*Response Models Comparation Models Comparation Get*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|

<aside class="success">
This operation does not require authentication
</aside>

## list_comparation_list_comparation_get

<a id="opIdlist_comparation_list_comparation_get"></a>

> Code samples

```shell
# You can also use wget
curl -X GET /api/list_comparation \
  -H 'Accept: application/json'

```

```http
GET /api/list_comparation HTTP/1.1

Accept: application/json

```

```javascript

const headers = {
  'Accept':'application/json'
};

fetch('/api/list_comparation',
{
  method: 'GET',

  headers: headers
})
.then(function(res) {
    return res.json();
}).then(function(body) {
    console.log(body);
});

```

```ruby
require 'rest-client'
require 'json'

headers = {
  'Accept' => 'application/json'
}

result = RestClient.get '/api/list_comparation',
  params: {
  }, headers: headers

p JSON.parse(result)

```

```python
import requests
headers = {
  'Accept': 'application/json'
}

r = requests.get('/api/list_comparation', headers = headers)

print(r.json())

```

```php
<?php

require 'vendor/autoload.php';

$headers = array(
    'Accept' => 'application/json',
);

$client = new \GuzzleHttp\Client();

// Define array of request body.
$request_body = array();

try {
    $response = $client->request('GET','/api/list_comparation', array(
        'headers' => $headers,
        'json' => $request_body,
       )
    );
    print_r($response->getBody()->getContents());
 }
 catch (\GuzzleHttp\Exception\BadResponseException $e) {
    // handle exception or api errors.
    print_r($e->getMessage());
 }

 // ...

```

```java
URL obj = new URL("/api/list_comparation");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("GET");
int responseCode = con.getResponseCode();
BufferedReader in = new BufferedReader(
    new InputStreamReader(con.getInputStream()));
String inputLine;
StringBuffer response = new StringBuffer();
while ((inputLine = in.readLine()) != null) {
    response.append(inputLine);
}
in.close();
System.out.println(response.toString());

```

```go
package main

import (
       "bytes"
       "net/http"
)

func main() {

    headers := map[string][]string{
        "Accept": []string{"application/json"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("GET", "/api/list_comparation", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

`GET /list_comparation`

*List Comparation*

> Example responses

> 200 Response

```json
[
  null
]
```

<h3 id="list_comparation_list_comparation_get-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|Inline|

<h3 id="list_comparation_list_comparation_get-responseschema">Response Schema</h3>

Status Code **200**

*Response List Comparation List Comparation Get*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|Response List Comparation List Comparation Get|[any]|false|none|none|

<aside class="success">
This operation does not require authentication
</aside>

## delete_app_delete_comparation_post

<a id="opIddelete_app_delete_comparation_post"></a>

> Code samples

```shell
# You can also use wget
curl -X POST /api/delete_comparation \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json'

```

```http
POST /api/delete_comparation HTTP/1.1

Content-Type: application/json
Accept: application/json

```

```javascript
const inputBody = '{
  "names": [
    "string"
  ]
}';
const headers = {
  'Content-Type':'application/json',
  'Accept':'application/json'
};

fetch('/api/delete_comparation',
{
  method: 'POST',
  body: inputBody,
  headers: headers
})
.then(function(res) {
    return res.json();
}).then(function(body) {
    console.log(body);
});

```

```ruby
require 'rest-client'
require 'json'

headers = {
  'Content-Type' => 'application/json',
  'Accept' => 'application/json'
}

result = RestClient.post '/api/delete_comparation',
  params: {
  }, headers: headers

p JSON.parse(result)

```

```python
import requests
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}

r = requests.post('/api/delete_comparation', headers = headers)

print(r.json())

```

```php
<?php

require 'vendor/autoload.php';

$headers = array(
    'Content-Type' => 'application/json',
    'Accept' => 'application/json',
);

$client = new \GuzzleHttp\Client();

// Define array of request body.
$request_body = array();

try {
    $response = $client->request('POST','/api/delete_comparation', array(
        'headers' => $headers,
        'json' => $request_body,
       )
    );
    print_r($response->getBody()->getContents());
 }
 catch (\GuzzleHttp\Exception\BadResponseException $e) {
    // handle exception or api errors.
    print_r($e->getMessage());
 }

 // ...

```

```java
URL obj = new URL("/api/delete_comparation");
HttpURLConnection con = (HttpURLConnection) obj.openConnection();
con.setRequestMethod("POST");
int responseCode = con.getResponseCode();
BufferedReader in = new BufferedReader(
    new InputStreamReader(con.getInputStream()));
String inputLine;
StringBuffer response = new StringBuffer();
while ((inputLine = in.readLine()) != null) {
    response.append(inputLine);
}
in.close();
System.out.println(response.toString());

```

```go
package main

import (
       "bytes"
       "net/http"
)

func main() {

    headers := map[string][]string{
        "Content-Type": []string{"application/json"},
        "Accept": []string{"application/json"},
    }

    data := bytes.NewBuffer([]byte{jsonReq})
    req, err := http.NewRequest("POST", "/api/delete_comparation", data)
    req.Header = headers

    client := &http.Client{}
    resp, err := client.Do(req)
    // ...
}

```

`POST /delete_comparation`

*Delete App*

> Body parameter

```json
{
  "names": [
    "string"
  ]
}
```

<h3 id="delete_app_delete_comparation_post-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|body|body|[Body_delete_app_delete_comparation_post](#schemabody_delete_app_delete_comparation_post)|true|none|

> Example responses

> 200 Response

```json
{}
```

<h3 id="delete_app_delete_comparation_post-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Successful Response|Inline|
|422|[Unprocessable Entity](https://tools.ietf.org/html/rfc2518#section-10.3)|Validation Error|[HTTPValidationError](#schemahttpvalidationerror)|

<h3 id="delete_app_delete_comparation_post-responseschema">Response Schema</h3>

Status Code **200**

*Response Delete App Delete Comparation Post*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|

<aside class="success">
This operation does not require authentication
</aside>

# Schemas

<h2 id="tocS_Body_delete_app_delete_comparation_post">Body_delete_app_delete_comparation_post</h2>
<!-- backwards compatibility -->
<a id="schemabody_delete_app_delete_comparation_post"></a>
<a id="schema_Body_delete_app_delete_comparation_post"></a>
<a id="tocSbody_delete_app_delete_comparation_post"></a>
<a id="tocsbody_delete_app_delete_comparation_post"></a>

```json
{
  "names": [
    "string"
  ]
}

```

Body_delete_app_delete_comparation_post

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|names|[string]|true|none|model id or all|

<h2 id="tocS_Body_get_model_models_get">Body_get_model_models_get</h2>
<!-- backwards compatibility -->
<a id="schemabody_get_model_models_get"></a>
<a id="schema_Body_get_model_models_get"></a>
<a id="tocSbody_get_model_models_get"></a>
<a id="tocsbody_get_model_models_get"></a>

```json
{
  "models": [
    "string"
  ]
}

```

Body_get_model_models_get

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|models|[string]|true|none|models name|

<h2 id="tocS_Body_launch_comparation_launch_comparation_post">Body_launch_comparation_launch_comparation_post</h2>
<!-- backwards compatibility -->
<a id="schemabody_launch_comparation_launch_comparation_post"></a>
<a id="schema_Body_launch_comparation_launch_comparation_post"></a>
<a id="tocSbody_launch_comparation_launch_comparation_post"></a>
<a id="tocsbody_launch_comparation_launch_comparation_post"></a>

```json
{
  "models": [
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
  ],
  "user": "string"
}

```

Body_launch_comparation_launch_comparation_post

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|models|[[ModelConfig](#schemamodelconfig)]|true|none|none|
|user|string|true|none|none|

<h2 id="tocS_Body_start_experimental_start_experimental_post">Body_start_experimental_start_experimental_post</h2>
<!-- backwards compatibility -->
<a id="schemabody_start_experimental_start_experimental_post"></a>
<a id="schema_Body_start_experimental_start_experimental_post"></a>
<a id="tocSbody_start_experimental_start_experimental_post"></a>
<a id="tocsbody_start_experimental_start_experimental_post"></a>

```json
{
  "models": {
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
}

```

Body_start_experimental_start_experimental_post

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|models|any|true|none|none|

anyOf

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|[ModelConfig](#schemamodelconfig)|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|string|false|none|none|

<h2 id="tocS_Body_update_comparation_update_comparation_post">Body_update_comparation_update_comparation_post</h2>
<!-- backwards compatibility -->
<a id="schemabody_update_comparation_update_comparation_post"></a>
<a id="schema_Body_update_comparation_update_comparation_post"></a>
<a id="tocSbody_update_comparation_update_comparation_post"></a>
<a id="tocsbody_update_comparation_update_comparation_post"></a>

```json
{
  "models": [
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
  ],
  "name": "string"
}

```

Body_update_comparation_update_comparation_post

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|models|[[ModelConfig](#schemamodelconfig)]|true|none|none|
|name|string|true|none|none|

<h2 id="tocS_Body_update_model_update_serving_post">Body_update_model_update_serving_post</h2>
<!-- backwards compatibility -->
<a id="schemabody_update_model_update_serving_post"></a>
<a id="schema_Body_update_model_update_serving_post"></a>
<a id="tocSbody_update_model_update_serving_post"></a>
<a id="tocsbody_update_model_update_serving_post"></a>

```json
{
  "model": {
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
}

```

Body_update_model_update_serving_post

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|model|[ModelConfig](#schemamodelconfig)|true|none|none|

<h2 id="tocS_HTTPValidationError">HTTPValidationError</h2>
<!-- backwards compatibility -->
<a id="schemahttpvalidationerror"></a>
<a id="schema_HTTPValidationError"></a>
<a id="tocShttpvalidationerror"></a>
<a id="tocshttpvalidationerror"></a>

```json
{
  "detail": [
    {
      "loc": [
        "string"
      ],
      "msg": "string",
      "type": "string"
    }
  ]
}

```

HTTPValidationError

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|detail|[[ValidationError](#schemavalidationerror)]|false|none|none|

<h2 id="tocS_ModelConfig">ModelConfig</h2>
<!-- backwards compatibility -->
<a id="schemamodelconfig"></a>
<a id="schema_ModelConfig"></a>
<a id="tocSmodelconfig"></a>
<a id="tocsmodelconfig"></a>

```json
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

```

ModelConfig

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|model_id|string|true|none|none|
|model_task|string|true|none|none|
|model_revision|string|true|none|none|
|is_oob|boolean|true|none|none|
|scaling_config|[Scaling_Config_Simple](#schemascaling_config_simple)|true|none|none|

<h2 id="tocS_ModelIdentifier">ModelIdentifier</h2>
<!-- backwards compatibility -->
<a id="schemamodelidentifier"></a>
<a id="schema_ModelIdentifier"></a>
<a id="tocSmodelidentifier"></a>
<a id="tocsmodelidentifier"></a>

```json
{
  "model_id": "string",
  "model_revision": "main"
}

```

ModelIdentifier

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|model_id|string|true|none|none|
|model_revision|string|false|none|none|

<h2 id="tocS_Prompt">Prompt</h2>
<!-- backwards compatibility -->
<a id="schemaprompt"></a>
<a id="schema_Prompt"></a>
<a id="tocSprompt"></a>
<a id="tocsprompt"></a>

```json
{
  "prompt": null,
  "use_prompt_format": true
}

```

Prompt

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|prompt|any|false|none|none|
|use_prompt_format|boolean|false|none|none|

<h2 id="tocS_Scaling_Config_Simple">Scaling_Config_Simple</h2>
<!-- backwards compatibility -->
<a id="schemascaling_config_simple"></a>
<a id="schema_Scaling_Config_Simple"></a>
<a id="tocSscaling_config_simple"></a>
<a id="tocsscaling_config_simple"></a>

```json
{
  "num_workers": 0,
  "num_gpus_per_worker": 1,
  "num_cpus_per_worker": 1
}

```

Scaling_Config_Simple

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|num_workers|integer|true|none|none|
|num_gpus_per_worker|number|false|none|none|
|num_cpus_per_worker|number|false|none|none|

<h2 id="tocS_ValidationError">ValidationError</h2>
<!-- backwards compatibility -->
<a id="schemavalidationerror"></a>
<a id="schema_ValidationError"></a>
<a id="tocSvalidationerror"></a>
<a id="tocsvalidationerror"></a>

```json
{
  "loc": [
    "string"
  ],
  "msg": "string",
  "type": "string"
}

```

ValidationError

### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|loc|[anyOf]|true|none|none|

anyOf

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|string|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|integer|false|none|none|

continued

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|msg|string|true|none|none|
|type|string|true|none|none|

