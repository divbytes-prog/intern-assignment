# FastAPI response model

Original study note based on the [FastAPI response model guide](https://fastapi.tiangolo.com/tutorial/response-model/).

## Define the public output shape

Annotate the return type with a Pydantic model or set `response_model=PublicItem` on a route decorator. FastAPI uses that declaration to generate an OpenAPI output schema, validate and serialize the return value, and filter fields outside the declared model. That last behavior is important when application data contains internal fields that should not appear in the HTTP response.

```python
from pydantic import BaseModel

class PublicUser(BaseModel):
    username: str

@app.get("/users/{username}", response_model=PublicUser)
def read_user(username: str):
    return {"username": username, "internal_note": "not returned"}
```

The response contains `username`, but not `internal_note`. A common design uses one model for account creation, including a password, and a separate response model that omits the password. Output filtering helps prevent accidental exposure; it does not replace sound data-access controls.

## `response_model` and return annotations

Either the decorator's `response_model` parameter or a return type annotation can describe the response. The decorator is useful when the function returns a broader object in Python but the HTTP output must be narrower. FastAPI's automatic `/docs` page then shows both the request schema and the response schema, which helps an API client understand what it should send and what it can expect back.

For endpoints that can abstain or return errors, document the successful response model and use meaningful status codes for invalid input or provider failures. A response model applies to the successful response; an `HTTPException` still produces its own error body.
