# FastAPI request body

Original study note based on the [FastAPI request body guide](https://fastapi.tiangolo.com/tutorial/body/).

## Declaring JSON input with a model

A request body is data sent by a client to an API operation. A Pydantic `BaseModel` describes its fields and gives FastAPI a schema for parsing, validation, and the generated OpenAPI documentation. Fields with no default are required; a field with a default such as `description: str | None = None` is optional in the incoming JSON.

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    description: str | None = None

@app.post("/items")
def create_item(item: Item):
    return {"name": item.name, "price": item.price}
```

For this operation, the client sends a JSON object such as `{"name":"Notebook","price":12.5}`. FastAPI creates an `Item` instance before running the function. Missing required fields or values that cannot pass the model's validation produce a request validation response rather than an `Item` instance. Model validation is more reliable than manually reading arbitrary JSON keys in every route.

## Combining path, query, and body data

FastAPI determines each parameter's source from the route path and its type. A name embedded in the URL pattern, such as `item_id` in `/items/{item_id}`, is a path parameter. A simple typed value like `q: str | None = None` that is not in the path is interpreted as a query parameter. A parameter typed as a Pydantic model is taken from the request body. For unusual or multiple body parameters, use FastAPI's `Body` declaration explicitly.

The `/docs` page displays the JSON schema and lets reviewers send a valid or invalid example. A `GET` request body is generally unsuitable because clients and intermediaries do not have a consistently defined behavior for it; use `POST`, `PUT`, or `PATCH` when sending JSON data.
