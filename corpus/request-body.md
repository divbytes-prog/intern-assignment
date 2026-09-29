# FastAPI request body

This is an original, concise study note based on https://fastapi.tiangolo.com/tutorial/body/.

## Pydantic model

Define a `BaseModel` with typed fields, then annotate a path operation
parameter with that model: `def create_item(item: Item)`. FastAPI reads the
request body as JSON, converts types where valid, validates the fields, and
supplies the resulting model to the function. Invalid body data generates a
validation error response.

## Parameters from different locations

Names declared in the route are path parameters. Simple typed parameters not
in the route are generally query parameters. A Pydantic model parameter is
treated as a request body. Use `Body` for explicit body parameter declarations.
