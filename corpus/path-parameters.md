# FastAPI path parameters

Original study note based on the [FastAPI path parameter guide](https://fastapi.tiangolo.com/tutorial/path-params/). The examples below are explanatory, not copied documentation.

## Route variables and parsing

A route pattern such as `@app.get("/items/{item_id}")` captures the matching URL segment and passes it to a parameter named `item_id` in the path operation function. If the function declares `item_id: int`, FastAPI parses the text segment into an integer before calling the function. A request to `/items/3` therefore gives the function the Python integer `3`, rather than the string `"3"`.

```python
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
```

If the caller sends `/items/abc`, integer parsing fails and the API returns a validation error identifying the path parameter. A decimal such as `/items/4.2` is also not a valid integer for this declaration. This conversion and validation come from the type annotation; application code does not need to call `int()` for the usual case. The generated OpenAPI schema and `/docs` page describe the parameter type.

## Route order and predefined values

Path operations are matched in declaration order. Register a fixed route like `/users/me` before `/users/{user_id}` so the variable route does not interpret `me` as an ID. If only a few text values are valid, annotate the parameter with a string `Enum`; FastAPI then validates membership and lists the possible values in API documentation.

## Parameters that themselves contain a path

The route `/files/{file_path:path}` uses a path converter to capture slashes inside `file_path`. This differs from a normal parameter, which captures a single URL segment. Think about route precedence and URL shape carefully when combining this converter with other file routes.
