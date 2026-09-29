# FastAPI dependencies

Original study note based on the [FastAPI dependency guide](https://fastapi.tiangolo.com/tutorial/dependencies/) and its linked sections on sub-dependencies and yield-based cleanup.

## Declare reusable behavior

`Depends(callable)` tells FastAPI to call a reusable function and inject its result into a path operation. The callable can read and validate request parameters like a route can. This is useful for pagination defaults, a current-user lookup, authorization rules, or constructing a database session.

```python
from fastapi import Depends

def common_page(limit: int = 20):
    return min(limit, 100)

@app.get("/items")
def list_items(page_size: int = Depends(common_page)):
    return {"limit": page_size}
```

In this example, the dependency receives `limit` as a query parameter and returns a bounded value. The route receives the result as `page_size`; it does not need to call `common_page` itself. FastAPI includes dependency parameters in the generated OpenAPI documentation.

## Dependency trees and caching

A dependency may declare its own dependencies, forming a tree that FastAPI resolves for the current request. This allows shared validation logic to be assembled without repeating it in every route. If the same dependency appears more than once in one request, FastAPI normally reuses its result for that request. Set `use_cache=False` on a specific `Depends` declaration if it really must run again. This cache is request-scoped, not a persistent application cache.

## Resource cleanup

A dependency implemented as a generator can yield a resource for the route, then clean it up after the request handling phase. A database session is a common example: create the session, `yield` it, and close it in `finally`. Keep authorization checks separate from data access when that separation makes failures easier to reason about.
