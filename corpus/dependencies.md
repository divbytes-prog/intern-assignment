# FastAPI dependencies

This is an original, concise study note based on https://fastapi.tiangolo.com/tutorial/dependencies/.

## Depends

Use `Depends(function)` in a path operation to declare a dependency.
FastAPI calls the dependency and passes its return value into the route.
Dependencies can accept request parameters and can depend on other
dependencies, forming a reusable tree.

## Common uses

Dependency injection is useful for shared query parsing, authorization, and
database sessions. Dependencies are integrated into OpenAPI documentation.
Within one request, dependency results are cached by default; `use_cache=False`
requests a fresh call.
