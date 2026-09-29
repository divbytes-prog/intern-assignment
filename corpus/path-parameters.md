# FastAPI path parameters

This is an original, concise study note based on https://fastapi.tiangolo.com/tutorial/path-params/.

## Define a path parameter

A route such as `@app.get("/items/{item_id}")` takes `item_id` from the URL.
Declaring `item_id: int` in the function signature asks FastAPI to parse and
validate the URL segment as an integer. Invalid values get a validation error.

## Order matters

Declare a fixed path such as `/users/me` before `/users/{user_id}`, otherwise
the variable route may catch the word `me`. A path parameter can use an Enum
to restrict values. OpenAPI documentation is generated from the route and types.
