# FastAPI response model

This is an original, concise study note based on https://fastapi.tiangolo.com/tutorial/response-model/.

## Declare and validate output

Set `response_model=ItemOut` on the route decorator, or use a return type
annotation. FastAPI uses the model to document, validate, and serialize the
returned data. The response model can filter fields that are not declared in
it, which helps avoid exposing private fields such as passwords.

## Output differs from input

An input model may contain a password while the output model omits it.
The output schema controls the response shape regardless of extra fields in
the returned object.
