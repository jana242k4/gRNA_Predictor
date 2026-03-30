from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class InvalidSequenceError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=422, detail=detail)


class SequenceTooLongError(HTTPException):
    def __init__(self, max_len: int):
        super().__init__(
            status_code=413,
            detail=f"Sequence exceeds maximum allowed length of {max_len} bp.",
        )


async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )
