import httpx
import pytest
import respx

from src.domain.exceptions import RateLimitExceededError, StorageError
from src.infrastructure.github.rest_client import GitHubRestClient


@pytest.mark.asyncio
async def test_rest_client_fetch_pr_files_paginates_and_maps_status() -> None:
    page_1 = [
        {
            "filename": f"libs/core/f{i}.py",
            "status": "modified",
            "additions": 1,
            "deletions": 0,
            "changes": 1,
            "patch": "@@ -1 +1 @@\n-a\n+b",
        }
        for i in range(100)
    ]
    page_2 = [
        {"filename": "libs/core/new.py", "status": "added"},
        {"filename": "libs/core/old.py", "status": "removed"},
        {
            "filename": "libs/core/moved.py",
            "status": "renamed",
            "previous_filename": "libs/core/orig.py",
        },
    ]
    async with httpx.AsyncClient() as client:
        with respx.mock(base_url="https://api.github.com") as respx_mock:
            route = respx_mock.get("/repos/o/r/pulls/5/files")
            route.side_effect = [
                httpx.Response(200, json=page_1),
                httpx.Response(200, json=page_2),
            ]
            gh = GitHubRestClient(token="t", client=client)
            files = await gh.fetch_pr_files("o", "r", 5)

            assert route.call_count == 2
            assert route.calls[0].request.headers["Authorization"] == "Bearer t"

    assert len(files) == 103
    assert files[0].patch is not None
    assert files[100].status == "ADDED"
    assert files[101].status == "REMOVED"
    assert files[102].status == "RENAMED"
    assert files[102].previous_filename == "libs/core/orig.py"


@pytest.mark.asyncio
async def test_rest_client_fetch_file_content_raw_and_missing() -> None:
    async with httpx.AsyncClient() as client:
        with respx.mock(base_url="https://api.github.com") as respx_mock:
            ok_route = respx_mock.get("/repos/o/r/contents/libs/core/a%20b.py").respond(
                200, content=b"def f():\n    return 1\n"
            )
            respx_mock.get("/repos/o/r/contents/libs/core/missing.py").respond(404)

            gh = GitHubRestClient(token="t", client=client)
            content = await gh.fetch_file_content("o", "r", "libs/core/a b.py", "abc123")
            missing = await gh.fetch_file_content("o", "r", "libs/core/missing.py", "abc123")

            request = ok_route.calls[0].request
            assert request.url.params["ref"] == "abc123"
            assert request.headers["Accept"] == "application/vnd.github.raw+json"

    assert content == "def f():\n    return 1\n"
    assert missing is None


@pytest.mark.asyncio
async def test_rest_client_rate_limit_and_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("src.infrastructure.github.rest_client.asyncio.sleep", no_sleep)

    async with httpx.AsyncClient() as client:
        with respx.mock(base_url="https://api.github.com") as respx_mock:
            limited = respx_mock.get("/repos/o/r/pulls/1/files").respond(
                403, headers={"x-ratelimit-remaining": "0"}, text="API rate limit exceeded"
            )
            respx_mock.get("/repos/o/r/pulls/2/files").respond(500, text="boom")

            gh = GitHubRestClient(token="t", client=client, max_retries=2)
            with pytest.raises(RateLimitExceededError):
                await gh.fetch_pr_files("o", "r", 1)
            assert limited.call_count == 3

            with pytest.raises(StorageError):
                await gh.fetch_pr_files("o", "r", 2)


@pytest.mark.asyncio
async def test_rest_client_owns_and_closes_internal_client() -> None:
    gh = GitHubRestClient(token="")
    internal = gh._get_client()
    assert "Authorization" not in gh._headers("application/json")
    await gh.aclose()
    assert internal.is_closed
