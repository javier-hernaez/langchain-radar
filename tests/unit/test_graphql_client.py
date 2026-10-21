import httpx
import pytest
import respx

from src.domain.exceptions import RateLimitExceededError
from src.infrastructure.github.graphql_client import GitHubGraphQLClient

MOCK_GRAPHQL_RESPONSE = {
    "data": {
        "repository": {
            "pullRequests": {
                "pageInfo": {"hasNextPage": True, "endCursor": "cursor_xyz"},
                "nodes": [
                    {
                        "number": 1234,
                        "title": "feat(core): new agent executor",
                        "body": "This introduces a breaking change to the old agent syntax.",
                        "author": {"login": "dev_expert"},
                        "mergedAt": "2026-09-07T12:00:00Z",
                        "baseRefName": "main",
                        "headRefName": "feat-agent",
                        "mergeCommit": {"oid": "a" * 40},
                        "labels": {"nodes": [{"name": "breaking change"}, {"name": "core"}]},
                        "files": {
                            "totalCount": 2,
                            "pageInfo": {"hasNextPage": False},
                            "nodes": [
                                {
                                    "path": "libs/core/agent.py",
                                    "changeType": "ADDED",
                                    "additions": 45,
                                    "deletions": 0,
                                    "patch": "class AgentExecutor: pass",
                                },
                                {
                                    "path": "poetry.lock",
                                    "changeType": "MODIFIED",
                                    "additions": 100,
                                    "deletions": 50,
                                    "patch": None,
                                },
                            ],
                        },
                    }
                ],
            }
        }
    }
}


@pytest.mark.asyncio
async def test_graphql_client_fetch_and_parse_success() -> None:
    async with httpx.AsyncClient() as client:
        with respx.mock(base_url="https://api.github.com") as respx_mock:
            respx_mock.post("/graphql").respond(json=MOCK_GRAPHQL_RESPONSE)

            gh_client = GitHubGraphQLClient(token="mock_token", client=client)
            prs, cursor, has_next = await gh_client.fetch_merged_prs(
                owner="langchain-ai", repo="langchain", page_size=10
            )

            assert len(prs) == 1
            assert cursor == "cursor_xyz"
            assert has_next is True

            pr = prs[0]
            assert pr.pr_number == 1234
            assert pr.author == "dev_expert"
            assert pr.is_breaking_change is True
            assert len(pr.files_changed) == 2
            assert pr.files_changed[0].file_path == "libs/core/agent.py"
            assert pr.files_changed[0].status == "ADDED"


@pytest.mark.asyncio
async def test_graphql_client_rate_limit_retry() -> None:
    async with httpx.AsyncClient() as client:
        with respx.mock(base_url="https://api.github.com") as respx_mock:
            # Siempre responde 429
            respx_mock.post("/graphql").respond(status_code=429)

            gh_client = GitHubGraphQLClient(token="mock_token", client=client)
            with pytest.raises(RateLimitExceededError):
                await gh_client.fetch_merged_prs(max_retries=1)
