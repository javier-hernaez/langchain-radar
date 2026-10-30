from datetime import UTC, datetime

from src.application.splitters.diff_splitter import GitDiffSplitter


def test_diff_splitter_empty_and_whitespace() -> None:
    splitter = GitDiffSplitter()
    now = datetime.now(UTC)

    assert splitter.parse_patch_hunks("test.py", "", "a" * 40, now) == []
    assert splitter.parse_patch_hunks("test.py", "   \n  \n", "a" * 40, now) == []


def test_diff_splitter_multi_hunk_parsing() -> None:
    splitter = GitDiffSplitter()
    now = datetime.now(UTC)

    patch = """@@ -15,6 +15,9 @@ def bind_tools(self, tools):
-    return self._old_bind(tools)
+    if strict:
+        validate_tools(tools)
+    return self._new_bind(tools)
@@ -45,4 +48,6 @@ class ChatModel:
+    def with_structured_output(self, schema):
+        return StructuredOutput(self, schema)
"""

    chunks = splitter.parse_patch_hunks(
        file_path="libs/core/language_models/chat.py",
        patch_content=patch,
        commit_sha="b" * 40,
        commit_date=now,
        release_version="v0.3.0",
        pr_number=123,
        pr_title="feat: improve tool binding",
        author="alice",
    )

    assert len(chunks) == 2
    c1, c2 = chunks

    # Hunk 1
    assert c1.start_line == 15
    assert c1.end_line == 23
    assert c1.metadata.entity_type == "diff_hunk"
    assert c1.metadata.language == "diff"
    assert "bind_tools" in c1.metadata.entity_name
    assert "PR #123: feat: improve tool binding" in c1.content
    assert "+    if strict:" in c1.content

    # Hunk 2
    assert c2.start_line == 48
    assert "ChatModel" in c2.metadata.entity_name
    assert "with_structured_output" in c2.content


def test_diff_splitter_breaking_change_detection() -> None:
    splitter = GitDiffSplitter()
    now = datetime.now(UTC)

    # 1. Breaking change in PR title
    patch = "@@ -1,3 +1,3 @@\n-old\n+new\n"
    chunks = splitter.parse_patch_hunks(
        file_path="core.py",
        patch_content=patch,
        commit_sha="c" * 40,
        commit_date=now,
        pr_title="feat!: breaking change in core abstractions",
    )
    assert len(chunks) == 1
    assert chunks[0].metadata.breaking_change is True

    # 2. Breaking change in hunk text
    patch_with_warning = "@@ -1,3 +1,4 @@\n+# BREAKING CHANGE: deprecated method removed\n+pass\n"
    chunks2 = splitter.parse_patch_hunks(
        file_path="core.py",
        patch_content=patch_with_warning,
        commit_sha="c" * 40,
        commit_date=now,
        pr_title="refactor: cleanup",
    )
    assert len(chunks2) == 1
    assert chunks2[0].metadata.breaking_change is True


def test_diff_splitter_secret_scrubbing_and_fallback() -> None:
    splitter = GitDiffSplitter()
    now = datetime.now(UTC)

    # Secret inside patch
    patch_with_secret = (
        "@@ -1,2 +1,3 @@\n+api_key = 'sk-proj-1234567890abcdef1234567890abcdef12345678'\n"
    )
    chunks = splitter.parse_patch_hunks(
        file_path="config.py",
        patch_content=patch_with_secret,
        commit_sha="d" * 40,
        commit_date=now,
    )
    assert len(chunks) == 1
    assert "[REDACTED_SECRET]" in chunks[0].content
    assert "sk-proj-" not in chunks[0].content

    # Fallback when no @@ header exists
    raw_patch = "+added line 1\n+added line 2"
    fallback_chunks = splitter.parse_patch_hunks(
        file_path="README.md",
        patch_content=raw_patch,
        commit_sha="d" * 40,
        commit_date=now,
    )
    assert len(fallback_chunks) == 1
    assert fallback_chunks[0].start_line == 1
    assert fallback_chunks[0].metadata.entity_name == "README.md#1-2"
