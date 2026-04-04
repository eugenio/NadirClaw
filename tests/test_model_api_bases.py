"""Tests for per-model API base URL resolution (multi-region support)."""

import os

import pytest


class TestModelApiBases:
    """Test NADIRCLAW_MODEL_API_BASES parsing and resolution."""

    def test_empty_returns_no_entries(self, monkeypatch):
        monkeypatch.delenv("NADIRCLAW_MODEL_API_BASES", raising=False)
        from nadirclaw.settings import Settings

        s = Settings()
        assert s.MODEL_API_BASES == []

    def test_single_entry(self, monkeypatch):
        monkeypatch.setenv(
            "NADIRCLAW_MODEL_API_BASES",
            "openai/deepseek.*=https://bedrock-mantle.eu-west-2.api.aws/v1",
        )
        from nadirclaw.settings import Settings

        s = Settings()
        bases = s.MODEL_API_BASES
        assert len(bases) == 1
        assert bases[0][0].match("openai/deepseek.v3.2")
        assert bases[0][1] == "https://bedrock-mantle.eu-west-2.api.aws/v1"

    def test_multiple_entries(self, monkeypatch):
        monkeypatch.setenv(
            "NADIRCLAW_MODEL_API_BASES",
            "openai/deepseek.*=https://eu-west-2.example.com/v1,"
            "openai/moonshotai.*=https://us-east-1.example.com/v1",
        )
        from nadirclaw.settings import Settings

        s = Settings()
        bases = s.MODEL_API_BASES
        assert len(bases) == 2

    def test_resolve_matches_first_pattern(self, monkeypatch):
        monkeypatch.setenv(
            "NADIRCLAW_MODEL_API_BASES",
            "openai/deepseek.*=https://eu-west-2.example.com/v1,"
            "openai/*=https://us-east-1.example.com/v1",
        )
        monkeypatch.setenv("NADIRCLAW_API_BASE", "https://default.example.com/v1")
        from nadirclaw.settings import Settings

        s = Settings()
        # Specific match wins
        assert s.resolve_api_base("openai/deepseek.v3.2") == "https://eu-west-2.example.com/v1"
        # Wildcard catch-all
        assert s.resolve_api_base("openai/moonshotai.kimi-k2.5") == "https://us-east-1.example.com/v1"

    def test_resolve_falls_back_to_api_base(self, monkeypatch):
        monkeypatch.setenv(
            "NADIRCLAW_MODEL_API_BASES",
            "openai/deepseek.*=https://eu-west-2.example.com/v1",
        )
        monkeypatch.setenv("NADIRCLAW_API_BASE", "https://default.example.com/v1")
        from nadirclaw.settings import Settings

        s = Settings()
        # No pattern matches → falls back to global API_BASE
        assert s.resolve_api_base("anthropic/claude-sonnet") == "https://default.example.com/v1"

    def test_resolve_returns_empty_when_no_match_and_no_global(self, monkeypatch):
        monkeypatch.setenv(
            "NADIRCLAW_MODEL_API_BASES",
            "openai/deepseek.*=https://eu-west-2.example.com/v1",
        )
        monkeypatch.delenv("NADIRCLAW_API_BASE", raising=False)
        from nadirclaw.settings import Settings

        s = Settings()
        assert s.resolve_api_base("anthropic/claude-sonnet") == ""

    def test_malformed_entries_skipped(self, monkeypatch):
        monkeypatch.setenv(
            "NADIRCLAW_MODEL_API_BASES",
            "no-equals-sign,=empty-pattern,openai/*=https://good.example.com/v1",
        )
        from nadirclaw.settings import Settings

        s = Settings()
        bases = s.MODEL_API_BASES
        assert len(bases) == 1
        assert bases[0][1] == "https://good.example.com/v1"

    def test_exact_model_match(self, monkeypatch):
        monkeypatch.setenv(
            "NADIRCLAW_MODEL_API_BASES",
            "openai/deepseek.v3.2=https://specific.example.com/v1",
        )
        from nadirclaw.settings import Settings

        s = Settings()
        assert s.resolve_api_base("openai/deepseek.v3.2") == "https://specific.example.com/v1"
        # Different model should NOT match (dots are escaped, then \. allows any char in regex,
        # but the exact name still matches because dots match dots)
        assert s.resolve_api_base("openai/deepseek.v3.2-beta") == ""

    def test_question_mark_wildcard(self, monkeypatch):
        monkeypatch.setenv(
            "NADIRCLAW_MODEL_API_BASES",
            "openai/deepseek.v?.2=https://specific.example.com/v1",
        )
        from nadirclaw.settings import Settings

        s = Settings()
        assert s.resolve_api_base("openai/deepseek.v3.2") == "https://specific.example.com/v1"
        assert s.resolve_api_base("openai/deepseek.v4.2") == "https://specific.example.com/v1"


class TestModelInfoEndpoint:
    """Test that /model/info returns correct context windows from MODEL_REGISTRY."""

    def test_glm5_context_window_in_registry(self):
        from nadirclaw.routing import MODEL_REGISTRY

        info = MODEL_REGISTRY.get("openai/zai.glm-5")
        assert info is not None
        assert info["context_window"] == 1_000_000

    def test_glm47_context_window_in_registry(self):
        from nadirclaw.routing import MODEL_REGISTRY

        info = MODEL_REGISTRY.get("openai/zai.glm-4.7")
        assert info is not None
        assert info["context_window"] == 202_000

    def test_model_info_response_format(self, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_SIMPLE_MODEL", "openai/deepseek.v3.2")
        monkeypatch.setenv("NADIRCLAW_MID_MODEL", "openai/moonshotai.kimi-k2.5")
        monkeypatch.setenv("NADIRCLAW_COMPLEX_MODEL", "openai/zai.glm-5")

        from nadirclaw.routing import MODEL_REGISTRY
        from nadirclaw.settings import Settings

        s = Settings()
        # Simulate what the endpoint builds
        for model_id in s.tier_models:
            registry_info = MODEL_REGISTRY.get(model_id, {})
            ctx = registry_info.get("context_window", 128_000)
            assert ctx > 0, f"{model_id} has no context window"
            assert isinstance(ctx, int)

        # GLM-5 must report 1M, not a default
        glm5_info = MODEL_REGISTRY["openai/zai.glm-5"]
        assert glm5_info["context_window"] == 1_000_000
