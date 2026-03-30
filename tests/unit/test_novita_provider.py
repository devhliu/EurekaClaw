"""Tests for the Novita AI LLM provider integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestNovitaAdapter:
    def test_adapter_inherits_openai_compat(self):
        from eurekaclaw.llm.novita_adapter import NovitaAdapter
        from eurekaclaw.llm.openai_compat import OpenAICompatAdapter

        assert issubclass(NovitaAdapter, OpenAICompatAdapter)

    def test_adapter_default_model(self):
        from eurekaclaw.llm.novita_adapter import NovitaAdapter

        adapter = NovitaAdapter(api_key="test-key")
        assert adapter._default_model == "moonshotai/kimi-k2.5"

    def test_adapter_custom_model(self):
        from eurekaclaw.llm.novita_adapter import NovitaAdapter

        adapter = NovitaAdapter(api_key="test-key", default_model="zai-org/glm-5")
        assert adapter._default_model == "zai-org/glm-5"


class TestNovitaConfig:
    def test_config_defines_novita_env_vars(self):
        config_text = Path("eurekaclaw/config.py").read_text()
        assert "NOVITA_API_KEY" in config_text
        assert "NOVITA_MODEL" in config_text
