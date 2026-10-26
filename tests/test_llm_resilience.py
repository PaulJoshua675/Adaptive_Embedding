from unittest.mock import MagicMock

from openai import OpenAIError

from src.llm import SimpleLLM, generate_local_fallback, query_groq_inference_api


def test_groq_inference_uses_openai_client_and_primary_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-token")
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Groq response"))
    ]
    openai_client = MagicMock(return_value=client)
    monkeypatch.setattr("src.llm.OpenAI", openai_client)

    assert query_groq_inference_api("Test prompt") == "Groq response"
    openai_client.assert_called_once_with(
        api_key="test-token",
        base_url="https://api.groq.com/openai/v1",
    )
    client.chat.completions.create.assert_called_once_with(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=512,
        temperature=0.3,
    )


def test_groq_api_errors_are_logged_and_return_empty_response(monkeypatch, caplog):
    monkeypatch.setenv("GROQ_API_KEY", "test-token")
    client = MagicMock()
    client.chat.completions.create.side_effect = OpenAIError("Rate limit exceeded")
    monkeypatch.setattr("src.llm.OpenAI", MagicMock(return_value=client))

    assert query_groq_inference_api("Test prompt") == ""
    assert "Rate limit exceeded" in caplog.text


def test_missing_groq_key_skips_request(monkeypatch, caplog):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(
        "src.llm.OpenAI",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("client should not initialize")),
    )

    assert query_groq_inference_api("Test prompt") == ""
    assert "GROQ_API_KEY is missing" in caplog.text


def test_groq_failures_use_extractive_fallback_without_loading_local_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-token")
    monkeypatch.setattr(
        "src.llm.OpenAI",
        lambda **kwargs: (_ for _ in ()).throw(OpenAIError("request timed out")),
    )
    llm = SimpleLLM()
    llm._get_generator = lambda: (_ for _ in ()).throw(
        AssertionError("local weights should not be loaded after hosted failure")
    )

    answer = llm.generate_answer(
        "What does retrieval use?",
        [{"text": "Retrieval uses indexed passages to ground answers."}],
    )

    assert "indexed passages" in answer


def test_local_fallback_bounds_prompt_and_generation():
    local_pipeline = MagicMock()
    local_pipeline.tokenizer.eos_token_id = 50256
    local_pipeline.return_value = [{"generated_text": "Safe response"}]

    result = generate_local_fallback("A" * 5000, local_pipeline)

    assert result == "Safe response"
    args, kwargs = local_pipeline.call_args
    assert len(args[0]) <= 2500
    assert "[Prompt truncated for model limits]" in args[0]
    assert kwargs["truncation"] is True
    assert kwargs["max_length"] == 1024
    assert kwargs["pad_token_id"] == 50256
