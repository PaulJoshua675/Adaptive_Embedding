from unittest.mock import MagicMock

from openai import OpenAIError

from src.llm import (
    HF_BASE_URL,
    HF_MODEL,
    HF_MODELS,
    GROQ_MODELS,
    SimpleLLM,
    generate_response,
    generate_local_fallback,
    query_groq_inference_api,
    query_huggingface_inference_api,
)


def test_simple_llm_generate_delegates_to_generate_response(monkeypatch):
    requested_prompts = []
    monkeypatch.setattr(
        "src.llm.generate_response",
        lambda prompt: requested_prompts.append(prompt) or "Generated response",
    )

    assert SimpleLLM().generate("Test prompt") == "Generated response"
    assert requested_prompts == ["Test prompt"]


def test_huggingface_inference_uses_token_and_primary_model(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    assert HF_MODELS == (
        "google/gemma-2-2b-it",
        "HuggingFaceH4/zephyr-7b-beta",
    )
    assert HF_MODEL == HF_MODELS[0]
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Hugging Face response"))
    ]
    openai_client = MagicMock(return_value=client)
    monkeypatch.setattr("src.llm.OpenAI", openai_client)

    assert query_huggingface_inference_api("Test prompt") == "Hugging Face response"
    openai_client.assert_called_once_with(
        api_key="hf-test-token",
        base_url=HF_BASE_URL,
    )
    client.chat.completions.create.assert_called_once_with(
        model=f"{HF_MODEL}:fastest",
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=350,
        temperature=0.3,
    )


def test_huggingface_timeout_is_logged_and_returns_empty_response(monkeypatch, caplog):
    from openai import APITimeoutError

    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    client = MagicMock()
    client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
    monkeypatch.setattr("src.llm.OpenAI", MagicMock(return_value=client))

    assert query_huggingface_inference_api("Test prompt") == ""
    assert "Hugging Face inference failed" in caplog.text


def test_missing_huggingface_token_skips_request(monkeypatch, caplog):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        "src.llm.OpenAI",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("client should not initialize")),
    )

    assert query_huggingface_inference_api("Test prompt") == ""
    assert "HF_TOKEN is missing" in caplog.text


def test_generate_response_prefers_huggingface_and_falls_back_to_groq(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setattr("src.llm.query_huggingface_inference_api", lambda prompt: "HF response")
    monkeypatch.setattr(
        "src.llm.query_groq_inference_api",
        lambda prompt: (_ for _ in ()).throw(AssertionError("Groq should not be called")),
    )

    assert generate_response("Test prompt") == "HF response"

    monkeypatch.setattr("src.llm.query_huggingface_inference_api", lambda prompt: "")
    monkeypatch.setattr("src.llm.query_groq_inference_api", lambda prompt: "Groq fallback")
    assert generate_response("Test prompt") == "Groq fallback"


def test_huggingface_model_lookup_failure_falls_back_to_groq(monkeypatch, caplog):
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    client = MagicMock()
    client.chat.completions.create.side_effect = OpenAIError("Model not found")
    monkeypatch.setattr("src.llm.OpenAI", MagicMock(return_value=client))
    monkeypatch.setattr("src.llm.query_groq_inference_api", lambda prompt: "Groq fallback")

    assert generate_response("Test prompt") == "Groq fallback"
    assert "Model not found" in caplog.text


def test_huggingface_rejected_model_advances_to_next_candidate(monkeypatch, caplog):
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    rejected = OpenAIError("Invalid request")
    rejected.status_code = 400
    rejected.type = "invalid_request_error"
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        rejected,
        MagicMock(choices=[MagicMock(message=MagicMock(content="Fallback model response"))]),
    ]
    monkeypatch.setattr("src.llm.OpenAI", MagicMock(return_value=client))

    assert query_huggingface_inference_api("Test prompt") == "Fallback model response"
    assert [call.kwargs["model"] for call in client.chat.completions.create.call_args_list] == [
        f"{HF_MODELS[0]}:fastest",
        f"{HF_MODELS[1]}:fastest",
    ]
    assert "trying next model" in caplog.text


def test_huggingface_timeout_uses_extractive_fallback_without_loading_local_model(monkeypatch):
    from openai import APITimeoutError

    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = MagicMock()
    client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
    monkeypatch.setattr("src.llm.OpenAI", MagicMock(return_value=client))
    llm = SimpleLLM()
    llm._get_generator = lambda: (_ for _ in ()).throw(
        AssertionError("local weights should not be loaded after hosted failure")
    )

    answer = llm.generate_answer(
        "What does retrieval use?",
        [{"text": "Retrieval uses indexed passages to ground answers."}],
    )

    assert "indexed passages" in answer


def test_groq_inference_uses_openai_client_and_primary_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-token")
    assert GROQ_MODELS == (
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
    )
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
        model=GROQ_MODELS[0],
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=350,
        temperature=0.3,
    )


def test_groq_missing_model_advances_to_next_candidate(monkeypatch, caplog):
    monkeypatch.setenv("GROQ_API_KEY", "test-token")
    rejected = OpenAIError("Model not found")
    rejected.status_code = 404
    rejected.code = "model_not_found"
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        rejected,
        MagicMock(choices=[MagicMock(message=MagicMock(content="Fallback model response"))]),
    ]
    monkeypatch.setattr("src.llm.OpenAI", MagicMock(return_value=client))

    assert query_groq_inference_api("Test prompt") == "Fallback model response"
    assert [call.kwargs["model"] for call in client.chat.completions.create.call_args_list] == [
        GROQ_MODELS[0],
        GROQ_MODELS[1],
    ]
    assert "trying next model" in caplog.text


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
    assert kwargs["max_new_tokens"] == 350
    assert kwargs["pad_token_id"] == 50256


def test_hosted_generation_streams_tokens_with_bounded_output(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    tokens = ("First ", "tokens")
    events = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=token))]
        )
        for token in tokens
    ]
    client = MagicMock()
    client.chat.completions.create.return_value = iter(events)
    monkeypatch.setattr("src.llm.OpenAI", MagicMock(return_value=client))

    output = list(
        SimpleLLM().generate_answer_stream(
            "What does retrieval use?",
            [{"text": "Retrieval uses indexed passages to ground answers."}],
        )
    )

    assert output == list(tokens)
    client.chat.completions.create.assert_called_once()
    assert client.chat.completions.create.call_args.kwargs["stream"] is True
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 350
