# src/providers/openai_like.py
"""
OpenAI-like provider for accessing non-OpenAI models through OpenAI-compatible APIs.

This provider uses llama_index.llms.openai_like.OpenAILike to connect to any service
that implements the OpenAI API protocol, such as:
- Alibaba DashScope (Qwen series)
- Zhipu AI (GLM series)
- Moonshot AI (Kimi)
- DeepSeek
- 01.AI (Yi series)
- Local deployments (Ollama, vLLM, etc.)
"""

from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike


def initialize_embed_model(
    embed_endpoint: str,
    embed_api_key: str,
    embed_model: str,
    **embed_extra: Any,  # noqa: ANN401
) -> BaseEmbedding:
    """
    Create OpenAI-compatible embedding model.

    This uses the standard OpenAI embedding client with a custom endpoint,
    which works with any OpenAI-compatible embedding API.

    Args:
        embed_endpoint: The API endpoint for the OpenAI-compatible API.
        embed_api_key: The API key for the service.
        embed_model: The name of the embedding model.
        embed_extra: Extra parameters for the embedding model.

    Returns:
        The initialized embed_model.

    """
    # Use OpenAIEmbedding with custom endpoint for compatibility
    # This works with any OpenAI-compatible embedding API

    # Extract known parameters to avoid conflicts
    dimensions = embed_extra.pop("dimensions", None)
    max_retries = embed_extra.pop("max_retries", 3)

    # Remove any conflicting keys that might be in embed_extra
    conflicting_keys = ["model", "model_name", "api_base", "api_key", "base_url"]
    for key in conflicting_keys:
        if key in embed_extra:
            del embed_extra[key]

    # Use a workaround for non-OpenAI models:
    # Set model to a valid OpenAI model name to bypass enum validation,
    # but use model_name parameter to specify the actual model for API calls.
    # This is the official way to use custom models with OpenAIEmbedding.
    try:
        # Try to initialize with the actual model name first
        return OpenAIEmbedding(
            model=embed_model,
            api_base=embed_endpoint,
            api_key=embed_api_key,
            dimensions=dimensions,
            max_retries=max_retries,
            **embed_extra,
        )
    except (ValueError, TypeError) as e:
        error_msg = str(e)
        if "not a valid OpenAIEmbeddingModelType" in error_msg or "not a valid OpenAI model" in error_msg:
            # If the model name is not recognized, use the model_name parameter workaround:
            # - model: A valid OpenAI model name to pass enum validation
            # - model_name: The actual model name to use in API requests (overrides model)
            import warnings

            warnings.warn(
                f"Model '{embed_model}' is not in OpenAI's list. "
                f"Using workaround with model_name parameter to specify custom model."
            )
            return OpenAIEmbedding(
                model="text-embedding-3-small",  # Valid model name to pass enum validation
                api_base=embed_endpoint,
                api_key=embed_api_key,
                dimensions=dimensions,
                max_retries=max_retries,
                model_name=embed_model,  # Official parameter to override the actual model used
                **embed_extra,
            )
        else:
            # Re-raise if it's a different error
            raise


def initialize_llm_model(
    llm_endpoint: str,
    llm_api_key: str,
    llm_model: str,
    **llm_extra: Any,  # noqa: ANN401
) -> LLM:
    """
    Create OpenAI-like LLM model for non-OpenAI providers.

    This uses OpenAILike which is specifically designed for OpenAI-compatible
    APIs from other providers. It automatically handles model validation and
    works with any service implementing the OpenAI protocol.

    Args:
        llm_endpoint: The API endpoint for the OpenAI-compatible API.
        llm_api_key: The API key for the service.
        llm_model: The name of the LLM model.
        llm_extra: Extra parameters for the LLM model.

    Returns:
        The initialized llm_model.

    Examples:
        ```python
        # Zhipu GLM
        llm = initialize_llm_model(
            llm_endpoint="https://open.bigmodel.cn/api/paas/v4",
            llm_api_key="your-glm-api-key",
            llm_model="glm-4-flash",
        )

        # Alibaba Qwen
        llm = initialize_llm_model(
            llm_endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",
            llm_api_key="your-dashscope-api-key",
            llm_model="qwen-plus",
        )

        # Moonshot Kimi
        llm = initialize_llm_model(
            llm_endpoint="https://api.moonshot.cn/v1",
            llm_api_key="your-moonshot-api-key",
            llm_model="moonshot-v1-8k",
        )
        ```

    """
    from libs.logger import logger

    # Extract context_window if provided, default to 32k for unknown models
    context_window = llm_extra.pop("context_window", 32768)

    # Remove any conflicting keys that might be in llm_extra
    conflicting_keys = ["model", "model_name", "api_base", "api_key", "base_url", "context_window"]
    for key in conflicting_keys:
        if key in llm_extra:
            logger.warning(f"Removing conflicting key '{key}' from llm_extra")
            del llm_extra[key]

    try:
        # Use OpenAILike which is designed for OpenAI-compatible APIs
        # This automatically handles model validation and works with any
        # service implementing the OpenAI protocol
        return OpenAILike(
            model=llm_model,
            api_base=llm_endpoint,
            api_key=llm_api_key,
            context_window=context_window,
            is_chat_model=True,  # Most modern LLMs are chat models
            **llm_extra,
        )
    except Exception as e:
        logger.error(f"Failed to initialize OpenAILike model: {e}", exc_info=True)
        raise

