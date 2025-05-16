from typing import Tuple
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
import os


def get_default_llm(
    temperature: float = 0.1, use_most_competent_llm=False, max_tokens=512
) -> Tuple[BaseChatModel, str]:
    """
    Get the THL model for chat-based interaction.

    Args:
        temperature (float): The temperature parameter for controlling the randomness of the model's output. Default is 0.1.
        use_most_competent_llm (bool): Whether to use the most competent LLM model. Default is False.

    Returns:
        Tuple[BaseChatModel, str]: A tuple containing the THL chat model and the model name.

    """
    api_base = os.getenv("LARGE_CUSTOM_LLM_URL") if use_most_competent_llm else os.getenv("DEFAULT_CUSTOM_LLM_URL")
    model_name = "chat-default" if use_most_competent_llm else "chat-large"
    return (
        ChatOpenAI(
            model=model_name,
            openai_api_base=api_base,
            openai_api_key="-",
            temperature=temperature,
            max_tokens=max_tokens,  # Embedding models are trained on 512 sequence length, so we use this as a max output length for chat responses.
            model_kwargs={"seed": 42},
        ),
        model_name,
    )


def get_mistral_model(
    mistral_api_key: str = None, temperature: float = 0.1, use_most_competent_llm=False, max_tokens=512
) -> Tuple[BaseChatModel, str]:
    """
    Get the Mistral chat model.

    Args:
        mistral_api_key (str, optional): API key for the Mistral service. Defaults to None.
        temperature (float, optional): The temperature parameter controls the randomness of the model's output.
            Higher values (e.g., 0.8) make the output more random, while lower values (e.g., 0.2) make it more deterministic.
            Defaults to 0.1.
        use_most_competent_llm (bool, optional): Whether to use the most competent LLM model or the small one.
            Defaults to False.

    Returns:
        Tuple[BaseChatModel, str]: A tuple containing the Mistral chat model instance and the model name.

    """
    model_name = "mistral-large" if use_most_competent_llm else "mistral-small"
    return (
        ChatMistralAI(
            model=model_name,
            mistral_api_key=mistral_api_key,
            temperature=temperature,
            random_seed=42,  # We use a seed of 42 to get reproducible results.
            max_tokens=max_tokens,
        ),
        model_name,
    )


def get_openai_model(
    openai_api_key: str = None, temperature: float = 0.1, use_most_competent_llm=False, max_tokens=512
) -> Tuple[BaseChatModel, str]:
    """
    Returns an instance of the ChatOpenAI model and the model name.

    Parameters:
        openai_api_key (str): The API key for OpenAI.
        temperature (float): The temperature parameter for generating responses.
        use_most_competent_llm (bool): Whether to use the most competent language model.

    Returns:
        Tuple[BaseChatModel, str]: A tuple containing the ChatOpenAI model instance and the model name.
    """
    model_name = (
        "gpt-4o" if use_most_competent_llm else "gpt-4o-mini"
    )
    return (
        ChatOpenAI(
            model=model_name,
            openai_api_key=openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs={"seed": 42},
        ),
        model_name,
    )


def get_llm(
    openai_api_key: str = None,
    mistral_api_key: str = None,
    temperature: float = 0.1,
    use_most_competent_llm=False,
    max_tokens=512,
) -> Tuple[BaseChatModel, str]:
    """
    Retrieves the appropriate language model for chat responses.

    Args:
        openai_api_key (str): The OpenAI API key.
        mistral_api_key (str): The Mistral API key.
        temperature (float): The temperature for the language model.
        use_most_competent_llm (bool): Whether to use the most competent language model.

    Returns:
        tuple: A tuple containing the language model and its name.
    """

    if mistral_api_key:
        model, model_name = get_mistral_model(
            mistral_api_key, temperature, use_most_competent_llm, max_tokens
        )
        fallback, fallback_name = get_default_llm(temperature, use_most_competent_llm, max_tokens)
        return model.with_fallbacks([fallback]), model_name
    if openai_api_key:
        model, model_name = get_openai_model(
            openai_api_key, temperature, use_most_competent_llm, max_tokens
        )
        fallback, fallback_name = get_default_llm(temperature, use_most_competent_llm, max_tokens)
        return model.with_fallbacks([fallback]), model_name

    model, model_name = get_default_llm(temperature, use_most_competent_llm, max_tokens)
    if os.getenv("MISTRAL_API_KEY"):
        fallback, fallback_name = get_mistral_model(temperature, use_most_competent_llm, max_tokens)
        return model.with_fallbacks([fallback]), model_name

    return model, model_name
