import pytest
from unittest.mock import patch, MagicMock

# Ensure Scripts directory is in path for imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Scripts')))

from llm.service import HuggingFaceLLM, LLMRegistry, LLMService

# --- Fixtures ---

@pytest.fixture
@patch('transformers.AutoModelForCausalLM.from_pretrained')
@patch('transformers.AutoTokenizer.from_pretrained')
def mock_hf_llm(mock_tokenizer_load, mock_model_load):
    # Mock the tokenizer and model instances
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 50256 # A common EOS token ID

    mock_model = MagicMock()

    # Configure the mocks to be returned by from_pretrained
    mock_tokenizer_load.return_value = mock_tokenizer
    mock_model_load.return_value = mock_model

    llm = HuggingFaceLLM(model_name="distilgpt2") # Use a real (but small) model name for config loading
    # Replace the loaded model and tokenizer with our mocks for testing behavior
    llm.tokenizer = mock_tokenizer
    llm.model = mock_model
    return llm

@pytest.fixture
def llm_registry():
    return LLMRegistry()

# --- Tests for HuggingFaceLLM ---

def test_hf_llm_instantiation(mock_hf_llm: HuggingFaceLLM):
    assert mock_hf_llm is not None
    assert mock_hf_llm.tokenizer is not None
    assert mock_hf_llm.model is not None
    # Check that from_pretrained was called for both model and tokenizer
    # The mock_hf_llm fixture itself uses these patches.
    # We can assert they were called during fixture setup.
    # This requires accessing the original patch objects if we passed them into the fixture.
    # Easier to assert on the mock_hf_llm.model.to being called if device matters.
    mock_hf_llm.model.to.assert_called_once_with(mock_hf_llm.device)


def test_hf_llm_generate(mock_hf_llm: HuggingFaceLLM):
    prompt = "Hello, world!"
    expected_output_text = "Hello, world! This is a test."

    # Mock tokenizer's behavior
    mock_inputs = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    mock_hf_llm.tokenizer.return_value = mock_inputs

    # Mock model's behavior
    # The model.generate returns token IDs, so we mock that.
    # Let's say our mock_tokenizer.decode will turn these IDs into expected_output_text
    mock_output_ids = MagicMock()
    mock_hf_llm.model.generate.return_value = mock_output_ids

    # Mock tokenizer.decode
    mock_hf_llm.tokenizer.decode.return_value = expected_output_text

    response = mock_hf_llm.generate(prompt, max_length=50)

    assert response == expected_output_text
    mock_hf_llm.tokenizer.assert_called_once_with(prompt, return_tensors="pt")
    mock_inputs["input_ids"].to.assert_called_once_with(mock_hf_llm.device) # Check input tensor moved to device
    mock_hf_llm.model.generate.assert_called_once_with(
        **mock_inputs,
        max_length=50,
        temperature=0.7, # Default temperature
        pad_token_id=mock_hf_llm.tokenizer.eos_token_id
    )
    mock_hf_llm.tokenizer.decode.assert_called_once_with(mock_output_ids[0], skip_special_tokens=True)


def test_hf_llm_batch_generate(mock_hf_llm: HuggingFaceLLM):
    prompts = ["Prompt 1", "Prompt 2"]
    expected_outputs = ["Output 1", "Output 2"]

    # We will mock the 'generate' method of the HuggingFaceLLM instance itself for simplicity in batch.
    # This means generate() needs to be an instance method that can be patched.

    # Configure the mock for generate to return different values on subsequent calls
    mock_hf_llm.generate = MagicMock(side_effect=expected_outputs)

    responses = mock_hf_llm.batch_generate(prompts, max_length=30)

    assert responses == expected_outputs
    assert mock_hf_llm.generate.call_count == len(prompts)
    mock_hf_llm.generate.assert_any_call(prompts[0], max_length=30, temperature=0.7)
    mock_hf_llm.generate.assert_any_call(prompts[1], max_length=30, temperature=0.7)


# --- Tests for LLMRegistry ---

def test_llm_registry_register_and_get(llm_registry: LLMRegistry, mock_hf_llm: HuggingFaceLLM):
    llm_registry.register_model("test_model", mock_hf_llm)
    retrieved_model = llm_registry.get_model("test_model")
    assert retrieved_model == mock_hf_llm

def test_llm_registry_get_non_existent(llm_registry: LLMRegistry):
    with pytest.raises(KeyError):
        llm_registry.get_model("non_existent_model")

def test_llm_registry_list_models(llm_registry: LLMRegistry, mock_hf_llm: HuggingFaceLLM):
    assert llm_registry.list_models() == []
    llm_registry.register_model("model_a", mock_hf_llm)
    llm_registry.register_model("model_b", MagicMock(spec=LLMService)) # Another mock LLM

    model_list = llm_registry.list_models()
    assert len(model_list) == 2
    assert "model_a" in model_list
    assert "model_b" in model_list

```
