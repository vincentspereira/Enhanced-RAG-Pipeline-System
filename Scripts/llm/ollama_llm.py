import ollama
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class OllamaLLM:
    def __init__(
        self,
        host: str = "http://localhost:11434", # Default Ollama API endpoint
        completion_model_name: str = "llama2", # Default model for completions
        embedding_model_name: Optional[str] = None, # Optional: specific model for embeddings
        request_timeout: float = 120.0 # Timeout for requests to Ollama server
    ):
        """
        Initializes the Ollama LLM client.

        Args:
            host (str): The host URL for the Ollama API.
            completion_model_name (str): The default Ollama model to use for text generation.
            embedding_model_name (Optional[str]): The Ollama model to use for generating embeddings.
                                                 If None, completion_model_name might be used or embeddings might not be supported.
            request_timeout (float): Timeout in seconds for requests to the Ollama server.
        """
        self.client = ollama.Client(host=host, timeout=request_timeout)
        self.completion_model_name = completion_model_name
        # If no specific embedding model, it's up to the user to ensure the completion model supports embeddings
        # or to handle cases where embedding generation is requested without a suitable model.
        self.embedding_model_name = embedding_model_name if embedding_model_name else completion_model_name
        self.request_timeout = request_timeout
        logger.info(f"OllamaLLM initialized with host='{host}', completion_model='{completion_model_name}', embedding_model='{self.embedding_model_name}'")

        # Verify connection and model availability (optional, can be done on first use)
        try:
            self.list_models() # A simple check to see if client can connect
            logger.info("Successfully connected to Ollama and listed models.")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama at {host} or list models during init: {e}. Ensure Ollama server is running.")

    def list_models(self) -> List[Dict[str, Any]]:
        """Lists models available via the Ollama instance."""
        try:
            models_info = self.client.list() # This is a synchronous call
            return models_info.get('models', [])
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            raise

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_length: Optional[int] = None, # Renamed from max_tokens for consistency with LLMService
        **kwargs
    ) -> str:
        """
        Generates a text completion using the specified Ollama model.
        This method is synchronous.

        Args:
            prompt (str): The prompt to send to the LLM.
            model (Optional[str]): The model to use. Defaults to self.completion_model_name.
            system_message (Optional[str]): An optional system message.
            temperature (float): The temperature for sampling.
            max_length (Optional[int]): Max tokens to generate (maps to num_predict).
            **kwargs: Additional options for ollama.chat.

        Returns:
            str: The generated text content.
        """
        target_model = model if model else self.completion_model_name
        messages = []
        if system_message:
            messages.append({'role': 'system', 'content': system_message})
        messages.append({'role': 'user', 'content': prompt})

        options = kwargs.pop('options', {})
        options['temperature'] = temperature
        if max_length is not None:
            options['num_predict'] = max_length

        logger.debug(f"Sending completion request to Ollama model '{target_model}' with prompt: '{prompt[:100]}...' and options: {options}")

        try:
            response = self.client.chat( # ollama.Client.chat is synchronous
                model=target_model,
                messages=messages,
                stream=False,
                options=options,
                timeout=self.request_timeout, # Pass along timeout
                **kwargs
            )
            logger.debug(f"Received response from Ollama model '{target_model}': {response}")

            if response and 'message' in response and 'content' in response['message']:
                return response['message']['content']
            else:
                logger.error(f"Unexpected response structure from Ollama: {response}")
                return "" # Or raise an error
        except Exception as e:
            logger.error(f"Error during Ollama completion for model '{target_model}': {e}")
            raise

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generates text for multiple prompts by calling generate for each.
        This method is synchronous.
        """
        results = []
        for prompt in prompts:
            # Pass through relevant kwargs like model, system_message, temperature, max_length
            results.append(self.generate(prompt, **kwargs))
        return results

    def generate_embeddings( # This method remains, not part of LLMService ABC directly
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs
    ) -> List[List[float]]:
        """
        Generates embeddings for a list of texts using the specified Ollama model.
        This method is synchronous.

        Args:
            texts (List[str]): A list of texts to embed.
            model (Optional[str]): The model to use. Defaults to self.embedding_model_name.
            **kwargs: Additional options for ollama.embeddings.

        Returns:
            List[List[float]]: A list of embeddings.

        Raises:
            ValueError: If no suitable embedding model is configured or available.
        """
        target_model = model if model else self.embedding_model_name
        if not target_model:
            logger.error("No embedding model specified for OllamaLLM.")
            raise ValueError("Ollama embedding model name not configured.")

        logger.debug(f"Sending embedding request to Ollama model '{target_model}' for {len(texts)} texts.")

        all_embeddings: List[List[float]] = []
        try:
            for text_content in texts:
                response = self.client.embeddings( # ollama.Client.embeddings is synchronous
                    model=target_model,
                    prompt=text_content,
                    timeout=self.request_timeout, # Pass along timeout
                    **kwargs
                )
                all_embeddings.append(response['embedding'])

            logger.debug(f"Successfully generated {len(all_embeddings)} embeddings using Ollama model '{target_model}'.")
            return all_embeddings
        except Exception as e:
            logger.error(f"Error during Ollama embedding generation for model '{target_model}': {e}")
            raise

# Example Usage (for testing purposes, typically not run directly like this)
if __name__ == '__main__':
    def test_ollama_sync():
        # Ensure Ollama server is running (e.g., `ollama serve`)
        # and the model 'llama2' (or your chosen models) are pulled.
        # ollama pull llama2
        # ollama pull mxbai-embed-large (if using this for embeddings)

        try:
            ollama_llm = OllamaLLM(completion_model_name="llama2", embedding_model_name="llama2")

            print("Available Ollama models:")
            models = ollama_llm.list_models()
            for m in models:
                print(f"- {m['name']} (Size: {m['size']//1024**2}MB, Modified: {m['modified_at']})")

            print("\n--- Testing Completion (sync) ---")
            completion_prompt = "What is the capital of France? Respond concisely."
            completion_text = ollama_llm.generate(prompt=completion_prompt, max_length=10) # Using generate now
            print(f"Prompt: {completion_prompt}")
            print(f"Completion: {completion_text}")

            print("\n--- Testing Batch Completion (sync) ---")
            batch_prompts = ["What is 1+1?", "What is the color of the sky?"]
            batch_completions = ollama_llm.batch_generate(prompts=batch_prompts, max_length=5)
            for p, c in zip(batch_prompts, batch_completions):
                print(f"Prompt: {p} -> Completion: {c}")


            print("\n--- Testing Embeddings (sync) ---")
            embedding_texts = ["Hello world", "Ollama is cool"]
            try:
                embeddings = ollama_llm.generate_embeddings(texts=embedding_texts)
                for text, emb in zip(embedding_texts, embeddings):
                    print(f"Text: {text}, Embedding (first 3 dims): {emb[:3]}..., Length: {len(emb)}")
            except ValueError as ve:
                print(f"Embedding error: {ve}")
            except Exception as e:
                print(f"An unexpected error occurred during embedding: {e}")
                logger.error("Make sure your Ollama server is running and the embedding model is pulled.")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Ensure the Ollama server is running and accessible at http://localhost:11434, and that the specified models are pulled.")

    test_ollama_sync()
