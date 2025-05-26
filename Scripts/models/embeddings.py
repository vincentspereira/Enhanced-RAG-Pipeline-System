from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import numpy as np
from transformers import AutoModel, AutoTokenizer

class EmbeddingModel(ABC):
    """Base class for embedding models."""
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text into embeddings."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimension of the embeddings."""
        pass

class HuggingFaceEmbedding(EmbeddingModel):
    """Wrapper for HuggingFace embedding models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize and get model outputs
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                              return_tensors="pt", max_length=512)
        outputs = self.model(**inputs)
        
        # Use mean pooling of last hidden states
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.detach().numpy()
    
    def get_dimension(self) -> int:
        return self.model.config.hidden_size

class EmbeddingModelRegistry:
    """Registry for managing different embedding models."""
    
    def __init__(self):
        self._models: Dict[str, EmbeddingModel] = {}
        
    def register_model(self, name: str, model: EmbeddingModel):
        """Register a new embedding model."""
        self._models[name] = model
        
    def get_model(self, name: str) -> EmbeddingModel:
        """Get a registered model by name."""
        if name not in self._models:
            raise KeyError(f"Model {name} not found in registry")
        return self._models[name]
        
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())

# Create global registry instance
model_registry = EmbeddingModelRegistry()

# Register default models
model_registry.register_model(
    "snowflake-arctic-embed2",
    HuggingFaceEmbedding("Snowflake-Labs/arctic-embed2")
)
