from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Extra, Field
from langchain.embeddings.base import Embeddings

DEFAULT_MODEL_NAME = "bert-base-chinese"

class BertEmbeddings(BaseModel, Embeddings):
    """Reimplementation of HuggingFaceEmbeddings to use Bert instead of SentenceTransformer.

    Example:
        .. code-block:: python

            import BertEmbeddings

            model_name = "bert-base-chinese"
            model_kwargs = {'device': 'cpu'}
            hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    """

    client: Any  #: :meta private:
    model_name: str = DEFAULT_MODEL_NAME
    """Model name to use."""
    cache_folder: Optional[str] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass when calling the `encode` method of the model."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        try:
            import transformers

        except ImportError as exc:
            raise ValueError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            ) from exc

        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        from transformers import BertConfig

        bert_config = BertConfig(max_position_embeddings=4096)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, config=bert_config)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, config=bert_config)
        self.client = pipeline("feature-extraction", model=model, tokenizer=tokenizer)


    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = [self.client(text, return_tensors=False, **self.encode_kwargs) for text in texts]
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.client(text, return_tensors=False, **self.encode_kwargs)
        return embedding

