# pylint:disable=all

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

class Embedding:
    def __init__(self, api_key: str = None):
        """
        Initializes the Embedding object with Pinecone and OpenAI clients.

        :param api_key: API key for the OpenAI service.
        :param model_name: The name of the model to use for generating embeddings.
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.embedding_model_name = "text-embedding-3-small"

    def generate(self, content, dimensions=None):
        """
        Generates an embedding for the given content using the specified model.

        :param content: The text content to generate an embedding for.
        :return: A list representing the generated embedding.
        """
        content = content.replace("\n", " ").strip()
        res = self.client.embeddings.create(
            input=[content], model=self.embedding_model_name,
            dimensions=dimensions if dimensions else 1536
        )
        embed = res.data[
            0
        ].embedding  # Assuming the response contains a list of embeddings
        return embed

    def generate_multiple(self, contents):
        """
        Generates embeddings for multiple pieces of content using the specified model.

        :param contents: A list of text content to generate embeddings for.
        :return: A list of embeddings corresponding to the input content.
        """
        contents = [content.replace("\n", " ").strip() for content in contents]
        res = self.client.embeddings.create(
            input=contents, model=self.embedding_model_name
        )
        embeddings = [item.embedding for item in res.data]
        return embeddings

