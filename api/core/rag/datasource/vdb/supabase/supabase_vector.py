from typing import Any

from core.rag.datasource.vdb.vector_base import BaseVector
from core.rag.datasource.vdb.vector_factory import AbstractVectorFactory
from core.rag.models.document import Document
from models.dataset import Dataset
from core.rag.datasource.entity.embedding import Embeddings
from core.rag.datasource.vdb.vector_type import VectorType
from pydantic import BaseModel, model_validator
import json
from configs import dify_config
from supabase import Client


class SupabaseVectorConfig(BaseModel):
    api_key: str
    url: str

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, values: dict) -> dict:
        if not values["api_key"]:
            raise ValueError("config Supabase API_KEY is required")
        if not values["url"]:
            raise ValueError("config Supabase URL is required")
        return values


SQL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {table_name} (
    id UUID PRIMARY KEY,
    text TEXT NOT NULL,
    meta JSONB NOT NULL,
    embedding vector({dimension}) NOT NULL
) using heap;
"""


class SupabaseVector(BaseVector):
    def __init__(self, collection_name: str, config: SupabaseVectorConfig):
        super().__init__(collection_name)
        self.table_name = f"embedding_{collection_name}"
        self.client = Client(supabase_url=config.url, supabase_key=config.api_key)

    def create(self, texts: list[Document], embeddings: list[list[float]], **kwargs):
        pass

    def add_texts(self, documents: list[Document], embeddings: list[list[float]], **kwargs):
        pass

    def text_exists(self, id: str) -> bool:
        pass

    def delete_by_ids(self, ids: list[str]) -> None:
        pass

    def delete_by_metadata_field(self, key: str, value: str) -> None:
        pass

    def search_by_vector(self, query_vector: list[float], **kwargs: Any) -> list[Document]:
        pass

    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        pass

    def delete(self) -> None:
        drop_table_sql = f"DROP TABLE IF EXISTS {self.table_name};"
        self.client.rpc("execute_sql", {"sql": drop_table_sql}).execute()

    def get_type(self) -> str:
        return VectorType.SUPABASE


class SupabaseVectorFactory(AbstractVectorFactory):
    def init_vector(self, dataset: Dataset, attributes: list, embeddings: Embeddings) -> SupabaseVector:
        if dataset.index_struct_dict:
            class_prefix: str = dataset.index_struct_dict["vector_store"]["class_prefix"]
            collection_name = class_prefix
        else:
            dataset_id = dataset.id
            collection_name = Dataset.gen_collection_name_by_id(dataset_id)
            dataset.index_struct = json.dumps(self.gen_index_struct_dict(VectorType.PGVECTOR, collection_name))

        return SupabaseVector(
            collection_name=collection_name,
            config=SupabaseVectorConfig(
                api_key=dify_config.SUPABASE_VECTOR_API_KEY,
                url=dify_config.SUPABASE_VECTOR_URL,
            ),
        )
