
from pydantic import BaseModel, model_validator
from supabase import Client

class SupabaseVectorConfig(BaseModel):
    api_key: str
    url: str
    dimension: int

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, values: dict) -> dict:
        if not values["api_key"]:
            raise ValueError("config Supabase API_KEY is required")
        if not values["url"]:
            raise ValueError("config Supabase URL is required")
        if not values["dimension"]:
            raise ValueError("config VECTOR DIMENSION is required")
        return values


SQL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {table_name} (
    id UUID PRIMARY KEY,
    text TEXT NOT NULL,
    meta JSONB NOT NULL,
    embedding vector({dimension}) NOT NULL
) using heap;
"""

FUNC_SQL_OF_EXECUTE_SQL = """
CREATE OR REPLACE FUNCTION execute_sql(sql TEXT)
RETURNS VOID AS $$
BEGIN
    EXECUTE sql;
END;
$$ LANGUAGE plpgsql;
"""


class SupabaseVector():
    def __init__(self, collection_name: str, config: SupabaseVectorConfig):
        # super().__init__(collection_name)
        self.table_name = f"v_{collection_name}"
        self.client = Client(supabase_url=config.url, supabase_key=config.api_key)
        self._create_table_if_not_exists(dimension=config.dimension)

    def _create_table_if_not_exists(self, dimension: int):
        create_table_sql = SQL_CREATE_TABLE.format(table_name=self.table_name, dimension=dimension)
        print(create_table_sql)
        check_response = self.client.rpc("execute_sql", {"sql": create_table_sql}).execute()
        print(check_response)
        # check_response = self.client.rpc('sql', {'query': create_table_sql}).execute()
        print("create table")

        # 返回结果的 True 或 False 表明表是否存在
        if check_response.data and check_response.data[0]["exists"]:
            print(f"Table '{self.table_name}' exists.")
        else:
            print(f"Failed to create or verify table '{self.table_name}'.")

s = SupabaseVector(
            collection_name="vector_index_7f8bc509_096f_49b1_80cb_28207d6b364c_node",
            config=SupabaseVectorConfig(
                api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyAgCiAgICAicm9sZSI6ICJhbm9uIiwKICAgICJpc3MiOiAic3VwYWJhc2UtZGVtbyIsCiAgICAiaWF0IjogMTY0MTc2OTIwMCwKICAgICJleHAiOiAxNzk5NTM1NjAwCn0.dc_X5iR_VP_qT0zsiyj_I_OZ2T9FtRU2BBNWN8Bu4GE",
                url="http://172.28.86.41:8000",
                dimension="1024"
            ),
        )
