import pandas as pd
import streamlit as st
from typing import Optional
from google.cloud import bigquery
from google.oauth2 import service_account
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from src.code_interpreter import CodeInterpreterClient


class SqlTableInfoInput(BaseModel):
    table_name: str = Field()


class ExecSqlInput(BaseModel):
    query: str = Field()
    limit: Optional[int] = Field(default=None)


class BigQueryClient:
    def __init__(
        self,
        code_interpreter: CodeInterpreterClient,
        project_id: str = "youtube-api-client-480202",  ## 자신의 Google Cloud 프로젝트 ID로 변경
        dataset_project_id: str = "bigquery-public-data",  ## Google 공공 데이터셋
        dataset_id: str = "google_trends",  ## Google Trends 데이터
    ) -> None:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        self.client = bigquery.Client(credentials=credentials, project=project_id)
        self.dataset_project_id = dataset_project_id
        self.dataset_id = dataset_id
        self.table_names_str = self._fetch_table_names()
        self.code_interpreter = code_interpreter

    def _fetch_table_names(self) -> str:
        """사용 가능한 테이블명을 쉼표 구분 문자열로 반환"""
        query = f"""
        SELECT table_name
        FROM `{self.dataset_project_id}.{self.dataset_id}.INFORMATION_SCHEMA.TABLES`
        """
        table_names = self._exec_query(query).table_name.tolist()
        return ", ".join(table_names)

    def _exec_query(self, query: str, limit: int = None) -> pd.DataFrame:
        """SQL 실행 → Pandas DataFrame 반환"""
        if limit is not None:
            query += f"\nLIMIT {limit}"
        query_job = self.client.query(query)
        return query_job.result().to_dataframe(create_bqstorage_client=True)

    def exec_query_and_upload(self, query: str, limit: int = None) -> str:
        """SQL 실행 → 결과를 CSV로 Code Interpreter Container에 업로드"""
        try:
            df = self._exec_query(query, limit)
            csv_data = df.to_csv().encode("utf-8")
            file_name, file_path = self.code_interpreter.upload_file(csv_data)
            return f"sql:\n```\n{query}\n```\n\nsample results:\n{df.head()}\n\nfull result was uploaded with File Name: {file_name} (accessible in Code Interpreter: {file_path})"
        except Exception as e:
            return f"SQL execution failed. Error message is as follows:\n```\n{e}\n```"

    def _generate_sql_for_table_info(self, table_name: str) -> tuple:
        """테이블의 스키마 조회 SQL과 샘플 데이터 조회 SQL을 생성"""
        get_schema_sql = f"""
        SELECT
            TO_JSON_STRING(
                ARRAY_AGG(
                    STRUCT(
                        IF(is_nullable = 'YES', 'NULLABLE', 'REQUIRED'
                    ) AS mode,
                    column_name AS name,
                    data_type AS type
                )
                ORDER BY ordinal_position
            ), TRUE) AS schema
        FROM
            `{self.dataset_project_id}.{self.dataset_id}.INFORMATION_SCHEMA.COLUMNS`
        WHERE
            table_name = "{table_name}"
        """

        sample_data_sql = f"""
        SELECT
            *
        FROM
            `{self.dataset_project_id}.{self.dataset_id}.{table_name}`
        LIMIT
            3
        """
        return get_schema_sql, sample_data_sql

    def get_table_info(self, table_name: str) -> str:
        """테이블의 스키마 + 샘플 데이터(3행)를 문자열로 반환"""
        get_schema_sql, sample_data_sql = self._generate_sql_for_table_info(table_name)
        schema = self._exec_query(get_schema_sql).to_string(index=False)
        sample_data = self._exec_query(sample_data_sql).to_string(index=False)
        table_info = f"""
        ### schema
        ```
        {schema}
        ```

        ### sample_data
        ```
        {sample_data}
        ```
        """
        return table_info

    def exec_query_tool(self):
        exec_query_tool_description = f"""
        BigQuery에서 SQL 쿼리를 실행하고, 결과를 Code Interpreter Container에 CSV로 저장하는 도구.
        저장된 CSV는 Code Interpreter에서 Python으로 분석 가능.

        ## 사용 전 필수 사항
        - 반드시 `sql_table_info` 도구로 테이블 스키마를 먼저 확인할 것

        ## 쿼리 작성 규칙
        - project_id, dataset_id, table_id를 반드시 명시
        - SQL은 줄바꿈을 포함하여 가독성 있게 작성
        - 최빈값 계산 시 "Mod" 함수 사용

        ## 현재 BigQuery 정보
        - project_id: {self.dataset_project_id}
        - dataset_id: {self.dataset_id}
        - table_id: {self.table_names_str}
        """
        return StructuredTool.from_function(
            name="exec_query",
            func=self.exec_query_and_upload,
            description=exec_query_tool_description,
            args_schema=ExecSqlInput,
        )

    def get_table_info_tool(self):
        sql_table_info_tool_description = f"""
        BigQuery 테이블의 스키마와 샘플 데이터(3행)를 조회하는 도구.
        SQL 작성 전 테이블 구조를 파악할 때 사용.

        이용 가능한 테이블: {self.table_names_str}
        """
        return Tool.from_function(
            name="sql_table_info",
            func=self.get_table_info,
            description=sql_table_info_tool_description,
            args_schema=SqlTableInfoInput,
        )
