# helper.py

import os
import json
import logging
import openai
import requests
from dotenv import load_dotenv

# Configure the logger for "helper"
# All logs from these helper classes will be prefixed with "helper" in the log file
logger = logging.getLogger("helper")

###################################
# 1. FlattenHelper
###################################
class FlattenHelper:
    """
    A class to handle flattening (serialization) of different field types into textual representation
    based on the "fieldType" property.
    """

    @staticmethod
    def flatten_checkbox_field(field_name: str, value: bool) -> str:
        status = "Checked" if value else "Unchecked"
        return f"{field_name}: {status}"

    @staticmethod
    def flatten_text_field(field_name: str, value: str) -> str:
        return f"{field_name}: {value}"

    @staticmethod
    def flatten_number_field(field_name: str, value) -> str:
        return f"{field_name}: {value}"

    @staticmethod
    def flatten_password_field(field_name: str) -> str:
        return f"{field_name}: [REDACTED]"

    @staticmethod
    def flatten_date_field(field_name: str, value: str) -> str:
        return f"{field_name}: {value}"

    @staticmethod
    def flatten_address_field(field_name: str, address_value: dict) -> str:
        line1 = address_value.get("line1", "")
        line2 = address_value.get("line2", "")
        city = address_value.get("city", "")
        state = address_value.get("state", "")
        zip_code = address_value.get("zip", "")
        parts = [line1, line2, city, state, zip_code]
        joined = ", ".join(p for p in parts if p)
        return f"{field_name}: {joined}"

    @staticmethod
    def flatten_table_field(field_name: str, rows: list) -> str:
        lines = [f"{field_name}:"]
        for i, row in enumerate(rows, start=1):
            row_content = ", ".join(f"{k}={v}" for k, v in row.items())
            lines.append(f"  Row{i}: [{row_content}]")
        return "\n".join(lines)

    @staticmethod
    def flatten_signature_field(field_name: str, value: dict) -> str:
        ts = value.get("timestamp", "")
        file_ref = value.get("fileRef", "")
        text = f"{field_name}: Signature provided"
        if ts:
            text += f" at {ts}"
        if file_ref:
            text += f", file: {file_ref}"
        return text

    @staticmethod
    def flatten_location_field(field_name: str, value: dict) -> str:
        lat = value.get("lat")
        lon = value.get("lon")
        if lat is not None and lon is not None:
            return f"{field_name}: (Lat={lat}, Lon={lon})"
        return f"{field_name}: [Location data]"

    @staticmethod
    def flatten_field(field_data: dict) -> str:
        """
        Decide which flatten_* method to call based on fieldType.
        """
        field_type = field_data.get("fieldType")
        field_name = field_data.get("fieldName", "UnknownField")
        value = field_data.get("value")

        if field_type == "checkbox":
            return FlattenHelper.flatten_checkbox_field(field_name, bool(value))
        elif field_type == "text":
            return FlattenHelper.flatten_text_field(field_name, str(value))
        elif field_type == "number":
            return FlattenHelper.flatten_number_field(field_name, value)
        elif field_type == "password":
            return FlattenHelper.flatten_password_field(field_name)
        elif field_type == "date":
            return FlattenHelper.flatten_date_field(field_name, str(value))
        elif field_type == "address":
            if isinstance(value, dict):
                return FlattenHelper.flatten_address_field(field_name, value)
            return f"{field_name}: [Invalid address data]"
        elif field_type == "table":
            if isinstance(value, list):
                return FlattenHelper.flatten_table_field(field_name, value)
            return f"{field_name}: [Invalid table data]"
        elif field_type == "signature":
            if isinstance(value, dict):
                return FlattenHelper.flatten_signature_field(field_name, value)
            return f"{field_name}: [Signature provided]"
        elif field_type == "location":
            if isinstance(value, dict):
                return FlattenHelper.flatten_location_field(field_name, value)
            return f"{field_name}: [Location data]"
        return f"{field_name}: {value}"

    @staticmethod
    def flatten_submission(submission_doc: dict) -> str:
        """
        Flatten an entire submission into a text block.
        """
        lines = []
        submission_id = submission_doc.get("_id", "")
        folder_id = submission_doc.get("folder_id", "")
        document_id = submission_doc.get("document_id", "")
        user_id = submission_doc.get("user_id", "")
        timestamp = submission_doc.get("timestamp", "")

        lines.append(f"Submission (Result ID: {submission_id})")
        lines.append(f"For Document: {document_id} in Folder: {folder_id}.")
        lines.append(f"Owned by user: {user_id}.")
        lines.append(f"Timestamp: {timestamp}.")

        field_values = submission_doc.get("fieldValues", [])
        lines.append("Field Values:")

        for field_entry in field_values:
            flattened_text = FlattenHelper.flatten_field(field_entry)
            lines.append("  " + flattened_text)

        return "\n".join(lines)

###################################
# 2. AzureSearchRESTHelper
###################################
class AzureSearchRESTHelper:
    def __init__(self, env_file: str = "local.env", index_definition_file: str = "index_definition.json"):
        load_dotenv(dotenv_path=env_file)
        
        self.endpoint = os.getenv("ACS_ENDPOINT")
        self.api_key = os.getenv("ACS_API_KEY")
        self.index_name = os.getenv("ACS_INDEX_NAME", "knowledge-index")
        self.api_version = "2024-07-01"
        self.index_definition_file = index_definition_file
        
        if not all([self.endpoint, self.api_key]):
            error_msg = "Azure Cognitive Search configuration is incomplete. Check your .env."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Read index definition
        try:
            with open(self.index_definition_file, "r", encoding="utf-8") as f:
                self.index_definition = json.load(f)
            logger.info(f"Loaded index definition from '{self.index_definition_file}'.")
        except FileNotFoundError:
            error_msg = f"Index definition file '{self.index_definition_file}' not found."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Error decoding JSON from '{self.index_definition_file}': {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def create_or_update_index(self):
        url = f"{self.endpoint}/indexes/{self.index_name}?api-version={self.api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        response = requests.put(url, headers=headers, json=self.index_definition)
        
        if response.status_code in [200, 201]:
            logger.info(f"Index '{self.index_name}' created/updated successfully.")
            print(f"Index '{self.index_name}' created/updated successfully.")
        else:
            try:
                error_message = response.json().get("error", {}).get("message", response.text)
            except json.JSONDecodeError:
                error_message = response.text
            logger.error(f"Failed to create/update index '{self.index_name}': {error_message}")
            raise Exception(f"Failed to create/update index '{self.index_name}': {error_message}")
    
    def delete_index(self):
        url = f"{self.endpoint}/indexes/{self.index_name}?api-version={self.api_version}"
        headers = {"api-key": self.api_key}
        response = requests.delete(url, headers=headers)
        
        if response.status_code == 204:
            logger.info(f"Index '{self.index_name}' deleted successfully.")
            print(f"Index '{self.index_name}' deleted successfully.")
        elif response.status_code == 404:
            logger.warning(f"Index '{self.index_name}' does not exist.")
            print(f"Index '{self.index_name}' does not exist.")
        else:
            try:
                error_message = response.json().get("error", {}).get("message", response.text)
            except json.JSONDecodeError:
                error_message = response.text
            logger.error(f"Failed to delete index '{self.index_name}': {error_message}")
            raise Exception(f"Failed to delete index '{self.index_name}': {error_message}")

###################################
# 3. EmbeddingHelper
###################################
class EmbeddingHelper:
    def __init__(self, env_file: str = "local.env"):
        load_dotenv(dotenv_path=env_file)
        
        self.api_type = "azure"
        self.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-03-15-preview")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.engine = os.getenv("AZURE_OPENAI_ENGINE", "text-embedding-ada-002")
        
        if not all([self.api_base, self.api_key, self.engine]):
            raise ValueError("Azure OpenAI configuration is incomplete. Check your .env.")
        
        openai.api_type = self.api_type
        openai.api_base = self.api_base
        openai.api_version = self.api_version
        openai.api_key = self.api_key

    def get_embedding(self, text: str) -> list:
        try:
            response = openai.Embedding.create(input=[text], engine=self.engine)
            embedding = response["data"][0]["embedding"]
            logger.info(f"Generated embedding for text length {len(text)}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

###################################
# 4. IndexingHelper
###################################
class IndexingHelper:
    def __init__(self, env_file: str = "local.env"):
        load_dotenv(dotenv_path=env_file)
        
        self.endpoint = os.getenv("ACS_ENDPOINT")
        self.api_key = os.getenv("ACS_API_KEY")
        self.index_name = os.getenv("ACS_INDEX_NAME")
        self.api_version = "2024-07-01"

        if not all([self.endpoint, self.api_key, self.index_name]):
            raise ValueError("Azure Cognitive Search config incomplete. Check ACS_ENDPOINT, ACS_API_KEY, ACS_INDEX_NAME.")

    def index_document(
        self,
        doc_id: str,
        user_id: str,
        folder_id: str,
        document_id: str,
        doc_type: str,
        content_text: str,
        field_values: str,  # JSON string
        embedding_vector: list,
        metadata: str = ""
    ) -> None:
        url = f"{self.endpoint}/indexes/{self.index_name}/docs/index?api-version={self.api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        search_doc = {
            "@search.action": "upload",
            "id": doc_id,
            "userId": user_id,
            "folderId": folder_id,
            "documentId": document_id,
            "type": doc_type,
            "content": content_text,
            "fieldValues": field_values,
            "contentVector": embedding_vector,
            "metadata": metadata
        }
        payload = {"value": [search_doc]}

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            logger.info(f"Indexed document ID: {doc_id}")
            print(f"Successfully indexed document ID: {doc_id}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to index document ID: {doc_id}. Error: {e}")
            raise

###################################
# 5. SearchHelper
###################################
class SearchHelper:
    def __init__(self, env_file: str = "local.env", index_definition_file: str = "index_definition.json"):
        load_dotenv(dotenv_path=env_file)
        self.endpoint = os.getenv("ACS_ENDPOINT")
        self.api_key = os.getenv("ACS_API_KEY")
        self.index_name = os.getenv("ACS_INDEX_NAME", "knowledge-index")
        self.api_version = os.getenv("ACS_API_VERSION", "2024-07-01")
        
        if not all([self.endpoint, self.api_key, self.index_name]):
            error_msg = "Azure Cognitive Search configuration is incomplete. Check .env."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Initialized SearchHelper.")
    
    def vector_search(self, embedding: list, top_k: int = 5, user_id: str = None, additional_filters: str = None) -> dict:
        select_fields = "id,userId,folderId,documentId,type,content,fieldValues,metadata"
        
        url = f"{self.endpoint}/indexes('{self.index_name}')/docs/search.post.search?api-version={self.api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        body = {
            "search": "*",
            "vectorQueries": [
                {
                    "kind": "vector",
                    "vector": embedding,
                    "fields": "contentVector",
                    "k": top_k,
                    "exhaustive": False
                }
            ],
            "select": select_fields,
            "top": top_k
        }
        
        filters = []
        if user_id:
            filters.append(f"userId eq '{user_id}'")
        if additional_filters:
            filters.append(additional_filters)
        if filters:
            body["filter"] = " and ".join(filters)
            
        try:
            response = requests.post(url, headers=headers, json=body)
            response.raise_for_status()
            results = response.json()
            logger.info(f"Vector search successful. Retrieved {len(results.get('value', []))} docs.")
            return results
        except requests.exceptions.HTTPError as http_err:
            try:
                error_message = response.json().get("error", {}).get("message", str(http_err))
            except json.JSONDecodeError:
                error_message = str(http_err)
            logger.error(f"HTTP error during vector search: {error_message}")
            raise Exception(f"HTTP error during vector search: {error_message}")
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            raise
