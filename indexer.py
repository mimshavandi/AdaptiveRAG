# indexer.py

import os
import json
import logging
from dotenv import load_dotenv

# Import from your unified helper file
from helpers import FlattenHelper, EmbeddingHelper, IndexingHelper

# Configure logging for "indexer"
logging.basicConfig(
    filename='application.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("indexer")

def load_env(env_file: str = "local.env"):
    load_dotenv(env_file)
    logger.info("Loaded environment variables in indexer script.")

def load_submission(file_path: str) -> dict:
    """
    Load the submission document from a JSON file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            submission = json.load(f)
        logger.info(f"Loaded submission from '{file_path}'.")
        return submission
    except FileNotFoundError:
        logger.error(f"Submission file '{file_path}' not found.")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{file_path}': {e}")
        raise

def main():
    # 1. Load env
    load_env("local.env")

    # 2. Configuration
    SUBMISSION_FILE = "form.json"

    # Step 1: Load submission
    try:
        submission = load_submission(SUBMISSION_FILE)
    except Exception as e:
        logger.error(f"Error loading submission: {e}")
        print(f"Error loading submission: {e}")
        return

    # Step 2: Flatten the submission
    try:
        text_summary = FlattenHelper.flatten_submission(submission)
        logger.info("Flattened Submission:")
        logger.info(text_summary)
        print("Flattened Submission:")
        print(text_summary)
        print("----")
    except Exception as e:
        logger.error(f"Error flattening submission: {e}")
        print(f"Error flattening submission: {e}")
        return

    # Step 3: Generate embedding
    try:
        embedder = EmbeddingHelper(env_file="local.env")
        embedding_vector = embedder.get_embedding(text_summary)
        logger.info(f"Generated embedding vector of length: {len(embedding_vector)}")
        print(f"Generated embedding vector of length: {len(embedding_vector)}")
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        print(f"Error generating embedding: {e}")
        return

    # Step 4: Index the document
    try:
        indexing_helper = IndexingHelper(env_file="local.env")
        doc_id = submission.get("_id")
        user_id = submission.get("user_id")
        folder_id = submission.get("folder_id")
        document_id = submission.get("document_id")
        doc_type = "result"

        if not all([doc_id, user_id, folder_id, document_id]):
            logger.error("Missing required fields: '_id', 'user_id', 'folder_id', 'document_id'.")
            print("Missing required fields in submission.")
            return

        field_values_str = json.dumps(submission.get("fieldValues", []))
        metadata = ""

        indexing_helper.index_document(
            doc_id=doc_id,
            user_id=user_id,
            folder_id=folder_id,
            document_id=document_id,
            doc_type=doc_type,
            content_text=text_summary,
            field_values=field_values_str,
            embedding_vector=embedding_vector,
            metadata=metadata
        )
        logger.info(f"Successfully indexed document ID: {doc_id}")
        print(f"Successfully indexed document ID: {doc_id}")
    except Exception as e:
        logger.error(f"Error indexing document: {e}")
        print(f"Error indexing document: {e}")
        return

if __name__ == "__main__":
    main()
