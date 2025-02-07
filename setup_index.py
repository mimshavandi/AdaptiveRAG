# setup_index.py

from azure_vector_helper import AzureSearchRESTHelper

def main():
    # Initialize Azure Search REST Helper
    try:
        azure_search_helper = AzureSearchRESTHelper(
            env_file="local.env",
            index_definition_file="RAG/index_definition.json"
        )
        azure_search_helper.create_or_update_index()
        print("Index created or updated successfully.")
    except Exception as e:
        print(f"Error creating/updating index: {e}")

if __name__ == "__main__":
    main()
