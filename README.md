# AdaptiveRAG

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview
AdaptiveRAG is a Retrieval-Augmented Generation (RAG) system designed for embedding, indexing, querying, and visualizing form data. This project enables users to process structured forms, retrieve relevant data via Azure AI Search, and generate insightful responses with OpenAI's GPT-4. Additionally, it supports resolving ambiguities and provides automatic data visualization with Matplotlib.

## Features
- **Embedding & Indexing**: Utilizes Azure OpenAI's `davinci` model for text embeddings and indexes data in Azure AI Search.
- **Query Processing**: Handles user queries by searching the indexed data and responding with AI-generated answers.
- **Disambiguation Handling**: Resolves ambiguous results across different forms or submissions.
- **Data Visualization**: Generates Bar, Line, Pie, and Scatter plots using Matplotlib.
- **Short-term Memory**: Maintains a session history to enhance conversational responses.
- **Logging & Debugging**: Logs operations in `application.log` and stores conversation history in `conversation_history.json`.
- **Command Line Friendly**: Accepts input via text files (`query.txt`, `form.json`) for ease of use.

## Repository Structure
```
AdaptiveRAG/
│── helpers.py               # Utilities for flattening, embedding, indexing, and searching
│── indexer.py               # Embeds and indexes form submissions in Azure AI Search
│── chat.py                  # Handles queries, disambiguation, and visualization
│── index_setup.py           # Creates the Azure AI Search index
│── index_definition.json    # Configuration for index schema
│── form.json                # Example form data (modifiable for new entries)
│── conversation_history.json# Stores user interactions
│── application.log          # Logs events for debugging
│── query.txt                # User query input file
│── README.md                # Documentation
```

## Installation & Setup
### Prerequisites
- Python 3.8+
- Azure OpenAI API access
- Azure AI Search instance

### Install Dependencies
```sh
pip install openai python-dotenv requests matplotlib
```

### Environment Configuration
Create a `.env` file in the root directory and set up the following:
```env
# Azure OpenAI
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=your_openai_endpoint
AZURE_OPENAI_ENGINE=text-embedding-davinci-002

# Azure AI Search
ACS_ENDPOINT=your_azure_search_endpoint
ACS_API_KEY=your_search_api_key
ACS_INDEX_NAME=knowledge-index
```

### Index Setup
Run the following command to create the index in Azure AI Search:
```sh
python index_setup.py
```

### Submit a Form
Modify `form.json` to reflect the form structure and submit it for indexing:
```sh
python indexer.py
```

### Run Chat for Querying
Provide a query in `query.txt` and execute:
```sh
python chat.py
```

## Querying & Visualization
- To retrieve answers based on indexed form submissions, update `query.txt` with a question.
- If the query contains `bar`, `line`, `pie`, or `scatter`, a visualization is generated and saved as `visualization.png`.

## Example Queries
1. **Retrieve customer details:**
   ```
   What is the contact info for Google?
   ```
2. **Find total quantity ordered:**
   ```
   Show a bar chart of ordered items.
   ```
3. **Identify order trends:**
   ```
   Visualize the trend of laptop orders over time.
   ```

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss the modifications.


