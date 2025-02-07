# chat.py

import os
import json
import logging
import openai
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Import from your unified helper file
from helpers import (
    EmbeddingHelper,
    SearchHelper
)

logging.basicConfig(
    filename='application.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("chat")

CONVERSATION_HISTORY_FILE = "conversation_history.json"

def load_env(env_file: str = "local.env"):
    load_dotenv(env_file)
    logger.info("Loaded environment variables in chat script.")

def load_conversation_history() -> list:
    if os.path.exists(CONVERSATION_HISTORY_FILE):
        try:
            with open(CONVERSATION_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")
            return []
    else:
        return []

def save_conversation_history(history: list) -> None:
    try:
        with open(CONVERSATION_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)
        logger.info("Conversation history saved.")
    except Exception as e:
        logger.error(f"Error saving conversation history: {e}")

def get_user_query(file_path: str = "query.txt") -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            query = f.read().strip()
        logger.info(f"User query loaded from '{file_path}': {query}")
        print(query)
        return query
    except FileNotFoundError:
        logger.error(f"Query file '{file_path}' not found.")
        raise
    except Exception as e:
        logger.error(f"Error reading query from '{file_path}': {e}")
        raise

##############################
# Chat + Aggregation
##############################

def process_results_with_openai_chat(results: list, user_query: str, conversation_history: list, aggregated_data: dict = None) -> str:
    """
    Use OpenAI's chat model to generate an answer from search results + conversation history,
    optionally including aggregated numeric data.
    """
    try:
        conversation_history.append({"role": "user", "content": user_query})
        doc_content = "\n\n".join(doc.get("content", "") for doc in results)
        system_msg = (
            "You are an AI assistant that helps answer questions based on provided documents. "
            "Use the information from the documents + conversation history to answer."
        )

        messages = [{"role": "system", "content": system_msg}] + conversation_history
        messages.append({"role": "system", "content": f"Documents:\n{doc_content}"})
        
        if aggregated_data:
            merged_json = json.dumps(aggregated_data)
            msg_text = (
                "We have aggregated numeric data across all chosen documents. "
                f"Here is a JSON summary of item totals:\n{merged_json}\n"
                "Consider these integrated totals when answering the user's question."
            )
            messages.append({"role": "system", "content": msg_text})

        response = openai.ChatCompletion.create(
            engine=os.getenv("OPENAI_CHAT_MODEL", "gpt-4"),
            messages=messages,
            max_tokens=300,
            temperature=0.5
        )
        answer = response.choices[0]["message"]["content"].strip()
        conversation_history.append({"role": "assistant", "content": answer})
        logger.info("Generated ChatCompletion answer.")
        return answer
    except Exception as e:
        logger.error(f"ChatCompletion failed: {e}")
        return "I'm sorry, I couldn't generate a response at this time."

def check_for_visualization_command(query: str) -> str:
    q = query.lower()
    chart_types = ["bar", "line", "pie", "scatter"]
    for ctype in chart_types:
        if ctype in q and ("chart" in q or "plot" in q or "visualize" in q):
            return ctype
    return None

def llm_extract_numeric_data(doc_fields: list, user_query: str) -> dict:
    fields_json = json.dumps(doc_fields, indent=2)
    system_prompt = (
        "You are an expert at interpreting structured field data. The user wants numeric data relevant to their query.\n"
        "Return JSON with a 'data' array where each element is {label: <string>, value: <number>} for relevant numeric fields."
    )
    user_prompt = f"User query: {user_query}\n\nFields:\n{fields_json}\nPlease respond ONLY with valid JSON."

    try:
        response = openai.ChatCompletion.create(
            engine=os.getenv("OPENAI_CHAT_MODEL", "gpt-4"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0
        )
        raw_output = response.choices[0]["message"]["content"].strip()
        parsed = json.loads(raw_output)
        if "data" in parsed and isinstance(parsed["data"], list):
            return parsed
        else:
            return {"data": []}
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return {"data": []}

def aggregate_docs_numeric_data(docs: list, user_query: str) -> dict:
    aggregated = {}
    for doc in docs:
        fv_str = doc.get("fieldValues", "")
        if not fv_str:
            continue
        try:
            doc_fields = json.loads(fv_str)
        except json.JSONDecodeError:
            doc_fields = []
        
        llm_result = llm_extract_numeric_data(doc_fields, user_query)
        data_list = llm_result.get("data", [])
        for item in data_list:
            lbl = item["label"]
            val = item["value"]
            aggregated[lbl] = aggregated.get(lbl, 0) + val
    return aggregated

##############################
# Visualization
##############################

def short_title_from_query(query: str) -> str:
    import re
    q = query.lower()
    q = re.sub(r"[^\w\s]", "", q)
    fillers = ["draw", "the", "chart", "bar", "line", "pie", "scatter", 
               "plot", "visualize", "of", "for"]
    tokens = q.split()
    cleaned_tokens = [t for t in tokens if t not in fillers]
    return " ".join(cleaned_tokens).strip().title()

def guess_y_axis_label(query: str) -> str:
    q = query.lower()
    if "order" in q:
        return "Orders"
    elif "sale" in q or "revenue" in q or "amount" in q:
        return "Sales"
    return "Value"

def fallback_word_count_chart(documents: list, chart_type: str):
    x_labels = []
    y_values = []
    for i, doc in enumerate(documents, start=1):
        doc_id = doc.get("id", f"Doc{i}")
        x_labels.append(doc_id)
        combined_text = doc.get("content", "") + " " + doc.get("fieldValues", "")
        word_count = len(combined_text.split())
        y_values.append(word_count)
    
    plt.figure(figsize=(10, 6))
    if chart_type == "bar":
        plt.bar(x_labels, y_values, color="skyblue")
    elif chart_type == "line":
        plt.plot(x_labels, y_values, marker="o", color="green")
    elif chart_type == "pie":
        plt.pie(y_values, labels=x_labels, autopct="%1.1f%%", startangle=140)
    elif chart_type == "scatter":
        plt.scatter(x_labels, y_values, color="red")
    else:
        logger.info(f"Unsupported chart type: {chart_type}")
        return
    
    plt.xlabel("Document ID")
    plt.ylabel("Word Count")
    plt.title(f"Default Visualization - {chart_type.capitalize()} Chart")
    plt.tight_layout()
    out_file = "visualization.png"
    plt.savefig(out_file)
    plt.close()
    logger.info(f"Fallback visualization saved to {out_file}.")

def visualize_aggregated_data(aggregated: dict, chart_type: str, user_query: str):
    if not aggregated:
        logger.info("No aggregated numeric data for visualization.")
        return
    
    labels = list(aggregated.keys())
    values = [aggregated[k] for k in labels]

    chart_title = short_title_from_query(user_query)
    y_label = guess_y_axis_label(user_query)

    plt.figure(figsize=(10, 6))
    if chart_type == "bar":
        bars = plt.bar(labels, values, color="skyblue")
        for i, bar in enumerate(bars):
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, h + 0.5, f"{values[i]}", ha="center", va="bottom", fontsize=9)
    elif chart_type == "line":
        plt.plot(labels, values, marker="o", color="green")
        for i, val in enumerate(values):
            plt.text(i, val+0.5, f"{val}", ha="center", va="bottom", fontsize=9)
    elif chart_type == "scatter":
        plt.scatter(labels, values, color="red")
        for i, val in enumerate(values):
            plt.text(i, val+0.5, f"{val}", ha="center", va="bottom", fontsize=9)
    elif chart_type == "pie":
        plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    else:
        logger.info(f"Unsupported chart type: {chart_type}")
        return
    
    plt.xlabel("")
    plt.ylabel(y_label)
    plt.title(chart_title)
    plt.tight_layout()
    out_file = "visualization.png"
    plt.savefig(out_file)
    plt.close()
    logger.info(f"Visualization saved to {out_file}.")

##############################
# Disambiguation
##############################

def ask_user_for_options(options: list, prompt_text: str) -> (list, bool):
    print(prompt_text)
    for i, opt in enumerate(options, start=1):
        print(f"  {i} - {opt}")
    
    while True:
        choice = input("Please enter an option number (or multiple comma-separated, or 'all'): ").strip().lower()
        if choice == "all":
            return (options, True)
        nums = choice.split(",")
        selected = []
        valid = True
        for n in nums:
            n = n.strip()
            if not n.isdigit():
                valid = False
                break
            idx = int(n)
            if idx < 1 or idx > len(options):
                valid = False
                break
            selected.append(options[idx-1])
        if valid and selected:
            return (selected, False)
        print("Invalid choice. Try again, or type 'all'.")

def disambiguate_documents(docs: list) -> list:
    from collections import defaultdict
    group_by_doc = defaultdict(list)
    for doc in docs:
        did = doc.get("documentId")
        group_by_doc[did].append(doc)

    doc_ids = list(group_by_doc.keys())
    if len(doc_ids) > 1:
        prompt = "Multiple document IDs found. Which do you want to select or integrate?"
        chosen_doc_ids, doc_all_flag = ask_user_for_options(doc_ids, prompt)
        if doc_all_flag:
            return docs
    else:
        chosen_doc_ids = doc_ids
        doc_all_flag = False

    final_docs = []
    for did in chosen_doc_ids:
        doc_group = group_by_doc[did]
        if len(doc_group) > 1 and not doc_all_flag:
            possible_results = [d.get("id") for d in doc_group]
            prompt = f"For DocumentID '{did}', multiple result IDs found. Which do you want?"
            chosen_results, res_all_flag = ask_user_for_options(possible_results, prompt)
            if res_all_flag:
                final_docs.extend(doc_group)
            else:
                final_docs.extend(d for d in doc_group if d.get("id") in chosen_results)
        else:
            final_docs.extend(doc_group)
    return final_docs

##############################
# Main Chat
##############################

def main():
    # 1. Load environment
    load_env()
    openai.api_base = os.getenv("OPENAI_API_BASE")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # 2. Initialize
    embedder = EmbeddingHelper()
    searcher = SearchHelper()

    # 3. Conversation history
    conversation_history = load_conversation_history()

    # 4. User query
    try:
        user_query = get_user_query()
    except Exception as e:
        logger.error("Error loading user query: %s", e)
        print("Error loading user query:", e)
        return

    # 5. Check for chart
    chart_type = check_for_visualization_command(user_query)
    if chart_type:
        logger.info(f"Visualization request detected: {chart_type} chart.")
        print(f"Visualization request detected: {chart_type} chart.")

    # 6. Generate embedding
    try:
        embedding = embedder.get_embedding(user_query)
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        print("Error generating embedding:", e)
        return

    # 7. Vector search
    user_id = os.getenv("CURRENT_USER_ID", "userXYZ")
    try:
        results = searcher.vector_search(embedding=embedding, top_k=5, user_id=user_id)
        docs = results.get("value", [])
        if not docs:
            logger.info("No relevant documents found.")
            print("No relevant documents found.")
            return
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        print("Error performing search:", e)
        return

    # 8. Show docs
    print("\nTop Relevant Documents:")
    for idx, doc in enumerate(docs, start=1):
        print(f"\nDocument {idx}:")
        print(f"ID: {doc.get('id')}")
        print(f"Document ID: {doc.get('documentId')}")
        print(f"Type: {doc.get('type')}")
        print(f"Content: {doc.get('content')}")
        print(f"fieldValues (as JSON string): {doc.get('fieldValues', '')}")
        print(f"Metadata: {doc.get('metadata')}")

    # 9. Disambiguate
    final_docs = disambiguate_documents(docs)
    if not final_docs:
        print("No documents selected. Exiting.")
        return

    # 10. Aggregate numeric data
    aggregated_data = aggregate_docs_numeric_data(final_docs, user_query)

    # 11. Visualization
    if chart_type:
        if aggregated_data:
            visualize_aggregated_data(aggregated_data, chart_type, user_query)
        else:
            fallback_word_count_chart(final_docs, chart_type)

    # 12. Chat
    try:
        answer = process_results_with_openai_chat(
            final_docs, 
            user_query, 
            conversation_history,
            aggregated_data
        )
        print("\nAnswer:\n", answer)
    except Exception as e:
        logger.error("Failed to generate answer: %s", e)
        print("Error generating answer:", e)

    # 13. Save conversation
    save_conversation_history(conversation_history)

if __name__ == "__main__":
    main()