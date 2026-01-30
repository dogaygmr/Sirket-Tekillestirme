import sys
import os
import json
import time

import pandas as pd

from langgraph.pregel.remote import RemoteGraph

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(project_root)

# Import the required module
from src.agent.state import DEFAULT_EXTRACTION_SCHEMA as EXTRACTION_SCHEMA
from src.agent.configuration import Configuration

AGENT_URL = "http://127.0.0.1:2024/"
GRAPH_ID = "company_researcher"
OUTPUT_PATH = "../data/output.json"

API_KEY_NAMES = [
    "GOOGLE_API_KEY_FOR_SEARCH",
    "GOOGLE_API_KEY_FOR_SEARCH_2",
    "GOOGLE_API_KEY_FOR_SEARCH_3",
    "GOOGLE_API_KEY_FOR_SEARCH_4"
]
BATCH_SIZE = 60  # Number of companies to process per API key
SLEEP_TIME = 86400  # 24 hours in seconds

def make_agent_runner(graph_id: str, agent_url: str):
    """Wrapper that transforms inputs/outputs to match the expected eval schema and invokes the agent."""
    agent_graph = RemoteGraph(graph_id, url=agent_url)

    def run_agent(inputs: dict) -> dict:
        """Run the agent on the inputs from the LangSmith dataset record, return outputs conforming to the LangSmith dataset output schema."""
        transformed_inputs = transform_dataset_inputs(inputs)
        response = agent_graph.invoke(transformed_inputs)
        return transform_agent_outputs(response)

    return run_agent


def transform_dataset_inputs(inputs: dict) -> dict:
    """Transform LangSmith dataset inputs to match the agent's input schema."""
    company_name = inputs.get("company", {}).get("name", "") if isinstance(inputs.get("company"), dict) else inputs.get("company", "")

    return {
        "company": company_name,
        "extraction_schema": inputs.get("extraction_schema"),
        "user_notes": inputs.get("user_notes"),
        "api_key_name": inputs.get("api_key_name")
    }

def transform_agent_outputs(outputs: dict) -> dict:
    """Transform agent outputs to match the LangSmith dataset output schema."""
    # Modify if needed to align with the expected output structure

    return outputs


def get_company_data(input: dict):
    """Fetch company data by interacting with the agent."""

    # Use the agent runner (created using make_agent_runner) to process the input
    run_agent = make_agent_runner(graph_id=GRAPH_ID, agent_url=AGENT_URL)
    return run_agent(input)

def read_companies(extraction_schema: str, user_notes: str, api_key_name: str, dataset: str = None, affiliate_list: list = None):
    company_inputs = []
    
    # Load existing tax numbers from output.json
    existing_tax_numbers = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as json_file:
            try:
                existing_data = json.load(json_file)
                if isinstance(existing_data, list):
                    existing_tax_numbers = {entry.get("tax_no") for entry in existing_data if entry.get("tax_no")}
            except json.JSONDecodeError:
                existing_data = []

    if dataset:
        df = pd.read_csv(dataset, dtype={"Org_Company_Tax_Number": str})
        df = df[["Org_Company_Tax_Number", "Merged_Org_Company_Name"]]

        for _, row in df.iterrows():
            tax_no = row["Org_Company_Tax_Number"]
            
            # Skip companies whose Org_Company_Tax_Number is in output.json
            if tax_no in existing_tax_numbers:
                continue

            company_inputs.append({
                "company": {"name": row['Merged_Org_Company_Name']},
                "extraction_schema": extraction_schema,
                "user_notes": user_notes,
                "tax_no": tax_no,
                "api_key_name": api_key_name
            })

    if affiliate_list:
        for affiliate in affiliate_list:
            company_inputs.append({
                "company": {"name": affiliate},
                "extraction_schema": extraction_schema,
                "user_notes": user_notes,
                "tax_no": None,
                "api_key_name": api_key_name
            })

    return company_inputs

def run_pipeline(dataset: str = None, affiliate_list: list = None):
    config = Configuration()
    prompt = (
        "Since our dataset includes sole proprietorships, company names might consist only of a first and last name "
        "(e.g., Mehmet Yıldız). In such cases, include a note in the completed_notes field specifying that this is a sole proprietorship. "
        "Additionally, search for companies primarily within Turkey (you can conduct a global search if the company is a global entity operating in Turkey). "
        "Provide your responses in English."
    )

    if dataset:
        inputs = read_companies(EXTRACTION_SCHEMA, prompt, API_KEY_NAMES[0], dataset=dataset)
    elif affiliate_list:
        inputs = read_companies(EXTRACTION_SCHEMA, prompt, API_KEY_NAMES[0], affiliate_list=affiliate_list)
    else:
        print("No dataset or affiliate list provided.")
        return

    total_companies = len(inputs)
    existing_data = []
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as json_file:
            try:
                existing_data = json.load(json_file)
            except json.JSONDecodeError:
                existing_data = []

    companies = existing_data if isinstance(existing_data, list) else []

    while True:  # Infinite loop
        for api_key_index, api_key in enumerate(API_KEY_NAMES):
            start_idx = api_key_index * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            batch_inputs = inputs[start_idx:end_idx]

            if not batch_inputs:
                print("All companies processed. Restarting cycle after 24 hours...")
                time.sleep(SLEEP_TIME)
                break

            print(f"Processing {len(batch_inputs)} companies with API key: {api_key}")

            for idx, input in enumerate(batch_inputs, start=start_idx):
                input["api_key_name"] = api_key
                start_time = time.time()
                
                while True:
                    try:
                        company_data = get_company_data(input)
                        break  # If successful, exit the loop
                    except ValueError as e:
                        error_message = str(e)
                        if "finish_reason" in error_message and "copyrighted material" in error_message:
                            print(f"Error encountered: {error_message}. Retrying in 5 minutes...")
                            time.sleep(300)  # Sleep for 5 minutes
                        else:
                            raise  # If it's a different error, don't retry—just crash

                end_time = time.time()

                affiliates = company_data["info"]["affiliations"]

                filtered_data = {
                    # "company_name": company_data["company"],
                    "metadata": {
                        "configuration": {
                            "max_search_queries": config.max_search_queries,
                            "max_search_results": config.max_search_results,
                            "batch_size": config.batch_size,
                            "max_reflection_steps": config.max_reflection_steps,
                            "max_character_count_for_one_page": config.max_character_count_for_one_page,
                        },
                        "execution_time_in_seconds": round(end_time - start_time, 2),
                        "number_of_affiliates": len(affiliates["parents"]) + len(affiliates["subsidiaries"]),
                        "failed_url_attempts": company_data.get("failed_url_attempts", 0),
                    },
                    "info": company_data["info"],
                    # "completed_notes": company_data["completed_notes"],
                    "tax_no": input["tax_no"]
                }

                companies.append(filtered_data)

                with open(OUTPUT_PATH, "w", encoding="utf-8") as json_file:
                    json.dump(companies, json_file, indent=4, ensure_ascii=False)

                print(f"{idx + 1}/{total_companies} | {input['company']['name']}: finished...")

        print("All API keys used. Sleeping for 24 hours before restarting...")
        time.sleep(SLEEP_TIME)  # Sleep for 24 hours

if __name__ == "__main__":
    dataset_path = "../data/shuffled_sample.csv" # Buraya original .csv'nin path'ini koyacağız
    run_pipeline(dataset=dataset_path)