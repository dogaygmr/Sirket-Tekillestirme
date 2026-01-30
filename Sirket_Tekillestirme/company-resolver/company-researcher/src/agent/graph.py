import asyncio
from typing import cast, Any, Literal, Optional
import json
from math import ceil

import time
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import urllib.parse

from agent.configuration import Configuration
from agent.state import InputState, OutputState, OverallState
from agent.utils import deduplicate_and_format_sources, format_all_notes
from agent.prompts import (
    EXTRACTION_PROMPT,
    REFLECTION_PROMPT,
    INFO_PROMPT,
    QUERY_WRITER_PROMPT,
)

# LLMs

rate_limiter = InMemoryRateLimiter(
    requests_per_second=4,
    check_every_n_seconds=0.1,
    max_bucket_size=10,  # Controls the maximum burst size.
)

"""
Çıkardım
claude_3_5_sonnet = ChatAnthropic(
    model="claude-3-5-sonnet-latest", temperature=0, rate_limiter=rate_limiter
)
"""

# Ekledim
gemini_1_5_flash = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", temperature=0, rate_limiter=rate_limiter
)


class Affiliations(BaseModel):
    parents: list[str] = Field(default=[], description="List of parent companies. Leave the list empty ([]) if you can't find a parent company.")
    subsidiaries: list[str] = Field(default=[], description="List of subsidiary or child companies. Leave the list empty ([]) if you can't find a subsidiary or child company.")
    
class CompanyInfo(BaseModel):
    company_name: str = Field(description="Official name of the company. Always required.")
    founding_year: Optional[int] = Field(
        description="Year the company was founded. Leave it null if the information is not available. If the year is approximate, extract the closest exact year."
    )
    founder_names: list[str] = Field(description="Names of the founding team members. Leave the list empty ([]) if you can't find the names.")
    product_description: str = Field(description="Brief description of the company's main product or service.")
    branch_dealer: list[str] = Field(description="List of branch dealers associated with the company. Leave the list empty ([]) if the company has no branch dealers.")
    affiliations: Affiliations = Field(default_factory=Affiliations, description="Structured relationships of the company with other organizations.")

class Queries(BaseModel):
    queries: list[str] = Field(
        description="List of search queries.",
    )


class ReflectionOutput(BaseModel):
    is_satisfactory: bool = Field(
        description="True if all required fields are well populated, False otherwise"
    )
    missing_fields: list[str] = Field(
        description="List of field names that are missing or incomplete"
    )
    search_queries: list[str] = Field(
        description="If is_satisfactory is False, provide 1-3 targeted search queries to find the missing information"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")


def generate_queries(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Generate search queries based on the user input and extraction schema."""
    print("\n\n----------------------------generate_queries başladı---------------------------------")


    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_search_queries = configurable.max_search_queries
    print(f"Number of max_search_queries: {max_search_queries}")

    # Generate search queries
    # structured_llm = claude_3_5_sonnet.with_structured_output(Queries)
    structured_llm = gemini_1_5_flash.with_structured_output(Queries)

    # Format system instructions
    query_instructions = QUERY_WRITER_PROMPT.format(
        company=state.company,
        info=json.dumps(state.extraction_schema, indent=2),
        user_notes=state.user_notes,
        max_search_queries=max_search_queries,
    )

    # Generate queries
    results = cast(
        Queries,
        structured_llm.invoke(
            [
                {"role": "system", "content": query_instructions},
                {
                    "role": "user",
                    "content": "Please generate a list of search queries related to the schema that you want to populate.",
                },
            ]
        ),
    )

    # Queries
    query_list = [query for query in results.queries]
    print(f"Number of queries: {len(query_list)}")
    print("\n\n----------------------------generate_queries bitti---------------------------------")
    return {"search_queries": query_list[:max_search_queries]}


def get_ddg_links(query, num_results):
    chrome_options = Options()
    #chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Set up the Selenium WebDriver (e.g., Chrome)
    driver = webdriver.Chrome(options=chrome_options)  # Ensure chromedriver is in your PATH or specify the full path

    encoded_query = urllib.parse.quote(query)
    driver.get(f'https://duckduckgo.com/?q={encoded_query}&t=h_&ia=web')
    
    """
    # Find the search bar and input the query
    search_box = driver.find_element(By.NAME, 'q')
    search_box.send_keys(query)
    search_box.send_keys(Keys.RETURN)
    """
    
    # Wait for the page to load
    time.sleep(3)  # Adjust time if necessary to ensure the page is fully loaded
    
    # Scrape the URLs
    result_links = driver.find_elements(By.XPATH, '//a[@data-testid="result-title-a"]')
    
    # Get the URLs from the links
    urls = [link.get_attribute('href') for link in result_links[:num_results]]
    
    # Close the browser
    driver.quit()
    
    return urls


def get_page_content(url):
    retries = 1 
    delay = 1  
    failed = False
    
    # Skip known problematic domains
    problematic_domains = ["crunchbase.com", "reuters.com", "dnb.com", "bloomberg.com"]
    if any(domain in url for domain in problematic_domains):
        print(f"Skipping known problematic domain: {url}")
        return None, None, True
    
    for attempt in range(retries):
        try:
            # Rotate user agents to avoid detection
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
            ]
            headers = {
                'User-Agent': user_agents[attempt % len(user_agents)],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Referer': 'https://www.google.com/',
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'lxml')
            title = soup.title.string if soup.title else "Başlık bulunamadı"

            return soup, title, failed

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}, retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  
    
    failed = True  # If we've exhausted all retries, mark as failed
    print(f"Failed to retrieve after {retries} attempts: {url}")
    return None, None, failed


def get_api_key(env_path, key_name):
    with open(env_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(key_name):
                return line.split('=')[1].strip()
    return None


def clean_page_content(page_content, query, api_key_name="GOOGLE_API_KEY_FOR_SEARCH"):
    GOOGLE_API_KEY_FOR_SEARCH = get_api_key("../company-researcher/.env", api_key_name)

    genai.configure(api_key=GOOGLE_API_KEY_FOR_SEARCH)
    model = genai.GenerativeModel("gemini-1.5-flash")

    search_instructions = f"""Aşağıda HTML formatında bir web sayfasının içeriği bulunmaktadır.

--- START ---
{page_content}
--- FINISH ---

Metni "{query}" sorgusunu baz alarak HTML'den temizleyip bir insanın okuyabileceği düz text haline getir. Metne göre sorguyu cevapladıktan sonra yeni bir paragrafa geç ve konuyla çok az alakası olan metinleri bu kısma yaz. Düz, okunaklı bir paragraf olmalı. Özellikle metinde bu şirketin bağlı olduğu çatı şirketler, bu şirkete bağlı olan iştirak (yan) şirketler ya da bu şirketin ortağı olan şirketler hakkında bir liste söz konusu olduğunda ilgili tüm şirketleri liste olarak belirtin. Ne kadar uzun olursa olsun tüm listeyi yazın."""

    print(f"Length of search instructions for Gemini is {len(search_instructions)}")

    response = model.generate_content(search_instructions)
    
    return response.text


async def research_company(
    state: OverallState, config: RunnableConfig
) -> dict[str, Any]:
    """Execute a multi-step web search and information extraction process.

    This function performs the following steps:
    1. Executes concurrent web searches using the Tavily API
    2. Deduplicates and formats the search results
    """
    print("\n\n----------------------------research_company başladı---------------------------------")


    configurable = Configuration.from_runnable_config(config)
    batch_size = configurable.batch_size
    max_search_results = configurable.max_search_results
    max_character_count_for_one_page = configurable.max_character_count_for_one_page
    api_key_name = state.api_key_name

    async def async_search(query: str):
        # Make this asynchronous
        urls = await asyncio.to_thread(get_ddg_links, query, 2 * max_search_results)
        contents = []
        failed_attempts = 0
        primary_urls = urls[:max_search_results]  # First set of URLs
        backup_urls = urls[max_search_results:]  # Backup URLs
        processed_urls = 0

        # Process URLs concurrently
        async def process_url(url):
            nonlocal failed_attempts, processed_urls
            if not url:
                return None
                
            # Make these operations asynchronous
            page_content, title, failed = await asyncio.to_thread(get_page_content, url)
            if failed:
                failed_attempts += 1
                return None

            text_content = page_content.get_text()
            if len(text_content) > max_character_count_for_one_page:
                return None

            cleaned_content = await asyncio.to_thread(clean_page_content, text_content, query, api_key_name)
            processed_urls += 1
            return {
                "title": title,
                "url": url,
                "content": cleaned_content,
                "raw_content": None
            }
        
        # Process primary URLs concurrently
        primary_tasks = [process_url(url) for url in primary_urls]
        primary_results = await asyncio.gather(*primary_tasks)
        contents.extend([r for r in primary_results if r is not None])
        
        # If needed, process backup URLs
        if processed_urls < max_search_results:
            remaining = max_search_results - processed_urls
            backup_tasks = [process_url(url) for url in backup_urls[:remaining]]
            backup_results = await asyncio.gather(*backup_tasks)
            contents.extend([r for r in backup_results if r is not None])
        
        return contents, failed_attempts

    
    async def batch_process_searches(queries, batch_size=batch_size, sleep_time=60):
        all_results = []
        total_failed_attempts = 0
        total_batches = ceil(len(queries) / batch_size)
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(queries))
            current_batch = queries[start_idx:end_idx]
            
            print(f"Processing batch {batch_num + 1}/{total_batches} with {len(current_batch)} queries")
            
            # Create tasks for current batch
            batch_tasks = [async_search(query) for query in current_batch]
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Separate results and failed counts
            for results, failed_count in batch_results:
                all_results.append(results)
                total_failed_attempts += failed_count
            
            # Sleep between batches (including the final batch)
            print(f"Sleeping for {sleep_time} seconds before proceeding")
            await asyncio.sleep(sleep_time)
        
        return all_results, total_failed_attempts

    transformed_results, failed_attempts = await batch_process_searches(state.search_queries)

    # Deduplicate and format sources
    source_str = deduplicate_and_format_sources(
        transformed_results, max_tokens_per_source=1000, include_raw_content=False
    )

    # Generate structured notes relevant to the extraction schema
    p = INFO_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2),
        content=source_str,
        company=state.company,
        user_notes=state.user_notes,
    )
    # result = await claude_3_5_sonnet.ainvoke(p)
    result = await gemini_1_5_flash.ainvoke(p)

    print("\n\n----------------------------research_company bitti---------------------------------")
    return {
        "completed_notes": [str(result.content)],
        "failed_url_attempts": failed_attempts
    }


def gather_notes_extract_schema(state: OverallState) -> dict[str, Any]:
    print("\n\n----------------------------gather_notes başladı---------------------------------")

    # Format all notes
    notes = format_all_notes(state.completed_notes)

    # Extract schema fields using CompanyInfo model
    system_prompt = EXTRACTION_PROMPT.format(
        info=json.dumps(CompanyInfo.model_json_schema(), indent=2), 
        notes=notes
    )
    
    structured_llm = gemini_1_5_flash.with_structured_output(CompanyInfo)
    
    result = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Produce a structured output from these notes."},
        ]
    )
    
    # Convert to dictionary and ensure all fields are present
    info_dict = result.model_dump()
    
    print("\n\n----------------------------gather_notes bitti---------------------------------")
    return {"info": info_dict}


def reflection(state: OverallState) -> dict[str, Any]:
    """Reflect on the extracted information and generate search queries to find missing information."""
    print("\n\n----------------------------reflection başladı---------------------------------")


    # structured_llm = claude_3_5_sonnet.with_structured_output(ReflectionOutput)
    structured_llm = gemini_1_5_flash.with_structured_output(ReflectionOutput)

    # Format reflection prompt
    system_prompt = REFLECTION_PROMPT.format(
        schema=json.dumps(state.extraction_schema, indent=2),
        info=state.info,
    )

    # Invoke
    result = cast(
        ReflectionOutput,
        structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Produce a structured reflection output."},
            ]
        ),
    )

    if result.is_satisfactory:
        print("\n\n----------------------------reflection bitti---------------------------------")
        return {"is_satisfactory": result.is_satisfactory}
    else:
        print("\n\n----------------------------reflection bitti---------------------------------")
        return {
            "is_satisfactory": result.is_satisfactory,
            "search_queries": result.search_queries,
            "reflection_steps_taken": state.reflection_steps_taken + 1,
        }


def route_from_reflection(
    state: OverallState, config: RunnableConfig
) -> Literal[END, "research_company"]:  # type: ignore
    """Route the graph based on the reflection output."""
    print("\n\n----------------------------route_from_reflection başladı---------------------------------")


    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # If we have satisfactory results, end the process
    if state.is_satisfactory:
        return END

    # If results aren't satisfactory but we haven't hit max steps, continue research
    if state.reflection_steps_taken <= configurable.max_reflection_steps:
        return "research_company"

    # If we've exceeded max steps, end even if not satisfactory
    print("\n\n----------------------------route_from_reflection bitti---------------------------------")
    return END


# Add nodes and edges
builder = StateGraph(
    OverallState,
    input=InputState,
    output=OutputState,
    config_schema=Configuration,
)
builder.add_node("gather_notes_extract_schema", gather_notes_extract_schema)
builder.add_node("generate_queries", generate_queries)
builder.add_node("research_company", research_company)
builder.add_node("reflection", reflection)

builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", "research_company")
builder.add_edge("research_company", "gather_notes_extract_schema")
builder.add_edge("gather_notes_extract_schema", "reflection")
builder.add_conditional_edges("reflection", route_from_reflection)

# Compile
graph = builder.compile()
