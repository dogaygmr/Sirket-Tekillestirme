import asyncio
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import time
import sys
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import urllib.parse

async def get_ddg_links(query, num_results):
    # Get all links at once
    chrome_options = Options()
    #chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Set up the Selenium WebDriver
    driver = webdriver.Chrome(options=chrome_options)

    encoded_query = urllib.parse.quote(query)
    driver.get(f'https://duckduckgo.com/?q={encoded_query}&t=h_&ia=web')
    
    # Wait for the page to load
    time.sleep(3)
    
    # Scrape all URLs - get double what we need to have backup URLs
    result_links = driver.find_elements(By.XPATH, '//a[@data-testid="result-title-a"]')
    
    # Get URLs (we'll need at least 2*num_results)
    urls = [link.get_attribute('href') for link in result_links[:num_results * 2]]
    
    # Close the browser
    driver.quit()
    
    return urls

def get_api_key(env_path, key_name):
    with open(env_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(key_name):
                return line.split('=')[1].strip()
    return None

async def get_page_content(url):
    """Fetches page content with retry mechanism"""
    retries = 1
    delay = 1
    
    # Rotate user agents to avoid detection
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
    ]
    
    for attempt in range(retries):
        try:
            headers = {
                'User-Agent': user_agents[attempt % len(user_agents)],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Referer': 'https://www.google.com/',
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            print(f"Retrieved: {url}")

            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else "Title not found"

            return soup, title, False

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}, retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2
    
    print(f"Failed to retrieve after {retries} attempts.")
    return None, None, True

async def clean_page_content(page_content, query):
    """Clean the content using Gemini - made async compatible"""
    GOOGLE_API_KEY = get_api_key("../company-researcher/.env", "GOOGLE_API_KEY_FOR_SEARCH_4")

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = (
        f"Below is the content of a web page in HTML format.\n\n"
        "--- START ---\n"
        f"{page_content}\n"
        "--- FINISH ---\n\n"
        f'Based on the query "{query}", clean the HTML content and convert it into plain, '
        "human-readable text. After answering the query using the relevant part of the content, "
        "start a new paragraph and include any slightly related or tangential information in that section. "
        "The result should be a clean and readable paragraph. "
        "For example, if the query asks about a company's parent organizations or subsidiaries, and the text "
        "includes a list of parent companies or subsidiary companies, provide the complete list. "
        "No matter how long it is, write out the full list."
    )

    # Run in executor to avoid blocking
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, lambda: model.generate_content(prompt)
    )
    
    return response.text

async def process_url(url, query):
    """Process a single URL asynchronously"""
    if not url:
        return None
    
    # Get page content
    page_content, title, failed = await get_page_content(url)
    if failed or not page_content:
        return None
    
    # Clean the content
    cleaned_content = await clean_page_content(page_content.get_text(), query)
    
    return {
        "title": title,
        "url": url,
        "content": cleaned_content,
        "raw_content": None
    }

async def get_content(query, num_results):
    """Main function to get search results with proper backup handling"""
    # Get all URLs (primary + backup)
    all_urls = await get_ddg_links(query, num_results)
    contents = []
    successful_urls = 0
    url_index = 0
    
    # Continue processing URLs until we have enough content or run out of URLs
    while successful_urls < num_results and url_index < len(all_urls):
        url = all_urls[url_index]
        url_index += 1
        
        result = await process_url(url, query)
        if result:
            contents.append(result)
            successful_urls += 1
    
    return contents

async def main(query, num_results):
    contents = await get_content(query, num_results)
    
    # Print summary
    print(f"\nSummary: Retrieved {len(contents)} successful results out of {num_results} requested")
    
    with open("../data/isolated_ddg_search_result.json", "w", encoding="utf-8") as f:
        json.dump(contents, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]  # python src/agent/isolated_ddg_search.py "query"
    else:
        print("Usage: python src/agent/isolated_ddg_search.py \"your search query here\"")
        sys.exit(1)

    num_results = 4

    asyncio.run(main(query, num_results))