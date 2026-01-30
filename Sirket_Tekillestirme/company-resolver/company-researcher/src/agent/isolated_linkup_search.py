import json
from langchain_linkup import LinkupSearchRetriever

def get_api_key(env_path, key_name):
    with open(env_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(key_name):
                return line.split('=')[1].strip()
    return None

def research_company():
    search_results = []
    for query in ["What is kariyernet", "Kariyer.net company info"]:
        results = retriever.invoke(query)
        search_results.append(results)

    return search_results

# utils.py dosyasında bulunan deduplicate_and_format_sources beklediği formata çevirmemiz gerkiyor
def transform_linkup_response(linkup_responses):
    transformed_results = []

    for response in linkup_responses:
        transformed_results.append({
            "title": response["metadata"].get("name", "No Title"),
            "url": response["metadata"].get("url", "No URL"),
            "content": response.get("page_content", "No Content"),
            "raw_content": None  # LinkUp API'da bu bilgi yok
        })

    return {"results": transformed_results}  # Tavily API formatına çeviriyoruz

if __name__ == "__main__":
    env_path = ".env"
    key_name = "LINKUP_API_KEY"
    LINKUP_API_KEY = get_api_key(env_path, key_name)

    retriever = LinkupSearchRetriever(
        depth="standard",  # "standard" or "deep"
        linkup_api_key=LINKUP_API_KEY,
    )

    results = research_company()
    results = [doc.__dict__ for doc_list in results for doc in doc_list]
    transformed_results = transform_linkup_response(results)

    with open("../data/isolated_linkup_search_result.json", "w", encoding="utf-8") as f:
        json.dump(transformed_results, f, indent=4, ensure_ascii=False)

    print("Results saved to ../data/isolated_linkup_search_result.json")
