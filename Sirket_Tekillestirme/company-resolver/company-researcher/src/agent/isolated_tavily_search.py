import asyncio
import json
from tavily import AsyncTavilyClient

def get_api_key(env_path, key_name):
    with open(env_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(key_name):
                return line.split('=')[1].strip()
    return None

# Bu kısmı graph.py dosyasından aldım. Tavily async call destekliyor (AsyncTavilyClient), diğer API'ler desteklemeyebilir
async def research_company():
    search_tasks = []
    # graph.py dosyasında query'ler state'ten alınıyor ama biz izole hale getirdiğimiz için sentetik query oluşturdum, yapı aynı
    for query in ["What is kariyernet", "Kariyer.net company info"]:
        search_tasks.append(
            # Asıl search işlemi burada gerçekleşiyor. Diğer kodlar sadece kurulum amaçlı.
            # API çağrısının nasıl yapıldığı ve kullanılan parametreler diğer API'lerde farklılık gösterebilir.  
            # graph.py dosyasında, max_results parametresi configuration.py dosyasındaki max_search_results değerinden alınıyor.  
            # Ancak, bu kod izole çalıştığı için burada doğrudan 3 olarak belirledim (configuration.py'da da 3 belirlemiştik).  
            # (Not: Kredi kullanımını bu max_results parametresi etkilemiyor, kaç sorgu (query) gönderdiğimiz krediyi etkiliyor.  
            # Kaç sorgu göndereceğimizi configuration.py'da max_search_queries parametresi (6 olarak ayarladığımız) belirliyor, 
            # dolayısıyla buradaki max_results değeriyle kredinin ilgisi yok. 
            tavily_async_client.search(
                query,
                max_results=3,
                include_raw_content=True,
                topic="general",
            )
        )

    search_docs = await asyncio.gather(*search_tasks)

    return search_docs

if __name__ == "__main__":
    env_path = ".env"
    key_name = "TAVILY_API_KEY"
    TAVILY_API_KEY = get_api_key(env_path, key_name)

    tavily_async_client = AsyncTavilyClient(TAVILY_API_KEY)

    results = asyncio.run(research_company())

    with open("../data/isolated_tavily_search_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("Results saved to ../data/isolated_tavily_search_result.json")