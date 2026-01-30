import google.generativeai as genai
import re
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_api_key(env_path, key_name):
    with open(env_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(key_name):
                return line.split('=')[1].strip()
    return None


def load_map(path):
    with open(path, 'r') as f:
        json_data = json.load(f)
    
    # We converted numpy arrays to list of numbers for json serialization, we revert that here
    # It is easier to work with numpy arrays for similarity and distance calculation
    for key, value in json_data.items():
        json_data[key] = np.array(value)
    
    return json_data


def get_similarity(embedding_map, company1, company2):
    return cosine_similarity([embedding_map[company1]], [embedding_map[company2]])[0][0]


def contains_yes(s):
    return bool(re.search(r'\b(yes)\b', s, re.IGNORECASE))

    
def get_llm_judgement(company1, company2, model, prompt):
    response = model.generate_content(prompt + f"[{company1}, {company2}]")
    return contains_yes(response.text)


# path burada değişebilir, eğer MongoDB'ye koyacaksak uri verip, connection açarız
# şimdilik direkt json üzerine çalışacağımızı varsaydım
def resolve_companies(path, threshold, error, model, prompt):
    # we are using dfs for finding matched groups
    def dfs(company_set, visited, company, group, embedding_map):
        visited.add(company)
        group.append(company)
        for other_company in company_set:
            if other_company not in visited:
                score = get_similarity(embedding_map, company, other_company)
                if score > threshold + error:
                    dfs(company_set, visited, other_company, group)
                elif (
                    threshold + error >= score >= threshold - error and 
                    get_llm_judgement(company, other_company, model, prompt)
                ):                    
                    dfs(company_set, visited, other_company, group)


    # assume we load a file where the structure is identical the dummy data
    company_sets = [
        ["comp_0_0", "comp_0_1", "comp_0_2"],
        ["comp_1_0", "comp_1_1", "comp_1_2"]
    ]

    # this should load the embedding map
    embedding_map_path = "../data/embedding_map.json"
    embedding_map = load_map(embedding_map_path)

    merged_companies = []

    # main loop
    for company_set in company_sets:
        visited = set()
        for company in company_set:
            if company not in visited:
                group = []
                dfs(company_set, visited, company, group, embedding_map)
                merged_companies.append(group)


    return merged_companies


if __name__ == "__main__":
    env_path = "../company-researcher/.env"
    key_name = "GOOGLE_API_KEY"
    GOOGLE_API_KEY = get_api_key(env_path, key_name)
    print(GOOGLE_API_KEY)

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = """I want you to answer if the following companies given in this format [company1, company2] are the same companies or not. 
                YOU WILL ONLY ANSWER WITH yes or no, you will give no other answer, just one word. the following is the input: """

    print(get_llm_judgement("turkcell (Muğla)", "turkcell (antalya)", model, prompt))
    print(get_llm_judgement("turkcell", "Turkcell İletişim Hizmetleri A.Ş.", model, prompt))

    print("end")