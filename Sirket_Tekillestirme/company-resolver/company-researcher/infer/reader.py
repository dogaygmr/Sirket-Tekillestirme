import pandas as pd
import itertools
from infer import run_pipeline

def research(companies):
    run_pipeline(affiliate_list = companies)

if __name__ == "__main__":
    output = "../data/dummy.json"
    
    researched_companies = set()

    depth = 1 # depth parametresi
    for _ in range(depth):
        data = pd.read_json(output, encoding='utf-8').to_dict(orient="records")
        researched_companies.update(set([company["info"]["company_name"] for company in data]))
        researched_companies.update(set([company["company_name"] for company in data]))

        current_parents = set(itertools.chain(*[company["info"]["affiliations"]["parents"] for company in data]))
        current_children = set(itertools.chain(*[company["info"]["affiliations"]["children"] for company in data]))
        current_partners = set(itertools.chain(*[company["info"]["affiliations"]["partners"] for company in data]))

        current = current_parents | current_children | current_partners
        research_list = list(current - researched_companies)

        print(research_list)

        if research_list:
            research(research_list)