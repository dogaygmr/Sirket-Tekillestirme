import pandas as pd
import json
from collections import defaultdict

class UnionFind:
    def __init__(self):
        self.parent = {}  # Maps each company to its parent
        self.rank = {}  # Keeps track of tree depth

    def find(self, company):
        """Find the root parent of a company with path compression."""
        if self.parent[company] != company:
            self.parent[company] = self.find(self.parent[company])  # Path compression
        return self.parent[company]

    def union(self, company1, company2):
        """Unifies two companies into the same cluster."""
        root1 = self.find(company1)
        root2 = self.find(company2)

        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

    def add_company(self, company):
        """Adds a new company to the data structure if it's not already there."""
        if company not in self.parent:
            self.parent[company] = company
            self.rank[company] = 1

    def get_clusters(self):
        """Groups companies into clusters based on their affiliations and returns a dictionary."""
        clusters = defaultdict(list)
        for company in self.parent:
            root = self.find(company)
            clusters[root].append(company)
        return dict(clusters)

def process_companies(json_data):
    uf = UnionFind()

    # Add all companies first
    for entry in json_data:
        company = entry["company_name"]
        uf.add_company(company)
        
        # Add affiliations
        affiliations = entry["info"]["affiliations"]
        related_companies = affiliations["parents"] + affiliations["children"] + affiliations["partners"]
        
        for related in related_companies:
            uf.add_company(related)
            uf.union(company, related)

    return uf

df = pd.read_json("../data/dummy.json")  
json_data = df.to_dict(orient="records")

uf = process_companies(json_data)
clusters = uf.get_clusters()

# Save the clusters as a JSON file
with open("../data/clusters.json", "w") as outfile:
    json.dump(clusters, outfile, indent=4)

print("Clusters have been saved to '../data/clusters.json'.")