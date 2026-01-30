EXTRACTION_PROMPT = """Your task is to extract structured information from the web research notes based on the provided schema. **It is critical that you do not omit any details, especially regarding affiliations such as parent companies and subsidiary (child) companies.** 

#### **Key Requirements:**
- If the research notes mention **any** parent or subsidiary (child) companies, you **must** include the **entire** list in the corresponding fields under `affiliations`.
- **Do not summarize, filter, or shorten the lists.** Include every company name exactly as mentioned.
- Under 'affiliations', each 'parents' and 'subsidiaries' should be list of strings on their own.
- If no parent or subsidiary companies are mentioned in web search notes, leave the corresponding list empty (`[]`). Otherwise, you have to fill the corresponding list with the provided list in web search notes.

---

#### **Schema Definition:**
<schema>
{info}
</schema>

---

Below are the research notes from which you will extract information:

<web_research_notes>
{notes}
</web_research_notes>
"""

QUERY_WRITER_PROMPT = """You are a search query generator tasked with creating targeted search queries to gather specific company information.

Here is the company you are researching: {company}

Generate at most {max_search_queries} search queries that will help gather the following information (No matter what, the number of queries you generated should NOT exceed {max_search_queries}):

<schema>
{info}
</schema>

<user_notes>
{user_notes}
</user_notes>

Your query should:
1. Focus on finding factual, up-to-date company information
2. Target official sources, news, and reliable business databases
3. Prioritize finding information that matches the schema requirements
4. Include the company name and relevant business terms
5. Be specific enough to avoid irrelevant results

Create a focused query that will maximize the chances of finding schema-relevant information."""

INFO_PROMPT = """You are doing web research on a company, {company}. 

The following schema shows the type of information we're interested in:

<schema>
{info}
</schema>

You have just scraped website content. Your task is to take clear, organized notes about the company, focusing on topics relevant to our interests.

<Website contents>
{content}
</Website contents>

Here are any additional notes from the user:
<user_notes>
{user_notes}
</user_notes>

Please provide detailed research notes that:
1. Are well-organized and easy to read
2. Focus on topics mentioned in the schema
3. Include specific facts, dates, and figures when available
4. Maintain accuracy of the original content
5. Note when important information appears to be missing or unclear
6. Do not omit any information especially when it is about the list of parent companies and subsidiary (child) companies. Write the whole list no matter how long it is

Remember: Don't try to format the output to match the schema - just take clear notes that capture all relevant information."""

REFLECTION_PROMPT = """You are a research analyst tasked with reviewing the quality and completeness of extracted company information.

Compare the extracted information with the required schema:

<Schema>
{schema}
</Schema>

Here is the extracted information:
<extracted_info>
{info}
</extracted_info>

Analyze if all required fields are present and sufficiently populated. Consider:
1. Are any required fields missing?
2. Are any fields incomplete or containing uncertain information?
3. Are there fields with placeholder values or "unknown" markers?
"""
