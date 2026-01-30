from dataclasses import dataclass, field
from typing import Any, Optional, Annotated
import operator


DEFAULT_EXTRACTION_SCHEMA = {
    "title": "CompanyInfo",
    "description": "Basic information about a company",
    "type": "object",
    "properties": {
        "company_name": {
            "type": "string",
            "description": "Official name of the company. Always required.",
        },
        "founding_year": {
            "type": "integer",
            "description": "Year the company was founded. Leave it null if the information is not available. If the year is approximate, extract the closest exact year.",
        },
        "founder_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names of the founding team members. Leave the list empty ([]) if you can't find the names.",
        },
        "product_description": {
            "type": "string",
            "description": "Brief description of the company's main product or service.",
        },
        "branch_dealer": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of branch dealers associated with the company. Leave the list empty ([]) if the company has no branch dealers.",
        },
        "affiliations": {
            "type": "object",
            "description": "Structured relationships of the company with other organizations.",
            "properties": {
                "parents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of parent companies. Leave the list empty ([]) if you can't find a parent company.",
                },
                "subsidiaries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of subsidiary or child companies. Leave the list empty ([]) if you can't find a subsidiary or child company.",
                },
            },
        },
    },
    "required": ["company_name"],
}


@dataclass(kw_only=True)
class InputState:
    """Input state defines the interface between the graph and the user (external API)."""

    company: str
    "Company to research provided by the user."

    extraction_schema: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_EXTRACTION_SCHEMA
    )
    "The json schema defines the information the agent is tasked with filling out."

    user_notes: Optional[dict[str, Any]] = field(default=None)
    "Any notes from the user to start the research process."

    api_key_name: str = field(default="GOOGLE_API_KEY_FOR_SEARCH")
    "Name of the API key to use from .env file"


@dataclass(kw_only=True)
class OverallState:
    """Input state defines the interface between the graph and the user (external API)."""

    company: str
    "Company to research provided by the user."

    extraction_schema: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_EXTRACTION_SCHEMA
    )
    "The json schema defines the information the agent is tasked with filling out."

    user_notes: str = field(default=None)
    "Any notes from the user to start the research process."

    api_key_name: str = field(default="GOOGLE_API_KEY_FOR_SEARCH")
    "Name of the API key to use from .env file"

    search_queries: list[str] = field(default=None)
    "List of generated search queries to find relevant information"

    completed_notes: Annotated[list, operator.add] = field(default_factory=list)
    "Notes from completed research related to the schema"

    info: dict[str, Any] = field(default=None)
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """

    is_satisfactory: bool = field(default=None)
    "True if all required fields are well populated, False otherwise"

    reflection_steps_taken: int = field(default=0)
    "Number of times the reflection node has been executed"

    failed_url_attempts: int = 0
    "Number of URL addresses whose content couldn't be fetched"

@dataclass(kw_only=True)
class OutputState:
    """The response object for the end user.

    This class defines the structure of the output that will be provided
    to the user after the graph's execution is complete.
    """

    info: dict[str, Any]
    """
    A dictionary containing the extracted and processed information
    based on the user's query and the graph's execution.
    This is the primary output of the enrichment process.
    """
