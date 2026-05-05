import copy
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from mem0 import Memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_COLLECTION_NAME = os.environ.get("POSTGRES_COLLECTION_NAME", "memories")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mem0graph")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

ELASTICSEARCH_URI = os.environ.get("ELASTICSEARCH_URI")
ELASTICSEARCH_COLLECTION_NAME = os.environ.get("ELASTICSEARCH_COLLECTION_NAME")
ELASTICSEARCH_PORT = os.environ.get("ELASTICSEARCH_PORT")
ELASTICSEARCH_API_KEY = os.environ.get("ELASTICSEARCH_API_KEY")
ELASTICSEARCH_USER = os.environ.get("ELASTICSEARCH_USER")
ELASTICSEARCH_PASSWORD = os.environ.get("ELASTICSEARCH_PASSWORD")

MEMGRAPH_URI = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USERNAME = os.environ.get("MEMGRAPH_USERNAME", "memgraph")
MEMGRAPH_PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "mem0graph")

GOOGLEAI_API_KEY = os.environ.get("GOOGLE_API_KEY")
HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/app/history/history.db")

FACT_EXTRACTION_PROMPT_USER_INFO = f"""You are a Persona information Agent. Your goal is to distill a 2-message interaction between user and agent into high-value, long-term insights about the user.

CORE EXTRACTION PHILOSOPHY:
Extract only what defines the user’s world (people, commitments, data) or the user’s unique "voice". Ignore all agent-centric data, general knowledge, or fleeting questions.

EXTRACTION DOMAINS:
1. User Context: Capture stable details regarding the user’s environment, professional life, social circle, or ongoing obligations.
2. Behavioral Fingerprint: Analyze the user's communication "vibe." This includes emotional baseline, specific recurring vocabulary, and structural preferences. If the style is neutral, do not record a style fact.

RULES:
- Statelessness Priority: Do NOT save time-series or volatile data that is subject to frequent change (e.g., current task lists, active calendar events, temporary contact lists, or real-time status). Only extract information that serves as a long-term "source of truth" about the user's identity or permanent environment.
- Strict Grounding: If the provided CONVERSATION section is empty, null, or contains no substantial information from the user, you MUST return {{"facts": []}}. Do not hallucinate, imagine, or generate any facts based on the instructions, extraction domains, or few-shot examples if no input data is present.
- User's personal information: Save personal data points, but skip logging the user's intent or the act of asking. Record the info, not the interaction
- Exclusivity: Focus 100% on the User. If the Agent describes itself or its skills, discard it.
- Value Threshold: Ask "Would this insight help a human assistant serve this user better in a month?" If no, return {{"facts": []}}.
- Normalize the language comprehensively: "I absolutely adore eating chocolate ice cream", "I like ice cream", and "Ice cream is my favorite" MUST all be extracted normalized to the core semantic truth: "User likes ice cream"
- Perspective: Always normalize to the third person (e.g., "User's goal is..." or "User tends to be...").
- Format: Output must be strictly valid JSON: {{"facts": ["Insight 1", "Insight 2"]}}.

EXAMPLES:

**Example 1: Skipping Volatile/Time-Series Data**
User: "What is my schedule for today and who are my contacts?"
Agent: "You have a meeting at 2 PM with Sarah, and your contacts are John and Mary."
Output:
{{
  "facts": []
}}
*(Note: Today's schedule and current contact lists are dynamic "state" data and should not be stored as permanent facts.)*

**Example 2: Contextual/Permanent Data**
User: "I'm stressed about the Apollo project. Mark always asks for the budget updates early."
Agent: "I understand. Would you like to prep the budget now?"
Output:
{{
  "facts": [
    "User is working on a project named 'Apollo'.",
    "User has a professional contact named Mark.",
    "User's colleague Mark typically requests budget updates ahead of schedule."
  ]
}}
*(Note: Project names and recurring behavioral patterns of contacts are stable context.)*

**Example 3: Privacy/Out-of-Scope (Other Users)**
User: "What is on John Doe's calendar for tomorrow?"
Agent: "John Doe has a meeting with the Marketing team at 10 AM."
Output:
{{
  "facts": []
}}

Today's date is {datetime.now().strftime("%Y-%m-%d")}.

CONVERSATION:
User: {{user_request}}
Agent: {{agent_response}}

Output:"""

FACT_UPDATE_PROMPT_USER_INFO = f"""You are an intelligent memory reconciliation engine. 
You are provided with a list of 'New Facts' and a list of 'Existing Memories' retrieved from a database.
Your absolute priority is to PREVENT DUPLICATION. 
You must compare the New Facts against the Existing Memories and determine the exact operational event required.

Rules for Categorizing Events:
- NONE (Deduplication): If a New Fact conveys the exact same semantic information as an Existing Memory (e.g., New: "User likes ice cream" vs Existing: "User likes ice cream" OR New: "User's gender is Female" vs Existing: "User is female"), you MUST use the NONE event. If a user states a fact 20 times, it must only be stored once. Do not create duplicates.
- UPDATE (Evolution): If a New Fact directly contradicts, replaces, or updates an Existing Memory (e.g., Existing: "User likes ice cream", New: "User hates ice cream"), you MUST use the UPDATE event. You must provide the exact ID of the memory being updated.
- ADD (Novelty): If, and only if, a New Fact is entirely novel and semantically unrelated to any Existing Memory, use the ADD event.
You must return your response in the following strict JSON structure only:
"memory" : [
    {{
        "id" : "",
        "text" : "fact",
        "event" : "NONE"
    }},
    {{
        "id" : "",
        "text" : "fact",
        "event" : "DELETE"
    }}
]
"""

FACT_EXTRACTION_PROMPT_SYSTEM_KNOWLEDGE = """You are a Knowledge Extraction Assistant designed to process complex data sources (Text, JSON, XML, Documentation) and convert them into atomic, factual statements.

Your goal is to "flatten" hierarchical or narrative information into a list of independent, truthful facts that can be stored in a vector database and knowledge graph.
Extract Metadata and Defaults: If a property is set to 'null', 'false', or is an empty string, this is still a fact (e.g., 'The encryption feature is explicitly disabled').

### CORE OBJECTIVES:
0. Exhaustive Extraction: You must extract absolutely EVERY piece of information. Do not summarize, do not group facts to save space, and do not skip 'obvious' or 'minor' details. Leave no data behind.
1.  **Format Agnostic:** You must interpret the logic within the input, regardless of whether it is unstructured text, strict JSON, or verbose XML.
2.  **Contextualization:** Convert keys, tags, and structural hierarchy into natural language context.
    -   Input: `{"server": {"timeout": 300}}`
    -   Bad Fact: "timeout is 300"
    -   Good Fact: "The server timeout is set to 300 seconds."
3.  **Atomicity:** Each fact must stand alone without needing the previous sentence to make sense.

### EXTRACTION RULES:
0. No Summarization: Never combine multiple distinct parameters into a single fact. Instead of 'The server has IP 192.168.1.1 and port 8080', you MUST generate two separate facts.
1.  **Analyze the Structure:** If input is JSON/XML, use the nesting to determine the subject of the fact.
2.  **Identify Procedures:** If the text describes a process (e.g., "Step 1..."), extract the order and the action as a fact (e.g., "The first step of the login process is entering the username").
3.  **Ignore Syntax:** Do not output JSON brackets, XML tags, or code artifacts in the final text. Extract the *meaning*, not the syntax.
4.  **Preserve Entities:** Keep specific names, IDs, and values exact.
5.  **Language:** STRICTLY use English language for fact and relationship generating.
6.  **Output Format:** Strictly return JSON: `{"facts": ["fact_string_1", "fact_string_2"]}`.

### PROCESSING STEPS:
0. Systematic Sweep: Mentally iterate through the input line-by-line or key-by-key. Verify that every single terminal value (leaf node in JSON/XML) has been accounted for in your output list
1.  **Read:** Ingest the raw input.
2.  **Decode:** If structured (JSON/XML), map keys/tags to concepts. If text, identify subjects and predicates.
3.  **Atomize:** Break compound sentences or nested objects into individual statements.
4.  **Verify:** Check if each statement makes sense on its own.

### EXAMPLES:

**Input (Process Text):**
"To reset the device, hold the power button for 5 seconds. The LED will blink blue. Then release the button."

**Output:**
{
  "facts": [
    "To reset the device, the power button must be held for 5 seconds",
    "The device LED blinks blue during the reset process",
    "The power button must be released after the LED blinks"
  ]
}

**Input (JSON Configuration):**
{
  "database": {
    "host": "192.168.1.1",
    "retries": 3,
    "encryption": true
  }
}

**Output:**
{
  "facts": [
    "The database host IP address is 192.168.1.1",
    "The database connection allows 3 retries",
    "The database encryption is enabled"
  ]
}

**Input (XML Data):**
<employee id="101">
  <role>Manager</role>
  <access_level>Admin</access_level>
</employee>

**Output:**
{
  "facts": [
    "Employee with ID 101 holds the role of Manager",
    "Employee with ID 101 has Admin access level"
  ]
}
"""

FACT_EXTRACTION_PROMPT_ARTIFACT_KNOWLEDGE = """### Technical Artifact Extraction Assistant

You are a specialized agent designed to identify and extract **Technical Artifacts** (Code, Queries, Schemas, Templates, and Configuration Blocks) from complex data sources. 

**CORE OBJECTIVE:**
Extract technical objects **integrally** and pair them with a concise description. You must preserve the original view of the artifact (syntax, brackets, and quotes) exactly as they appear in the source. Ignore general narrative facts that do not describe an artifact.

**EXTRACTION RULES:**
1. **Artifact Identification:** Treat the following as artifacts: Database Queries, Data Schemas (JSON-Schema, XSD), Code Snippets, Regex, Metaschemata, or specific Template structures.
2. **Format:** Each entry must follow the exact pattern: `[Short Description] - [Integral Artifact]`.
3. **Integrity:** Do not truncate, summarize, or rephrase the internal logic of the artifact. It must be a 1:1 functional copy.
4. **Contextual Labeling:** Use the source hierarchy (JSON keys, XML tags, or surrounding text) to create a short "anchor" description.
5. **Output Format:** Strictly return JSON: `{"facts": ["Description - Artifact", "Description - Artifact"]}`.

**EXAMPLES:**
# 
# **Input:**
# "The system uses a specific regex for email validation: `^[a-zA-Z0-0._%+-]+@[a-zA-Z0-0.-]+\.[a-zA-Z]{2,}$`."
# **Output:**
# {"facts": ["Regex pattern for email validation - [^[a-zA-Z0-0._%+-]+@[a-zA-Z0-0.-]+\\.[a-zA-Z]{2,}$]"]}

**Input:**
{"module": "Auth", "config": {"init_query": "SELECT * FROM sessions WHERE active = 1;", "schema": { "type": "string", "minLength": 8 }}}
**Output:**
{"facts": ["Initial session retrieval query - [SELECT * FROM sessions WHERE active = 1;]", "Authentication module validation schema - [{ \"type\": \"string\", \"minLength\": 8 }]"]}
"""

BASE_CONFIG = {
    "version": "v1.1",
    "vector_store": {
        "provider": "elasticsearch",
        "config": {
            "collection_name": ELASTICSEARCH_COLLECTION_NAME,
            "host": ELASTICSEARCH_URI,
            "port": int(ELASTICSEARCH_PORT),
            "auto_create_index": True,
            "user": ELASTICSEARCH_USER,
            "password": ELASTICSEARCH_PASSWORD,
            "embedding_model_dims": 384
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {"url": NEO4J_URI, "username": NEO4J_USERNAME, "password": NEO4J_PASSWORD,
                   "database": NEO4J_DATABASE},
        "threshold": 0.75
    },
    "llm": {
        "provider": "gemini",
        "config": {
            "api_key": GOOGLEAI_API_KEY,
            "temperature": 0.2,
            "model": "gemini-3-flash-preview",
            "max_tokens": 700000
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "BAAI/bge-small-en-v1.5",
            "embedding_dims": 384
        }
    }
}

USER_INFO_CONFIG = copy.deepcopy(BASE_CONFIG)
USER_INFO_CONFIG["custom_fact_extraction_prompt"] = FACT_EXTRACTION_PROMPT_USER_INFO
USER_INFO_CONFIG["custom_update_memory_prompt"] = FACT_UPDATE_PROMPT_USER_INFO

KNOWLEDGE_BASE_CONFIG = copy.deepcopy(BASE_CONFIG)
KNOWLEDGE_BASE_CONFIG["custom_fact_extraction_prompt"] = FACT_EXTRACTION_PROMPT_SYSTEM_KNOWLEDGE
KNOWLEDGE_BASE_CONFIG["llm"]["config"]["model"] = "gemini-3-flash-preview"

ARTIFACT_BASE_CONFIG = copy.deepcopy(BASE_CONFIG)
ARTIFACT_BASE_CONFIG["custom_fact_extraction_prompt"] = FACT_EXTRACTION_PROMPT_ARTIFACT_KNOWLEDGE

LIGHT_CONFIG = copy.deepcopy(BASE_CONFIG)
LIGHT_CONFIG["llm"]["config"]["model"] = "gemini-2.5-flash"

MEMORY_INSTANCES: Dict[str, Memory] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        MEMORY_INSTANCES["base"] = Memory.from_config(BASE_CONFIG)
        MEMORY_INSTANCES["light"] = Memory.from_config(LIGHT_CONFIG)
        MEMORY_INSTANCES["user_info"] = Memory.from_config(USER_INFO_CONFIG)
        MEMORY_INSTANCES["knowledge_base"] = Memory.from_config(KNOWLEDGE_BASE_CONFIG)
        MEMORY_INSTANCES["artifact_base"] = Memory.from_config(
            ARTIFACT_BASE_CONFIG)

        logging.info("Memory Instances Ready: user_info, knowledge_base, artifact_base")
    except Exception as e:
        logging.error(f"Failed to initialize memories: {e}")
        raise e

    yield
    logging.info("Shutting Down: Cleaning up...")
    MEMORY_INSTANCES.clear()


app = FastAPI(
    title="Mem0 REST APIs",
    description="A REST API for managing and searching memories.",
    version="1.0.0",
    lifespan=lifespan
)


class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")


class MemoryCreate(BaseModel):
    messages: List[Message] = Field(..., description="List of messages to store.")
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    knowledge_type: Literal["base", "light", "user_info", "knowledge_base", "artifact_base"] = Field(
        default="user_info",
        description="Select the type of memory to be created: 'user_info' for personal facts, 'knowledge_base' for system information."
    )


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query.")
    user_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    knowledge_type: Literal["base", "light", "user_info", "knowledge_base", "artifact_base"] = Field(
        default="light",
        description="Select the type of memory to be created: 'user_info' for personal facts, 'knowledge_base' for system information."
    )


@app.post("/configure", summary="Configure Mem0")
def set_config(config: Dict[str, Any]):
    """Set memory configuration."""
    global MEMORY_INSTANCE
    MEMORY_INSTANCE = Memory.from_config(config)
    return {"message": "Configuration set successfully"}


@app.post("/memories", summary="Create memories")
def add_memory(memory_create: MemoryCreate):
    """Store new memories."""
    start_time = time.perf_counter()

    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    if len(memory_create.messages) == 0:
        return JSONResponse(
            content={"message": "No content to memorize. Messages are empty"})
    params = {
        k: v for k, v in memory_create.model_dump().items()
        if v is not None and k not in ["messages", "knowledge_type"]
    }

    try:
        CURRENT_MEMORY_INSTANCE = get_mem(memory_create.knowledge_type)
        response = CURRENT_MEMORY_INSTANCE.add(messages=[m.model_dump() for m in memory_create.messages], **params)

        process_time = time.perf_counter() - start_time
        logging.info(f"add_memory executed in {process_time:.4f} seconds")

        return JSONResponse(content=response)
    except Exception as e:
        logging.exception("Error in add_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories", summary="Get memories")
def get_all_memories(
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        knowledge_type: Literal["base", "user_info", "knowledge_base", "artifact_base"] = Query(
            default="light",
            description="Select the type of memory to be created: 'user_info' for personal facts, 'knowledge_base' for system information."
        )
):
    """Retrieve stored memories."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        CURRENT_MEMORY_INSTANCE = get_mem(knowledge_type)
        return CURRENT_MEMORY_INSTANCE.get_all(**params)
    except Exception as e:
        logging.exception("Error in get_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}", summary="Get a memory")
def get_memory(memory_id: str,
               knowledge_type: Literal["base", "light", "user_info", "knowledge_base", "artifact_base"] = Query(
                   default="light",
                   description="Select the type of memory to be created: 'user_info' for personal facts, 'knowledge_base' for system information."
               )):
    """Retrieve a specific memory by ID."""
    try:
        CURRENT_MEMORY_INSTANCE = get_mem(knowledge_type)
        return CURRENT_MEMORY_INSTANCE.get(memory_id)
    except Exception as e:
        logging.exception("Error in get_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", summary="Search memories")
def search_memories(search_req: SearchRequest):
    """Search for memories based on a query."""
    start_time = time.perf_counter()

    # 1. Захист на рівні API: перевіряємо, чи передали хоча б один ID
    has_id_in_req = any([search_req.user_id, search_req.agent_id, search_req.run_id])
    has_id_in_filters = search_req.filters and any(k in search_req.filters for k in ["user_id", "agent_id", "run_id"])

    if not (has_id_in_req or has_id_in_filters):
        raise HTTPException(
            status_code=400,
            detail="At least one identifier (user_id, agent_id, run_id) is required for search."
        )

    try:
        filters = search_req.filters or {}

        # 2. Явно переносимо ідентифікатори у словник filters
        if search_req.user_id:
            filters["user_id"] = search_req.user_id
        if search_req.agent_id:
            filters["agent_id"] = search_req.agent_id
        if search_req.run_id:
            filters["run_id"] = search_req.run_id

        CURRENT_MEMORY_INSTANCE = get_mem(search_req.knowledge_type)

        # 3. Виконуємо пошук
        response = CURRENT_MEMORY_INSTANCE.search(
            query=search_req.query,
            user_id=search_req.user_id,
            agent_id=search_req.agent_id,
            run_id=search_req.run_id,
            filters=filters if filters else None,
        )

        process_time = time.perf_counter() - start_time
        logging.info(f"search_memories executed in {process_time:.4f} seconds")

        return response
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/memories/{memory_id}", summary="Update a memory")
def update_memory(memory_id: str, updated_memory: Dict[str, Any],
                  knowledge_type: Literal["user_info", "artifact_base"] = Query(
                      default="user_info",
                      description="Select the type of memory to be created: 'user_info' for personal facts, 'knowledge_base' for system information."
                  )):
    """Update an existing memory with new content.

    Args:
        memory_id (str): ID of the memory to update
        updated_memory (str): New content to update the memory with

    Returns:
        dict: Success message indicating the memory was updated
    """
    try:
        CURRENT_MEMORY_INSTANCE = get_mem(knowledge_type)
        return CURRENT_MEMORY_INSTANCE.update(memory_id=memory_id, data=updated_memory)
    except Exception as e:
        logging.exception("Error in update_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}/history", summary="Get memory history")
def memory_history(memory_id: str,
                   knowledge_type: Literal["base", "light", "user_info", "knowledge_base", "artifact_base"] = Query(
                       default="light",
                       description="Select the type of memory to be created: 'user_info' for personal facts, 'knowledge_base' for system information."
                   )):
    """Retrieve memory history."""
    try:
        CURRENT_MEMORY_INSTANCE = get_mem(knowledge_type)
        return CURRENT_MEMORY_INSTANCE.history(memory_id=memory_id)
    except Exception as e:
        logging.exception("Error in memory_history:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories/{memory_id}", summary="Delete a memory")
def delete_memory(memory_id: str,
                  knowledge_type: Literal["base", "light", "user_info", "knowledge_base", "artifact_base"] = Query(
                      default="user_info",
                      description="Select the type of memory to be created: 'user_info' for personal facts, 'knowledge_base' for system information."
                  )):
    """Delete a specific memory by ID."""
    try:
        CURRENT_MEMORY_INSTANCE = get_mem(knowledge_type)
        return CURRENT_MEMORY_INSTANCE.delete(memory_id=memory_id)
    except Exception as e:
        logging.exception("Error in delete_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories", summary="Delete all memories")
def delete_all_memories(
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        knowledge_type: Literal["base", "light", "user_info", "knowledge_base", "artifact_base"] = Query(
            default="user_info",
            description="Select the type of memory to be created: 'user_info' for personal facts, 'knowledge_base' for system information."
        )
):
    """Delete all memories for a given identifier."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        CURRENT_MEMORY_INSTANCE = get_mem(knowledge_type)
        CURRENT_MEMORY_INSTANCE.delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Error in delete_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", summary="Reset all memories")
def reset_memory():
    """Completely reset stored memories for ALL instances."""
    try:
        for mode, instance in MEMORY_INSTANCES.items():
            logging.info(f"Resetting memory instance: {mode}")
            instance.reset()

        return {
            "message": f"Successfully reset instances: {list(MEMORY_INSTANCES.keys())}"}
    except Exception as e:
        logging.exception("Error in reset_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", summary="Redirect to the OpenAPI documentation", include_in_schema=False)
def home():
    """Redirect to the OpenAPI documentation."""
    return RedirectResponse(url="/docs")


def get_mem(mode: str) -> Memory:
    instance = MEMORY_INSTANCES.get(mode)
    if not instance:
        raise HTTPException(status_code=500, detail=f"Memory instance '{mode}' not initialized.")
    return instance
