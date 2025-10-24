import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
from datetime import datetime
import time
import os
from strands import Agent, tool
from bedrock_agentcore.memory.integrations.strands.config import (
    AgentCoreMemoryConfig,
    RetrievalConfig,
)
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from datetime import datetime, timedelta
import json
import boto3


app = BedrockAgentCoreApp()

MEMORY_ID = os.getenv("BEDROCK_AGENTCORE_MEMORY_ID")
REGION = os.getenv("AWS_REGION")
MODEL_ID = (
    "arn:aws:bedrock:us-east-2:746630811346:inference-profile/us.amazon.nova-micro-v1:0"
)
bedrock_client = boto3.client('bedrock-runtime', region_name=REGION)
ci_sessions = {}
CACHE_TTL_DAYS = 1
CACHE_DIR = "cache/"

current_session = None
memory_session_manager = None  # store globally

# --------------------------------
# CONFIG
# --------------------------------
BASE_URL = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
ROLE = "Data Scientist"
LOCATION = "United States"
MAX_PAGES = 2  # scrape first 5 pages
OUTPUT_FILE = "data_scientist_jobs.csv"

DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "Job-market-agent-db")
CACHE_DURATION_SECONDS = 5 * 24 * 60 * 60  # 5 days

# --- NEW: Initialize Boto3 clients ---
dynamodb = boto3.resource('dynamodb', region_name=REGION)
cache_table = dynamodb.Table(DYNAMODB_TABLE_NAME)


# --------------------------------
# HELPER FUNCTIONS
# --------------------------------


def get_job_listings(role, location, pages=1):
    """Fetch job IDs from LinkedIn guest job listings."""
    job_ids = []

    for page in range(pages):
        start = page * 25
        url = f"{BASE_URL}?keywords={role.replace(' ', '%20')}&location={location.replace(' ', '%20')}&start={start}"
        print(f"Fetching: {url}")
        response = requests.get(url)
        if response.status_code != 200:
            print(f"⚠️ Skipped page {page} — status code {response.status_code}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        for li in soup.find_all("li"):
            div = li.find("div", {"class": "base-card"})
            if div and div.get("data-entity-urn"):
                job_id = div["data-entity-urn"].split(":")[-1]
                job_ids.append(job_id)

        time.sleep(1)  # avoid hitting LinkedIn too fast
    return job_ids


def get_job_details(job_id):
    """Fetch full job details from a given job_id."""
    job_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
    res = requests.get(job_url)
    if res.status_code != 200:
        return None

    soup = BeautifulSoup(res.text, "html.parser")

    def extract_text(selector, class_name):
        tag = soup.find(selector, {"class": class_name})
        return tag.text.strip() if tag else None

    job = {
        "job_id": job_id,
        "title": extract_text("h2", "top-card-layout__title"),
        "company": extract_text("a", "topcard__org-name-link"),
        "description": clean_html(
            str(soup.find("div", {"class": "show-more-less-html__markup"}))
        ),
        "url": f"https://www.linkedin.com/jobs/view/{job_id}",
        "fetched_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return job


def clean_html(text):
    """Remove HTML tags and clean up whitespace."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(" ")
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    return clean_text


def scrape_jobs(role=ROLE, location=LOCATION, pages=MAX_PAGES):
    """Scrape and compile job data."""
    job_ids = get_job_listings(role, location, pages)
    print(f"Found {len(job_ids)} job IDs")

    jobs = []
    for jid in job_ids:
        job = get_job_details(jid)
        if job:
            jobs.append(job)
        time.sleep(1)
    return jobs


def save_to_csv(jobs, file_name):
    df = pd.DataFrame(jobs)
    df.drop_duplicates(subset="job_id", inplace=True)
    df.to_csv(file_name, index=False)
    print(f"✅ Saved {len(df)} jobs to {file_name}")


def load_from_cache(role):
    path = f"{CACHE_DIR}/{role.replace(' ', '_')}.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            if datetime.now() - datetime.fromisoformat(data["timestamp"]) < timedelta(
                days=30
            ):
                return data["jobs"]
    return None

import re

def extract_skills_from_jobs(jobs_list: list) -> dict:
    """
    Extracts and counts predefined skills from a list of job dictionaries.

    Args:
        jobs_list: A list of 'job' dictionaries, where each dict is expected
                   to have a 'description' key with a string value.

    Returns:
        A dictionary where keys are standardized skill names (e.g., "Python")
        and values are the count of jobs mentioning that skill.
    """
    
    # --- Your Skill Database ---
    SKILL_DATABASE = {
        "Python": ['python'],
        "Java": ['java'],
        "JavaScript": ['javascript', 'js'],
        "SQL": ['sql'],
        "R": [r'\b r\b'],
        "C++": ['c++', 'cpp'],
        "C#": ['c#', 'csharp'],
        "AWS": ['aws', 'amazon web services'],
        "Azure": ['azure'],
        "Google Cloud (GCP)": ['gcp', 'google cloud'],
        "Tableau": ['tableau'],
        "Power BI": ['power bi', 'powerbi'],
        "Spark": ['spark', 'apache spark'],
        "Pandas": ['pandas'],
        "NumPy": ['numpy'],
        "Excel": ['excel'],
        "React": ['react'],
        "Node.js": ['node.js', 'nodejs'],
        "Docker": ['docker'],
        "Kubernetes": ['kubernetes', 'k8s'],
        "Git": ['git'],
        "Agile": ['agile', 'scrum'],
    }

    skill_counts = {}
    
    if not isinstance(jobs_list, list):
        print("Error: Input must be a list of job dictionaries.")
        return {}

    # --- This is the key change ---
    for job in jobs_list:
        if not isinstance(job, dict):
            continue  # Skip any item that isn't a dictionary
        
        # Get the description string from the job dictionary
        desc = job.get("description")

        if not desc or not isinstance(desc, str):
            continue # Skip if description is missing or not a string
            
        skills_found_in_this_job = set()

        for skill_name, keywords in SKILL_DATABASE.items():
            for keyword in keywords:
                # Search for the keyword as a whole word, ignoring case
                if re.search(r'\b' + re.escape(keyword) + r'\b', desc, re.IGNORECASE):
                    skills_found_in_this_job.add(skill_name)
                    break 
        
        # Update the master count
        for skill in skills_found_in_this_job:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1

    return skill_counts



# @tool
# def scrape_job_market(role: str) -> str:
#     session_id = current_session or 'default'
#     jobs = scrape_jobs(role=role)
#     save_to_csv(jobs, f"{role.replace(' ', '_')}_jobs.csv")
#     return f"Scraped and saved {len(jobs)} {role} postings."

@tool
def scrape_and_extract_skills(role: str) -> str:
    """
    Scrapes job postings for a specific role, extracts the most common 
    skills from the descriptions, and returns a summary of those skills.
    This tool caches results for 5 days to avoid re-scraping.

    :param role: The job title or keyword to search for (e.g., 'data scientist').
    :return: A string summarizing the job count and top skills found.
    """
    global current_session
    session_id = current_session or 'default'
    print(f"[TOOL LOG] 'scrape_and_extract_skills' CALLED for role '{role}'")
    
    # Standardize the role key for caching (e.g., "data analyst")
    # cache_key = role.lower().strip()
    # current_time = int(time.time())

    # try:
        # --- 1. CHECK CACHE FIRST ---
        # print(f"[CACHE LOG] Checking DynamoDB table '{DYNAMODB_TABLE_NAME}' for key '{cache_key}'")
        # cache_response = cache_table.get_item(Key={'role': cache_key})
        
        # if 'Item' in cache_response:
        #     item = cache_response['Item']
        #     item_timestamp = int(item.get('timestamp', 0))
            
        #     # Check if cache is still valid
        #     if (current_time - item_timestamp) < CACHE_DURATION_SECONDS:
        #         print("[CACHE LOG] Cache HIT. Returning cached data.")
        #         return item['skills_summary']
        #     else:
        #         print("[CACHE LOG] Cache STALE. Proceeding to scrape.")
        # else:
        #     print("[CACHE LOG] Cache MISS. Proceeding to scrape.")

        # # --- 2. CACHE MISS/STALE: RUN THE REAL LOGIC ---
    print("[TOOL LOG] Scraping and extracting new data...")
    jobs = scrape_jobs(role=role)
    job_count = len(jobs)
    
    if not jobs:
        return f"No job postings found for the role: {role}."
    print(jobs)
    skills = extract_skills_from_jobs(jobs)
    if not skills:
        return f"Found {job_count} jobs for '{role}', but could not extract any common skills."

    # Format the skill summary
    sorted_skills = sorted(skills.items(), key=lambda item: item[1], reverse=True)
    
    # Now, just get the skill names (the keys) from the sorted list
    skill_names_list = [skill for skill, count in sorted_skills]
    
    # Join them into a simple, comma-separated string
    skill_summary_str = ", ".join(skill_names_list)

    
    # This is the final string we will return AND cache
    response_str = f"From {job_count} '{role}' job postings, I extracted the following jobs\n{jobs}"

    
    print(f"[TOOL LOG] Returning jobs:\n{response_str}")
    return response_str

    # except Exception as e:
    #     print(f"[TOOL ERROR] An exception occurred: {str(e)}")
    #     traceback.print_exc(file=sys.stdout)
    #     return f"Error: The tool failed to run. Check logs for details: {str(e)}"

@app.entrypoint
def invoke(payload, context):
    global current_session

    if not MEMORY_ID:
        return {"error": "Memory not configured"}

    actor_id = context.headers.get('X-Amzn-Bedrock-AgentCore-Runtime-Custom-Actor-Id', 'user') if hasattr(context, 'headers') else 'user'

    session_id = getattr(context, 'session_id', 'default')
    current_session = session_id

    memory_config = AgentCoreMemoryConfig(
        memory_id=MEMORY_ID,
        session_id=session_id,
        actor_id=actor_id,
        retrieval_config={
            f"/users/{actor_id}/facts": RetrievalConfig(top_k=3, relevance_score=0.5),
            f"/users/{actor_id}/preferences": RetrievalConfig(top_k=3, relevance_score=0.5)
        }
    )

    system_prompt = """
    <Instructions>
    You are a specialized Job Market Agent. Your primary function is to use the available tools to find and process job postings.
    1. When a user asks for jobs, your first step is to identify the specific job role or keyword from their request.
    2. Once you have identified the role, you MUST call the `scrape_job_market` tool.
    3. Pass the identified role as the `role` parameter to the tool.
    4. After the tool runs, and gets job description, your job is to find the most common skills from the job description extracted by the tool. donot hallucinate that.
    5. for job description, include list of most useful commonly used skills, it can be technical or soft skils.
    Do not answer questions that are not related to finding jobs.
    </Instructions>
    
    <example>
    User: "I want to be a product manager"
    Agent Action: Calls `scrape_job_market` with the parameter `role='product manager'`
    </example>
    """

    agent = Agent(
        model=MODEL_ID,
        session_manager=AgentCoreMemorySessionManager(memory_config, REGION),
        system_prompt=system_prompt,
        tools=[scrape_and_extract_skills]  
    )

    result = agent(payload.get("prompt", ""))
    return {"response": result.message.get('content', [{}])[0].get('text', str(result))}


if __name__ == "__main__":
    app.run()
    # role = "Software Developer"
    # result = scrape_and_extract_skills(role)
    # print(result)
