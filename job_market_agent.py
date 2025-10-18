import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
from datetime import datetime
import time
import os
from strands import Agent, tool
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

MEMORY_ID = os.getenv("BEDROCK_AGENTCORE_MEMORY_ID")
REGION = os.getenv("AWS_REGION")
MODEL_ID = "arn:aws:bedrock:us-east-2:746630811346:inference-profile/us.amazon.nova-micro-v1:0"

ci_sessions = {}
current_session = None


# --------------------------------
# CONFIG
# --------------------------------
BASE_URL = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
ROLE = "Data Scientist"
LOCATION = "United States"
MAX_PAGES = 5  # scrape first 5 pages
OUTPUT_FILE = "data_scientist_jobs.csv"

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
        "description": clean_html(str(soup.find("div", {"class": "show-more-less-html__markup"}))),
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

@tool
def scrape_job_market(role: str = "Data Scientist") -> str:
    session_id = current_session or 'default'
    jobs = scrape_jobs(role=role)
    save_to_csv(jobs, f"{role.replace(' ', '_')}_jobs.csv")
    return f"Scraped and saved {len(jobs)} {role} postings."


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

    agent = Agent(
        model=MODEL_ID,
        session_manager=AgentCoreMemorySessionManager(memory_config, REGION),
        # system_prompt="You are a helpful assistant. Use tools when appropriate.",
        system_prompt="You are a Job Market Agent. You fetch and summarize job postings for given roles using LinkedIn public data.",
        tools=[scrape_job_market]
    )

    result = agent(payload.get("prompt", ""))
    return {"response": result.message.get('content', [{}])[0].get('text', str(result))}

if __name__ == "__main__":
    app.run()