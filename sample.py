# @tool
# def scrape_job_market(role: str) -> str:
#     session_id = current_session or 'default'
#     return "Function called successfully"
    # jobs = scrape_jobs(role=role)
    # save_to_csv(jobs, f"{role.replace(' ', '_')}_jobs.csv")
    # return f"Scraped and saved {len(jobs)} {role} postings."
# @app.entrypoint
# def invoke(payload, context):
#     global current_session

#     if not MEMORY_ID:
#         return {"error": "Memory not configured"}

#     actor_id = context.headers.get('X-Amzn-Bedrock-AgentCore-Runtime-Custom-Actor-Id', 'user') if hasattr(context, 'headers') else 'user'

#     session_id = getattr(context, 'session_id', 'default')
#     current_session = session_id

#     memory_config = AgentCoreMemoryConfig(
#         memory_id=MEMORY_ID,
#         session_id=session_id,
#         actor_id=actor_id,
#         retrieval_config={
#             f"/users/{actor_id}/facts": RetrievalConfig(top_k=3, relevance_score=0.5),
#             f"/users/{actor_id}/preferences": RetrievalConfig(top_k=3, relevance_score=0.5)
#         }
#     )

#     agent = Agent(
#         model=MODEL_ID,
#         session_manager=AgentCoreMemorySessionManager(memory_config, REGION),
#         # system_prompt="You are a helpful assistant. Use tools when appropriate.",
#         system_prompt="""You are a Job Market Agent. You fetch and summarize job postings for given roles using LinkedIn public data.
#                     When the user asks for jobs, identify the role or keyword from their request (e.g., 'data scientist', 'product manager') and call the scrape_job_market tool with that role.
#         """,
#         tools=[scrape_job_market]
#     )

#     result = agent(payload.get("prompt", ""))
#     return {"response": result.message.get('content', [{}])[0].get('text', str(result))}
