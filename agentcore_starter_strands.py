"""
Strands Agent sample with AgentCore
"""
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

@tool
def calculate(code: str) -> str:
    """Execute Python code for calculations or analysis."""
    session_id = current_session or 'default'

    if session_id not in ci_sessions:
        ci_sessions[session_id] = {
            'client': CodeInterpreter(REGION),
            'session_id': None
        }

    ci = ci_sessions[session_id]
    if not ci['session_id']:
        ci['session_id'] = ci['client'].start(
            name=f"session_{session_id[:30]}",
            session_timeout_seconds=1800
        )

    result = ci['client'].invoke("executeCode", {
        "code": code,
        "language": "python"
    })

    for event in result.get("stream", []):
        if stdout := event.get("result", {}).get("structuredContent", {}).get("stdout"):
            return stdout
    return "Executed"

@tool
def summarize_text(content: str) -> str:
        """Summarize the provided text using the LLM."""
        from boto3 import client
        import json

        bedrock = client("bedrock-runtime", region_name=REGION)
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps({
                "messages": [
                    {"role": "user", "content": f"Summarize the following text:\n\n{content}"}
                ]
            }),
        )

        result = json.loads(response["body"].read())
        return result["output"]["message"]["content"][0]["text"]
    
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
        system_prompt="You are a helpful assistant that summarizes text clearly and concisely.",
        tools=[summarize_text]
    )

    result = agent(payload.get("prompt", ""))
    return {"response": result.message.get('content', [{}])[0].get('text', str(result))}

if __name__ == "__main__":
    app.run()