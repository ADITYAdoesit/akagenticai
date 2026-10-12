# Basic Autogen Agent example
# pip install "autogen-agentchat" "autogen-ext[openai]"
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv(override=True)

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini"
)

# Define a simple function tool that the agent can use.
async def get_forex_rate(target: str) -> float:
    """Get the forex rate of USD against a target currency."""
    return 90.10

# Define an AssistantAgent
agent = AssistantAgent(
    name="forex_agent",
    model_client=model_client,
    tools=[get_forex_rate],
    system_message="You are a helpful assistant.",
    reflect_on_tool_use=True,
    model_client_stream=True,
)

# Run the agent and stream the messages to the console.
async def main() -> None:
    await Console(agent.run_stream(task="What is the current forex rate of USD against INR?"))
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
