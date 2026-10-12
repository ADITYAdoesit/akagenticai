import os
# pip install autogen-agentchat autogen-core autogen-ext python-d

import asyncio
from PIL import Image
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import MultiModalMessage, TextMessage
from autogen_core import Image as AGImage
from autogen_agentchat.agents import AssistantAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(override=True)

# Load image
pil_image = Image.open(os.path.join(BASE_DIR, "1911_Solvay_conference.jpg"))
img = AGImage(pil_image)

# Create agents
desc_agent = AssistantAgent("desc_agent", OpenAIChatCompletionClient(model="gpt-4o"))
context_agent = AssistantAgent("context_agent", OpenAIChatCompletionClient(model="gpt-4o"))

async def main():
    # Get description
    print("\n" + "="*50 + "\nDESCRIPTION\n" + "="*50)
    desc = await desc_agent.on_messages([MultiModalMessage(
        content=["Describe this historical photograph in detail. What kind of event does this appear to be?", img], 
        source="User"
    )], cancellation_token=None)
    print(desc.chat_message.content)
    
    # Get historical context without direct identification
    print("\n" + "="*50 + "\nHISTORICAL CONTEXT\n" + "="*50)
    context = await context_agent.on_messages([
        MultiModalMessage(
            content=["Analyze the setting, time period, and context of this photograph.", img], 
            source="User"
        ),
        TextMessage(
            content="Based on the image analysis, if this is the 1911 Solvay Conference, who were the attendees? "
                    "List the participants by row position (front row left to right, back row left to right) "
                    "and describe their major contributions to physics.",
            source="User"
        )
    ], cancellation_token=None)
    print(context.chat_message.content)
    
    # Cleanup
    await desc_agent.close()
    await context_agent.close()

if __name__ == "__main__":
    asyncio.run(main())