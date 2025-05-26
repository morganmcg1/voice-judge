import os
from pathlib import Path
from openai import OpenAI
from voice_prompt_guidelines import GEN_VOICE_GUIDELINES
import weave
from pydantic import BaseModel, Field
import json
from datetime import datetime

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

voices = ["onyx"]  # Options: "onyx", "verse", "ash", etc.


class VoiceInstruction(BaseModel):
    reasoning: str = Field(description="The reasoning process that led to the voice instructions.")
    voice_instructions: str = Field(description="The voice instructions for the multi-modal LLM to generate audio.")
    voice_instructions_id: str = Field(description="A short, unique, lowercase, underscore-separated, descriptive \
identifier for the voice instructions, e.g. 'angry_cowboy'")

MODEL_NAME = "o3-2025-04-16"
TEMPERATURE = 1.0

# voice_prompt = "Please write a creative voice prompt for text to speech model that will \
# generate audio for a caucasian man in his 50s. Consider all aspects of the man's background, \
# life history, and character that might impact his voice."

voice_instructions_guidelines = weave.StringPrompt(f"{GEN_VOICE_GUIDELINES}")
weave.publish(voice_instructions_guidelines, name="voice_instructions_guidelines")

voice_guidelines_system_prompt = weave.StringPrompt(f"This system will write voice instructions for a multi-modal LLM to generate audio. \
Below is a description of how to write high quality voice prompts:\n\n{voice_instructions_guidelines}.\n\n")
weave.publish(voice_guidelines_system_prompt, name="voice_instructions_guidelines_system_prompt")


gen_man_voice_prompt = weave.StringPrompt("""Please write a creative voice prompt for text to speech model that will \
generate audio for a caucasian man in his 50s. Consider all aspects of the man's background, \
life history, and character that might impact his voice.""")
weave.publish(gen_man_voice_prompt, name="gta-VI_man_voice_prompt")


@weave.op
def gen_voice_instructions(
    system_prompt: str, 
    voice_prompt: str,
    response_format: BaseModel, 
    model_name: str = "o3-2025-04-16", 
    temperature: float = 1.0
    ) -> str:
    response = client.responses.parse(
        model=model_name,
        temperature=temperature,
        input=[
            {"role": "system", 
            "content": system_prompt,
            },
            {
                "role": "user",
                "content": voice_prompt,
            },
        ],
        text_format=response_format,
    )
    return response.output_parsed

voice_instructions = gen_voice_instructions(
    system_prompt=voice_guidelines_system_prompt, 
    voice_prompt=gen_man_voice_prompt, 
    response_format=VoiceInstruction,
    model_name=MODEL_NAME,
    temperature=TEMPERATURE
)

# Save to JSON file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_filename = f"voice_instructions_{timestamp}.json"

# Convert Pydantic model to dict for JSON serialization
voice_data = {
    "voice_instructions_id": voice_instructions.voice_instructions_id,
    "reasoning": voice_instructions.reasoning,
    "voice_instructions": voice_instructions.voice_instructions,
    "model_name": MODEL_NAME,
    "temperature": TEMPERATURE
}

# Save to JSON file
with open(json_filename, 'w', encoding='utf-8') as f:
    json.dump(voice_data, f, indent=2, ensure_ascii=False)

print(f"Voice instructions saved to {json_filename}")

# Create and publish Weave dataset
dataset_rows = [voice_data]  # Can be extended with multiple voice instructions

voice_dataset = weave.Dataset(
    name="voice_instructions_tmp",
    rows=dataset_rows
)

# Publish the dataset to Weave
dataset_ref = weave.publish(voice_dataset)
print(f"Voice instructions dataset published to Weave: {dataset_ref}")
print(f"Dataset contains {len(dataset_rows)} voice instruction(s)")