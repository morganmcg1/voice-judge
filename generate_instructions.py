import os
from pathlib import Path
from openai import AsyncOpenAI
from voice_prompt_guidelines import GEN_VOICE_GUIDELINES
import weave
from pydantic import BaseModel, Field
import json
from datetime import datetime
from dotenv import load_dotenv
import asyncio
import random

load_dotenv()

MODEL_NAME = "o3-2025-04-16"
# MODEL_NAME = "gpt-4.1-2025-04-14"
TEMPERATURE = 1.2
WEAVE_DATASET_NAME = "voice_instructions"
N_GENERATIONS = 20

weave.init("wandb-voice-ai/voice-judge")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

voice_instructions_guidelines = weave.StringPrompt(f"{GEN_VOICE_GUIDELINES}")
weave.publish(voice_instructions_guidelines, name="voice_instructions_guidelines")

voice_guidelines_system_prompt = weave.StringPrompt(
    f"This system will write voice instructions for a multi-modal LLM to generate audio. \
Below is a description of how to write high quality voice prompts:\n\n{voice_instructions_guidelines.format()}.\n\n"
)
weave.publish(voice_guidelines_system_prompt, name="voice_instructions_guidelines_system_prompt")

gen_man_voice_prompt = weave.StringPrompt("""Please write a creative voice prompt for text to speech model that will \
generate audio for a caucasian man in his 50s. Consider all aspects of the man's background, \
life history, and character that might impact his voice. 
                                          
The man is from {country}. He's currently feeling very {mood}. He is speaking {speed} and {volume}.
                                          
Write a concise voice prompt that will be used to generate audio for a text to speech model for this man.
""")
weave.publish(gen_man_voice_prompt, name="gta-VI_man_voice_prompt")

countries = ["posh England", "France", "Ireland", "Georgia, USA", "Russia"]
voice_speeds = ["very quickly", "rapidly", "super super quickly"]
voice_volumes = ["loudly", "quietly"]
moods = ["sarcastic", "angry", "excited", "afraid", \
         "fearful", "surprised", "confused"]

random_voice_parameters = [{
    "country": random.choice(countries),
    "mood": random.choice(moods),
    "speed": random.choice(voice_speeds),
    "volume": random.choice(voice_volumes),
} for _ in range(N_GENERATIONS)]

class VoiceInstruction(BaseModel):
    reasoning: str = Field(description="The reasoning process that led to the voice instructions.")
    voice_instructions: str = Field(
        description="The voice instructions for the multi-modal LLM to generate audio."
    )
    voice_instructions_id: str = Field(
        description="A short, unique, lowercase, underscore-separated, descriptive \
identifier for the voice instructions, e.g. 'angry_cowboy'"
    )


@weave.op
async def gen_voice_instructions(
    system_prompt: str,
    voice_prompt: str,
    response_format: BaseModel,
    voice_parameters: dict ,
    model_name: str = "o3-2025-04-16",
    temperature: float = 1.0,
) -> str:
    response = await client.responses.parse(
        model=model_name,
        # temperature=temperature,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": voice_prompt.format(**voice_parameters),
            },
        ],
        text_format=response_format,
    )
    return response.output_parsed


@weave.op
async def generate_multiple_voice_instructions(n_generations: int = N_GENERATIONS):
    tasks = [
        gen_voice_instructions(
            system_prompt=voice_guidelines_system_prompt.format(),
            voice_prompt=gen_man_voice_prompt,
            response_format=VoiceInstruction,
            voice_parameters=voice_params,
            model_name=MODEL_NAME,
            # temperature=TEMPERATURE,
        )
        for voice_params in random_voice_parameters
    ]

    return await asyncio.gather(*tasks)


voice_instructions_list = asyncio.run(generate_multiple_voice_instructions())

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
voice_instr_dir = Path("voice_instructions")
voice_instr_dir.mkdir(parents=True, exist_ok=True)
json_filename = voice_instr_dir / f"voice_instructions_{timestamp}.json"

voice_data_list = [
    {
        "voice_instructions_id": vi.voice_instructions_id,
        "reasoning": vi.reasoning,
        "voice_instructions": vi.voice_instructions,
        "model_name": MODEL_NAME,
    }
    for vi in voice_instructions_list
]

# Create and publish Weave dataset
voice_dataset = weave.Dataset(name=WEAVE_DATASET_NAME, rows=voice_data_list)

# Publish the dataset to Weave
dataset_ref = weave.publish(voice_dataset)
print(f"Voice instructions dataset published to Weave: {dataset_ref}")
print(f"Dataset contains {len(voice_data_list)} voice instruction(s)")

with open(json_filename, "w", encoding="utf-8") as f:
    json.dump(voice_data_list, f, indent=2, ensure_ascii=False)

print(f"Voice instructions saved to {json_filename}")
