# %% [markdown]
# # LLM Judge for Generated Speech
# This notebook generates speech from text using OpenAI's TTS API and 

# %% Install dependencies
import os
import json

# os.system("uv pip install openai weave genai")

# %% Imports
import wave
import io

from dotenv import load_dotenv

from pathlib import Path
import asyncio
from typing import List, Union, BinaryIO
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

import weave

from voice_prompt_guidelines import GEN_VOICE_GUIDELINES

load_dotenv()

SPEECH_AUDIO_DATASET_URI = "weave:///wandb-voice-ai/voice-judge/object/generated_speech_audio:v1"
# %% Configuration


class JudgeCriteria(BaseModel):
    thinking: str = Field(description="Thinking about which voice is better.")
    best_voice: List[str] = Field(description="The best voice.")

class JudgeBasics(BaseModel):
    thinking: str = Field(description="Thinking about the audio given the respective ranking.")
    how_to_judge_a_voice: List[str] = Field(description="The criteria needed for a LLM to judge \
the audio and come to the same decision as the human judge about the ranking.")
    

class JudgeRanking(BaseModel):
    thinking: str = Field(description="Reasoning about how to rank the voices given the criteria.")
    ranking: List[str] = Field(description="The ranking of the voices by their ID.")


# system_instruction = GEN_VOICE_GUIDELINES

BASE_SYSTEM_INSTRUCTION = """The task is to consider the voice characteristics that lead to the \
ranking of the audio samples."""

async def get_audio_parts(audio_ls: list, client: genai.Client, initial_prompt: str, prompt_divider: str = "\n\n"):
    audio_parts = [initial_prompt]
    for i, audio in enumerate(audio_ls):
        if i > 0:
            audio_parts.append(prompt_divider.format(input_order=i + 1))
            
        if isinstance(audio, (str, Path)):
            # File path - check size and upload if needed
            audio_path = Path(audio)
            if not audio_path.exists():
                raise ValueError(f"Audio file not found: {audio_path}")
                
            file_size_mb = audio_path.stat().st_size / (1024 * 1024)
            
            if file_size_mb > 20:
                # Upload large files
                uploaded_file = await client.aio.files.upload(path=str(audio_path))
                audio_parts.append(uploaded_file)
            else:
                # Read and inline small files
                with open(audio_path, 'rb') as f:
                    audio_bytes = f.read()
                mime_type = _get_mime_type(audio_path.suffix)
                audio_parts.append(types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=mime_type
                ))
                
        elif isinstance(audio, bytes):
            # Raw bytes - assume MP3 if not specified
            audio_parts.append(types.Part.from_bytes(
                data=audio,
                mime_type='audio/mp3'
            ))
        else:
            raise ValueError(f"Unsupported audio data type: {type(audio)}")
    
    return audio_parts

@weave.op
async def run_speech_llm(
    model_name: str,
    temperature: float,
    response_model: BaseModel,
    audio_data: Union[str, bytes, Path, List[Union[str, bytes, Path]]],
    prompt: str,
    system_instruction: str = None,
    max_tokens: int = 4000,
    initial_audio_parts_prompt: str = "\n\nRank 1 voice:\n",
    audio_parts_prompt_divider: str = "\n\nRank {input_order} voice:\n"
) -> BaseModel:
    # Initialize client
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Handle audio data - convert to list if single item
    if not isinstance(audio_data, list):
        audio_data = [audio_data]
    
    # Process audio parts
    parts = [prompt]  # Start with text prompt

    audio_parts = await get_audio_parts(
        audio_ls=audio_data, 
        client=client,
        initial_prompt=initial_audio_parts_prompt,
        prompt_divider=audio_parts_prompt_divider)
    parts.extend(audio_parts)
    
    # Configure generation with structured output
    generation_config = types.GenerateContentConfig(
        temperature=temperature if temperature > 0 else 1.0,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
        response_schema=response_model,
        system_instruction=system_instruction
    )
    
    # Generate response
    response = await client.aio.models.generate_content(
        model=model_name,
        contents=parts,
        config=generation_config
    )
    
    # Parse and validate response
    json_content = json.loads(response.text)
    return response_model.model_validate(json_content)


def _get_mime_type(suffix: str) -> str:
    """Get MIME type from file extension."""
    mime_map = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mp3',
        '.aiff': 'audio/aiff',
        '.aac': 'audio/aac',
        '.ogg': 'audio/ogg',
        '.flac': 'audio/flac'
    }
    return mime_map.get(suffix.lower(), 'audio/mp3')


# Example usage
if __name__ == "__main__":
    weave.init("wandb-voice-ai/voice-judge")
    print("Downloading speech samples from Weave...")
    ds_ref = weave.ref(SPEECH_AUDIO_DATASET_URI).get()
    speech_samples = list(ds_ref.rows)

    audio_obj = speech_samples[0]["audio"]
    audio_obj_two = speech_samples[1]["audio"]
    
    audio_bytes = wave_read_to_wav_bytes(audio_obj)
    audio_bytes_two = wave_read_to_wav_bytes(audio_obj_two)

    async def example():
        # Example with file path
        result = await process_audio_with_gemini(
            system_instruction=BASE_SYSTEM_INSTRUCTION,
            prompt="Describe each voice in this audio.",
            model_name="gemini-2.0-flash",
            temperature=0.1,
            response_model=JudgeCriteria,
            audio_data=[audio_bytes, audio_bytes_two],
        )
        return result
        
    # Run example
    result = asyncio.run(example())
    print(result.thinking)
    print(result.best_voice)

