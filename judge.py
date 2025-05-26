# %% [markdown]
# # LLM Judge for Generated Speech
# This notebook generates speech from text using OpenAI's TTS API and 

# %% Install dependencies
import os

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


# system_instruction = GEN_VOICE_GUIDELINES


BASE_SYSTEM_INSTRUCTION = """The task is to evaluate generated speech audio on the style, tone, pace, affect and \
character of each voice sample."""

@weave.op
async def run_speech_judge(
    model_name: str,
    temperature: float,
    response_model: BaseModel,
    audio_data: Union[str, bytes, Path, List[Union[str, bytes, Path]]],
    prompt: str,
    system_instruction: str = None,
    max_tokens: int = 4000
) -> BaseModel:
    # Initialize client
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Handle audio data - convert to list if single item
    if not isinstance(audio_data, list):
        audio_data = [audio_data]
    
    # Process audio parts
    parts = [prompt]  # Start with text prompt
    
    for audio in audio_data:
        if isinstance(audio, (str, Path)):
            # File path - check size and upload if needed
            audio_path = Path(audio)
            if not audio_path.exists():
                raise ValueError(f"Audio file not found: {audio_path}")
                
            file_size_mb = audio_path.stat().st_size / (1024 * 1024)
            
            if file_size_mb > 20:
                # Upload large files
                uploaded_file = await client.aio.files.upload(path=str(audio_path))
                parts.append(uploaded_file)
            else:
                # Read and inline small files
                with open(audio_path, 'rb') as f:
                    audio_bytes = f.read()
                mime_type = _get_mime_type(audio_path.suffix)
                parts.append(types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=mime_type
                ))
                
        elif isinstance(audio, bytes):
            # Raw bytes - assume MP3 if not specified
            parts.append(types.Part.from_bytes(
                data=audio,
                mime_type='audio/mp3'
            ))
        else:
            raise ValueError(f"Unsupported audio data type: {type(audio)}")
    
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
    import json
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

