import os
import json

# os.system("uv pip install openai weave genai")

import wave
import io

from dotenv import load_dotenv

from pathlib import Path
import asyncio
from typing import List, Union, BinaryIO
from pydantic import BaseModel, Field
from google import genai
from google.genai import types as genai_types

import weave
from weave import Scorer

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

def update_pairwise_comparison_history(samples_to_rank, ranking):
    for r in ranking["rankings"]:
        competitor_id = [rr.get("id") for rr in ranking["rankings"] if rr.get("id") != r.get("id")][0]
        samples_to_rank[r.get("id")]["pairwise_comparison_history"][ranking["completed_at"]] = {
            "competitor_id": competitor_id,
            "sample_rank_in_this_pair": r.get("rank"),
        }
    return samples_to_rank


class BinaryVoiceRank(Scorer):
    ranking: list
    timestamp: str

    def __init__(self, rankings: dict, timestamp: str):
        # self.ranking = rankings["rankings"]
        # self.timestamp = rankings["completed_at"]
        super().__init__(
            ranking=rankings,  # Pass the list for the 'ranking' field
            timestamp=timestamp,  # Pass the string for the 'timestamp' field
        )

    @weave.op
    def score(self, output: str) -> dict:
        sample_one_preferred = False
        sample_two_preferred = False

        # Iterate through the ranked items to find the one with rank 1
        for item in self.ranking:
            if item["rank"] == 1:
                if item["original_input_order"] == 1:
                    sample_one_preferred = True
                elif item["original_input_order"] == 2:
                    sample_two_preferred = True
                preferred_id = item["id"]
                break

        return {
            "preferred_sample_id": preferred_id,
            "sample_one_preferred": sample_one_preferred,
            "sample_two_preferred": sample_two_preferred,
            "ranking_timestamp": self.timestamp,
        }


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
                audio_parts.append(genai_types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=mime_type
                ))
                
        elif isinstance(audio, bytes):
            # Raw bytes - assume MP3 if not specified
            audio_parts.append(genai_types.Part.from_bytes(
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
    prompt: str,
    audio_data: Union[str, bytes, Path, List[Union[str, bytes, Path]]] = [],
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
    if isinstance(prompt, str):
        parts = [prompt]  # Start with text prompt
    elif isinstance(prompt, list):
        parts = prompt
    else:
        raise ValueError(f"Unsupported prompt type: {type(prompt)}")

    if audio_data:
        audio_parts = await get_audio_parts(
            audio_ls=audio_data, 
            client=client,
            initial_prompt=initial_audio_parts_prompt,
            prompt_divider=audio_parts_prompt_divider)
        parts.extend(audio_parts)
    
    # Configure generation with structured output
    generation_config = genai_types.GenerateContentConfig(
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