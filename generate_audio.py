import asyncio
import json
import os
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

import weave
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from openai.helpers import LocalAudioPlayer
from pydantic import BaseModel, Field

load_dotenv()


TTS_MODEL_NAME = "gpt-4o-mini-tts"
VOICE = "onyx"
VOICE_INSTRUCTIONS_DATASET_NAME = "voice_instructions"
N_AUDIO_GENERATIONS = 5  # Number of audio files to generate
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

weave.init("wandb-voice-ai/voice-judge")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# character_text = weave.StringPrompt(
#     """Hey boy! You fixing that satellite dish already? Good! I been missing my stories for three weeks!"""
# )
character_text = weave.StringPrompt(
    """What are you doing up there? GTA 6 was supposed to be out by now — you coding the whole thing by hand or what?!"""
)
weave.publish(character_text, name="character_text")


class AudioGeneration(BaseModel):
    voice_instructions_id: str = Field(description="The ID of the voice instructions used")
    voice_instructions: str = Field(description="The voice instructions text")
    audio_file_path: str = Field(description="Path to the generated audio file")
    tts_model_name: str = Field(description="The TTS model used")
    voice: str = Field(description="The voice used for generation")


@weave.op
async def generate_speech_from_instructions(
    character_text: str,
    voice_instructions: str,
    voice_instructions_id: str,
    tts_model_name: str = TTS_MODEL_NAME,
    voice: str = VOICE,
    response_format: str = "wav",
    timestamp: str = TIMESTAMP,
) -> dict[Any, str]:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    audio_dir = Path("generated_audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    speech_file_path = audio_dir / f"audio_{voice_instructions_id}_{timestamp}.wav"
    try:
        resp = await client.audio.speech.with_raw_response.create(
            model=tts_model_name,
            voice=voice,
            input=character_text,
            instructions=voice_instructions,
            response_format=response_format,
        )
        content = resp.content if hasattr(resp, "content") else await resp.aread()
        with open(speech_file_path, "wb") as f:
            f.write(content)
        return {
            "audio": wave.open(speech_file_path.as_posix(), "rb"),
            "audio_file_path": str(speech_file_path),
        }
    except Exception as e:
        print(f"Exception occurred while generating audio: {e}")
        return {"audio": None, "audio_file_path": ""}


@weave.op
async def download_voice_dataset() -> list:
    """Download the voice instructions dataset from Weave."""
    try:
        # Get the dataset reference
        dataset_ref = weave.ref(VOICE_INSTRUCTIONS_DATASET_NAME).get()

        if dataset_ref is None:
            print(f"Dataset '{VOICE_INSTRUCTIONS_DATASET_NAME}' not found")
            return []

        # Access the rows from the dataset
        if hasattr(dataset_ref, "rows"):
            # If it's a Table or WeaveTable, convert to list
            if hasattr(dataset_ref.rows, "to_pylist_notags"):
                return dataset_ref.rows.to_pylist_notags()
            elif hasattr(dataset_ref.rows, "__iter__"):
                return list(dataset_ref.rows)
            else:
                return dataset_ref.rows
        else:
            return list(dataset_ref)

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return []


@weave.op
async def generate_speech_batch(voice_instructions_data: list, n_generations: int = N_AUDIO_GENERATIONS):
    selected_instructions = voice_instructions_data[:n_generations]
    results = []
    for instruction_data in selected_instructions:
        res = await generate_speech_from_instructions(
            character_text=character_text.format(),
            voice_instructions=instruction_data["voice_instructions"],
            voice_instructions_id=instruction_data["voice_instructions_id"],
            tts_model_name=TTS_MODEL_NAME,
            voice=VOICE,
        )
        results.append(res.get("audio_file_path"))
    return results


async def main():
    """Main function to download dataset and generate audio."""
    print("Downloading voice instructions dataset from Weave...")
    # voice_instructions_data = await download_voice_dataset()
    ds_ref = weave.ref(VOICE_INSTRUCTIONS_DATASET_NAME).get()
    voice_instructions_data = list(ds_ref.rows)

    if not voice_instructions_data:
        print("No voice instructions found in dataset. Please run generate_instructions.py first.")
        return

    print(f"Found {len(voice_instructions_data)} voice instructions in dataset")
    print(f"Generating audio for {min(N_AUDIO_GENERATIONS, len(voice_instructions_data))} instructions...")

    # Generate audio files
    audio_results = await generate_speech_batch(voice_instructions_data, N_AUDIO_GENERATIONS)

    print(f"Successfully generated {len(audio_results)} audio files")

    # Create summary data for logging
    audio_generations = []
    for i, (instruction_data, audio_result) in enumerate(
        zip(voice_instructions_data[:N_AUDIO_GENERATIONS], audio_results)
    ):
        audio_file_path = audio_result[1] if audio_result else ""
        audio_gen = AudioGeneration(
            voice_instructions_id=instruction_data["voice_instructions_id"] + "_" + TIMESTAMP,
            voice_instructions=instruction_data["voice_instructions"],
            audio_file_path=audio_file_path,
            tts_model_name=TTS_MODEL_NAME,
            voice=VOICE,
        )
        audio_generations.append(audio_gen.model_dump())

    # Create and publish audio dataset to Weave
    audio_dataset = weave.Dataset(
        name="generated_speech", rows=weave.Dataset.convert_to_table(audio_generations)
    )

    # Publish the dataset to Weave
    dataset_ref = weave.publish(audio_dataset)
    print(f"Audio generation dataset published to Weave: {dataset_ref}")

    # # Save to local JSON file as backup
    # audio_dir = Path("generated_audio")
    # audio_dir.mkdir(parents=True, exist_ok=True)
    # json_filename = audio_dir / f"audio_generations_{timestamp}.json"

    # with open(json_filename, "w", encoding="utf-8") as f:
    #     json.dump(audio_generations, f, indent=2, ensure_ascii=False)

    # print(f"Audio generation metadata saved to {json_filename}")


if __name__ == "__main__":
    asyncio.run(main())
