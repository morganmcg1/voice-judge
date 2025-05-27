import asyncio
import os
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

import weave
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()


TTS_MODEL_NAME = "gpt-4o-mini-tts"
VOICE = "onyx"
VOICE_INSTRUCTIONS_DATASET_NAME = "voice_instructions"
GENERATED_AUDIO_DATASET_NAME = "generated_speech_audio_test"
N_AUDIO_GENERATIONS = 20  # Number of audio files to generate
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

weave.init("wandb-voice-ai/voice-judge")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@weave.op
async def generate_speech_from_instructions(
    character_text: str,
    voice_instructions: str,
    voice_instructions_id: str,
    tts_model_name: str = TTS_MODEL_NAME,
    voice: str = VOICE,
    response_format: str = "wav",
    timestamp: str = TIMESTAMP,
) -> dict[str, Any]:
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
        
        # Open the audio file with wave
        audio_obj = wave.open(speech_file_path.as_posix(), "rb")
                
        return {
            "audio": audio_obj,
            "success": True,
            "audio_file_path": str(speech_file_path),
            "voice_instructions_id": voice_instructions_id,
        }
    except Exception as e:
        print(f"Exception occurred while generating audio for instruction ID {voice_instructions_id}: {e}")
        return {
            "audio": None,
            "success": False,
            "audio_file_path": "",
            "voice_instructions_id": voice_instructions_id,
            "error_message": str(e),
        }


@weave.op
async def generate_speech_batch(voice_instructions_data: list, character_text: str = None, n_generations: int = N_AUDIO_GENERATIONS):
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
        results.append(res)
    return results


async def main():
    """Main function to download dataset and generate audio."""
    print("Downloading voice instructions dataset from Weave...")
    ds_ref = weave.ref(VOICE_INSTRUCTIONS_DATASET_NAME).get()
    voice_instructions_data = list(ds_ref.rows)

    if not voice_instructions_data:
        print("No voice instructions found in dataset. Please run generate_instructions.py first.")
        return

    print(f"Found {len(voice_instructions_data)} voice instructions in dataset")
    print(f"Generating audio for {min(N_AUDIO_GENERATIONS, len(voice_instructions_data))} instructions...")

    character_text = weave.ref("character_text").get()

    audio_generation_results = await generate_speech_batch(
        voice_instructions_data=voice_instructions_data,
        character_text=character_text,
        n_generations=N_AUDIO_GENERATIONS
    )

    successful_generations_count = 0
    audio_generations = []

    for i, instruction_data in enumerate(voice_instructions_data[:N_AUDIO_GENERATIONS]):
        # Find the corresponding result for the current instruction_data
        current_result = next((r for r in audio_generation_results if r["voice_instructions_id"] == instruction_data["voice_instructions_id"]), None)

        if current_result and current_result["success"]:
            audio_file_path = current_result["audio_file_path"]
            audio_gen = {
                "voice_instructions_id": instruction_data["voice_instructions_id"] + "_" + TIMESTAMP,
                "voice_instructions": instruction_data["voice_instructions"],
                "audio_file_path": audio_file_path,
                "tts_model_name": TTS_MODEL_NAME,
                "voice": VOICE,
                "audio": current_result["audio"]
            }
            audio_generations.append(audio_gen)
            
            successful_generations_count += 1
        elif current_result:
            print(f"Skipping instruction ID {instruction_data['voice_instructions_id']} due to error: {current_result.get('error_message', 'Unknown error')}")
        else:
            # This case should ideally not happen if generate_speech_batch returns a result for every input
            print(f"Warning: No result found for instruction ID {instruction_data['voice_instructions_id']}")

    print(f"Successfully generated {successful_generations_count} audio files out of {len(voice_instructions_data[:N_AUDIO_GENERATIONS])} attempts.")

    # Create and publish audio dataset to Weave only if there are successful generations
    if audio_generations:
        # Dataset with metadata only (original approach)
        audio_dataset = weave.Dataset(
            name=GENERATED_AUDIO_DATASET_NAME, 
            rows=weave.Dataset.convert_to_table(audio_generations)
        )
        dataset_ref = weave.publish(audio_dataset)
        print(f"Audio generation dataset published to Weave: {dataset_ref}")
    else:
        print("No audio files were successfully generated. Skipping dataset publication.")

    # # Save to local JSON file as backup
    # audio_dir = Path("generated_audio")
    # audio_dir.mkdir(parents=True, exist_ok=True)
    # json_filename = audio_dir / f"audio_generations_{timestamp}.json"

    # with open(json_filename, "w", encoding="utf-8") as f:
    #     json.dump(audio_generations, f, indent=2, ensure_ascii=False)

    # print(f"Audio generation metadata saved to {json_filename}")


if __name__ == "__main__":
    asyncio.run(main())
