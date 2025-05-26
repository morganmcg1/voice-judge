# %% [markdown]
# # OpenAI Text-to-Speech Generation
# This notebook generates speech from text using OpenAI's TTS API

# %% Install dependencies
!uv pip install openai weave genai

# %% Imports
import os
from pathlib import Path
from openai import OpenAI
import asyncio
from typing import List, Union, BinaryIO
from pydantic import BaseModel
from google import genai
from google.genai import types

# %% Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# %% Configuration
voice = "onyx"  # Options: "onyx", "verse", "ash", etc.
speech_file_path = Path(f"{voice}_speech.mp3")

# Input text for speech generation
input_text = "What the hell are you doing up there??? Where's my rent?!"

# %% Voice instructions presets
# Friendly cowboy preset (commented out)
friendly_instructions = """Voice: Warm, relaxed, and friendly, with a steady cowboy drawl that feels approachable.

Punctuation: Light and natural, with gentle pauses that create a conversational rhythm without feeling rushed.

Delivery: Smooth and easygoing, with a laid-back pace that reassures the listener while keeping things clear.

Phrasing: Simple, direct, and folksy, using casual, familiar language to make technical support feel more personable.

Tone: Lighthearted and welcoming, with a calm confidence that puts the caller at ease."""

# Angry cowboy preset (active)
angry_instructions = """Voice: Deep, Gritty and forceful, old man cowboy drawl but tight-jawed and crackling with irritation, every word carrying a hard edge.

Punctuation: Sharp and punchy—short pauses like clipped breaths, abrupt sentence stops that underline the speaker's impatience.

Delivery: Fast and tense, words fired off in quick bursts; no spare breath wasted on politeness.

Phrasing: Blunt, no-nonsense, rough-and-ready slang ("Listen up," "Cut the nonsense") that replaces folksy warmth with direct confrontation.

Tone: Openly angry and exasperated, projecting controlled intensity that demands attention and leaves no room for doubt."""

# %% Generate speech
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice=voice,
    input=input_text,
    instructions=angry_instructions
) as response:
    response.stream_to_file(speech_file_path)
    
print(f"Speech generated and saved to: {speech_file_path}")

async def process_audio_with_gemini(
    model_name: str,
    temperature: float,
    response_model: BaseModel,
    audio_data: Union[str, bytes, Path, List[Union[str, bytes, Path]]],
    prompt: str = "Analyze this audio",
    system_instruction: str = None,
    max_tokens: int = 4000
) -> BaseModel:
    """
    Process audio with Gemini and return structured output.
    
    Args:
        model_name: Gemini model name (e.g., "gemini-2.0-flash")
        temperature: Temperature for generation (0.0 to 2.0)
        response_model: Pydantic model class for structured output
        audio_data: Can be:
            - str/Path: path to audio file
            - bytes: raw audio data
            - List of any above for multiple audio files
        prompt: Text prompt to accompany the audio
        system_instruction: Optional system instruction
        max_tokens: Maximum output tokens
        
    Returns:
        Instance of response_model with parsed data
    """
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
    # Define response model
    class AudioAnalysis(BaseModel):
        transcript: str
        sentiment: str
        summary: str
        key_points: List[str]
        
    async def example():
        # Example with file path
        result = await process_audio_with_gemini(
            model_name="gemini-2.0-flash",
            temperature=0.7,
            response_model=AudioAnalysis,
            audio_data="path/to/audio.mp3",
            prompt="Transcribe and analyze this audio. Provide sentiment, summary, and key points.",
        )
        print(result)
        
        # Example with multiple audio files
        result = await process_audio_with_gemini(
            model_name="gemini-2.0-flash", 
            temperature=0.5,
            response_model=AudioAnalysis,
            audio_data=["audio1.mp3", "audio2.mp3"],
            prompt="Analyze these audio clips together.",
        )
        print(result)
        
    # Run example
    # asyncio.run(example())