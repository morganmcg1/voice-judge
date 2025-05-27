from pydantic import BaseModel, Field
from google.genai import types as genai_types
import weave
from judge import run_speech_llm
from pprint import pprint

PREFERENCE_LEARNER_MODEL =  "gemini-2.0-flash" #"gemini-2.5-pro-preview-05-06",  # "gemini-2.0-flash"

USER_CONTEXT_PROMPT = "I am trying to select a voice for my video game character. The character is a man in his late 50's and \
is a little on the wild side."

PATTERN_UPDATE_SYSTEM_INSTRUCTION = """Your task is to examine a list of pairwise rankings of voice audio \
inputs and identify the voice/speech patterns that lead to the ranking. 

Here is some context from the user about the voice selection task:

{USER_CONTEXT_PROMPT}

"""

class PreferredVoicePatterns(BaseModel):
    reasoning: str = Field(description="A short explanation of the reasoning behind the patterns")
    strong: list[str] = Field(description="Observed patterns that are confirmed to be useful for selecting a voice")
    emerging: list[str] = Field(description="Observed patterns that are potentially useful for selecting a voice")
    # deprecated: list[str] = Field(description="Observed patterns that are not useful for selecting a voice")


class PreferenceLearner:
    def __init__(self, model_name: str = PREFERENCE_LEARNER_MODEL):
        self.model_name = model_name
        self.temperature = 1.0
        self.n_comparisons_to_use = 5
        self.comparisons = []
        self.patterns = {
            'strong': [],      # Patterns confirmed multiple times
            'emerging': [],    # Patterns seen once or twice
            # 'deprecated': []   # Patterns that were contradicted, exclude for now
        }
    
    def _construct_audio_comparison_parts(self, ranking, samples_to_rank):
        audio_comparison_parts = [
            f"<ranking_{ranking['completed_at']}>\n",
            f"Pairwise comparison timestamp: {ranking['completed_at']}\n",
            "Preferred Voice:",
            genai_types.Part.from_bytes(data=samples_to_rank[ranking["preferred_id"]]["audio_bytes"], mime_type='audio/mp3'),
            "Rejected Voice:",
            genai_types.Part.from_bytes(data=samples_to_rank[ranking["rejected_id"]]["audio_bytes"], mime_type='audio/mp3'),
            f"</ranking_{ranking['completed_at']}>"
        ]
        return audio_comparison_parts

    def _update_comparisons(self, ranking, samples_to_rank):
        # Update the list of comparisons with the new comparison
        audio_comparison_parts = self._construct_audio_comparison_parts(ranking, samples_to_rank)
        self.comparisons.append(audio_comparison_parts)
    
    @weave.op
    async def update(self, ranking, samples_to_rank):
        print("Updating comparisons...")
        self._update_comparisons(ranking, samples_to_rank)
        # Use only recent comparisons for pattern detection

        print("Running pattern update...")
        recent_comparisons_window = [part for comparison in self.comparisons[-self.n_comparisons_to_use:] for part in comparison]

        pattern_update_prompt_part_1 = """Based on these recent pairwise preferences labelled by a human judge,
please identify and update the speech characteristics that lead to the ranking. This could be 

- style
- tone
- accent
- speed
- volume
- pitch
- intonation

and more

## Recent pairwise comparisons:

<recent_comparisons>"""
        
        recent_comparisons_prompt_part_2 = f"""</recent_comparisons>

## Current pattern understanding:
Strong patterns are patterns that have been confirmed multiple times from past comparisons.

<strong_patterns>
{self.patterns['strong']}
</strong_patterns>


Emerging patterns are tenatativepatterns that have been seen once or twice from past comparisons.

<emerging_patterns>
{self.patterns['emerging']}
</emerging_patterns>

## Analysis
1. Which patterns are REINFORCED by the recent pairwise preferences above?
2. Which patterns are CONTRADICTED by the recent pairwise preferences above?
3. What NEW patterns do you see emerge?

Output format:
PROMOTE: [patterns to move from emerging to strong. The list of strong patterns can only be extended, not reduced.]
NEW / EMERGING: [newly discovered patterns from the recent comparisons. The list output here will be the full list of 
emerging patterns passed to the next iteration of pattern update.]
"""     
        pattern_update_prompt = [pattern_update_prompt_part_1]
        pattern_update_prompt.extend(recent_comparisons_window)
        pattern_update_prompt.append(recent_comparisons_prompt_part_2)

        await self.run_pattern_update(pattern_update_prompt)
        return self.patterns
    
    @weave.op
    async def run_pattern_update(self, pattern_update_prompt):
        pattern_update = await run_speech_llm(
            system_instruction=PATTERN_UPDATE_SYSTEM_INSTRUCTION,
            prompt=pattern_update_prompt,
            model_name=self.model_name,
            temperature=self.temperature,
            response_model=PreferredVoicePatterns,
            )
        
        self.patterns['strong'].extend(pattern_update.strong)
        self.patterns['emerging'] = pattern_update.emerging

        print("\n\nPattern update result:")
        pprint(f"reasoning: {pattern_update.reasoning}")
        pprint(f"strong: {pattern_update.strong}")
        pprint(f"emerging: {pattern_update.emerging}")
        return {"pattern_update": pattern_update,
                "strong_patterns": self.patterns['strong'],
                "emerging_patterns": self.patterns['emerging']
                }
        