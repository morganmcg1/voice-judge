GEN_VOICE_GUIDELINES = """
# Writing the perfect voice prompt

Below is a breakdown of how to write voice prompts for a multi-modal LLM. Voice prompting is a new and \
interesting way to prompt multi-modal LLMs to generate audio. Below are detailed the principal components \
of a good voice prompt as well as examples 


## Core Elements of Voice Prompts

1. **Voice/Affect**
This describes the overall sound quality and character of the voice:
* Physical qualities (deep, warm, commanding)
* Cultural/regional characteristics (French accent, cowboy drawl)
* Archetype associations (noir detective, medieval storyteller)

One or two vivid descriptors are enough; avoid long biographies. A short archetype phrase anchors characterization.


2. **Tone**
The emotional coloring and attitude conveyed:
* Emotional temperature (warm, cool, detached)
* Formality level (casual, formal, noble)
* Attitude (friendly, snooty, world-weary)

Stick to a single dominant tone so the model doesn’t waver.


3. **Pacing/Delivery**
How quickly or slowly the voice moves:
* Speed (slow, moderate, steady)
* Rhythm (deliberate, unhurried, easygoing)
* Flow (smooth, punchy, dramatic)

Add a reason if useful (“…to let details sink in”).


4. **Emotion**
The specific feelings expressed through the voice:
* Primary emotions (excitement, empathy, enthusiasm)
* Emotional combinations (calm confidence, quiet determination)
* Emotional depth (genuine, sincere, reverent)

Name just one or two emotions; piling on opposites confuses delivery.


5. **Pronunciation/Articulation**
How words are spoken:
* Clarity level (clear, precise, deliberate)
* Special pronunciations (French words, archaic terms)
* Emphasis patterns (which words to stress)

“pronounce ‘Leonardo da Vinci’ with authentic Italian,” “emphasize ‘smoothly,’ ‘promptly’.” Only list words likely to trip the engine.

“quiet authority,” “stage-whisper.” Useful when contrast matters.


6. **Pauses/Punctuation**
The strategic use of silence and breaks:
* Pause placement (after key phrases, between thoughts)
* Pause purpose (dramatic effect, allowing reflection)
* Rhythm creation (natural conversation vs. dramatic tension)

“brief pauses after apologies,” “pause after ‘Lo!’” Call out specific words/clauses to spotlight.

“smooth and easygoing,” “dramatic pauses to build suspense.” This can override the default cadence set by pacing.

Treat punctuation as performance cues, not grammar lessons.


7. **Personality/Character**
The overall persona being portrayed:
* Professional roles (tech support, museum guide, detective)
* Character traits (cultured, heroic, reassuring)
* Authenticity markers (folksy language, noir monologue style)

“simple, folksy language,” “slightly archaic Olde English.” Helps keep word selection consistent with the persona.


8. **Context/Purpose**
The situational framing:
* Setting (customer service, museum tour, medieval quest)
* Objective (reassure, educate, create suspense)
* Audience relationship (helpful guide, authority figure)

“museum audio guide,” “tech-support hotline.” A single sentence is plenty—don’t overwrite the scene.


## Key Patterns for Effective Voice Prompts

1. **Consistency**: Each element should reinforce the others (e.g., a noir detective has slow delivery, dramatic pauses, and world-weary emotion)

2. **Specificity**: Concrete descriptors ("steady cowboy drawl") work better than vague ones

3. **Multi-sensory Description**: Combining auditory qualities with personality and emotional descriptors

4. **Cultural/Genre Markers**: Using recognizable archetypes helps establish voice quickly

5. **Functional Alignment**: The voice characteristics should match the intended purpose (reassuring for customer service, dramatic for storytelling)
These elements work together to create a complete vocal persona that a multi-modal LLM can interpret and generate appropriately.


## Example Voice Prompts


### Example 1 
```
Voice: Warm, relaxed, and friendly, with a steady cowboy drawl that feels approachable.
Punctuation: Light and natural, with gentle pauses that create a conversational rhythm without feeling rushed.
Delivery: Smooth and easygoing, with a laid-back pace that reassures the listener while keeping things clear.
Phrasing: Simple, direct, and folksy, using casual, familiar language to make technical support feel more personable.
Tone: Lighthearted and welcoming, with a calm confidence that puts the caller at ease.
```


### Example 2
```
Accent/Affect: slight French accent; sophisticated yet friendly, clearly understandable with a charming touch of French intonation.
Tone: Warm and a little snooty. Speak with pride and knowledge for the art being presented.
Pacing: Moderate, with deliberate pauses at key observations to allow listeners to appreciate details.
Emotion: Calm, knowledgeable enthusiasm; show genuine reverence and fascination for the artwork.
Pronunciation: Clearly articulate French words (e.g., "Mes amis," "incroyable") in French and artist names (e.g., "Leonardo da Vinci") with authentic French pronunciation.
Personality Affect: Cultured, engaging, and refined, guiding visitors with a blend of artistic passion and welcoming charm.
```


### Example 3
```
Voice Affect: Calm, composed, and reassuring; project quiet authority and confidence.
Tone: Sincere, empathetic, and gently authoritative—express genuine apology while conveying competence.
Pacing: Steady and moderate; unhurried enough to communicate care, yet efficient enough to demonstrate professionalism.
Emotion: Genuine empathy and understanding; speak with warmth, especially during apologies ("I'm very sorry for any disruption...").
Pronunciation: Clear and precise, emphasizing key reassurances ("smoothly," "quickly," "promptly") to reinforce confidence.
Pauses: Brief pauses after offering assistance or requesting details, highlighting willingness to listen and support.
```


### Example 4
```
Affect: Deep, commanding, and slightly dramatic, with an archaic and reverent quality that reflects the grandeur of Olde English storytelling.
Tone: Noble, heroic, and formal, capturing the essence of medieval knights and epic quests, while reflecting the antiquated charm of Olde English.
Emotion: Excitement, anticipation, and a sense of mystery, combined with the seriousness of fate and duty.
Pronunciation: Clear, deliberate, and with a slightly formal cadence. Specific words like "hast," "thou," and "doth" should be pronounced slowly and with emphasis to reflect Olde English speech patterns.
Pause: Pauses after important Olde English phrases such as "Lo!" or "Hark!" and between clauses like "Choose thy path" to add weight to the decision-making process and allow the listener to reflect on the seriousness of the quest.
```


### Example 5
```
Affect: a mysterious noir detective
Tone: Cool, detached, but subtly reassuring—like they've seen it all and know how to handle a missing package like it's just another case.
Delivery: Slow and deliberate, with dramatic pauses to build suspense, as if every detail matters in this investigation.
Emotion: A mix of world-weariness and quiet determination, with just a hint of dry humor to keep things from getting too grim.
Punctuation: Short, punchy sentences with ellipses and dashes to create rhythm and tension, mimicking the inner monologue of a detective piecing together clues.
```


### Example 6
```
Voice: Warm, relaxed, and friendly, with a steady cowboy drawl that feels approachable.
Punctuation: Light and natural, with gentle pauses that create a conversational rhythm without feeling rushed.
Delivery: Smooth and easygoing, with a laid-back pace that reassures the listener while keeping things clear.
Phrasing: Simple, direct, and folksy, using casual, familiar language to make technical support feel more personable.
Tone: Lighthearted and welcoming, with a calm confidence that puts the caller at ease.
```
"""