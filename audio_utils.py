import io
import uuid
import wave  # Ensure wave is imported
import hashlib # For hashing names/ids
from typing import Any
import datetime # Added import
import asyncio # Added import for async operations

import ipywidgets as widgets
import weave
from IPython.display import HTML as IPythHTML
from IPython.display import Audio as IPythAudio
from IPython.display import display
from weave.flow.annotation_spec import AnnotationSpec

from judge import BinaryVoiceRank


def save_wave_read_to_file(wave_read_obj, filename):
    with wave.open(filename, "wb") as out_wav:
        out_wav.setparams(wave_read_obj.getparams())
        wave_read_obj.rewind()
        out_wav.writeframes(wave_read_obj.readframes(wave_read_obj.getnframes()))


def wave_read_to_wav_bytes(wave_read_obj):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out_wav:
        out_wav.setparams(wave_read_obj.getparams())
        wave_read_obj.rewind()
        out_wav.writeframes(wave_read_obj.readframes(wave_read_obj.getnframes()))
    return buf.getvalue()


class AudioRanker:
    """
    An IPython widget to sequentially rank N audio samples.

    The widget displays all audio samples with players. It then prompts
    the user to assign Rank 1, Rank 2, etc., by selecting an audio
    sample from the remaining unranked options.
    It can also log the final ranking as a Weave human annotation.
    """

    def __init__(
        self,
        audio_samples_input: list,
        image_path: str = None,
        weave_client: Any = None,
        target_call: Any = None, # Changed from target_call_id
        scorer_name: str = "AudioRanker",
        scorer_description: str = "Sequential ranking of multiple audio samples.",
    ):
        """
        Initializes the AudioRanker widget.

        Args:
            audio_samples_input: A list of audio samples. Each item can be:
                - A tuple: (name: str, audio_data: wave.Wave_read object or bytes)
                - A dict: {'name': str, 'audio': wave.Wave_read object or bytes}
                          (tries to find 'name'/'id' and 'audio'/'data' keys)
                          This format is suitable for data like Weave dataset rows.
            image_path: Path to an image associated with the audio samples.
            weave_client: Optional initialized Weave client for logging feedback.
                          If None, Weave logging is disabled.
            target_call: Optional Weave call object to associate this ranking feedback with.
                         Required if weave_client is provided and logging/scoring is desired.
            scorer_name: Name for the Weave Human Annotation Scorer.
            scorer_description: Description for the Weave Human Annotation Scorer.
        """
        self.weave_client = weave_client
        self.target_call = target_call # Store the call object
        self.target_call_id = target_call.id if target_call else None # Extract ID for existing logic
        self.scorer_name = scorer_name
        self.scorer_description = scorer_description
        self.scorer_uri = None  # Will be set after publishing
        self.raw_audio_samples = []  # Stores more info now
        self.image_path = image_path # Store image_path

        for i, item in enumerate(audio_samples_input):
            original_input_name = None # This will be the name or id from the input dict/tuple
            audio_data_source = None

            if isinstance(item, tuple) and len(item) == 2:
                original_input_name, audio_data_source = item
                if not isinstance(original_input_name, str):
                    original_input_name = f"Sample_{i + 1}"  # Fallback
            elif isinstance(item, dict):
                original_input_name = str(item.get('name', item.get('id', f"Sample_{i + 1}")))
                audio_data_source = item.get("audio", item.get("data"))
            else:
                print(f"Warning: Skipping item {i} due to unrecognized format: {type(item)}.")
                continue

            if audio_data_source is None:
                print(f"Warning: Could not find audio data for {original_input_name}. Skipping.")
                continue
            
            # Sanitize original_input_name for use as part of widget's internal ID
            # This internal_widget_id is used for selections in dropdowns etc.
            safe_name_part = "".join(c if c.isalnum() else "_" for c in str(original_input_name))
            internal_widget_id = f"{safe_name_part}_{i}"

            # Generate anonymized label for UI display
            short_hash = hashlib.sha256(str(original_input_name).encode()).hexdigest()[:8]
            anonymized_label = f"Sample {i + 1} ({short_hash})"

            audio_bytes = None
            if hasattr(audio_data_source, "getparams") and hasattr(
                audio_data_source, "rewind"
            ):  # wave.Wave_read like
                audio_bytes = self._convert_wave_to_bytes(audio_data_source)
            elif isinstance(audio_data_source, bytes):
                audio_bytes = audio_data_source  # Assume already WAV bytes
            else:
                print(
                    f"Warning: Unsupported audio data type for {original_input_name}: {type(audio_data_source)}. Skipping."
                )
                continue
            
            if audio_bytes:
                self.raw_audio_samples.append({
                    "widget_internal_id": internal_widget_id, # Widget's internal unique ID for selection
                    "id": original_input_name, # Original name/id from input, for logging and final output
                    "original_input_order": item.get("original_input_order", i) if isinstance(item, dict) else i, # Get from input or use index
                    "data": audio_bytes,
                    "anonymized_label": anonymized_label, # For UI display
                    "short_hash": short_hash # Store the generated hash
                })
            else:
                print(f"Warning: Failed to process audio bytes for {original_input_name}. Skipping.")

        if not self.raw_audio_samples:
            self.widget = widgets.Label("No valid audio samples to display for ranking.")
            return

        self.num_samples = len(self.raw_audio_samples)
        self._initialize_state()
        self._build_ui()

        # Setup Weave scorer now that UI elements like messages_output exist
        if self.weave_client and self.target_call: # Check for target_call object
            self._setup_weave_scorer()
        elif self.weave_client and not self.target_call: # If client but no call object
            with self.messages_output:
                display(
                    IPythHTML(
                        "<p style='color:orange;'>Warning: Weave client provided, but no target_call object. Weave ranking annotation will be disabled.</p>"
                    )
                )

    def _setup_weave_scorer(self):
        """Defines and publishes the Human Annotation Scorer for audio rankings."""
        if not self.weave_client:  # Should not happen if called conditionally
            print("Internal Warning: Weave client not provided to _setup_weave_scorer.")
            return

        spec = AnnotationSpec(
            name=self.scorer_name,
            description=self.scorer_description,
            field_schema={
                "type": "object",
                "properties": {
                    "ranking_session_id": {"type": "string"},
                    "target_call_id": {"type": "string"},
                    "ranked_samples": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "rank": {"type": "integer"},
                                "id": {"type": "string"},  # Original ID provided by user
                                "original_input_order": {"type": "integer"}, # Original index
                                "short_hash": {"type": "string"}, # Added short_hash
                            },
                            "required": ["rank", "id", "original_input_order", "short_hash"], # Added short_hash
                        },
                    },
                    "all_presented_sample_widget_ids": {"type": "array", "items": {"type": "string"}}, # IDs used internally by widget
                },
                "required": ["ranking_session_id", "target_call_id", "ranked_samples"],
            },
        )
        try:
            # weave.publish uses the globally initialized client context from weave.init()
            # The second argument `name` is the object ID for the scorer, allowing upserts.
            scorer_ref = weave.publish(spec, name=self.scorer_name)
            self.scorer_uri = scorer_ref.uri()  # Get the URI of the Ref object

            if hasattr(self, "messages_output") and self.messages_output:
                with self.messages_output:
                    display(
                        IPythHTML(
                            f"<p style='color:green;'>Weave Annotation Scorer '{self.scorer_name}' setup/updated successfully. URI: {self.scorer_uri}</p>"
                        )
                    )
            else:  # Fallback if messages_output not yet available (should be, due to call order)
                print(f"Weave Annotation Scorer '{self.scorer_name}' setup/updated. URI: {self.scorer_uri}")

        except Exception as e:
            self.scorer_uri = None  # Ensure it's None if setup fails
            error_message = f"Error setting up Weave scorer '{self.scorer_name}': {e}. Weave feedback logging will be disabled."
            if hasattr(self, "messages_output") and self.messages_output:
                with self.messages_output:
                    display(IPythHTML(f"<p style='color:red;'>{error_message}</p>"))
            else:
                print(error_message)

    def _convert_wave_to_bytes(self, wave_read_obj):
        if wave_read_obj is None:
            return None
        if hasattr(wave_read_obj, "rewind"):
            wave_read_obj.rewind()

        buf = io.BytesIO()
        try:
            params = wave_read_obj.getparams()
        except Exception as e:
            print(f"Error getting params from wave object: {e}")
            return None

        with wave.open(buf, "wb") as out_wav:
            out_wav.setparams(params)
            if hasattr(wave_read_obj, "rewind"):  # Rewind again before reading frames
                wave_read_obj.rewind()
            try:
                frames = wave_read_obj.readframes(wave_read_obj.getnframes())
                out_wav.writeframes(frames)
            except Exception as e:
                print(f"Error reading/writing frames for wave object: {e}")
                return None
        return buf.getvalue()

    def _initialize_state(self):
        self.ranked_assignments = {}
        self.unranked_audio_ids = [sample['widget_internal_id'] for sample in self.raw_audio_samples] # Use widget_internal_id here
        self.current_rank_to_assign = 1
        self.final_ranking_list = []
        self.ranking_completion_timestamp = None # Added timestamp
        # Clear previous scorer messages if any, on reset
        if hasattr(self, "messages_output") and self.messages_output:
            # This might be too aggressive if other messages are important.
            # Consider a more targeted way to clear scorer-specific messages if needed.
            # For now, clearing all messages on reset to avoid stale scorer status.
            self.messages_output.clear_output(wait=True)

    def _get_unranked_sample_options(self):
        options = [("Select an audio...", None)]
        for sample_widget_id in self.unranked_audio_ids: # Iterate using widget_internal_id
            sample_details = next((s for s in self.raw_audio_samples if s['widget_internal_id'] == sample_widget_id), None)
            if sample_details:
                display_label = sample_details['anonymized_label']
                options.append((display_label, sample_widget_id)) # Value is widget_internal_id
            else:
                options.append((f"Unknown Sample (Widget ID: {sample_widget_id})", sample_widget_id))
        return options

    def _build_ui(self):
        audio_display_items = [widgets.HTML("<h3>Which voice sample suits this character?</h3>")]
        for i, sample in enumerate(self.raw_audio_samples):
            audio_player_out = widgets.Output()
            with audio_player_out:
                display(IPythAudio(data=sample['data'], autoplay=False))
            
            item_box = widgets.VBox([
                widgets.Label(sample['anonymized_label']),
                audio_player_out
            ], layout=widgets.Layout(margin='0 0 10px 0'))
            audio_display_items.append(item_box)
        audio_players_vbox = widgets.VBox(audio_display_items) # Renamed for clarity

        # Image display
        image_widget = None
        if self.image_path:
            try:
                with open(self.image_path, "rb") as f:
                    image_data = f.read()
                image_widget = widgets.Image(value=image_data, format='jpg', width=800, height=300) # Set width and height
            except FileNotFoundError:
                image_widget = widgets.Label(f"Image not found: {self.image_path}")
            except Exception as e:
                image_widget = widgets.Label(f"Error loading image: {e}")
        
        # Two-column layout for audio players and image
        if image_widget:
            content_hbox = widgets.HBox([audio_players_vbox, image_widget])
        else:
            content_hbox = audio_players_vbox # Fallback if no image

        self.rank_assignment_label = widgets.Label(value="")
        self.rank_selection_dropdown = widgets.Dropdown(options=[], value=None, layout={'width': 'auto', 'min_width': '250px'})
        self.assign_rank_button = widgets.Button(description="Assign This Rank", icon="check", button_style='primary')
        self.assign_rank_button.on_click(self._on_assign_rank_click)
        
        ranking_controls_box = widgets.VBox([
            self.rank_assignment_label,
            self.rank_selection_dropdown,
            self.assign_rank_button
        ], layout=widgets.Layout(margin='20px 0'))

        self.current_rankings_display_html = widgets.HTML(value="")
        self.reset_button = widgets.Button(description="Reset Rankings", icon="refresh", button_style='warning')
        self.reset_button.on_click(self._on_reset_click)
        self.messages_output = widgets.Output()

        self.widget = widgets.VBox([
            content_hbox, # Use the HBox here
            widgets.HTML("<hr style='margin: 20px 0;'>"),
            ranking_controls_box,
            widgets.HTML("<h3>Current Rankings:</h3>"),
            self.current_rankings_display_html,
            widgets.HTML("<br>"),
            self.reset_button,
            self.messages_output
        ])
        self._update_ranking_controls_state()

    def _get_current_rankings_html(self):
        if not self.ranked_assignments:
            return "<p><i>No rankings assigned yet.</i></p>"
        
        html = "<ol style='padding-left: 20px;'>"
        for rank_num in sorted(self.ranked_assignments.keys()):
            audio_widget_id = self.ranked_assignments[rank_num] # This is widget_internal_id
            sample_details = next((s for s in self.raw_audio_samples if s['widget_internal_id'] == audio_widget_id), None)
            display_label = sample_details['anonymized_label'] if sample_details else f"Unknown (Widget ID: {audio_widget_id})"
            html += f"<li style='margin-bottom: 5px;'><b>Rank {rank_num}:</b> {display_label}</li>"
        html += "</ol>"
        return html

    def _update_ranking_controls_state(self):
        self.messages_output.clear_output()
        if not self.unranked_audio_ids: 
            self.rank_assignment_label.value = "🎉 All samples have been ranked! 🎉"
            self.rank_selection_dropdown.options = [("All ranked", None)]
            self.rank_selection_dropdown.value = None
            self.rank_selection_dropdown.disabled = True
            self.assign_rank_button.disabled = True
            self.ranking_completion_timestamp = datetime.datetime.now() # Set timestamp
            
            self.final_ranking_list = []
            for rank_num in sorted(self.ranked_assignments.keys()):
                audio_widget_id = self.ranked_assignments[rank_num] # This is widget_internal_id
                sample_details = next((s for s in self.raw_audio_samples if s['widget_internal_id'] == audio_widget_id), None)
                if sample_details:
                    self.final_ranking_list.append({
                        'rank': rank_num,
                        'id': sample_details['id'], # Use the original ID for the output
                        'original_input_order': sample_details['original_input_order'],
                        'short_hash': sample_details.get('short_hash', 'unknown_hash') # Add the short_hash
                    })
                else: # Should ideally not happen if logic is correct
                    self.final_ranking_list.append({'rank': rank_num, 'id': "Unknown", 'original_input_order': -1, 'widget_internal_id': audio_widget_id, 'short_hash': 'unknown_hash'})
            
            with self.messages_output:
                print("Final Rankings (UI display based on anonymized labels):")
                for rank_num in sorted(self.ranked_assignments.keys()):
                    audio_widget_id = self.ranked_assignments[rank_num]
                    sample_details = next((s for s in self.raw_audio_samples if s['widget_internal_id'] == audio_widget_id), None)
                    display_label = sample_details['anonymized_label'] if sample_details else f"Unknown (Widget ID: {audio_widget_id})"
                    print(f"  Rank {rank_num}: {display_label}") # (Original ID: {sample_details['id'] if sample_details else 'N/A'}

                # Log to Weave if client, target_call object, and scorer_uri are available
                if self.weave_client and self.target_call and self.scorer_uri: # Check for self.target_call
                    try:
                        # We already have self.target_call, no need to fetch it again
                        ranking_session_id = str(uuid.uuid4())
                        all_sample_widget_ids = [s["widget_internal_id"] for s in self.raw_audio_samples]
                        
                        data_for_scorer = {
                            "ranking_session_id": ranking_session_id,
                            "target_call_id": self.target_call_id, # Still log the ID
                            "ranked_samples": self.final_ranking_list, # Already in the new format
                            "all_presented_sample_widget_ids": all_sample_widget_ids,
                        }

                        payload_to_send = {"value": data_for_scorer}

                        self.target_call.feedback.add( # Use self.target_call directly
                            feedback_type=f"wandb.annotation.{self.scorer_name}",
                            payload=payload_to_send, # Use the wrapped payload
                            annotation_ref=self.scorer_uri
                        )
                        with self.messages_output: # Ensure messages_output is used here too
                            display(IPythHTML(f"<p style='color:green;'>Ranking successfully logged as annotation to Weave call '{self.target_call_id}' (Session ID: {ranking_session_id}).</p>"))
                        
                        # Now, also apply the scorer asynchronously
                        asyncio.create_task(self._apply_scorer_async())

                    except Exception as e:
                        with self.messages_output: # Ensure messages_output is used here too
                            display(IPythHTML(f"<p style='color:red;'>Error logging ranking to Weave or preparing to apply scorer: {e}</p>"))
                elif self.weave_client:
                    missing_parts = []
                    if not self.target_call: # Check for target_call object
                        missing_parts.append("target_call object")
                    if not self.scorer_uri:
                        missing_parts.append("scorer_uri (scorer setup may have failed)")
                    with self.messages_output: # Ensure messages_output is used here too
                        display(
                            IPythHTML(
                                f"<p style='color:orange;'>Skipping Weave annotation log and scorer application: Missing {', '.join(missing_parts)}.</p>"
                            )
                        )
        else:
            self.rank_assignment_label.value = f"Select audio for Rank {self.current_rank_to_assign}:"
            self.rank_selection_dropdown.options = self._get_unranked_sample_options()
            self.rank_selection_dropdown.value = None
            self.rank_selection_dropdown.disabled = False
            self.assign_rank_button.disabled = False

        self.current_rankings_display_html.value = self._get_current_rankings_html()

    def _on_assign_rank_click(self, button):
        self.messages_output.clear_output()
        selected_audio_id = self.rank_selection_dropdown.value

        if selected_audio_id is None:
            with self.messages_output:
                display(
                    IPythHTML(
                        "<p style='color:red;'>❗ Please select an audio sample for the current rank.</p>"
                    )
                )
            return

        self.ranked_assignments[self.current_rank_to_assign] = selected_audio_id
        self.unranked_audio_ids.remove(selected_audio_id)
        self.current_rank_to_assign += 1

        self._update_ranking_controls_state()

    def _on_reset_click(self, button):
        self.messages_output.clear_output()
        self._initialize_state()
        self._update_ranking_controls_state()
        with self.messages_output:
            print("Rankings have been reset.")

    def display_widget(self):
        """Displays the ranking widget in a Jupyter environment."""
        if hasattr(self, "widget"):
            display(self.widget)
        else:
            print("AudioRanker widget could not be initialized properly (e.g., no valid audio samples).")

    def get_final_rankings(self):
        """
        Returns the final list of ranked audio samples if all samples have been ranked.
        Each item in the list is a dict: {'rank': int, 'id': str, 'original_input_order': int, 'short_hash': str}
        Returns None if ranking is not complete.
        """
        if not self.unranked_audio_ids and self.final_ranking_list and self.ranking_completion_timestamp:
            preferred_id = None
            rejected_id = None
            if self.final_ranking_list:
                for item in self.final_ranking_list:
                    if item['rank'] == 1:
                        preferred_id = item['id']
                    if item['rank'] == self.num_samples: # Last rank
                        rejected_id = item['id']
            
            return {
                "rankings": self.final_ranking_list,
                "completed_at": self.ranking_completion_timestamp.strftime("%Y-%m-%d_%H-%M"),
                "preferred_id": preferred_id,
                "rejected_id": rejected_id
            }
        else:
            # print("Ranking is not yet complete or has been reset.") # Console noise
            return None

    async def _apply_scorer_async(self):
        """Applies the scorer asynchronously using the target_call object."""
        if not hasattr(self, 'messages_output') or not self.messages_output:
            print("Debug: messages_output not available for _apply_scorer_async")
            # Fallback to print if messages_output isn't ready, though it should be.
            def display_html_fallback(html_str):
                print(html_str.replace("<p style='color:red;'>", "ERROR: ").replace("<p style='color:orange;'>", "WARNING: ").replace("<p style='color:green;'>", "INFO: ").replace("</p>", ""))
        else:
            def display_html_fallback(html_str): # pylint: disable=function-redefined
                with self.messages_output:
                    display(IPythHTML(html_str))

        if not self.target_call:
            display_html_fallback(
                "<p style='color:orange;'>Warning: No target_call available to apply scorer.</p>"
            )
            return

        final_rankings_data = self.get_final_rankings()
        if not final_rankings_data:
            display_html_fallback(
                "<p style='color:orange;'>Warning: Final rankings not available. Scorer not applied.</p>"
            )
            return

        ranking_list = final_rankings_data.get("rankings")
        ranking_timestamp = final_rankings_data.get("completed_at")
        if not ranking_list:
            display_html_fallback(
                "<p style='color:red;'>Error: 'rankings' key missing in final_rankings_data. Scorer not applied.</p>"
            )
            return

        try:
            # Check if BinaryVoiceRank is defined. If not, we can't proceed.
            if 'BinaryVoiceRank' not in globals() and 'BinaryVoiceRank' not in locals():
                 raise NameError("BinaryVoiceRank class is not defined or imported.")

            scorer_instance = BinaryVoiceRank(ranking_list, ranking_timestamp) # type: ignore # noqa

            with self.messages_output: # Ensure this is a clear message
                self.messages_output.clear_output(wait=True) # Clear previous messages for this step
                display(IPythHTML("<p>Applying scorer (e.g., BinaryVoiceRank)...</p>"))
            
            res = await self.target_call.apply_scorer(scorer_instance)
            
            # Append to messages_output, don't clear it here
            with self.messages_output:
                display(IPythHTML(f"<p style='color:green;'>BinaryVoiceRank applied successfully to weave call. Result: {res}</p>"))

        except NameError as ne: # Specifically for BinaryVoiceRank not being defined
             display_html_fallback(f"<p style='color:red;'>Error: {ne}. Scorer `BinaryVoiceRank` cannot be applied. Please ensure it's imported/defined in your notebook or environment.</p>")
        except Exception as e:
            display_html_fallback(f"<p style='color:red;'>Error applying scorer: {e}</p>")
