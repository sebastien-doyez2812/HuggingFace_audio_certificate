from datasets import load_dataset
from datasets import Audio
import librosa

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")

# --- Sampling at 16kHz ---
minds = minds.cast_column("audio", Audio(sampling_rate=16000))
# print(minds)

# --- Filter: long audio ---
MAX_DURATION_IN_SECONDS = 20.0

def is_audio_length_in_range(input_length):
    return input_length < MAX_DURATION_IN_SECONDS

new_column = [
    librosa.get_duration(y=x["array"], sr=x["sampling_rate"]) for x in minds["audio"]
]
minds = minds.add_column("duration", new_column)

minds = minds.filter(is_audio_length_in_range, input_columns=["duration"])

minds = minds.remove_columns(["duration"])
# print(minds)



# --- Preprocess for have the good sapling Rate ---

from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

def prepare_dataset(example):
    audio = example["audio"]

    if audio["sampling_rate"] != 16000:
        audio_array = librosa.resample(
            audio["array"], orig_sr=audio["sampling_rate"], target_sr=16000
        )
        audio = {"array": audio_array, "sampling_rate": 16000}

    features = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"], padding=True
    )
    return features

minds = minds.map(prepare_dataset)
print(minds)