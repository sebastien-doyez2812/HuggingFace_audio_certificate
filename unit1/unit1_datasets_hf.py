from datasets import load_dataset
import gradio as gr

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")

print("-------Structure-------\n")
print(minds)

print("-------Example-------\n")
example = minds[0]
print(example)

id2label = minds.features["intent_class"].int2str
print(id2label(example["intent_class"]))

print("----Cleaning-----")
columns_to_remove = ["lang_id", "english_transcription"]
print(f"Before:\n{minds}")
minds = minds.remove_columns(columns_to_remove)
print(f"After:\n{minds}")

print("----Generate-----")

def generate_audio():
    example = minds.shuffle()[0]
    audio = example["audio"]
    return (
        audio["sampling_rate"],
        audio["array"],
    ), id2label(example["intent_class"])


with gr.Blocks() as demo:
    with gr.Column():
        for _ in range(4):
            audio, label = generate_audio()
            output = gr.Audio(audio, label=label)

demo.launch(debug=True)