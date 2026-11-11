from datasets import load_dataset

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