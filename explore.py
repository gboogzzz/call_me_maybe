from llm_sdk import Small_LLM_Model
from src.vocab import VocabIndex

model = Small_LLM_Model()
vocab_index = VocabIndex(model)

print("Tamanho id_to_str:", len(vocab_index.id_to_str))
print("Tamanho digits_id:", len(vocab_index.digits_id))
print("Digits:", {vocab_index.id_to_str[i] for i in vocab_index.digits_id})

print("\nStructural ids:")
for simbolo, id_ in vocab_index.structural_ids.items():
    print(f"{simbolo!r} -> {id_}")

print("\nForced sequences:")
for chave, ids in vocab_index.forced_sequences.items():
    print(f"{chave!r} -> {ids}")