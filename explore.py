from llm_sdk import Small_LLM_Model
from src.vocab import VocabIndex

model = Small_LLM_Model()
vocab_index = VocabIndex(model)

print("space_id:", vocab_index.space_id)
print("  -> id_to_str:", vocab_index.id_to_str[vocab_index.space_id])

print("\ndot_id:", vocab_index.dot_id)
print("  -> id_to_str:", vocab_index.id_to_str[vocab_index.dot_id])

print("\nminus_id:", vocab_index.minus_id)
print("  -> id_to_str:", vocab_index.id_to_str[vocab_index.minus_id])

print("\nforced_sequences_first:")
for chave, ids in vocab_index.forced_sequences_first.items():
    print(f"  {chave!r} -> {ids}")

print("\nforced_sequences_next:")
for chave, ids in vocab_index.forced_sequences_next.items():
    print(f"  {chave!r} -> {ids}")

print("\n--- confirmar que as chaves batem certo entre os dois dicionários ---")
print("Chaves iguais nos dois dicts?",
      set(vocab_index.forced_sequences_first.keys()) == set(vocab_index.forced_sequences_next.keys()))