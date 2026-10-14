from llm_sdk import Small_LLM_Model
from src.vocab import VocabIndex
from src.decoding import generate_string

model = Small_LLM_Model()
vocab = VocabIndex(model)

key = "replacement"

# valores ja gerados com sucesso pelos parametros anteriores (source_string, regex),
# tal como viste nos testes anteriores -- aqui simulamos o que o pipeline real ja
# teria decidido antes de chegar ao "replacement"
casos = [
    (
        "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
        {"source_string": "Hello 34 I'm 233 years old", "regex": "Hello [0-9] I'm [0-9] years old"},
    ),
    (
        "Replace all vowels in 'Programming is fun' with asterisks",
        {"source_string": "Programming is fun", "regex": ".*[aeiouAEIOU].*"},
    ),
    (
        "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'",
        {"source_string": "The cat sat on the mat with another cat", "regex": "cat"},
    ),
]


def variante_A_sem_irmaos(prompt, prior):
    # controlo: formato vencedor anterior, SEM ver os parametros ja gerados
    return f"Request: {prompt}\n{key}: \""


def variante_H(prompt, prior):
    # contexto cumulativo, com labels "Request:"
    linhas = "".join(f'{k}: "{v}"\n' for k, v in prior.items())
    return f"Request: {prompt}\n{linhas}{key}: \""


def variante_I(prompt, prior):
    # contexto cumulativo, sem label "Request:"
    linhas = "".join(f'{k}: "{v}"\n' for k, v in prior.items())
    return f"{prompt}\n{linhas}{key}: \""


variantes = {
    "A (controlo, sem irmaos)": variante_A_sem_irmaos,
    "H (Request: + irmaos ja gerados)": variante_H,
    "I (so prompt + irmaos ja gerados)": variante_I,
}

for prompt, prior in casos:
    print("=" * 100)
    print(f"PROMPT: {prompt}")
    print(f"ja gerado antes: {prior}")
    for nome, builder in variantes.items():
        contexto = builder(prompt, prior)
        prompt_ids = vocab.encode_to_ids(contexto)
        gen, _ = generate_string(model, vocab, list(prompt_ids), list(prompt_ids))
        print(f"\n  Variante: {nome}")
        print(f"    contexto: {contexto!r}")
        print(f"    gerado: {gen!r}")
    print()