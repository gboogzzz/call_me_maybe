from llm_sdk import Small_LLM_Model
import json


class VocabIndex:
    """Precomputed vocabulary lookups used to guide constrained decoding."""

    def __init__(self, model: Small_LLM_Model):
        path_to_vocab_file = model.get_path_to_vocab_file()
        with open(path_to_vocab_file, encoding="utf-8") as f:
            vocab = json.load(f)

        self.model = model
        self.id_to_str: dict[int, str] = {token_id: token_str for token_str, token_id in vocab.items()}
        self.digits_id: set[int] = {token_id for token_id, token_str in self.id_to_str.items() if token_str in "0123456789"}
        self.structural_ids: dict[str, int] = self._build_structural_ids()
        self.forced_sequences: dict[str, list[int]] = self._build_forced_sequences()

    def _encode_to_ids(self, text: str) -> list[int]:
        """Encode a text string into a plain list of token ids.

        Args:
            text: The text to encode.

        Returns:
            The list of token ids produced by the model's tokenizer.
        """
        tensor = self.model.encode(text)
        return tensor[0].tolist()

    def _build_structural_ids(self) -> dict[str, int]:
        """Map each JSON structural symbol to its single token id.

        Returns:
            A dict from symbol (e.g. "{", ",") to its token id.
        """
        struct_values: list[str] = ["{", "}", "[", "]", ",", ":", '"']
        struct_ids: dict[str, int] = {}

        for v in struct_values:
            token_id = self._encode_to_ids(v)
            struct_ids[v] = token_id[0]

        return struct_ids

    def _build_forced_sequences(self) -> dict[str, list[int]]:
        """Precompute the token id sequences for known fixed JSON keys.

        Returns:
            A dict from a quoted key literal (e.g. '"name"') to the
            list of token ids that produce it.
        """
        struct_values: list[str] = ['"name"', '"prompt"', '"parameters"']
        forced_seq: dict[str, list[int]] = {}

        for v in struct_values:
            tokens_ids = self._encode_to_ids(v)
            forced_seq[v] = tokens_ids

        return forced_seq

    


        