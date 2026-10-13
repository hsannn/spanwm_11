from dataclasses import dataclass

@dataclass(frozen=True)
class StructuralSpan:
    sentence_index: int
    role: str
    start_char: int
    end_char: int
    text: str
    parser_token_start: int  # spaCy token index (inclusive)
    parser_token_end: int    # spaCy token index (exclusive)

def _trim(a: int, b: int, text: str) -> tuple[int, int]:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b

class TokenSpanMapper:

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def encode_with_offsets(self, text: str):
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        return enc["input_ids"], enc["offset_mapping"]

    def map_positions(self, text: str, span: StructuralSpan) -> tuple[list[int], list[int]]:
        input_ids, offsets = self.encode_with_offsets(text)
        positions: list[int] = []
        for i, (a, b) in enumerate(offsets):
            if a == b:  # zero-length / special token
                continue
            ta, tb = _trim(a, b, text)
            if ta >= tb:
                continue
            if tb > span.start_char and ta < span.end_char:
                positions.append(i)
        return positions, input_ids

    def count(self, text: str, span: StructuralSpan) -> int:
        return len(self.map_positions(text, span)[0])

