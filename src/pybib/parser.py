"""Parser to convert between block objects and plain text.

Parser grammar from
https://github.com/aclements/biblib
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Optional, Self

import attrs


class ParserError(Exception):
    """Problem when parsing BibTeX file."""


WHITESPACE_RE = re.compile(r"[ \t\n]*")
NUMBER_RE = re.compile(r"[0-9]+")
IDENTIFIER_RE = re.compile(r"(?![0-9])(?:(?![ \t\"#%\'(),={}])[\x20-\x7f])+")
KEY_PAREN_RE = re.compile(r"[^, \t\n()]*")
KEY_BRACE_RE = re.compile(r"[^, \t\n{}]*")


def _scan_whitespace(text: str, offset: int) -> int:
    """Scan for whitespace."""
    return WHITESPACE_RE.match(text, offset).end()


def _try_scan_literal(
    text: str,
    offset: int,
    valid_literals: Optional[str] = None,
    *,
    skip_whitespace: bool = True,
) -> tuple[int, bool]:
    """Try to match a single character."""
    scan_pos = _scan_whitespace(text, offset) if skip_whitespace else offset
    if scan_pos < len(text):
        literal = text[scan_pos]
    else:
        return offset, False
    if valid_literals is not None and literal not in valid_literals:
        return offset, False
    return scan_pos + 1, True


def _scan_literal(
    text: str,
    offset: int,
    valid_literals: Optional[str] = None,
    *,
    skip_whitespace: bool = True,
) -> tuple[int, str]:
    """Scan for a single character, possibly skipping leading whitespace."""
    offset, found = _try_scan_literal(
        text, offset, valid_literals, skip_whitespace=skip_whitespace
    )
    if not found:
        raise ParserError(f"Expected {valid_literals}")
    # Scan position was advanced past the matched token
    literal = text[offset - 1]
    return offset, literal


def _scan_identifier(
    text: str, offset: int, *, skip_whitespace: bool = True
) -> tuple[int, str]:
    """Scan for an identifier, possibly skipping leading whitespace."""
    if skip_whitespace:
        offset = _scan_whitespace(text, offset)
    match = IDENTIFIER_RE.match(text, offset)
    if not match:
        raise ParserError("Expected identifier")
    return match.end(), match.group(0)


# todo: note that enclosing '"' within '{}' is allowed. Thus we need to track {} even when scanning for "
def _scan_balanced(text: str, scan_offs: int) -> int:
    """Scan for balanced parens; leading whitespace not allowed."""
    open_tok = text[scan_offs]
    close_tok = {"{": "}", "(": ")", '"': '"'}[open_tok]
    paren_count = 1
    # Balance the parens
    while paren_count > 0:
        open_idx = text.find(open_tok, scan_offs + 1)
        close_idx = text.find(close_tok, scan_offs + 1)
        if open_idx >= 0 and close_idx >= 0:
            # Both found
            paren_count += 1 if open_idx < close_idx else -1
            scan_offs = min(open_idx, close_idx)
        elif open_idx >= 0:
            paren_count += 1
            scan_offs = open_idx
        elif close_idx >= 0:
            paren_count -= 1
            scan_offs = close_idx
        else:
            # No more parens left
            raise ParserError("Unable to balance parens")
    return scan_offs


def _scan_value(
    text: str, offset: int, *, skip_whitespace: bool = True
) -> tuple[int, str]:

    def _scan_field_piece(_offs: int) -> int:
        _offs = _scan_whitespace(text, _offs)
        # Match number
        if match := NUMBER_RE.match(text, _offs):
            return match.end()
        # Match quoted text
        if text[_offs] in '{"':
            return _scan_balanced(text, _offs) + 1
        # Match identifier
        _offs, _ = _scan_identifier(text, _offs)
        return _offs

    if skip_whitespace:
        offset = _scan_whitespace(text, offset)
    value_begin_offs = offset
    try:
        offset = _scan_field_piece(offset)
    except ParserError as exe:
        raise ParserError("Expected string, number, or macro") from exe

    # Try searching for additional pieces
    while True:
        lookahead_offs = _scan_whitespace(text, offset)
        if lookahead_offs == len(text) or text[lookahead_offs] != "#":
            break
        offset = _scan_field_piece(lookahead_offs + 1)

    return offset, text[value_begin_offs:offset]


def _unwrap_entry(text: str) -> tuple[tuple[int, int], str, str]:
    """Match an entry and return only the body.

    Obeys the following grammar:

        entry = ws '@' ws ident ws (
            '{' ws entry_body ws '}'
            | '(' ws entry_body ws ')'
        ) ws

        entry_body = balanced_parens

    where `ident` and `ws` are defined globally.
    """
    scan_offs = 0
    scan_offs, _ = _scan_literal(text, scan_offs, "@")
    scan_offs, entry_kind = _scan_identifier(text, scan_offs)
    # Now match the wrapped body of the entry
    scan_offs, open_tok = _scan_literal(text, scan_offs, "{(")
    body_begin_offs = scan_offs
    # Paren matching expects starting point on opening paren
    scan_offs -= 1
    scan_offs = _scan_balanced(text, scan_offs)
    body_end_offs = scan_offs
    scan_offs = _scan_whitespace(text, scan_offs + 1)
    if scan_offs != len(text):
        raise ParserError("Expected no further content")
    return (
        (body_begin_offs - 1, body_end_offs),
        entry_kind,
        text[body_begin_offs:body_end_offs],
    )


class BibtexBlock(ABC):
    """Block object in a bibtex file."""

    @abstractmethod
    def render(self) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_text(cls, text: str) -> Self:
        raise NotImplementedError


@attrs.define(order=False)
class EntryBlock(BibtexBlock):
    text: str
    entry_type: str
    entry_key: str
    fields: Mapping[str, str]

    def render(self) -> str:
        return self.text

    @classmethod
    def from_text(cls, text: str) -> Self:
        """Match entry in the BibTeX database.

        Uses the grammar:

            entry = '@' ws ident ws (
                '{' ws key ws entry_body? ws '}'
                | '(' ws key_paren ws entry_body? ws ')'
            )

            key = [^, \t}\n]*

            key_paren = [^, \t\n]*

            entry_body = (',' ws ident ws '=' ws value ws)* ','?

        Note that we do not exactly follow the BibTeX spec: since parens can be included
        in the key without restriction, the entire entry body may not have balanced
        parens. This will make earlier stages of the parser more complex, so we ignore
        it for now and disallow '{}' and '()' in entries wrapped by those delimiters.
        """
        (begin_idx, _), entry_type, body_text = _unwrap_entry(text)
        open_tok = text[begin_idx]
        scan_offs = _scan_whitespace(body_text, 0)
        # Match entry key (may be empty)
        key_re = KEY_BRACE_RE if open_tok == "{" else KEY_PAREN_RE
        match = key_re.match(body_text, scan_offs)
        # Regex will always match
        entry_key = match.group(0)
        scan_offs = match.end()
        # Loop to match fields
        entry_fields = {}
        while True:
            scan_offs, found_comma = _try_scan_literal(body_text, scan_offs, ",")
            if not found_comma:
                break
            scan_offs = _scan_whitespace(body_text, scan_offs)
            if scan_offs == len(body_text):
                # Fine to reach end of body with a trailing comma
                break
            scan_offs, field_name = _scan_identifier(body_text, scan_offs)
            scan_offs, _ = _scan_literal(body_text, scan_offs, "=")
            scan_offs, field_value = _scan_value(body_text, scan_offs)
            entry_fields[field_name] = field_value
        scan_offs = _scan_whitespace(body_text, scan_offs)
        if scan_offs != len(body_text):
            raise ParserError("Expected no further content in entry")
        return cls(text, entry_type, entry_key, entry_fields)


@attrs.define(order=False)
class CommentBlock(BibtexBlock):
    """Comment included in the BibTeX file."""

    text: str

    def render(self) -> str:
        return self.text

    @classmethod
    def from_text(cls, text: str) -> Self:
        return cls(text)


@attrs.define(order=False)
class PreambleBlock(BibtexBlock):
    """Preamble content included in database."""

    text: str
    value: str

    def render(self) -> str:
        return self.text

    @classmethod
    def from_text(cls, text: str) -> Self:
        """Parse from BibTeX file.

        Uses the following grammar:

            preamble = '@' ws 'preamble' ws (
                '{' ws preamble_body ws '}'
                | '(' ws preamble_body ws ')'
            )

            preamble_body = value

        where ws and value rules are defined globally.
        """
        _, kind, body_text = _unwrap_entry(text)
        if kind != "preamble":
            raise ParserError("Expected preamble entry")
        scan_offs = 0
        scan_offs, preamble_value = _scan_value(body_text, scan_offs)
        scan_offs = _scan_whitespace(body_text, scan_offs)
        if scan_offs != len(body_text):
            raise ParserError("Expected no further content")
        return cls(text, preamble_value)


@attrs.define(order=False)
class StringBlock(BibtexBlock):
    """String definition included in the database."""

    text: str
    name: str
    value: str

    def render(self) -> str:
        return self.text

    @classmethod
    def from_text(cls, text: str) -> Self:
        """Parse from the BibTeX database.

        Use the following grammar:

            string = '@' ws 'string' ws (
                '{' ws string_body ws '}'
                | '(' ws string_body ws ')'
            )

            string_body = identifier ws '=' ws value

        where ws, identifier, and value rules are defined globally.
        """
        _, kind, body_text = _unwrap_entry(text)
        if kind != "string":
            raise ParserError("Expected string entry")
        scan_offs = 0
        scan_offs, str_name = _scan_identifier(body_text, scan_offs)
        scan_offs, _ = _scan_literal(body_text, scan_offs, "=")
        scan_offs, str_value = _scan_value(body_text, scan_offs)
        scan_offs = _scan_whitespace(body_text, scan_offs)
        if scan_offs != len(body_text):
            raise ParserError("Expected no further content")
        return cls(text, str_name, str_value)


@attrs.define(order=False)
class BadBlock(BibtexBlock):
    """Container for malformed BibTeX entries."""

    text: str

    def render(self) -> str:
        return ""

    @classmethod
    def from_text(cls, text: str) -> Self:
        return cls(text)


def _split_entries(lines: Iterable[str]) -> Iterator[str]:
    """Split a stream of text into BibTeX entries.

    Lines are an arbitrary divisions and but are assumed to be delimited by whitespace.
    """
    lines_iter = iter(lines)

    # Working data
    # Entries may span multiple lines, so new chunks are processed
    # until an entry is complete
    text_chunks: list[str] = []
    # Position within the first chunk where the current entry starts
    start_offs = 0
    # Position of parsing in the last block (used only in top level parser)
    scan_offs = 0

    def get_new_chunk() -> bool:
        """Try to get a new chunk from input."""
        nonlocal text_chunks
        try:
            text_chunks.append(next(lines_iter))
        except StopIteration:
            return False
        return True

    def finalize_block(_end_offs: int) -> str:
        """Group together chunks between the offsets."""
        nonlocal start_offs, text_chunks

        if len(text_chunks) == 0:
            return ""
        elif len(text_chunks) == 1:
            full_block = text_chunks[0][start_offs:_end_offs]
            start_offs = _end_offs
            return full_block

        full_block = (
            text_chunks[0][start_offs:]
            + "".join(text_chunks[1:-1])
            + text_chunks[-1][:_end_offs]
        )
        start_offs = _end_offs
        text_chunks = text_chunks[-1:]
        return full_block

    def scan_whitespace(_start_scan) -> int:
        """Match whitespace, possibly continuing to next chunk."""
        # This will always match
        match = WHITESPACE_RE.match(text_chunks[-1], _start_scan)
        if match.end() < len(text_chunks[-1]):
            return match.end()
        # Keep searching in next chunk
        if get_new_chunk():
            return scan_whitespace(0)
        return match.end()

    def scan_identifier(_start_scan) -> tuple[str, int]:
        """Match an identifier and any leading whitespace."""
        start_idx = scan_whitespace(_start_scan)
        # If we are at the end of a block, load a new chunk
        if start_idx == len(text_chunks[0]):
            if not get_new_chunk():
                raise ParserError("End of input while scanning for identifier")
            start_idx = 0
        if match := IDENTIFIER_RE.match(text_chunks[-1], start_idx):
            return match.group(0), match.end()
        raise ParserError("Could not match identifier")

    def scan_balanced(_start_scan: int) -> int:
        """Scan for balanced parentheses."""
        start_idx = scan_whitespace(_start_scan)
        # If we are at the end of a block, load a new chunk
        if start_idx == len(text_chunks[0]):
            if not get_new_chunk():
                raise ParserError("End of input while scanning for entry body")
            start_idx = 0

        # Identify paren type
        if text_chunks[-1][start_idx] not in "{(":
            raise ParserError("Entry body not found")
        open_tok = text_chunks[-1][start_idx]
        close_tok = "}" if open_tok == "{" else ")"
        paren_count = 1
        start_idx += 1

        # Search for balanced parens
        while paren_count > 0:
            open_idx = text_chunks[-1].find(open_tok, start_idx)
            close_idx = text_chunks[-1].find(close_tok, start_idx)
            if open_idx >= 0 and close_idx >= 0:
                # Both found
                paren_count += 1 if open_idx < close_idx else -1
                start_idx = 1 + min(open_idx, close_idx)
            elif open_idx >= 0:
                paren_count += 1
                start_idx = 1 + open_idx
            elif close_idx >= 0:
                paren_count -= 1
                start_idx = 1 + close_idx
            else:
                # No more parens left, try in next chunk
                if not get_new_chunk():
                    raise ParserError("End of input while scanning for entry body")
                start_idx = 0
        # Done, return end of entry
        return start_idx

    # Get first chunk
    if not get_new_chunk():
        return

    # Start of top level parsing algorithm
    while True:
        # Scan for '@' token
        if (scan_offs := text_chunks[-1].find("@", scan_offs)) < 0:
            # Keep searching
            if not get_new_chunk():
                # No more text, finalize what we have (an implicit comment) and exit
                yield finalize_block(len(text_chunks[-1]))
                return
            scan_offs = 0
        else:
            # Finalize current block (an implicit comment)
            yield finalize_block(scan_offs)
            # Begin scan for an entry
            # Scan for identifier; initial '@' not included in search
            kind, scan_offs = scan_identifier(scan_offs + 1)
            if kind == "comment":
                # Comments are ignored
                yield finalize_block(scan_offs)
            # Scan for entry body
            scan_offs = scan_balanced(scan_offs)
            # Finalize the entry
            yield finalize_block(scan_offs)


def _parse_entry(entry: str) -> BibtexBlock:
    """Parse text for a single entry into a block object."""
    # Expect no leading whitespace for entries
    if entry == "" or entry[0] != "@":
        return CommentBlock.from_text(entry)

    scan_offs = 1
    match = WHITESPACE_RE.match(entry, scan_offs)
    # This will always match
    scan_offs = match.end()
    match = IDENTIFIER_RE.match(entry, scan_offs)
    if not match:
        return BadBlock.from_text(entry)
    ident = match.group(0).lower()
    try:
        if ident == "comment":
            return CommentBlock.from_text(entry)
        if ident == "preamble":
            return PreambleBlock.from_text(entry)
        if ident == "string":
            return StringBlock.from_text(entry)
        return EntryBlock.from_text(entry)
    except ParserError:
        return BadBlock.from_text(entry)


def parse_lines(lines: Iterable[str]) -> list[BibtexBlock]:
    """Parse lines from a BibTex file into block objects."""
    return [_parse_entry(blk) for blk in _split_entries(lines)]


def parse_string(bibtex_str: str) -> list[BibtexBlock]:
    """Parse a string as BibTex."""
    return parse_lines([bibtex_str])


def parse_file(bibtex_file: Path) -> list[BibtexBlock]:
    """Parse the contents of a BibTex file."""
    with open(bibtex_file, encoding="utf8") as fh:
        return parse_lines([fh.read()])


def gen_lines(blocks: Iterable[BibtexBlock]) -> Iterator[str]:
    for block in blocks:
        yield from block.render()


def write_string(blocks: Iterable[BibtexBlock]) -> str:
    """Write structured reference data as BibTex."""
    return "".join(gen_lines(blocks))


def write_file(blocks: Iterable[BibtexBlock], bibtex_file: Path) -> None:
    """Write structured reference data to BibTex file."""
    with open(bibtex_file, "w", encoding="utf8") as fh:
        fh.writelines(gen_lines(blocks))
