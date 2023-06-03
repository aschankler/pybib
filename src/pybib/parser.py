"""Parser to convert between block objects and plain text."""

from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Self, Sequence

import attrs


class BibtexBlock(ABC):
    """Block object in a bibtex file."""

    @abstractmethod
    def render(self) -> list[str]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_lines(cls, lines: Iterable[str]) -> Self:
        raise NotImplementedError


def _split_block(lines: Iterable[str]) -> tuple[str, list[str]]:
    lines_iter = iter(lines)
    first_line = next(lines_iter).lstrip()
    assert first_line[0] == "@"
    tokens = first_line.split()
    if tokens[0] == "@":
        raise ValueError
    blk_type = tokens[0][1:]
    if "{" not in blk_type:
        blk_type = blk_type.lower()
        tokens = tokens[1:]
    else:
        i_open = len(blk_type) if "{" not in blk_type else blk_type.index("{")
        i_close = len(blk_type) if "}" not in blk_type else blk_type.index("}")
        if i_open > i_close:
            raise ValueError
        tokens[0] = blk_type[i_open:]
        blk_type = blk_type[:i_open].lower()

    line = " ".join(tokens) if tokens else next(lines_iter).lstrip()
    assert line[0] == "{"
    line = line[1:] if len(line) > 1 else next(lines_iter)
    paren_count = 1
    contents = []
    while paren_count != 0:
        for i, char in enumerate(line):
            if char == "{":
                paren_count += 1
            if char == "}":
                paren_count -= 1
            if paren_count == 0:
                if line[:i]:
                    contents.append(line[:i])
                assert len(line[i:]) == 1
                break
        else:
            contents.append(line)
            try:
                line = next(lines_iter)
            except StopIteration:
                raise ValueError
    return blk_type, contents


@attrs.frozen(order=False)
class EntryBlock(BibtexBlock):
    entry_type: str
    entry_key: str
    fields: Mapping[str, str]

    def render(self) -> list[str]:
        lines = list()
        lines.append(f"@{self.entry_type}{{{self.entry_key},\n")
        lines.extend(
            (f"  {f_key} = {f_val},\n" for f_key, f_val in self.fields.items())
        )
        lines.append("}\n")
        return lines

    @staticmethod
    def split_entry_parts(lines: Iterable[str]) -> list[str]:
        entry_str = " ".join(lines)
        parts = []
        last_split = 0
        paren_count = 0
        for i, char in enumerate(entry_str):
            if char == "{":
                paren_count += 1
            if char == "}":
                paren_count -= 1
            if paren_count == 0 and char == ",":
                parts.append(entry_str[last_split:i].strip())
                last_split = i + 1
        return parts

    @staticmethod
    def split_field(field: str) -> tuple[str, str]:
        assert "=" in field
        field_key, field_val = field.split("=", maxsplit=1)
        return field_key.strip(), field_val.strip()

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> Self:
        kind, contents = _split_block(lines)
        entry_key, *fields = cls.split_entry_parts(contents)
        fields_dict = dict(map(cls.split_field, fields))
        return cls(kind, entry_key, fields_dict)


@attrs.frozen(order=False)
class ExplicitCommentBlock(BibtexBlock):
    lines: Sequence[str]

    def render(self) -> list[str]:
        text = ["@comment{\n"]
        text.extend([f"    {ln}\n" for ln in self.lines])
        text.append("}\n")
        return text

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> Self:
        kind, contents = _split_block(lines)
        assert kind.lower() == "comment"
        return cls(contents)


@attrs.frozen(order=False)
class ImplicitCommentBlock(BibtexBlock):
    lines: Sequence[str]

    def render(self) -> list[str]:
        return [line + "\n" for line in self.lines]

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> Self:
        return cls([ln.strip() for ln in lines])


@attrs.frozen(order=False)
class PreambleBlock(BibtexBlock):
    lines: Sequence[str]

    def render(self) -> list[str]:
        text = ["@preamble{\n"]
        text.extend([f"    {ln}\n" for ln in self.lines])
        text.append("}\n")
        return text

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> Self:
        kind, contents = _split_block(lines)
        assert kind.lower() == "preamble"
        return cls(contents)


@attrs.frozen(order=False)
class BadBlock(BibtexBlock):
    lines: Sequence[str]

    def render(self) -> list[str]:
        return []

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> Self:
        return cls(list(lines))


@attrs.frozen(order=False)
class StringBlock(BibtexBlock):
    lines: Sequence[str]

    def render(self) -> list[str]:
        text = ["@string{\n"]
        text.extend([f"    {ln}\n" for ln in self.lines])
        text.append("}\n")
        return text

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> Self:
        kind, contents = _split_block(lines)
        assert kind.lower() == "string"
        return cls(contents)


def _split_blocks(lines: Iterable[str]) -> Iterator[tuple[str]]:
    """Split a stream of lines into a stream of blocks.

    Implementation can split multiple blocks on one line.
    Uses a state machine implementation.
    """

    class State(Enum):
        TOPLEVEL = auto()
        BLOCKTYPE = auto()
        BLOCKBODY = auto()
        IMPLICIT = auto()

    lines_iter = iter(lines)
    line = ""
    this_block = []
    paren_count = 0
    state = State.TOPLEVEL

    while True:
        if not line:
            try:
                line = next(lines_iter).strip()
            except StopIteration:
                return
            if not line:
                continue

        if state == State.TOPLEVEL:
            if line[0] == "@":
                state = State.BLOCKTYPE
            else:
                state = State.IMPLICIT
        elif state == State.IMPLICIT:
            if "@" in line:
                idx = line.index("@")
                this_block.append(line[:idx])
                line = line[idx:]
                yield tuple(this_block)
                this_block = []
                state = State.TOPLEVEL
            else:
                this_block.append(line)
                line = None
        elif state == State.BLOCKTYPE:
            if "{" in line:
                idx = line.index("{")
                this_block.append(line[: idx + 1])
                line = line[idx + 1 :].lstrip()
                paren_count = 1
                state = State.BLOCKBODY
            else:
                raise RuntimeError
        elif state == State.BLOCKBODY:
            if not any(x in line for x in "{}"):
                this_block.append(line)
                line = None
                continue
            for i, char in enumerate(line):
                if char == "{":
                    paren_count += 1
                if char == "}":
                    paren_count -= 1
                if paren_count == 0:
                    this_block.append(line[: i + 1])
                    line = line[i + 1 :].lstrip()
                    yield tuple(this_block)
                    this_block = []
                    state = State.TOPLEVEL
                    break
            else:
                this_block.append(line)
                line = None
        else:
            raise RuntimeError


def parse_lines(lines: Iterable[str]) -> list[BibtexBlock]:
    """Parse lines from a BibTex file into block objects."""

    def _get_block_type(block_lines: Sequence[str]) -> type[BibtexBlock]:
        if "@" not in block_lines[0]:
            return ImplicitCommentBlock
        tokens = block_lines[0].split()
        if tokens[0] == "@":
            return BadBlock
        blk_type = tokens[0][1:]
        i_open = len(blk_type) if "{" not in blk_type else blk_type.index("{")
        i_close = len(blk_type) if "}" not in blk_type else blk_type.index("}")
        if i_open > i_close:
            return BadBlock
        blk_type = blk_type[:i_open].lower()
        if blk_type == "comment":
            return ExplicitCommentBlock
        if blk_type == "preamble":
            return PreambleBlock
        if blk_type == "string":
            return StringBlock
        return EntryBlock

    return [_get_block_type(blk).from_lines(blk) for blk in _split_blocks(lines)]


def parse_string(bibtex_str: str) -> list[BibtexBlock]:
    """Parse a string as BibTex."""
    return parse_lines(bibtex_str.split("\n"))


def parse_file(bibtex_file: Path) -> list[BibtexBlock]:
    """Parse the contents of a BibTex file."""
    with open(bibtex_file, encoding="utf8") as fh:
        return parse_lines(fh.readlines())


def gen_lines(blocks: Iterable[BibtexBlock]) -> Iterator[str]:
    for block in blocks:
        yield from block.render()


def write_string(blocks: Iterable[BibtexBlock]) -> str:
    """Write structured reference data as BibTex."""
    return "\n".join(gen_lines(blocks))


def write_file(blocks: Iterable[BibtexBlock], bibtex_file: Path) -> None:
    """Write structured reference data to BibTex file."""
    with open(bibtex_file, "w", encoding="utf8") as fh:
        fh.writelines(gen_lines(blocks))
