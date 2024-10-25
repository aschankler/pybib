"""Programmatically modify a BibTex library."""

from abc import ABC
from copy import deepcopy
from typing import Callable, Collection, Sequence

import attrs

from pybib.parser import (
    BadBlock,
    BibtexBlock,
    EntryBlock,
    CommentBlock,
    PreambleBlock,
    StringBlock,
)

# Input transform: (pre parser)
# strip % comments

# Block transforms:
# validators
# compress spaces in entries
# Reformat

# Library transforms:
# dereference strings
# exclude comments
# merge whitespace into surrounding blocks


# noinspection PyMethodMayBeStatic
@attrs.frozen(eq=False)
class BlockTransform(ABC):
    """Update blocks with new data."""

    allow_inplace_modification: bool = attrs.field(default=False, kw_only=True)
    require_copy: bool = attrs.field(default=False, kw_only=True)

    def transform_block(self, block: BibtexBlock) -> BibtexBlock:
        if self.require_copy:
            block = deepcopy(block)

        if isinstance(block, EntryBlock):
            return self.transform_entry(block)
        if isinstance(block, CommentBlock):
            return self.transform_comment(block)
        if isinstance(block, StringBlock):
            return self.transform_string(block)
        if isinstance(block, PreambleBlock):
            return self.transform_preamble(block)
        if isinstance(block, BadBlock):
            return self.transform_bad_block(block)
        raise ValueError(f"Unrecognized type {block.__class__.__name__}")

    def transform_many(self, blocks: Sequence[BibtexBlock]) -> Sequence[BibtexBlock]:
        return [self.transform_block(block) for block in blocks]

    def transform_entry(self, block: EntryBlock) -> BibtexBlock:
        return block

    def transform_string(self, block: StringBlock) -> BibtexBlock:
        return block

    def transform_preamble(self, block: PreambleBlock) -> BibtexBlock:
        return block

    def transform_comment(self, block: CommentBlock) -> BibtexBlock:
        return block

    def transform_bad_block(self, block: BadBlock) -> BibtexBlock:
        return block


@attrs.frozen(eq=False)
class NormalizeKeyCase(BlockTransform):
    """Normalize the case of entry field keys."""

    case: str = attrs.field(
        default="lower",
        kw_only=True,
        validator=attrs.validators.in_(("upper", "lower", "title")),
    )
    _test_fn: Callable[[str], bool] = attrs.field(init=False)
    _convert_fn: Callable[[str], str] = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        if self.case == "upper":
            object.__setattr__(self, "_test_fn", str.isupper)
            object.__setattr__(self, "_convert_fn", str.upper)
        elif self.case == "lower":
            object.__setattr__(self, "_test_fn", str.islower)
            object.__setattr__(self, "_convert_fn", str.lower)
        elif self.case == "title":
            object.__setattr__(self, "_test_fn", str.istitle)
            object.__setattr__(self, "_convert_fn", str.title)

    def transform_entry(self, block: EntryBlock) -> BibtexBlock:
        if all(self._test_fn(f_key) for f_key in block.fields.keys()):
            return block
        if not self.require_copy and not self.allow_inplace_modification:
            block = deepcopy(block)
        block.fields = {
            self._convert_fn(f_key): f_val for f_key, f_val in block.fields.items()
        }
        return block


@attrs.frozen(eq=False)
class StripFields(BlockTransform):
    """Remove unwanted fields from entries."""

    exclude_fields: Collection[str] = attrs.field(
        kw_only=True, factory=tuple, converter=lambda val: tuple(x.lower() for x in val)
    )

    def transform_entry(self, block: EntryBlock) -> BibtexBlock:
        if not any(f_key in block.fields.keys() for f_key in self.exclude_fields):
            # No modification needed
            return block
        if not self.allow_inplace_modification and not self.require_copy:
            block = deepcopy(block)

        block.fields = {
            f_key: f_val
            for f_key, f_val in block.fields.items()
            if f_key.lower() not in self.exclude_fields
        }
        return block
