"""Extraction layer: .docx (OOXML) -> pure :class:`Doc` model.

Nothing above this package should import python-docx or lxml; the rules
consume the model only.
"""
from .docx_extractor import (
    build_doc,
    Block,
    Cell,
    CoreProps,
    Doc,
    Field,
    HeaderFooter,
    HeadingEntry,
    Paragraph,
    ParagraphProps,
    ResolvedFont,
    Row,
    Run,
    Section,
    StyleIndex,
    StyleInfo,
    Table,
    TableRef,
)

__all__ = [
    "build_doc",
    "Block",
    "Cell",
    "CoreProps",
    "Doc",
    "Field",
    "HeaderFooter",
    "HeadingEntry",
    "Paragraph",
    "ParagraphProps",
    "ResolvedFont",
    "Row",
    "Run",
    "Section",
    "StyleIndex",
    "StyleInfo",
    "Table",
    "TableRef",
]
