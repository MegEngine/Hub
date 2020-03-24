from typing import Dict

from attr import attrs

__all__ = ["TAGS"]


@attrs(auto_attribs=True)
class Tag:
    id: str
    name: Dict[str, str]  # Mapping from language to text


TAGS = [
    Tag(id="vision", name={"zh_CN": "视觉", "en_US": "Vision"}),
    Tag(id="detection", name={"zh_CN": "识别", "en_US": "Detection"}),
    Tag(id="classification", name={"zh_CN": "分类", "en_US": "Detection"}),
    Tag(id="nlp", name={"zh_CN": "自然语言处理", "en_US": "NLP"}),
]
