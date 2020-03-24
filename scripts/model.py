from datetime import datetime
from typing import List, Tuple, Dict, Any, Iterable, Optional

import attr

import consts
import tags


@attr.s()
class ModelMeta:
    id = attr.ib(validator=attr.validators.instance_of(str))
    title = attr.ib(validator=attr.validators.instance_of(str))
    author = attr.ib(validator=attr.validators.instance_of(str))
    summary = attr.ib()
    github_link = attr.ib(
        validator=attr.validators.instance_of(str)
    )  # TODO  change to URL validator
    tags = attr.ib()
    update_time = attr.ib(
        default=None,
        init=False,
        validator=attr.validators.optional(attr.validators.instance_of(datetime)),
    )

    @summary.validator
    def summary_validtor(self, attribute, value):
        if not isinstance(value, dict):
            raise ValueError(
                "summary must be a mapping from language to text, not {}".format(
                    type(value)
                )
            )

        for k, v in value.items():
            if k not in consts.ALL_LANGUAGES:
                raise ValueError("summary key must be a supported language")
            if not isinstance(v, str):
                raise ValueError("summary value must be text")

    @tags.validator
    def tags_validtor(self, attribute, value):
        if not isinstance(value, list):
            raise ValueError("tags must be a list")

        supported_tags = set(map(lambda t: t.id, tags.TAGS))
        for t in value:
            if t not in supported_tags:
                raise ValueError(
                    "tag '{}' is not in tag list, please update list in tags.py".format(
                        t
                    )
                )

    def to_jsondict(self):
        return {
            "id": self.id,
            "title": self.title,
            "author": self.author,
            "summary": self.summary,
            "githubLink": self.github_link,
            "tags": self.tags,
            "updateTime": self.update_time,
        }


@attr.s(auto_attribs=True)
class ModelContent:
    sample_code: str
    full_text: Dict[str, str]

    def to_jsondict(self):
        return {
            "sampleCode": self.sample_code,
            "fullText": self.full_text,
        }


@attr.s(auto_attribs=True)
class Model:
    meta: ModelMeta
    content: ModelContent

    def to_jsondict(self):
        return {"meta": self.meta.to_jsondict(), "content": self.content.to_jsondict()}
