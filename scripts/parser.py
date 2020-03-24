import re
import json
from typing import List, Tuple, Dict, Iterable

import attr
import yaml
import mistune

import consts
from model import ModelMeta, ModelContent, Model


section_re = re.compile(r"^\s*<!--\s*section:\s*(.+)\s*-->\s*$")

UNSPECIFIC_SECTION = "unspecific"

markdown_ast = mistune.create_markdown(
    renderer=mistune.AstRenderer(),
    plugins=["strikethrough", "table", "footnotes", "table", "url"],
)

markdown_html = mistune.create_markdown(
    renderer=mistune.HTMLRenderer(),
    plugins=["strikethrough", "table", "footnotes", "table", "url"],
)


def split_by_meta_and_content(lines: List[str]) -> Tuple[List[str], List[str]]:
    meta: List[str] = []
    if lines[0] != "---":
        return meta, lines
    lines.pop(0)

    for line in lines:
        if line in ("---", "..."):
            content_beginning = lines.index(line) + 1
            lines = lines[content_beginning:]
            break
        meta.append(line)
    return meta, lines


def split_sections(lines: List[str]) -> Dict[str, List[str]]:
    current_section_name = UNSPECIFIC_SECTION
    sections: Dict[str, List[str]] = {current_section_name: []}

    for line in lines:
        m = section_re.match(line)
        if m is None:
            sections[current_section_name].append(line)
        else:
            current_section_name = m.group(1).strip()
            if current_section_name in sections:
                raise ValueError(
                    "Duplicated section name: {}".format(current_section_name)
                )
            sections[current_section_name] = []

    return sections


def generate_meta(id: str, meta_lines: List[str]) -> ModelMeta:
    meta_input = yaml.load("\n".join(meta_lines), yaml.SafeLoader)

    if meta_input["template"] != "hub1":
        raise ValueError("unsupported template '{}'".format(meta_input["template"]))

    meta = ModelMeta(
        id=id,
        title=meta_input["title"],
        author=meta_input["author"],
        summary=meta_input["summary"],
        github_link=meta_input["github-link"],
        tags=meta_input["tags"],
    )

    return meta


def generate_content(sections: Dict[str, List[str]]) -> ModelContent:
    for key in sections.keys():
        if key not in [*consts.ALL_LANGUAGES, UNSPECIFIC_SECTION]:
            raise ValueError(
                "unsupported language section {}, please check supported language in consts".format(
                    key
                )
            )

    for key in consts.REQUIRED_LANGUAGES:
        if key not in sections:
            raise ValueError(
                "Missing required language sections, please check required language in consts"
            )

    # Check sample code in unspecific section.
    # Make sure it exist and is the one stuff in it.
    code_section = sections.pop(UNSPECIFIC_SECTION)
    code_ast = markdown_ast("\n".join(code_section))
    code_blocks = list(filter(lambda x: x["type"] != "newline", code_ast))
    if len(code_blocks) != 1 or code_blocks[0]["type"] != "block_code":
        raise ValueError(
            "document must start with sample code, follow by section comments"
        )

    sections_content: Dict[str, str] = {}

    # Use markdown engine process each section, confirm it's valid markdown.
    for key, section in sections.items():
        # Render all section decorate with sample code at beginning.
        sections_content[key] = markdown_html("\n".join(code_section + section))
    return ModelContent(sample_code=code_blocks[0]["text"], full_text=sections_content)


def parse_file(id: str, lines: Iterable[str]):
    # Strip right only to prevent strip indent.
    lines = list(map(str.rstrip, lines))

    meta_lines, content_lines = split_by_meta_and_content(lines)

    meta = generate_meta(id, meta_lines)

    sections = split_sections(content_lines)
    content = generate_content(sections)

    return Model(meta=meta, content=content)


if __name__ == "__main__":
    with open("../megengine_example.md") as f:
        model = parse_file("megengine_example", f)

        # TODO fill update time.
        print(json.dumps(attr.asdict(model)))
