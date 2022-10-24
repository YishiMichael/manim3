from __future__ import annotations

import os
import re


def generate_current_condition(conditions_stack: list[list[str | None]]) -> str:
    conditions = []
    for stack in conditions_stack:
        conditions.extend(
            f"! ( {condition} )"
            for condition in stack[:-1]
        )
        last_condition = stack[-1]
        if last_condition is not None:
            conditions.append(f"( {last_condition} )")
    return " && ".join(conditions)


def get_matched_results(full_content: str) -> list[tuple[str, str]]:
    pattern = re.compile("".join((
        r"\s*",
        "(",
        "|".join((
            "#if",
            "#ifdef",
            "#ifndef",
            "#elif",
            "#else",
            "#endif",
            "uniform",
            "varying"
        )),
        ")",
        r"\b",
        r"(.*)"
    )))

    result: list[tuple[str, str]] = []
    conditions_stack: list[list[str | None]] = []
    for line in full_content.split("\n"):
        match_obj = pattern.fullmatch(line)
        if match_obj is None:
            continue
        s = match_obj.group(1)
        if s == "#if":
            condition = match_obj.group(2).strip()
            conditions_stack.append([condition])
        elif s == "#ifdef":
            condition = f"defined( {match_obj.group(2).strip()} )"
            conditions_stack.append([condition])
        elif s == "#ifndef":
            condition = f"! defined( {match_obj.group(2).strip()} )"
            conditions_stack.append([condition])
        elif s == "#elif":
            condition = match_obj.group(2).strip()
            conditions_stack[-1].append(condition)
        elif s == "#else":
            condition = None
            conditions_stack[-1].append(condition)
        elif s == "#endif":
            conditions_stack.pop()
        else:
            condition = generate_current_condition(conditions_stack)
            result.append((line.strip(), condition))
    assert not conditions_stack
    return result


if __name__ == "__main__":
    shader_chunks = {}
    for filename in os.listdir("shader_chunk"):
        with open(os.path.join("shader_chunk", filename), "r", encoding="utf-8") as input_f:
            name = filename.split(".")[0]
            shader_chunks[name] = input_f.read()

    with open(f"shader_variables.glsl", "w") as output_f:
        for filename in os.listdir("shader_lib"):
            output_f.write(f"//// {filename} ////\n\n")
            with open(os.path.join("shader_lib", filename), "r", encoding="utf-8") as input_f:
                full_content = re.sub(
                    r"^\s*#include\s*<([\w./]+?)>",
                    lambda match_obj: shader_chunks[match_obj.group(1)],
                    input_f.read(),
                    flags=re.MULTILINE
                )
                for line, condition in get_matched_results(full_content):
                    if not condition:
                        output_f.write(f"{line}\n")
                    else:
                        output_f.write(f"#if {condition}\n")
                        output_f.write(f"\t{line}\n")
                        output_f.write(f"#endif\n")
            output_f.write("\n\n")
