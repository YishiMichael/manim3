import os
import re


# Automatically generate .glsl files from three/src/renderers/shaders/ShaderChunk, ShaderLib

FOLDER_PATH = "E:\\code\\HTML_projects\\by_others\\three\\src\\renderers\\shaders"


if __name__ == "__main__":
    try:
        os.mkdir("shader_chunk")
    except FileExistsError:
        pass
    for filename in os.listdir(os.path.join(FOLDER_PATH, "ShaderChunk")):
        name = filename.split(".")[0]
        with open(os.path.join(FOLDER_PATH, "ShaderChunk", filename), "r", encoding="utf-8") as input_f:
            contents = re.compile(r"`.*?`", flags=re.MULTILINE | re.DOTALL).findall(input_f.read())
            with open(os.path.join("shader_chunk", f"{name}.glsl"), "w", encoding="utf-8") as output_f:
                output_f.write(contents[0][2:-1])

    try:
        os.mkdir("shader_lib")
    except FileExistsError:
        pass
    for filename in os.listdir(os.path.join(FOLDER_PATH, "ShaderLib")):
        name = filename.split(".")[0]
        with open(os.path.join(FOLDER_PATH, "ShaderLib", filename), "r", encoding="utf-8") as input_f:
            contents = re.compile(r"`.*?`", flags=re.MULTILINE | re.DOTALL).findall(input_f.read())
            with open(os.path.join("shader_lib", f"{name}_vert.glsl"), "w", encoding="utf-8") as output_f:
                output_f.write(contents[0][2:-1])
            with open(os.path.join("shader_lib", f"{name}_frag.glsl"), "w", encoding="utf-8") as output_f:
                output_f.write(contents[1][2:-1])
