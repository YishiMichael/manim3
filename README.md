# manim3


## Introduction
Manim3 is a personal variant of [manim](https://github.com/3b1b/manim). It aims at standardizing the project structure, improving rendering performance and providing additional features.

Note: This project is still in its primary stage and not stable.


## Installation
Manim3 runs on Python 3.10 and OpenGL 4.3.

You may install manim3 directly via
```sh
pip install manim3
```
to install the latest version distributed on pypi. Or, to catch up with the latest development and edit the source code, you may clone this repository via
```sh
git clone https://github.com/YishiMichael/manim3.git
cd manim3
pip install -e .
```
Through either way most features of manim3 have become available. To enable more functionalities, these are their corresponding soft dependencies:

### Video generation (`write_video` config option)
Install ffmpeg.

### `Tex`, `MathTex`
Install LaTeX.

### `MathJax`
Install node modules by running `npm install` under `manim3/plugins/mathjax`.

### `Text`, `Markup`
Install `manimpango` using pip.

### `Code`
Install `manimpango` using pip. Plus, install sublime (which will generate highlight information), and copy the directory `manim3/plugins/code/export_highlight` under `~/Sublime Text/Packages/`.


## Using manim3

A demo file is provided as `examples.py`. After installing, just run
```sh
py examples.py
```
A window running a scene shall pop up.


## License
MIT license
