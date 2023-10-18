Note: This is a forked project. The [original project](https://github.com/bsdz/remarkable-layers) is no longer being maintained and is incompatable with the new ReMarkable Cloud API. It also has issues with (my version of) Inkscape.

# remarkable-layers

Python module for reading and writing Remarkable Lines files (*.rm).

Currently only supports version 5 of Remarkable Lines files.

This module and supporting routines are experimental.

## Installation

The install instruction of the original repository can be find over there.
This fork is not available as a Pip package (I have no idea how to do that; someone is welcome to create a package out of this).

### 1. The Python package
You can install this fork by cloning it and using [poetry install](https://python-poetry.org/docs/cli/#install).
```bash
git clone https://github.com/juupje/remarkable-layers.git
cd remarkable-layers
poetry install
```

### 2. rmapi
To upload documents to the reMarkable Cloud, `rmapi` is used. There exists a python implementation [rmapy](https://github.com/subutux/rmapy), but unfortunately it is not compatible with the new reMarkable Sync API (at the time of writing). Therefore, the GO implementation is used. Grab the binary (or build from source) from [https://github.com/juruen/rmapi/releases/tag/v0.0.25](https://github.com/juruen/rmapi/releases/tag/v0.0.25).
Make sure to put the executable in your `PATH`.
You need to run rmapi once to create the device and user token. Run it with:

`./rmapi`

The first time you run it, it will ask you to go to https://my.remarkable.com/ to enter a new activation code. You will see a prompt like this where you just need to introduce the activation code.

`Enter one-time code (go to https://my.remarkable.com):`

After entering the code, it should be connected to your cloud.

### 3. Inkscape plugins
For converting PDFs to RM Lines files (see below), you need Inkscape.
First, install inkscape if you haven't already [https://inkscape.org/](https://inkscape.org/).

This package requires two Inkscape plugins.
1. `Ungroup Deep` should be part of the default Inkscape extensions.
2. `Apply Transform` can be installed by grabbing the `.inx` and `.py` files from [https://github.com/Klowner/inkscape-applytransforms](https://github.com/Klowner/inkscape-applytransforms) and putting them in Inkscape's extensions folder (for linux users, this usually is `~/.config/inkscape/extensions`).

## Core Dependencies

The core module for reading & writing rm line files only uses core python standard library.

The SVG conversion module utilises numpy and lxml.

The scripts introduce other dependencies.

## Usage

Read a RM Lines binary file.

```python
from pathlib import Path
from rmlines import RMLines

p = Path("./samples/03f23a6e-c14b-4dba-836d-828707979356.rm")
rm0 = RMLines.from_bytes(p.open("rb"))
```

Dump internal structure of RM Lines to logger.

```python
import logging
logging.basicConfig(level="DEBUG")
# To reduce logging for some binary types, eg:
# logging.getLogger('rmlines.rmobject.segment').setLevel(logging.INFO)

rm0.dump()
```

Convert to SVG (as strokes).

```python
from io import StringIO
from IPython.display import SVG

sbuffer = StringIO()
rm0.to_svg(sbuffer)
SVG(sbuffer.getvalue().encode())
```

Convert simple SVG file into RM Lines format. Can only contain paths with simple line segments. For conversion of PDF to simple SVG see pdf_converter.py below.

```python
p = Path("./my_simple.svg")
rm0 = RMLines.from_svg(f.open("rb")
```

## Scripts

### pdf_converter
This is the main use-case of this project, I think.
This script converts a pdf to several intermediate SVG files, one per page, then generates RM Lines notebook that is uploaded to Remarkable Cloud. The SVG output files are simplified SVG that the RM Lines module can process. That is all beziers have been linearized and raster images traced into paths. This can be time consuming.

Typical usage:

```bash
pdf_converter my_file.pdf --first 10 --last 25 --upload
```

Use "--help" flag for more options. The most important options are 
- `--nsymb` the number of interpolation points of the bezier curves for symbols (letters, numbers and other glyphs). The default is 3, but can be increased for rounder symbols. This does increase the file size significantly.
- `--n` The number of interpolation points for bezier curves other than those for symbols. The default is equal to `--nsymb`. Increasing it can make vector pictures (like those drawn with `TikZ`) look more true to the original.
- `--upload-dest` Specifies the target folder to which the file is uploaded. Note, this should only be the path, the filename is determined from the input pdf.

You can put this script in your `PATH` to call it directly from the commandline, or in your `PYTHONPATH` to call it as a python module (with `python -m pdf_converter`).

#### Additional Dependencies

Applications: inkscape and pdfinfo. 
Python modules: potrace, svgpathtools, svgwrite, pillow. Note that in pyproject.toml some dependencies reference git branches / revisions and/or forks.

### pen_gallery

This script uses hershey stroke fonts to place text in Remarkable lines files with different pen styles. More fonts are available at [SVG Fonts repo](ttps://gitlab.com/oskay/svg-fonts).

## Known issues

Conversion from RM to SVG is very basic and doesn't support the Remarkable pen tips that are stored as PNG images.

Conversion from PDF to RM shows all text and objects as outlines. This is a bit annoying but a limitation of Remarkable's Lines format that doesn't support fills. I think the reason for Remarkable's decision to not support SVG directly is because they actually map PNG images over the stroke paths and this might be difficult/impossible to do with SVG.

The round trip mapping, eg read RM to binary and convert back to again, does not support all pen configurations.

This has only be tested on Linux and not Windows.

This is a very experimental package created so I can properly take notes and test mathematical texts. I'm still not sure if it will solve my use case.

## Demo

![demo](pdf_to_rm_format.gif "Demo")


