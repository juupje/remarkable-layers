import subprocess
from pathlib import Path
import logging
from argparse import ArgumentParser
from io import BytesIO
import base64
from copy import copy

from lxml import etree as ET
from svgpathtools import parse_path, Path as SVGPath, Line, CubicBezier
from svgpathtools.parser import parse_transform
from svgpathtools.path import transform
import numpy as np
from PIL import Image
import svgwrite
import potrace

from rmlines import RMLines, Layer, Stroke, Segment, Colour, Pen, Width, X_MAX, Y_MAX
from rmlines.svg import apply_transform
from rmlines.rm_io import save_rm_doc, upload_rm_doc

logger = logging.getLogger(__name__)

XML_PARSER = ET.XMLParser(huge_tree=True)


def pdf_info(filename):
    run = subprocess.run(["pdfinfo"] + [filename], capture_output=True,)
    return {
        line[0 : line.find(":")].strip(): line[line.find(":") + 1 :].strip()
        for line in run.stdout.decode("utf8").splitlines()
    }

def run_inkscape(filename, args=[], actions=[]):
    print(" ".join(["inkscape"]
        + [str(x) for x in args]
        + (["--actions=\"%s\"" % "; ".join(actions)] if actions else [])
        + [str(filename)]))
    run = subprocess.run(
        ["inkscape"]
        + args
        + (["--actions=%s" % "; ".join(actions)] if actions else [])
        + [filename],
        capture_output=True,
    )
    logger.info(run.stderr.decode("ascii"))


def resize_doc(stage_svg,to_file=None):
    if to_file is None:
        to_file = stage_svg #overwrite
    # resize to remarkable size
    root = ET.fromstring(stage_svg.read_bytes(), parser=XML_PARSER)
    x_min, y_min, x_max, y_max = [float(s) for s in root.attrib["viewBox"].split(" ")]

    X_MAX, Y_MAX = 1404.0, 1872.0
    svg_ratio = x_max / y_max
    rm_ratio = X_MAX / Y_MAX
    if svg_ratio > rm_ratio:
        # fit width
        factor = X_MAX / x_max
    else:
        # fit height
        factor = Y_MAX / y_max

    x_max_new, y_max_new = factor * x_max, factor * y_max

    # appending node to another group moves it (lxml)
    group = ET.Element("g", transform=f"scale({factor:0.2f})")
    for child in root:
        if "sodipodi" in child.tag: continue
        group.append(child)
    root.append(group)

    root.attrib["width"] = f"{x_max_new:.2f}pt"
    root.attrib["height"] = f"{y_max_new:.2f}pt"
    root.attrib["viewBox"] = f"0 0 {x_max_new:.2f} {y_max_new:.2f}"

    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t", level=0)
    tree.write(to_file, encoding="utf-8")
    #to_file.write_bytes(ET.tostring(root, pretty_print=True))
    return to_file


def trace_image(data, transform):
    im1 = Image.open(BytesIO(base64.b64decode(data)))

    # convert to b&w
    im2 = im1.convert("L").point(lambda x: 255 if x > 254 else 0, mode="1")

    bmp = potrace.Bitmap(np.array(im2))
    path = bmp.trace()

    dwg = svgwrite.Drawing(viewBox=(f"0 0 {im1.width} {im1.height}"))

    # TODO: as multiple svg paths? along these lines..
    # for curve in path:
    #     elements = ["M", *curve.start_point]
    #     for p in curve.tesselate():
    #         elements.extend(["L", *p])
    #     dwg.add(dwg.path(d=elements, stroke='black', stroke_width='1', fill='white'))

    # as single path
    elements = []
    for curve in path:
        elements.extend(
            ["M", *apply_transform(transform, np.array([curve.start_point]))[0]]
        )
        for p in apply_transform(transform, curve.tesselate()):
            elements.extend(["L", *p])
    if elements:
        return dwg.path(
            d=elements, stroke="black", stroke_width="1", fill="white"
        ).tostring()


def prepare_images(stage_svg, to_file=None):
    to_file = to_file or stage_svg
    """Traces bitmaps or removes them."""
    root = ET.fromstring(stage_svg.read_bytes(), parser=XML_PARSER)
    defs = root.find(".//defs", root.nsmap)

    # remove masks (appear to be redundant)
    for mask in defs.findall(".//mask", root.nsmap):
        mask_image_id = mask[0].attrib["{http://www.w3.org/1999/xlink}href"]
        mask_image_id = mask_image_id.replace("#", "")
        defs.remove(mask)
        mask_image = defs.find(f".//image[@id='{mask_image_id}']", root.nsmap)
        defs.remove(mask_image)

    # move images from defs into use (assume no duplicates)
    for image in defs.findall(".//image", root.nsmap):
        use = root.find(f".//use[@xlink:href='#{image.attrib['id']}']", root.nsmap)
        image_data = image.attrib["{http://www.w3.org/1999/xlink}href"]
        defs.remove(image)
        # image data should be base64 incoded image and start with
        # "data:image/(png|jpeg|..);base64,..." so take everything
        # after first comma.
        char_start = image_data.find(",") + 1
        image_data = image_data[char_start:]
        trans_matrix = parse_transform(use.attrib["transform"])
        traced_image_path = trace_image(image_data, trans_matrix)
        if traced_image_path:
            svg_path = ET.XML(traced_image_path)
            use.getparent().replace(use, svg_path)
        else:
            # TODO: failed to trace, delete instead
            use.getparent().remove(use)

    # TODO: replace clipped paths with rectangles for now
    for clip_path in defs.findall("./clipPath", root.nsmap):
        clip_path_id = clip_path.attrib["id"]
        clipping_path = clip_path.find("./path", root.nsmap)
        for g in root.findall(f".//g[@clip-path='url(#{clip_path_id})']", root.nsmap):
            g.getparent().replace(g, copy(clipping_path))
        defs.remove(clip_path)

    to_file.write_bytes(ET.tostring(root))
    return to_file

CUBIC_TO_POLY = np.array(
    [
        [-1, 3, -3, 1],  # transforms cubic bez to standard poly
        [3, -6, 3, 0],
        [-3, 3, 0, 0],
        [1, 0, 0, 0],
    ]
)

CUBIC_SAMPLE_SPACE = lambda n: np.linspace(0, 1, n)
CUBIC_TO_POLY_SAMPLE = lambda n: np.dot(
    CUBIC_TO_POLY, np.power(CUBIC_SAMPLE_SPACE(n), [[3], [2], [1], [0]])
)

CACHE = {3: CUBIC_TO_POLY_SAMPLE(3)}

def flatten_beziers(svg_d, n=3):
    sp = parse_path(svg_d)
    spn = SVGPath()

    for seg in sp:
        if isinstance(seg, Line):
            spn.append(seg)
        elif isinstance(seg, CubicBezier):
            B = [seg.bpoints()]
            try:
                sample_mat = CACHE[n]
            except KeyError:
                sample_mat = CUBIC_TO_POLY_SAMPLE(n)
                CACHE[n] = sample_mat
            foo = np.dot(B, sample_mat)
            spn.extend([Line(x, y) for x, y in zip(foo[0, :-1], foo[0, 1:])])
        else:
            raise RuntimeError(f"unsupported {seg}")

    return spn.d()


def transform_to_line_segments(stage_svg, to_file=None, n_symbol=3, n_other=None):
    n_other = n_other or n_symbol
    to_file = to_file or stage_svg
    """Converts bezier curves into straight line segments."""
    root = ET.fromstring(stage_svg.read_bytes(), parser=XML_PARSER)
    symbol_kids = {c for p in root.iter() for c in p if ("id" in p.attrib and p.attrib["id"].startswith("glyph"))}
    print(len(symbol_kids))
    for path in root.findall(".//path", root.nsmap):
        if "d" in path.attrib and path.attrib["d"]:
            if(path in symbol_kids):
                path.attrib["d"] = flatten_beziers(path.attrib["d"], n=n_symbol)
            else:
                path.attrib["d"] = flatten_beziers(path.attrib["d"], n=n_other)

    to_file.write_bytes(ET.tostring(root))
    return to_file

def transform_paths(stage_svg, to_file=None):
    to_file = to_file or stage_svg
    """Inkscapes deep ungroup doesn't handle paths with inline matrix
    transforms well. Transform them here instead"""
    root = ET.fromstring(stage_svg.read_bytes(), parser=XML_PARSER)
    for path in root.findall(".//path[@transform]", root.nsmap):
        trans_matrix = parse_transform(path.attrib.pop("transform"))
        svg_path = parse_path(path.attrib["d"])
        path.attrib["d"] = transform(svg_path, trans_matrix).d()

    to_file.write_bytes(ET.tostring(root))
    return to_file

def svgpathtools_flatten(stage_svg):
    # TODO: perhaps use this instead of inkscapes's deep ungroup?
    from svgpathtools import Document

    doc = Document(str(stage_svg))
    results = doc.flatten_all_paths()
    for result in results:
        # TODO: save result.path to new SVG document
        # and overwrite stage_svg?
        pass


def remove_groups(stage_svg, to_file=None):
    to_file = to_file or stage_svg
    """Some empty groups left behind. Clean them up."""
    # TODO: assert groups are actually empty
    root = ET.fromstring(stage_svg.read_bytes(), parser=XML_PARSER)
    i = 0
    for idx, elem in enumerate(list(root.iter())):
        tag = elem.tag[elem.tag.rindex("}")+1:]
        if(tag == "g"):
            path = elem.find("path", root.nsmap)
            root.insert(idx,path)
            root.remove(elem)

    to_file.write_bytes(ET.tostring(root))
    return to_file

def extract_svg_page(orig_pdf:Path, page_no:int, out_dir:Path, overwrite:bool, transform:dict={}):
    # NOTE: stage_svg is passed by reference and
    # mutated within calling functions.
    #
    stage_svg = out_dir / f"page_{page_no}.svg"

    if not overwrite and stage_svg.exists():
        logger.warning(f"SVG {stage_svg} already exists, skipping...")
        return stage_svg

    logger.info("Extracting to SVG page %s", page_no)

    # use inkscape to load in page using popplet/cairo import
    # TODO: use pdftocairo directly?
    #
    run_inkscape(
        orig_pdf,
        ["--pdf-poppler", f"--pdf-page={page_no}", f"--export-filename={stage_svg}"],
    )

    # capture document size and resize to width or height accordingly
    #
    stage_svg = resize_doc(stage_svg)
    
    # flatten all cubic bezier paths into lines
    #
    stage_svg = transform_to_line_segments(stage_svg, **transform)

    # remove image masks and prepare for bitmap tracing
    #
    stage_svg = prepare_images(stage_svg)

    # transform paths with in line matrix transforms
    # (inkscape ungroup doesn't work well with these)
    #
    stage_svg = transform_paths(stage_svg)

    # inkscape tasks
    # TODO: optimize by rewriting as pure xml transforms
    # TODO: potentially use svgpathtools new flatten_all_paths() function?
    # * un group everything (extensions / arrange / deep ungroup)
    # * select all (path / objects to path)
    # * file / clean up document
    # * apply path (extension / modify path / apply transform) - https://stackoverflow.com/questions/13329125/removing-transforms-in-svg-files
    # * save as plain SVG
    #
    
    run_inkscape(
        str(stage_svg),
        ["--batch-process"],
        [
            "mcepl.ungroup_deep.noprefs",
            "select-all",
            "object-to-path",
            "select-clear",
            "vacuum-defs",
            "com.klowner.filter.apply_transform",
            "FileSave",
            "file-close",
        ],
    )

    # remove remaining unecessary groups
    # and overwrite the original file
    stage_svg = remove_groups(stage_svg)
    return stage_svg


def generate_template_layers(vertical_lines=True, horizontal_lines=True):
    """Creates two layers for note taking with optional grid lines."""

    layers = []
    for name, x_min, y_min, x_max, y_max in [
        ("Top Notes", 0, 0, X_MAX, Y_MAX / 2),
        ("Bot Notes", 0, Y_MAX / 2, X_MAX, Y_MAX),
    ]:
        layer = Layer(name)

        # background
        for y in range(int(y_min), int(y_max), 10):
            st = Stroke(Pen.MARKER_2, Colour.WHITE, Width.LARGE)
            st.extend([Segment(x_min, y, width=20), Segment(x_max, y, width=20)])
            layer.append(st)

        # horiz lines
        if horizontal_lines:
            for y in range(int(y_min), int(y_max), 50):
                st = Stroke(Pen.FINELINER_2, Colour.GREY, Width.SMALL)
                st.extend([Segment(x_min, y, width=1), Segment(x_max, y, width=1)])
                layer.append(st)

        # vert lines
        if vertical_lines:
            for x in range(int(x_min), int(x_max), 50):
                st = Stroke(Pen.FINELINER_2, Colour.GREY, Width.SMALL)
                st.extend([Segment(x, y_min, width=1), Segment(x, y_max, width=1)])
                layer.append(st)

        layers.append(layer)

    return layers


def generate_rmlines(svgs, exclude_grid_layers=False):
    base_layers = [] if exclude_grid_layers else generate_template_layers()

    rms = []
    for f in svgs:
        logger.info("Creating Rm Lines file for %s", f)
        rm = RMLines.from_svg(f.open("rb"))
        print(str(rm))
        rm.objects[0].name = "Doc"
        rm.objects = base_layers + rm.objects
        rms.append(rm)
    return rms


def main():
    logging.basicConfig(level="INFO")

    parser = ArgumentParser("Convert PDF into simple SVG.")
    parser.add_argument(
        "pdf_input", metavar="pdf_path", type=str, help="Path to file for conversion"
    )
    parser.add_argument(
        "--first", dest="first", type=int, default=1, help="Start from this page number"
    )
    parser.add_argument(
        "--last", dest="last", type=int, default=None, help="Finish on this page number"
    )
    parser.add_argument("--nsymb", help="Number of interpolation points of bezier curves in symbols/glyphs",
                        type=int, default=5)
    parser.add_argument("--n", help="Number of interpolation points of bezier curves other than symbols/glyphs",
                        type=int, default=None)
    parser.add_argument(
        "--overwrite-svg",
        dest="overwrite_svg",
        action='store_true',
        help="Write over intermediate SVG files",
    )
    parser.add_argument(
        "--exclude-grid-layers",
        dest="exclude_grid_layers",
        action='store_true',
        help="Exclude note taking grid layers",
    )
    parser.add_argument(
        "--upload",
        dest="upload",
        action='store_true',
        help="Upload document to Remarkable Cloud",
    )
    parser.add_argument(
        "--upload-dest",
        dest="upload_dest",
        type=str,
        help="Destination folder of the uploaded file. Implies --upload.",
    )
    args = parser.parse_args()
    if(args.upload_dest):
        args.upload = True

    pdf_input = Path(args.pdf_input)
    if not pdf_input.exists():
        raise RuntimeError(f"PDF file '{pdf_input}' does not exist")
    out_dir = Path(pdf_input).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing PDF file '%s'", pdf_input)

    if not args.last:
        info = pdf_info(pdf_input)
        args.last = int(info["Pages"])

    svgs = []
    for pn in range(args.first, args.last + 1):
        svgs.append(extract_svg_page(
            orig_pdf=pdf_input,
            page_no=pn,
            out_dir=out_dir,
            overwrite=args.overwrite_svg,
            transform=dict(n_symbol = args.nsymb, n_other = args.n)
        ))
    rms = generate_rmlines(svgs, exclude_grid_layers=args.exclude_grid_layers)
    if args.upload:
        upload_rm_doc(pdf_input.stem+"_rm.zip", rms, dest_path=args.upload_dest)
    else:
        print("Saving as", pdf_input.stem+"_rm")
        save_rm_doc(pdf_input.stem+"_rm.zip", rms)

if __name__ == "__main__":
    main()
