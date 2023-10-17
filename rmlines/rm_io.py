from itertools import count
from pathlib import Path
from io import BytesIO
from uuid import uuid4

from rmapy.document import ZipDocument, RmPage

def create_rm_doc(name, rms):
    empty_jpg = Path(__file__).parent / "empty.jpg"
    empty_jpg_bytes = empty_jpg.read_bytes()

    rmps = []
    for rm in rms:
        layer_counter = count(1)
        buffer = BytesIO()
        rm.to_bytes(buffer)
        buffer.seek(0)

        uuid = str(uuid4())
        rmp = RmPage(
            buffer,
            metadata={
                "layers": [{"name": layer.name if layer.name else f"Layer {next(layer_counter)}"} for layer in rm.objects]
            },
            thumbnail=BytesIO(empty_jpg_bytes),
            order=uuid,
        )
        rmps.append(rmp)

    zd = ZipDocument()
    zd.content["fileType"] = "notebook"
    zd.content["pages"] = [rmp.order for rmp in rmps]
    zd.content["pageCount"] = len(rmps)
    zd.metadata["VissibleName"] = name
    zd.pagedata = "\n".join(["Blank"] * len(rmps))
    zd.rm.extend(rmps)
    return zd

def upload_rm_doc(name, rms, dest_path=None):
    dest_path = dest_path or name
    import subprocess
    save_rm_doc(name, rms)
    #Upload using rmapi
    run = subprocess.run(["rmapi", "put", f"{name}", f"{dest_path}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("UPLOAD:",run.stdout.decode("utf8"))
    error = run.stderr.decode("utf8")
    if(len(error.strip())>0):
        print(error)

def save_rm_doc(name, rms):
    zd = create_rm_doc(name, rms)
    #Save it to a file
    with open(name,mode='wb') as f:
        zd.dump(f)
