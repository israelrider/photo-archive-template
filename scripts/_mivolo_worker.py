"""Age 384px face crops with MiVOLO. Runs INSIDE MiVOLO's own venv, never in the main one.

Reads a directory of `<face_id>.png` crops, prints `{"<face_id>": age, ...}` as JSON on
stdout. Nothing else may go to stdout - the caller parses the whole stream.

    <mivolo_python> scripts\\_mivolo_worker.py <crop_dir>

with PYTHONPATH pointing at the MiVOLO clone. See _mivolo.py for the caller and
images-agent\\docs\\AGE_ESTIMATION.md for why this model and not `face.age`.

This script only reads the venv and the model cache: no installs, no downloads - HF_HUB_OFFLINE=1 is set by the caller so a missing
cache fails loudly instead of reaching the network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL = "iitolstykh/mivolo_v2"


def main() -> None:
    crop_dir = Path(sys.argv[1])
    crops = sorted(crop_dir.glob("*.png"))
    if not crops:
        print("{}")
        return

    model = AutoModelForImageClassification.from_pretrained(
        MODEL, trust_remote_code=True).eval()
    proc = AutoImageProcessor.from_pretrained(MODEL, trust_remote_code=True)

    out: dict[str, float] = {}
    for c in crops:
        img = cv2.imread(str(c))
        if img is None:
            continue
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (384, 384),
                         interpolation=cv2.INTER_AREA)
        # body_input is optional - we have face boxes only, which is what the archive's
        # existing ages were produced from.
        faces = proc([img])
        body = proc([None])
        with torch.no_grad():
            r = model(faces_input=faces["pixel_values"],
                      body_input=body.get("pixel_values"))
        out[c.stem] = round(float(r.age_output[0]), 1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
