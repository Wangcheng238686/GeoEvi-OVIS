"""S1.1 official demo: single-image text-prompt concept segmentation on GPU.

Completion evidence for P-1 S1.1 (implementation-v2): loads the official
checkpoint from $SAM3_CKPT (offline), runs one text prompt on one bundled
image, saves a mask overlay PNG and prints the output contract summary.

Run:  source sam3_ovis/scripts/load_environment.sh
      "$PYTHON" sam3_ovis/inference/demo_text.py [--prompt airplane]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--prompt", default="airplane")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from PIL import Image

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    # Official usage (examples/sam3_image_predictor_example.ipynb): run the
    # whole session under cuda bf16 autocast; the ViTDet FFN's fused
    # addmm_act emits bf16 unconditionally, so fp32-only inference breaks.
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    repo = Path(os.environ["SAM3_REPO"])
    image_path = args.image or str(
        sorted((repo / "assets" / "images").glob("*"))[0])
    out_path = args.out or os.path.join(
        os.environ.get("SAM3_OVIS_LOG_ROOT", "/tmp"),
        f"demo_text_{args.prompt.replace(' ', '_')}.png")

    model = build_sam3_image_model(
        checkpoint_path=os.environ["SAM3_CKPT"], load_from_HF=False)
    processor = Sam3Processor(model)

    im = Image.open(image_path).convert("RGB")
    # Pass the PIL object: set_image derives H/W via shape[-2:], which
    # mis-reads HWC numpy arrays (treats W,C as H,W).
    state = processor.set_image(im)
    state = processor.set_text_prompt(state=state, prompt=args.prompt)

    masks = state["masks"]  # [N, 1, H, W] bool
    scores = state["scores"]
    n = int(masks.shape[0])
    print(f"image      : {image_path}")
    print(f"prompt     : {args.prompt!r}")
    print(f"instances  : {n}")
    if n:
        print(f"scores     : {[round(float(s), 4) for s in scores[:8]]}")
        print(f"mask dtype : {masks.dtype}, shape {tuple(masks.shape)}")

        overlay = np.array(im).copy()
        colors = [(255, 64, 64), (64, 200, 64), (64, 128, 255), (255, 208, 0)]
        for i in range(min(n, len(colors))):
            m = masks[i, 0].cpu().numpy()
            overlay[m] = (0.5 * overlay[m] + 0.5 * np.array(colors[i])).astype(np.uint8)
        Image.fromarray(overlay).save(out_path)
        print(f"overlay    : {out_path}")

    print("S1.1 demo PASSED" if isinstance(n, int) else "demo failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
