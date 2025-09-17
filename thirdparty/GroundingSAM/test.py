import sam2
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# define an ontology to map class names to our Grounded SAM 2 prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = GroundedSAM2(
    ontology=CaptionOntology(
        {
            "person": "person",
            # "shipping container": "shipping container",
        }
    )
)

here = Path(__file__).parent
img_path = here / "context_images" / "image.png"
# run inference on a single image
results = base_model.predict(str(img_path))

vis = plot(
    image=cv2.imread(str(img_path)),
    classes=base_model.ontology.classes(),
    detections=results
)
out_dir = here / "context_images"
out_dir.mkdir(parents=True, exist_ok=True)
if vis is not None:
    cv2.imwrite(str(out_dir / "vis.png"), vis)

# save masks to files
masks = None
if hasattr(results, "mask"):
    masks = results.mask
elif hasattr(results, "masks"):
    masks = results.masks

if masks is not None:
    if isinstance(masks, np.ndarray) and masks.ndim == 3:
        mask_list = [masks[i] for i in range(masks.shape[0])]
    elif isinstance(masks, list):
        mask_list = masks
    else:
        mask_list = []
    for i, m in enumerate(mask_list):
        m_uint8 = (m.astype(np.uint8) * 255)
        Image.fromarray(m_uint8).save(str(out_dir / f"mask_{i}.png"))

# label all images in a folder called `context_images`
base_model.label("./context_images", extension=".png")