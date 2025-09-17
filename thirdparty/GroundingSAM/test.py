from pathlib import Path
import cv2
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology
from autodistill.utils import plot

here = Path(__file__).parent
img_path = here / "context_images" / "image.png"  # 确保这张图存在
assert img_path.exists(), f"not found: {img_path}"

base_model = GroundedSAM2(
    ontology=CaptionOntology({"person": "person", "shipping container": "shipping container"})
)

results = base_model.predict(str(img_path))

img = cv2.imread(str(img_path))
assert img is not None, "cv2.imread failed"

vis = plot(image=img, classes=base_model.ontology.classes(), detections=results)

out_dir = here / "context_images"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "vis.png"

if vis is None:
    print("No detections to visualize.")
else:
    cv2.imwrite(str(out_path), vis)
    print("Saved:", out_path)