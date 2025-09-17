import sam2
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

# define an ontology to map class names to our Grounded SAM 2 prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = GroundedSAM2(
    ontology=CaptionOntology(
        {
            "person": "person",
            "shipping container": "shipping container",
        }
    )
)

# run inference on a single image
results = base_model.predict("context_images/image.png")

vis = plot(
    image=cv2.imread("context_images/image.png"),
    classes=base_model.ontology.classes(),
    detections=results
)
cv2.imwrite("context_images/vis.png", vis)  # 输出可视化结果
# label all images in a folder called `context_images`
base_model.label("./context_images", extension=".png")