import re
import torch
from transformers import StoppingCriteria


BOI_TOKEN = '<|im_gen_start|>'
EOI_TOKEN = '<|im_gen_end|>'
IMG_TOKEN = '<|im_gen_{:04d}|>'
EOS_TOKEN = '<|endoftext|>'
BOV_TOKEN = '<|vision_start|>'
EOV_TOKEN = '<|vision_end|>'
IMG_PAD_TOKEN = '<|image_pad|>'


def remove_mismatched_weights(model, pretrained_state_dict):
    own_state = model.state_dict()
    mismatch_keys = []

    for name in list(pretrained_state_dict.keys()):
        if name not in own_state or own_state[name].shape != pretrained_state_dict[name].shape:
            mismatch_keys.append(name)
            pretrained_state_dict.pop(name)

    return pretrained_state_dict, mismatch_keys


def parse_coordinates_colors(cot_text):
    """
    Parse bounding box coordinates and their colors from the CoT text.

    Args:
        cot_text (str): Chain of Thought text containing bounding box information.

    Returns:
        list: A list of dictionaries with keys 'x1', 'y1', 'x2', 'y2', and 'color'.
    """
    # Regular expression to match bounding box and color patterns
    pattern = r"<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|> \((\w+)\)"

    # Parse all matches
    matches = re.findall(pattern, cot_text)

    # Extract bounding box coordinates and colors
    parsed_data = []
    for match in matches:
        x1, y1, x2, y2, color = match
        parsed_data.append({
            'position': [[int(x1), int(y1)], [int(x2), int(y2)]],
            'color': color
        })

    return parsed_data


class StopOnToken(StoppingCriteria):
    def __init__(self, token_id):
        self.token_id = token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if the last generated token is BOI_TOKEN
        return input_ids[0, -1] == self.token_id
