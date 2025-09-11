import torch
from omegaconf import DictConfig
import hydra
from peft import (
    LoraConfig,
    PeftModel,
    LoraModel,
    PeftModelForCausalLM,
    get_peft_model,
)


def get_peft_model_without_resize_embedding(model, peft_config=None, torch_dtype='bf16'):
    if torch_dtype == 'bf16' or torch_dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif torch_dtype == 'fp16' or torch_dtype == 'float16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if isinstance(model, DictConfig):
        model = hydra.utils.instantiate(model, torch_dtype=torch_dtype)

    print('peft config: ', peft_config)
    if isinstance(peft_config, DictConfig):
        peft_config = hydra.utils.instantiate(peft_config)
    peft_model = get_peft_model(model=model, peft_config=peft_config)

    # peft_model.print_trainable_parameters()

    return peft_model
