from transformers import AutoProcessor


BOI_TOKEN = '<|im_gen_start|>'
EOI_TOKEN = '<|im_gen_end|>'
IMG_TOKEN = '<|im_gen_{:04d}|>'


def get_processor(model_name, add_gen_token_num=64):
    processor = AutoProcessor.from_pretrained(model_name)
    add_token_list = [BOI_TOKEN, EOI_TOKEN]
    for i in range(add_gen_token_num):
        add_token_list.append(IMG_TOKEN.format(i))
    processor.tokenizer.add_tokens(add_token_list, special_tokens=True)
    return processor
