import os
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms
from transformers import StoppingCriteriaList
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
from .utils import (
    IMG_TOKEN, BOI_TOKEN, EOI_TOKEN, EOS_TOKEN, BOV_TOKEN, EOV_TOKEN, IMG_PAD_TOKEN,
    parse_coordinates_colors, StopOnToken
)


class GenCot(nn.Module):
    def __init__(self, mllm, output_projector, output_projector_add, scheduler, vae, unet, processor,
                 num_img_out_tokens=64, img_gen_start_id=151667, box_start_id=151648, box_end_id=151649) -> None:
        super().__init__()
        self.mllm = mllm  # qwen25-vl model
        self.output_projector = output_projector
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        self.output_projector_add = output_projector_add

        # uses an additional image for conditioning.
        # it uses 12 channels (instead of 4) in the first (conv) layer of the UNet.
        in_channels = 12
        self.unet.register_to_config(in_channels=in_channels)

        with torch.no_grad():
            conv = torch.nn.Conv2d(in_channels, self.unet.conv_in.out_channels, self.unet.conv_in.kernel_size,
                                   self.unet.conv_in.stride, self.unet.conv_in.padding)
            conv.weight.zero_()
            conv.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
            self.unet.conv_in = conv
        self.vae.requires_grad_(False)
        self.vae_batch = 1

        if is_xformers_available():
            import xformers
            unet.enable_xformers_memory_efficient_attention()

        self.img_gen_start_id = img_gen_start_id
        self.num_img_out_tokens = num_img_out_tokens
        self.box_start_id = box_start_id
        self.box_end_id = box_end_id
        self.diffusion_transform = None
        self.source_transform = None
        self.processor = processor

    def _get_add_time_ids(
            self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
                self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    @torch.no_grad()
    def generate(self,
                 text_input,
                 image=None,
                 max_new_tokens=1024,
                 num_inference_steps=50,
                 guidance_scale=7.5,
                 image_guidance_scale=1.0,
                 cond_image_guidance_scale=4.0,
                 height=1024,
                 width=1024,
                 input_token_num=256,
                 do_classifier_free_guidance=True,
                 crops_coords_top_left=(0, 0),
                 prompt_type='t2i',
                 random_seed=42,
                 got_input=None,
                 only_return_got=False,
                 cond_save_dir=None,
                 **generate_kwargs
                 ):
        """
        Generate text and optional images from the model.

        Args:
            text_input (str): The input text prompt.
            image (PIL.Image.Image, optional): A single image for Qwen2.5-VL context or editing.
            max_new_tokens (int): Maximum number of tokens to generate.
            num_inference_steps (int): Diffusion steps for stable diffusion.
            guidance_scale (float): CFG scale for stable diffusion.
            image_guidance_scale (float): Image guidance scale for stable diffusion.
            cond_image_guidance_scale (float): Conditional image guidance scale for stable diffusion.
            height (int): Height of the output image.
            width (int): Width of the output image.
            input_token_num (int): Number of image tokens in the input.
            do_classifier_free_guidance (bool): Whether to use classifier-free guidance during inference.
            crops_coords_top_left (Tuple[int, int]): The top-left coordinates of the crops.
            prompt_type (str): The prompt type to use.
            random_seed (int): Random seed for torch.random.
            got_input (Str): The customize got content. For interactive generation only.
            only_return_got (bool): Whether to return the got text for interactive generation.
            generate_kwargs: Additional kwargs for self.mllm.generate().

        Returns:
            A dict with:
                'text': str, the generated text.
                'images': List[PIL.Image.Image], the generated images if any.
        """
        device = next(self.parameters()).device
        vae_dtype = next(self.vae.parameters()).dtype

        if self.diffusion_transform is None:
            self.diffusion_transform = transforms.Compose([
                transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        if self.source_transform is None:
            self.source_transform = transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC)

        # Generate image tokens
        img_token_ids = [self.processor.tokenizer.convert_tokens_to_ids(IMG_TOKEN.format(i)) for i in
                         range(self.num_img_out_tokens)]
        img_token_ids = torch.tensor(img_token_ids, device=device).unsqueeze(0)  # [1, num_img_out_tokens]

        # input image tokens
        input_token_ids = [self.processor.tokenizer.convert_tokens_to_ids(IMG_PAD_TOKEN) for _ in
                           range(input_token_num)]
        input_token_ids = torch.tensor(input_token_ids, device=device).unsqueeze(0)  # [1, num_img_out_tokens]

        # Convert BOI_TOKEN to ID
        boi_token_id = self.processor.tokenizer.convert_tokens_to_ids(BOI_TOKEN)
        eos_token_id = self.processor.tokenizer.convert_tokens_to_ids(EOS_TOKEN)
        bov_token_id = self.processor.tokenizer.convert_tokens_to_ids(BOV_TOKEN)

        # Define stopping criteria to stop at BOI_TOKEN
        stopping_criteria = StoppingCriteriaList([
            StopOnToken(boi_token_id), StopOnToken(bov_token_id), StopOnToken(eos_token_id)
        ])
        ori_w, ori_h = image.size if image is not None else (width, height)
        input_images = [self.source_transform(image)] if image is not None else []
        original_images = [image] if image is not None else []
        generated_images = []
        output_text = ''

        if prompt_type == 't2i':
            prompt = f"Follow the caption to generate an image through a chain of thought process: {text_input}"
        elif prompt_type == 'edit':
            prompt = f"Follow the instruction to edit the given image through a chain of thought process: {text_input}"
        else:
            raise ValueError(f"Unknown prompt type {prompt_type}")

        # Prepare the conversation structure for Qwen2.5-VL
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        # If image is provided, add it to messages
        if image is not None:
            # Insert the image into the content
            messages[0]["content"].insert(0, {"type": "image"})

        # Apply chat template to form the prompt as Qwen2.5-VL expects
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=None if not input_images else input_images,
            padding=False,
            return_tensors="pt"
        ).to(device)
        input_ids = inputs.input_ids  # shape: [1, seq_len]

        # if the last token is not EOS_TOKEN, continue generating
        while input_ids[0, -1] != eos_token_id:
            input_length = input_ids.shape[1]
            image_inputs = None if not input_images \
                else self.processor.image_processor(images=input_images, return_tensors="pt").to(device)

            if got_input is None:
                partial_generation = self.mllm.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    pixel_values=image_inputs.pixel_values if hasattr(image_inputs, 'pixel_values') else None,
                    image_grid_thw=image_inputs.image_grid_thw if hasattr(image_inputs, 'image_grid_thw') else None,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_hidden_states=False,  # No need yet, we will do a second pass
                    stopping_criteria=stopping_criteria,
                    **generate_kwargs
                )

                input_ids = partial_generation['sequences']  # shape: [1, seq_len]
            else:
                input_ids = self.processor.tokenizer.encode(got_input)
                input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
                got_input = None

            if only_return_got:
                return {"got_text": self.processor.tokenizer.decode(input_ids[0])}

            # Decode the newly generated text
            cur_decoded_text = self.processor.tokenizer.decode(input_ids[0, input_length:], skip_special_tokens=False)
            output_text += cur_decoded_text\
                .replace(EOS_TOKEN, '').replace(EOI_TOKEN, '').replace(BOV_TOKEN, '').replace(EOV_TOKEN, '')

            # generate a image
            if input_ids[0, -1] == boi_token_id:
                input_ids = torch.cat([input_ids, img_token_ids], dim=1)  # now includes BOI_TOKEN + image tokens

                second_out = self.mllm(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    pixel_values=image_inputs.pixel_values if hasattr(image_inputs, 'pixel_values') else None,
                    image_grid_thw=image_inputs.image_grid_thw if hasattr(image_inputs, 'image_grid_thw') else None,
                    output_hidden_states=True,
                    return_dict=True
                )
                last_hidden_states = second_out['hidden_states'][-1]  # [batch_size, seq_len, hidden_size]

                img_gen_mask = torch.logical_and(
                    self.img_gen_start_id <= input_ids, input_ids < self.img_gen_start_id + self.num_img_out_tokens)

                gen_hidden_states = last_hidden_states[img_gen_mask].view(-1, self.num_img_out_tokens,
                                                                          last_hidden_states.shape[-1])
                gen_hidden_states = gen_hidden_states[-1:]  # only take the last batch 64 image tokens
                gen_hidden_states = gen_hidden_states.to(self.output_projector.projector.weight.dtype)

                gen_conditioning = self.output_projector(gen_hidden_states)
                gen_conditioning_add = self.output_projector_add(gen_hidden_states)  # [bz, gen_num, dim]
                null_conditioning = self.output_projector(torch.zeros_like(gen_hidden_states))
                gen_conditioning_pooled = torch.mean(gen_conditioning_add, dim=1)

                self.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = self.scheduler.timesteps

                # Prepare stable diffusion latents
                generator = torch.Generator(device=device).manual_seed(random_seed)

                latents = randn_tensor(
                    shape=(1, self.vae.config.latent_channels, height // 8, width // 8),
                    generator=generator,
                    device=device,
                    dtype=vae_dtype
                )
                latents = latents * self.scheduler.init_noise_sigma

                # The first 4 are the noisy latents, the next 4 are original image latents (for editing).
                # In tex-to-image generation scenario, we just provide zeros for original_image.
                original_image = original_images[-1] if original_images \
                    else Image.new('RGB', (width, height), (0, 0, 0))

                original_image_tensor = self.diffusion_transform(original_image).unsqueeze(0).to(device).to(vae_dtype)
                image_latents = self.vae.encode(original_image_tensor).latent_dist.mode()

                positions_colors = parse_coordinates_colors(cur_decoded_text)
                mask_num = max(len(positions_colors), 1)

                cond_images = [Image.new('RGB', (width, height), (0, 0, 0)) for _ in range(mask_num)]

                for i in range(len(positions_colors)):
                    p_c = positions_colors[i]
                    draw = ImageDraw.Draw(cond_images[i])
                    position = p_c['position']
                    color = p_c['color']
                    draw.rectangle(((position[0][0] / 1000 * width, position[0][1] / 1000 * height),
                                    (position[1][0] / 1000 * width, position[1][1] / 1000 * height)), fill=color)
                    del draw

                # Optionally save condition images for debugging/inspection
                if cond_save_dir is not None:
                    try:
                        os.makedirs(cond_save_dir, exist_ok=True)
                        for idx_save, img_save in enumerate(cond_images):
                            img_save.save(os.path.join(cond_save_dir, f"cond_{idx_save}.png"))
                    except Exception:
                        pass

                cond_images_tensor = []
                for c_image in cond_images:
                    c_image_tensor = self.diffusion_transform(c_image)
                    cond_images_tensor.append(c_image_tensor)

                # (1, mask_num, 3, target_size, target_size)
                cond_mask = torch.stack(cond_images_tensor, dim=0).unsqueeze(0)
                B, N, C, H, W = cond_mask.shape
                cond_mask = cond_mask.view(B * N, C, H, W)

                unet_cond_embeds = []
                for i in range(0, cond_mask.shape[0], self.vae_batch):
                    sub_batch = cond_mask[i: i + self.vae_batch]
                    embeds = self.vae.encode(sub_batch.to(device, dtype=vae_dtype)).latent_dist.mode()
                    embeds = embeds.to(device)
                    unet_cond_embeds.append(embeds)
                unet_cond_embeds = torch.cat(unet_cond_embeds, dim=0)
                unet_cond_embed = unet_cond_embeds.mean(dim=0, keepdim=True)

                if do_classifier_free_guidance:
                    uncond_image_latents = torch.zeros_like(image_latents)
                    image_latents = torch.cat([image_latents, image_latents, image_latents, uncond_image_latents],
                                              dim=0)

                    uncond_cond_image_latents = torch.zeros_like(unet_cond_embed)
                    unet_cond_embed = torch.cat([unet_cond_embed, uncond_cond_image_latents,
                                                 uncond_cond_image_latents, uncond_cond_image_latents], dim=0)

                combined_prompt_embeds = torch.cat(
                    [gen_conditioning, gen_conditioning, null_conditioning, null_conditioning],
                    dim=0) if do_classifier_free_guidance else gen_conditioning

                text_encoder_projection_dim = int(gen_conditioning_pooled.shape[-1])

                original_size = (height, width)
                target_size = (height, width)

                add_time_ids = self._get_add_time_ids(
                    original_size,
                    crops_coords_top_left,
                    target_size,
                    dtype=combined_prompt_embeds.dtype,
                    text_encoder_projection_dim=text_encoder_projection_dim,
                )

                added_cond_kwargs = {"text_embeds": gen_conditioning_pooled.to(device),
                                     "time_ids": add_time_ids.to(device)}

                for i, t in enumerate(tqdm(timesteps)):
                    latent_model_input = torch.cat([latents] * 4) if do_classifier_free_guidance else latents
                    scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents, unet_cond_embed],
                                                          dim=1)

                    noise_pred = self.unet(
                        scaled_latent_model_input,
                        t,
                        encoder_hidden_states=combined_prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False
                    )[0]

                    if do_classifier_free_guidance:
                        noise_pred_cond, noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(4,
                                                                                                                 dim=0)
                        noise_pred = (
                                noise_pred_uncond
                                + guidance_scale * (noise_pred_text - noise_pred_image)
                                + cond_image_guidance_scale * (noise_pred_cond - noise_pred_text)
                                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                        )

                    # step through scheduler
                    latents = self.scheduler.step(noise_pred, t, latents, generator=generator, return_dict=False)[0]

                final_latents = latents / self.vae.config.scaling_factor
                image_tensor = self.vae.decode(final_latents, generator=generator).sample
                image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
                pil_image = Image.fromarray(
                    (image_tensor[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype("uint8"))

                generated_images.append(pil_image)
                original_images.append(pil_image)
            elif input_ids[0, -1] == bov_token_id:
                input_images.append(self.source_transform(generated_images[-1]))
                input_ids = torch.cat([input_ids, input_token_ids], dim=1)

        # resize generated images with ori_w, and ori_h, with the shortest side being 1024
        if ori_w < ori_h:
            target_size = (width, int(height * ori_h / ori_w))
        else:
            target_size = (int(width * ori_w / ori_h), height)
        generated_images = [img.resize(target_size) for img in generated_images]

        return {"got_text": output_text, "images": generated_images}

    @classmethod
    def from_pretrained(cls, mllm, output_projector, scheduler, vae, unet, pretrained_model_path=None, **kwargs):
        model = cls(mllm=mllm, output_projector=output_projector, scheduler=scheduler, vae=vae, unet=unet, **kwargs)
        if os.environ.get('DEBUG_FLAG', 'False') == 'True':
            return model

        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            logs = model.load_state_dict(ckpt, strict=False)
            print(logs)
        return model
