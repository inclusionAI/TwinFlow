## Codebase Structure

```sh
.
├── configs/    # configuration files for distinct operational setups.
├── data/       # dataset definitions and data loading pipelines.
├── methods/    # core algorithm implementations and logic.
├── scripts/    # shell scripts for execution of training and sampling.
├── services/   # auxiliary utilities and shared tooling services.
└── steerers/   # primary control flows for training and sampling.
```

## Algorithm Guide

TwinFlow-trained model is an any-step model, which supports few-step, any-step, and multi-step sampling at the same time. To achieve this, there are two timestep conditions: timestep and target timestep.

During training, there are key configurations to control the target timestep distribution:

https://github.com/inclusionAI/TwinFlow/blob/f1231854d7aba806eb586a79d05aa9cf062e29ca/src/methodes/twinflow/twinflow.py#L132

![](../assets/probs_spec.png)

Common practices:

- `probs = {"e2e": 1, "mul": 1, "any": 1, "adv": 1}`: TwinFlow training
- `probs = {"e2e": 0, "mul": 0, "any": 1, "adv": 0}`: RCGM training (in theory)
- `probs = {"e2e": 1, "mul": 1, "any": 1, "adv": 0}`: RCGM training (in practice)
  - In this case, you need comment these lines to run correctly:
https://github.com/inclusionAI/TwinFlow/blob/f1231854d7aba806eb586a79d05aa9cf062e29ca/src/methodes/twinflow/twinflow.py#L358-L365

- `probs = {"e2e": 0, "mul": 1, "any": 0, "adv": 0}`: Flow Matching training
  - In this case, also set `consistc_ratio`, `enhanced_ratio`, `estimate_order` to 0

## Quick Start

### Full Training on OpenUni

To deploy TwinFlow on OpenUni, please follow the procedure outlined below.

### Environment Configuration
Begin by setting up the environment prerequisites as detailed in the official [OpenUni repository](https://github.com/wusize/OpenUni). Ensure all dependencies are correctly installed before proceeding.

### Model Configuration

- Generator Backbone

Download the [OpenUni generator backbone](https://huggingface.co/wusize/openuni/blob/main/openuni_l_internvl3_2b_sana_1_6b_512_hf_blip3o60k.pth) checkpoint locally. Once downloaded, update the configuration file `configs/openuni_task/openuni_full.yaml` to reflect the local path:

```yaml
model:
  type: ./networks/openuni/openuni_l_internvl3_2b_sana_1_6b_512_hf.py
  path: path/to/openuni_l_internvl3_2b_sana_1_6b_512_hf_blip3o60k.pth 
  in_chans: 16
```

- Other Components
  - OpenUni Encoder: [InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B)
  - SANA 1.6B: [Sana_1600M_512px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_1600M_512px_diffusers)
  - DC-AE: [dc-ae-f32c32-sana-1.1-diffusers](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers)

Prior to initiating training, define the environment variables pointing to your downloaded component models. You may modify `scripts/openuni/train_ddp.sh` directly or export them in your script:

```sh
export INTERNVL3_PATH="path/to/InternVL3-2B"
export SANA_1600M_512PX_PATH="path/to/Sana_1600M_512px_diffusers"
export DCAE_PATH="path/to/dc-ae-f32c32-sana-1.1-diffusers"
```

### Launch Training

- Standard Training (TwinFlow on OpenUni):

```sh
scripts/openuni/train_ddp.sh configs/openuni_task/openuni_full.yaml
```

- Data-Free Training (No Text-Image Pairs Required):

```sh
scripts/openuni/train_ddp.sh configs/openuni_task/openuni_full_imgfree.yaml
```

### Sampling Images

To directly run sampling:

```sh
scripts/openuni/sample_demo.sh configs/openuni_task/openuni_full.yaml
```

After training, the trained model supports 3 sampling modes: **few-step**, **any-step**, and **standard multi-step**. We provide different sampling configurations for each mode in the yaml for reference:

```yaml
# few-step sampling
sample:
  ckpt: "700" # <- change to the ckpt
  cfg_scale: 0
  cfg_interval: [0.00, 0.00]
  sampling_steps: 2 # 1
  stochast_ratio: 1.0 # 0.8
  extrapol_ratio: 0.0
  sampling_order: 1
  time_dist_ctrl: [1.0, 1.0, 1.0]
  rfba_gap_steps: [0.001, 0.7]
  sampling_style: few

# any-step sampling
sample:
  ckpt: "700"
  cfg_scale: 0
  cfg_interval: [0.00, 0.00]
  sampling_steps: 4 # 8
  stochast_ratio: 0.0
  extrapol_ratio: 0.0
  sampling_order: 1
  time_dist_ctrl: [1.0, 1.0, 1.0]
  rfba_gap_steps: [0.001, 0.5]
  sampling_style: any

# multi-step sampling
sample:
  ckpt: "700"
  cfg_scale: 0
  cfg_interval: [0.00, 0.00]
  sampling_steps: 30
  stochast_ratio: 0.0
  extrapol_ratio: 0.0
  sampling_order: 1
  time_dist_ctrl: [1.17, 0.8, 1.1]
  rfba_gap_steps: [0.001, 0.0]
  sampling_style: mul
```

### LoRA Training

- LoRA Training (TwinFlow on OpenUni), you need to comment L52 and use original transformer in L51:

https://github.com/inclusionAI/TwinFlow/blob/9fc59521017d329ed6aee8046ff71e523a26f68f/src/networks/openuni/openuni_l_internvl3_2b_sana_1_6b_512_hf.py#L51-L52

```sh
# 1 order
scripts/openuni/train_ddp_lora.sh configs/openuni_task/openuni_lora_1order.yaml
# 2 order
scripts/openuni/train_ddp_lora.sh configs/openuni_task/openuni_lora_2order.yaml
```

- LoRA Training (TwinFlow on SD3.5-M):

```sh
# 1 order
scripts/sd3/train_ddp_lora.sh configs/sd_task/sd35_lora_1order.yaml
# 2 order
scripts/sd3/train_ddp_lora.sh configs/sd_task/sd35_lora_2order.yaml
```

### QwenImage and QwenImage-Edit Training

- Full Training (TwinFlow on QwenImage):

```sh
scripts/qwenimage/train_fsdp.sh configs/qwenimage_task/qwenimage_full.yaml
```

> [!NOTE]
> Due to memory limit, we did not add separate EMA for QwenImage, thus, we set `ema_decay_rate: 0` in the config. Users could switch FSDP2, which could fully support this.
> https://github.com/inclusionAI/TwinFlow/blob/ab5808dfbbc7b3c5712d73652021e7318c93a39b/src/steerers/qwenimage/sft_fsdp.py#L132-L136

> [!NOTE]
> Switch to LoRA training is easy, we suggest you to refer to `src/steerers/stable_diffusion_3/sft_ddp_lora.py` to add LoRA, and set the method config like `configs/sd_task/sd35_lora_1order.yaml` or `configs/sd_task/sd35_lora_2order.yaml`

> [!NOTE]
> Current modeling supports single reference image input for editing, you need to modify minor code to support edit training.

1. In the config, change `QwenImage` to `QwenImageEdit`.
2. Modify the dataloader to support loading the reference image, e.g. `text, image, control_image = batch["text"], batch["image"].cuda(), batch["control_img"].cuda()`
3. Prepare inputs like:
```python
with torch.no_grad():
   (
       prompt_embeds,
       prompt_embeds_mask,
       uncond_prompt_embeds,
       uncond_prompt_embeds_mask,
   ) = model.encode_prompt(text, control_image, do_cfg=True)
   prompt_embeds = prompt_embeds.to(torch.float32)
   uncond_prompt_embeds = uncond_prompt_embeds.to(torch.float32)

   latents = model.pixels_to_latents(image).to(torch.float32)
   control_latents = model.pixels_to_latents(control_image).to(torch.float32)
```
4. Pass the inputs to `training_step` like:
```python
loss = method.training_step(
   model,
   latents,
   c=[prompt_embeds, prompt_embeds_mask, control_latents],
   e=[uncond_prompt_embeds, uncond_prompt_embeds_mask, control_latents],
   step=(global_step - 1),
   v=None
)
```
