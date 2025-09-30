# nano-banana-loop


Testing image models with recursive editing from simple prompts.

How to run jump to [Usage](#usage)

# Examples

## [Nano Banana](https://fal.ai/models/fal-ai/nano-banana/edit)
### Future




https://github.com/user-attachments/assets/67f6960e-2914-41bd-b0ba-c49dfb6054aa


https://github.com/user-attachments/assets/ca3564ed-8b02-4696-adb3-c340390e2cc8


https://github.com/user-attachments/assets/4cfea3b4-1493-40d8-912c-cc9d12e63ed5



https://github.com/user-attachments/assets/e47d5943-8f4e-484d-9d02-35bea95130e3


https://github.com/user-attachments/assets/09dae028-a2fe-4b7d-8d96-6cfb59ff6098


https://github.com/user-attachments/assets/a8488269-521b-4852-892a-818896bb9a97

### Funny

https://github.com/user-attachments/assets/5b017c8c-6515-4f19-9f33-c4c817449035



### Tilt

https://github.com/user-attachments/assets/4d6bcc2d-adaa-4afd-985d-bda024734ad3

## [Seedream](https://fal.ai/models/fal-ai/bytedance/seedream/v4/edit)

#### Future

https://github.com/user-attachments/assets/218cfbd4-a28f-4fbe-bd06-65339b8a5afe

### Funny



https://github.com/user-attachments/assets/66f7d83c-34c3-41b4-bfa8-d29fd9688de6


https://github.com/user-attachments/assets/98baf9bf-9f75-4a55-9176-baf691f822e1



## [Qwen Image Edit Plus](https://fal.ai/models/fal-ai/qwen-image-edit-plus)

### Funny

https://github.com/user-attachments/assets/7bcc79d2-0623-46e9-8b3f-b9326f16e533



# Usage

Get your Fal API key [here](https://fal.ai/dashboard/keys).

```bash
export FAL_KEY="your_fal_api_key_here"
```

```bash
# Basic usage - future
python app.py --image_url init_images/tengyart-DoqtEEn8SOo-unsplash.jpg --mode future --num_frames 50

# funny mode
python app.py --image_url init_images/tengyart-DoqtEEn8SOo-unsplash.jpg --mode funny --num_frames 50
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv run app.py --image_url init_images/tengyart-DoqtEEn8SOo-unsplash.jpg --mode future --num_frames 50

uv run app.py --image_url init_images/tengyart-DoqtEEn8SOo-unsplash.jpg --mode funny --num_frames 50
```

**Available modes:** `up`, `down`, `left`, `right`, `rotate-left`, `rotate-right`, `zoom-in`, `zoom-out`, `future`, `past`, `funny`, `serious`, `dramatic`, `peaceful`, `futuristic`, `nature`, `urban`, `minimalist`, `crowded`, `empty`
