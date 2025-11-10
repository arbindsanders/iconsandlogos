from io import BytesIO
from pathlib import Path
from typing import Optional

import modal

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.13",
    )
    .entrypoint([])  # remove verbose logging by base image on entry
    .apt_install(
        "git",
        "libwebp-dev",  # WebP support
        "libavif-dev",  # AVIF support
        "libaom-dev",  # AV1 codec for AVIF
        "libdav1d-dev",  # Additional AV1 decoder
    )
    .pip_install("uv")
    .run_commands(
        f"uv pip install --system --compile-bytecode --index-strategy unsafe-best-match hf_transfer Pillow~=11.2.1 safetensors~=0.5.3 transformers~=4.53.0 sentencepiece~=0.2.0 torch==2.7.1 torchvision torchaudio optimum-quanto==0.2.7 fastapi[standard]==0.115.4 python-multipart==0.0.12 git+https://github.com/huggingface/diffusers https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.1dev20250929/nunchaku-1.0.1.dev20250929+torch2.7-cp313-cp313-linux_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu128"
    )
)

MODEL_NAME = "Qwen/Qwen-Image-Edit-2509"

# Qwen model configuration
num_inference_steps = 4  # you can also use the 8-step model to improve the quality
rank = 32  # you can also use the rank=128 model to improve the quality

CACHE_DIR = Path("/cache")
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

secrets = [modal.Secret.from_name("flux-app-secrets", required_keys=["HF_TOKEN"])]

image = image.env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Allows faster model downloads
        "HF_HOME": str(CACHE_DIR),  # Points the Hugging Face cache to a Volume
    }
)

app = modal.App("nunchaku-qwen-image-fastapi")

with image.imports():
    import torch
    import os
    import math
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
    from diffusers.utils import load_image
    from nunchaku import NunchakuQwenImageTransformer2DModel
    from nunchaku.utils import get_precision, get_gpu_memory
    from PIL import Image
    from fastapi import FastAPI, Form, HTTPException
    from fastapi.responses import Response
    from pydantic import BaseModel


@app.cls(
    image=image,
    gpu="L40s",
    volumes=volumes,
    secrets=secrets,
    scaledown_window=120,
    timeout=10 * 60,  # 10 minutes
)
class NunchakuQwenImageModel:
    @modal.enter()
    def enter(self):
        print(f"Downloading {MODEL_NAME} and Nunchaku transformer if necessary...")

        self.dtype = torch.bfloat16
        self.device = "cuda"

        # Auto-detect precision (int4 or fp4) based on GPU
        self.precision = get_precision()
        print(f"Detected precision: {self.precision}")

        # Scheduler configuration for Qwen
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),  # We use shift=3 in distillation
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),  # We use shift=3 in distillation
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set shift_terminal to None
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        # Model path for the new 2509 model
        model_path = f"nunchaku-tech/nunchaku-qwen-image-edit-2509/svdq-{self.precision}_r{rank}-qwen-image-edit-2509-lightningv2.0-{num_inference_steps}steps.safetensors"

        # Load the quantized transformer
        self.transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
            model_path,
            cache_dir=CACHE_DIR,
        )

        # Load the full pipeline with the quantized transformer
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_NAME,
            transformer=self.transformer,
            scheduler=self.scheduler,
            torch_dtype=self.dtype,
            cache_dir=CACHE_DIR,
            token=os.environ.get("HF_TOKEN"),
        )

        # Memory optimization based on GPU memory
        if get_gpu_memory() > 18:
            self.pipe.enable_model_cpu_offload()
        else:
            # use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
            self.transformer.set_offload(
                True, use_pin_memory=False, num_blocks_on_gpu=1
            )  # increase num_blocks_on_gpu if you have more VRAM
            self.pipe._exclude_from_cpu_offload.append("transformer")
            self.pipe.enable_sequential_cpu_offload()

        print("Model loaded successfully!")

    @modal.method()
    def inference(
        self,
        prompt: str,
        images: list[bytes],
        true_cfg_scale: float = 1.0,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
    ) -> bytes:
        # Use provided seed or generate a random one
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = torch.Generator(device=self.device)

        print(f"Generating image with prompt: {prompt}")
        print(f"Using precision: {self.precision}")
        print(f"Processing {len(images)} input images")

        # Convert bytes to PIL Images
        pil_images = []
        for img_bytes in images:
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            pil_images.append(img)

        # Prepare inputs for the pipeline
        inputs = {
            "image": pil_images,
            "prompt": prompt,
            "true_cfg_scale": true_cfg_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "height": height,
            "width": width,
        }

        output = self.pipe(**inputs)
        output_image = output.images[0]

        byte_stream = BytesIO()
        output_image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        return image_bytes


# Create a separate FastAPI app to handle the web interface
@app.function(image=image, volumes=volumes, secrets=secrets, cpu="0.5", memory="2GiB")
@modal.asgi_app()
def fastapi_app():
    from fastapi import Depends, FastAPI, Form, HTTPException, status, UploadFile, File
    from fastapi.responses import Response
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel
    from typing import List

    class GenerateImageRequest(BaseModel):
        prompt: str
        negative_prompt: str = ""
        width: int = 1024
        height: int = 1024
        true_cfg_scale: float = 1.0
        seed: Optional[int] = None

    class EditImagesRequest(BaseModel):
        prompt: str
        images: List[str] = []  # base64 encoded images with data URL format
        true_cfg_scale: float = 1.0
        seed: Optional[int] = None
        width: Optional[int] = 1024
        height: Optional[int] = 1024

    web_app = FastAPI(
        title="Nunchaku Qwen Image Edit 2509",
        description="Edit multiple images using Nunchaku-quantized Qwen Image Edit 2509 model",
        version="1.0.0",
    )

    @web_app.post("/edit-images")
    async def edit_images(
        request: EditImagesRequest,
        token: Optional[HTTPAuthorizationCredentials] = Depends(
            HTTPBearer(auto_error=False)
        ),
    ):
        """
        Edit multiple images using Nunchaku-quantized Qwen Image Edit 2509 model.

        - **prompt**: Text description of how you want to edit the images
        - **true_cfg_scale**: True CFG scale (default: 1.0, optimal for Qwen)
        - **seed**: Optional seed for reproducible results
        - **images**: List of base64 encoded images in data URL format (e.g., data:image/webp;base64,...)
        """

        if os.environ.get("BEARER_TOKEN", False):
            if not token or token.credentials != os.environ["BEARER_TOKEN"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect bearer token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        try:
            # Validate that we have at least one image
            if not request.images:
                raise HTTPException(
                    status_code=400,
                    detail="At least one image is required.",
                )

            import base64
            import re

            image_bytes_list = []

            for i, image_data_url in enumerate(request.images):
                # Validate data URL format
                if not image_data_url.startswith("data:image/"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Image {i+1} is not a valid data URL. Expected format: data:image/[type];base64,[data]",
                    )

                # Extract base64 data
                try:
                    # Split on comma to separate header from data
                    header, data = image_data_url.split(",", 1)

                    # Validate supported image types
                    supported_types = ["jpeg", "jpg", "png", "webp", "avif"]
                    image_type = None
                    for img_type in supported_types:
                        if f"image/{img_type}" in header:
                            image_type = img_type
                            break

                    if not image_type:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Image {i+1} has unsupported type. Supported types: {', '.join(supported_types)}",
                        )

                    # Decode base64 data
                    image_bytes = base64.b64decode(data)
                    image_bytes_list.append(image_bytes)

                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error decoding image {i+1}: {str(e)}",
                    )

            model = NunchakuQwenImageModel()
            result_bytes = model.inference.remote(
                prompt=request.prompt,
                images=image_bytes_list,
                true_cfg_scale=request.true_cfg_scale,
                seed=request.seed,
                width=request.width or 1024,
                height=request.height or 1024,
            )

            return Response(
                content=result_bytes,
                media_type="image/png",
                headers={
                    "Content-Disposition": f"inline; filename=edited_image_{request.seed or 'random'}.png"
                },
            )

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error editing images: {str(e)}"
            )

    return web_app
