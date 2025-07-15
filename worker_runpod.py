import os
import sys
import types
import importlib.util
import base64
import traceback
import io

ROOT = os.path.dirname(os.path.abspath(__file__))
pkg = types.ModuleType('utils')
pkg.__path__ = [os.path.join(ROOT, 'utils')]
sys.modules['utils'] = pkg
util_path = os.path.join(ROOT, 'utils', 'install_util.py')
spec = importlib.util.spec_from_file_location('utils.install_util', util_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.modules['utils.install_util'] = mod

import json, requests, runpod

import torch
import random
from comfy import controlnet
from comfy.sd import load_checkpoint_guess_config
from nodes import LoadImage, VAEEncode
from nodes import NODE_CLASS_MAPPINGS
from comfy_extras import nodes_post_processing, nodes_differential_diffusion, nodes_upscale_model
import numpy as np
from PIL import Image
import asyncio
import execution
# 5. Явно импортируем нужные классы из server.py
from server import PromptServer
from nodes import load_custom_node
from math import ceil, floor

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET"]
)

def download_file(url, save_dir='/content/ComfyUI/input'):
    os.makedirs(save_dir, exist_ok=True)
    file_name = url.split('/')[-1]
    file_path = os.path.join(save_dir, file_name)

    # Настраиваем сессию с retry
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        # Устанавливаем таймаут (например, 30 секунд)
        response = session.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)
    except requests.exceptions.RequestException as e:
        # Логируем и пробрасываем дальше
        print(f"Ошибка при загрузке {url}: {e}")
        raise
    return file_path

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
server_instance = PromptServer(loop)
execution.PromptQueue(server_instance)

load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-AutomaticCFG")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-Custom-Scripts")
load_custom_node("/content/ComfyUI/custom_nodes/Derfuu_ComfyUI_ModdedNodes")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-Impact-Pack")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-Impact-Subpack")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-Inspire-Pack")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-KJNodes")
load_custom_node("/content/ComfyUI/custom_nodes/comfyui_controlnet_aux")
load_custom_node("/content/ComfyUI/custom_nodes/ComfyUI-TiledDiffusion")
load_custom_node("/content/ComfyUI/custom_nodes/was-node-suite-comfyui")

Automatic_CFG = NODE_CLASS_MAPPINGS["Automatic CFG"]()
ImageScaleToTotalPixels = nodes_post_processing.NODE_CLASS_MAPPINGS["ImageScaleToTotalPixels"]()
GetImageSizeAndCount = NODE_CLASS_MAPPINGS["GetImageSizeAndCount"]()
TTPlanet_TileSimple = NODE_CLASS_MAPPINGS["TTPlanet_TileSimple_Preprocessor"]()
TiledDiffusion = NODE_CLASS_MAPPINGS["TiledDiffusion"]()
KSampler_inspire = NODE_CLASS_MAPPINGS["KSampler //Inspire"]()
ControlNetApplyAdvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
UltralyticsDetectorProvider = NODE_CLASS_MAPPINGS["UltralyticsDetectorProvider"]()
SegmDetectorSEGS = NODE_CLASS_MAPPINGS["SegmDetectorSEGS"]()
DifferentialDiffusion = nodes_differential_diffusion.NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
DetailerForEach = NODE_CLASS_MAPPINGS["DetailerForEach"]()
VAEDecodeTiled = NODE_CLASS_MAPPINGS["VAEDecodeTiled"]()
ColorMatch = NODE_CLASS_MAPPINGS["ColorMatch"]()
ImageBlend = nodes_post_processing.NODE_CLASS_MAPPINGS["ImageBlend"]()
WAS_Image_Blending_Mode = NODE_CLASS_MAPPINGS["Image Blending Mode"]()
ImageScale = NODE_CLASS_MAPPINGS["ImageScale"]()
ImageScaleBy = NODE_CLASS_MAPPINGS["ImageScaleBy"]()
UpscaleModelLoader = nodes_upscale_model.NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
ImageUpscaleWithModel = nodes_upscale_model.NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()

with torch.inference_mode():
    model_patcher, clip, vae, clipvision = load_checkpoint_guess_config("/content/ComfyUI/models/checkpoints/dreamshaperXL_lightningDPMSDE.safetensors", output_vae=True, output_clip=True, embedding_directory=None)
    tile_control_net = controlnet.load_controlnet("/content/ComfyUI/models/controlnet/xinsir-controlnet-tile-sdxl-1.0.safetensors")
    segm_detector = UltralyticsDetectorProvider.doit(model_name="segm/PitEyeDetailer-v2-seg.pt")
    upscale_model = UpscaleModelLoader.load_model(model_name="4xRealWebPhoto_v4_dat2.safetensors")[0]
    model_patcher = Automatic_CFG.patch(model=model_patcher, hard_mode=True, boost=True)[0]

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_image = values['input_image_check']
    input_image = download_file(input_image)
    positive_prompt = values['positive_prompt']
    negative_prompt = values['negative_prompt']
    inspire_seed = values['inspire_seed']
    inspire_steps = values['inspire_steps']
    inspire_cfg = values['inspire_cfg']
    inspire_sampler_name = values['inspire_sampler_name']
    inspire_scheduler = values['inspire_scheduler']
    inspire_denoise = values['inspire_denoise']
    inspire_noise_mode = values['inspire_noise_mode']
    inspire_batch_seed_mode = values['inspire_batch_seed_mode']
    inspire_variation_seed = values['inspire_variation_seed']
    inspire_variation_strength = values['inspire_variation_strength']
    inspire_variation_method = values['inspire_variation_method']
    scale_factor = values['scale_factor']
    blur_strength = values['blur_strength']
    strength = values['strength']
    start_percent = values['start_percent']
    end_percent = values['end_percent']
    tile_method = values['tile_method']
    tile_overlap = values['tile_overlap']
    tile_size = values['tile_size']
    threshold = values['threshold']
    dilation = values['dilation']
    crop_factor = values['crop_factor']
    drop_size = values['drop_size']
    labels = values['labels']
    detailer_guide_size = values['detailer_guide_size']
    detailer_guide_size_for_bbox = values['detailer_guide_size_for_bbox']
    detailer_max_size = values['detailer_max_size']
    detailer_seed = values['detailer_seed']
    detailer_steps = values['detailer_steps']
    detailer_cfg = values['detailer_cfg']
    detailer_sampler_name = values['detailer_sampler_name']
    detailer_scheduler = values['detailer_scheduler']
    detailer_denoise = values['detailer_denoise']
    detailer_feather = values['detailer_feather']
    detailer_noise_mask = values['detailer_noise_mask']
    detailer_force_inpaint = values['detailer_force_inpaint']
    detailer_cycle = values['detailer_cycle']
    detailer_inpaint_model = values['detailer_inpaint_model']
    detailer_noise_mask_feather = values['detailer_noise_mask_feather']
    color_method = values['color_method']
    blend_factor = values['blend_factor']
    blend_mode = values['blend_mode']
    blending_mode = values['blending_mode']
    blending_blend_percentage = values['blending_blend_percentage']
    vram = values['vram']
    upscale_mp = values['upscale_mp']
    w_tiles = values['w_tiles']
    h_tiles = values['h_tiles']
    downscale_by = values['downscale_by']

    output_image, output_mask = LoadImage().load_image(input_image)
    output_image_s = ImageScaleToTotalPixels.upscale(image=output_image, upscale_method="nearest-exact", megapixels=1.0)[0]
    image_width = GetImageSizeAndCount.getsize(output_image_s)["result"][1]
    image_height = GetImageSizeAndCount.getsize(output_image_s)["result"][2]
    w_math = ceil((image_width * upscale_mp) / 8) * 8
    h_math = ceil((image_height * upscale_mp) / 8) * 8
    tile_width = ceil((w_math / w_tiles) / 8) * 8
    tile_height = ceil((h_math / h_tiles) / 8) * 8
    tile_batch_size = floor((vram-3) / ((tile_width*tile_height) / 1000000))
    upscale_image = ImageScaleBy.upscale(image=output_image, upscale_method="bilinear", scale_by=downscale_by)[0]
    upscaled_image = ImageUpscaleWithModel.upscale(upscale_model=upscale_model, image=upscale_image)[0]
    output_image = ImageScale.upscale(image=upscaled_image, upscale_method="bilinear", width=w_math, height=h_math, crop="disabled")[0]

    cond, pooled = clip.encode_from_tokens(clip.tokenize(positive_prompt), return_pooled=True)
    cond = [[cond, {"pooled_output": pooled}]]
    n_cond, n_pooled = clip.encode_from_tokens(clip.tokenize(negative_prompt), return_pooled=True)
    n_cond = [[n_cond, {"pooled_output": n_pooled}]]
    output_image_t = TTPlanet_TileSimple.execute(output_image, scale_factor=scale_factor, blur_strength=blur_strength)[0]
    positive, negative = ControlNetApplyAdvanced.apply_controlnet(positive=cond, negative=n_cond, control_net=tile_control_net, image=output_image_t, strength=strength, start_percent=start_percent, end_percent=end_percent)
    tile_model = TiledDiffusion.apply(model=model_patcher, method=tile_method, tile_width=tile_width, tile_height=tile_height, tile_overlap=tile_overlap, tile_batch_size=tile_batch_size)[0]
    latent_image = VAEEncode().encode(vae, output_image)[0]
    inspire_sample = KSampler_inspire.doit(model=tile_model, 
                                            seed=inspire_seed, 
                                            steps=inspire_steps, 
                                            cfg=inspire_cfg, 
                                            sampler_name=inspire_sampler_name, 
                                            scheduler=inspire_scheduler, 
                                            positive=positive, 
                                            negative=negative,
                                            latent_image=latent_image, 
                                            denoise=inspire_denoise,
                                            noise_mode=inspire_noise_mode,
                                            batch_seed_mode=inspire_batch_seed_mode,
                                            variation_seed=inspire_variation_seed,
                                            variation_strength=inspire_variation_strength,
                                            variation_method=inspire_variation_method)[0]
    tiled_decoded = VAEDecodeTiled.decode(vae=vae, samples=inspire_sample, tile_size=tile_size)[0]
    segs = SegmDetectorSEGS.doit(segm_detector=segm_detector[1], image=output_image, threshold=threshold, dilation=dilation, crop_factor=crop_factor, drop_size=drop_size, labels=labels)[0]
    dd_model_patcher = DifferentialDiffusion.apply(model_patcher)[0]
    detailer_image = DetailerForEach.do_detail(image=tiled_decoded, 
                                        segs=segs, 
                                        model=dd_model_patcher, 
                                        clip=clip, 
                                        vae=vae, 
                                        guide_size=detailer_guide_size, 
                                        guide_size_for_bbox=detailer_guide_size_for_bbox, 
                                        max_size=detailer_max_size, 
                                        seed=detailer_seed, 
                                        steps=detailer_steps, 
                                        cfg=detailer_cfg, 
                                        sampler_name=detailer_sampler_name, 
                                        scheduler=detailer_scheduler,
                                        positive=cond, 
                                        negative=n_cond, 
                                        denoise=detailer_denoise, 
                                        feather=detailer_feather, 
                                        noise_mask=detailer_noise_mask, 
                                        force_inpaint=detailer_force_inpaint,
                                        cycle=detailer_cycle,
                                        inpaint_model=detailer_inpaint_model,
                                        noise_mask_feather=detailer_noise_mask_feather)[0]
    color_image = ColorMatch.colormatch(image_ref=output_image, image_target=detailer_image, method=color_method)[0]
    blend_image = ImageBlend.blend_images(image1=color_image, image2=detailer_image, blend_factor=blend_factor, blend_mode=blend_mode)[0]
    blending_image = WAS_Image_Blending_Mode.image_blending_mode(image_a=blend_image, image_b=output_image, mode=blending_mode, blend_percentage=blending_blend_percentage)[0]
    # Image.fromarray(np.array(blending_image*255, dtype=np.uint8)[0]).save("/content/ultralytics.png")

    print("="*20)
    print("SEND IMAGE")
    print("="*20)
    
    result = "/content/ultralytics.png"
    
    try:
        pil_img = Image.fromarray(np.array(blending_image*255, dtype=np.uint8)[0])

        # Записываем изображение в буфер в формате PNG
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        # Кодируем в Base64 и декодируем в строку
        base64_str = base64.b64encode(image_bytes).decode("utf-8")

        return base64_str

        #default_filename = os.path.basename(result)
        #with open(result, "rb") as file:
        #    data = {
        #        "result_url": result,
        #        "original_url": input_image
        #    }
        #    
        #    files = {
        #        "file": (default_filename, file, "image/png")
        #    }
        #    response = requests.post(
        #        "https://fast-knotta.ru.tuna.am/callback",
        #        data=data,
        #        files=files
        #    )
        #    
        #    response.raise_for_status()
        #    return {"status": "DONE"}
    except Exception as err:
        err_str = traceback.format_exc()
        print("Произошла ошибка:\n", err_str)
        return {"result": f"FAILED: {str(err)}", "status": "FAILED"}
    #finally:
    #    if os.path.exists(result):
    #        try:
    #            os.remove(result)
    #        except OSError:
    #            pass

runpod.serverless.start({"handler": generate})
