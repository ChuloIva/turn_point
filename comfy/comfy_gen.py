import random
import torch
import sys

sys.path.append("/home/koalacrown/comfy/ComfyUI")
from nodes import (
    VAEDecode,
    KSamplerAdvanced,
    EmptyLatentImage,
    SaveImage,
    CheckpointLoaderSimple,
    CLIPTextEncode,
)


def main():
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="sd_xl_base_1.0.safetensors"
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(
            width=1024, height=1024, batch_size=1
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = cliptextencode.encode(
            text="evening sunset scenery blue sky nature, glass bottle with a galaxy in it",
            clip=checkpointloadersimple_4[1],
        )

        cliptextencode_7 = cliptextencode.encode(
            text="text, watermark", clip=checkpointloadersimple_4[1]
        )

        checkpointloadersimple_12 = checkpointloadersimple.load_checkpoint(
            ckpt_name="sd_xl_refiner_1.0.safetensors"
        )

        cliptextencode_15 = cliptextencode.encode(
            text="evening sunset scenery blue sky nature, glass bottle with a galaxy in it",
            clip=checkpointloadersimple_12[1],
        )

        cliptextencode_16 = cliptextencode.encode(
            text="text, watermark", clip=checkpointloadersimple_12[1]
        )

        ksampleradvanced = KSamplerAdvanced()
        vaedecode = VAEDecode()
        saveimage = SaveImage()

        # This loop will create 10 images total
        # Each iteration generates one image through the two-stage sampling process
        for q in range(3):
            ksampleradvanced_10 = ksampleradvanced.sample(
                add_noise="enable",
                noise_seed=random.randint(1, 2**64),
                steps=25,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                start_at_step=0,
                end_at_step=20,
                return_with_leftover_noise="enable",
                model=checkpointloadersimple_4[0],
                positive=cliptextencode_6[0],
                negative=cliptextencode_7[0],
                latent_image=emptylatentimage_5[0],
            )

            ksampleradvanced_11 = ksampleradvanced.sample(
                add_noise="disable",
                noise_seed=random.randint(1, 2**64),
                steps=25,
                cfg=8,
                sampler_name="euler",
                scheduler="normal",
                start_at_step=20,
                end_at_step=10000,
                return_with_leftover_noise="disable",
                model=checkpointloadersimple_12[0],
                positive=cliptextencode_15[0],
                negative=cliptextencode_16[0],
                latent_image=ksampleradvanced_10[0],
            )

            vaedecode_17 = vaedecode.decode(
                samples=ksampleradvanced_11[0], vae=checkpointloadersimple_12[2]
            )

            saveimage_19 = saveimage.save_images(
                filename_prefix="ComfyUI", images=vaedecode_17[0]
            )


if __name__ == "__main__":
    main()