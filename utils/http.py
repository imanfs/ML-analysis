import asyncio
import io
import aiohttp
from typing import List
from PIL import Image, ImageOps


async def fetch_image(session, url):
    async with session.get(url) as response:
        img_bytes = await response.read()
        return img_bytes


async def download_images(image_urls: List[str]) -> List[Image.Image]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_image(session, url) for url in image_urls]
        images_bytes = await asyncio.gather(*tasks)

    images = [
        ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        for image_bytes in images_bytes
    ]
    return images
