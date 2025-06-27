import json
from pathlib import Path
from typing import List
from PIL import Image, ImageOps
from tqdm import tqdm
import shutil
import urllib.request


def combine_and_save_annotations(root_path: Path):
    annotations = []
    anno_path = root_path / "mlannotations"
    render_path = root_path / "mlrender"
    for annotation in anno_path.glob("*.json"):
        annotations.append(json.load(open(annotation)))
    json.dump(annotations, open(render_path / "annotations.json", "w"), indent=4)
    print(
        f"Combined {len(annotations)} annotations into {render_path / 'annotations.json'}"
    )


def orientations_to_face_rotation(orientation, face=1):
    if orientation is None:
        return (face, 0)

    mapping = {
        0: (1, 0),
        1: (1, 90),
        2: (2, 0),
        3: (2, 90),
        4: (3, 0),
        5: (3, 90),
        6: (7, 0),
        7: (7, 90),
        8: (8, 0),
        9: (8, 90),
        10: (9, 0),
        11: (9, 90),
        12: (1, 180),
        13: (1, 270),
        14: (2, 180),
        15: (2, 270),
        16: (3, 180),
        17: (3, 270),
        18: (7, 180),
        19: (7, 270),
        20: (8, 180),
        21: (8, 270),
        22: (9, 180),
        23: (9, 270),
    }

    return mapping.get(orientation, (0, 0))


def get_uuids_to_ignore():
    path = Path("./image prep/real photos cleaned annos/cleaned annos")

    uuids = []
    annos = list(path.glob("**/*.mla"))
    for anno in annos:
        with open(anno, "r") as file:
            data = json.load(file)
        uuids.append(data["imageFileUuid"])

    return uuids


faceint_to_str = {1: "front", 2: "left", 7: "back", 8: "right"}


def images_from_vq(ignore: List[str]):
    # path = Path("./image prep/real photos cleaned annos/cleaned annos")
    # output_path = Path("./image prep/real photos cleaned annos/extracted")
    path = Path("./image prep/Murphy oil annotations/all")
    output_path = Path("./image prep/Murphy oil annotations/extracted")

    if output_path.exists():
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    count = 0

    for mla in tqdm(list(path.glob("**/*.mla"))):
        with open(mla, "r") as file:
            data = json.load(file)

        if data["imageFileUuid"] in ignore:
            tqdm.write(f"skipping {mla}")
            continue

        image_path = mla.parent.parent / "Images" / f"{mla.stem}.jpg"
        if not image_path.exists():
            image_path.parent.mkdir(parents=True, exist_ok=True)

            url = data["url"]
            with urllib.request.urlopen(url) as response, open(
                image_path, "wb"
            ) as out_file:
                shutil.copyfileobj(response, out_file)

        image = Image.open(image_path).convert("RGB")
        image = ImageOps.exif_transpose(image)

        for annotation in data["annotations"]:
            if "face" not in annotation:
                continue

            face = faceint_to_str[annotation["face"]]

            if "productUuid" not in annotation or annotation["productUuid"] is None:
                continue

            product_uuid = annotation["productUuid"]

            if image.width < image.height:
                width, height = image.width, image.height
            else:
                width, height = image.height, image.width

            xmin, xmax, ymin, ymax = (
                annotation["xmin"],
                annotation["xmax"],
                annotation["ymin"],
                annotation["ymax"],
            )

            xmin, xmax, ymin, ymax = (
                int(xmin * width),
                int(xmax * width),
                int(ymin * height),
                int(ymax * height),
            )

            product_out_folder = output_path / f"{product_uuid}_{face}_real-real"
            product_out_folder.mkdir(parents=True, exist_ok=True)

            i = len(list(product_out_folder.glob("*.jpg")))
            product_out_path = (
                product_out_folder / f"{product_uuid}_{face}_real_{i}.jpg"
            )

            image.crop((xmin, ymin, xmax, ymax)).save(product_out_path)
            count += 1

    print(count, "items")


def synth_ml_images(path, output_path):

    if output_path.exists():
        shutil.rmtree(output_path)

    count = 0

    with open(path / "annotations.json") as file:
        file_annotations = json.load(file)

    for file in tqdm(file_annotations):
        image_path = path / file["filename"]
        image = Image.open(image_path).convert("RGB")

        for annotation in file["bounding_boxes"]:
            if "orientation" not in annotation:
                continue
            face = orientations_to_face_rotation(annotation["orientation"])[0]
            face = faceint_to_str[face]
            if "productUuid" not in annotation or annotation["productUuid"] is None:
                continue

            product_uuid = annotation["productUuid"]

            xmin, xmax, ymin, ymax = (
                annotation["xmin"],
                annotation["xmax"],
                annotation["ymin"],
                annotation["ymax"],
            )

            product_out_folder = output_path / f"{product_uuid}_{face}"
            product_out_folder.mkdir(parents=True, exist_ok=True)

            i = len(list(product_out_folder.glob("*.jpg")))
            product_out_path = product_out_folder / f"{product_uuid}_{face}_{i}.jpg"

            # image.crop((xmin, ymin, xmax, ymax)).show()
            image.crop((xmin, ymin, xmax, ymax)).save(product_out_path)
            count += 1

    print(count, "items")


if __name__ == "__main__":
    # uuids = get_uuids_to_ignore()
    # images_from_vq(ignore=uuids)
    org = "telstra"
    dataset = "20250627"
    root_path = Path(f"/Users/iman/model-pipeline/cache/{org}-{dataset}_w00/renders")
    render_path = root_path / "mlrender"
    if "annotations.json" not in render_path.glob("*.json"):
        combine_and_save_annotations(root_path)
    output_path = (
        Path("/Users/iman/345-data/ml-datasets")
        / org
        / "recognition"
        / f"{org}-{dataset}"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    synth_ml_images(render_path, output_path)
