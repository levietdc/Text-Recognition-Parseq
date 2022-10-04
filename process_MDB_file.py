import io
import os

from PIL import Image
from tqdm import tqdm

from resources.lmdb_dataset import LMDBDataSet


def convert_mdb_to_image_and_label(data_dir, out_dir, dataset_type):
    dataset = LMDBDataSet(data_dir=os.path.join(data_dir, dataset_type))
    if not os.path.exists("imgs"):
        os.mkdir("imgs")
    with open(os.path.join(out_dir, dataset_type + ".txt"), "w") as f:
        for i, (image, label) in tqdm(enumerate(dataset)):
            # image = Image.open(io.BytesIO(byte_img))
            save_path = os.path.join("imgs", str(i) + "_" + dataset_type + ".png")
            image.save(out_dir + save_path)
            f.write(save_path + "\t" + label + "\n")


def main():
    data_dir = (
        "/home/bap/BAP/ocr/TextRecognition/parseq/data_gen/val/vie_data/"
    )
    out_dir = "/home/bap/BAP/ocr/TextRecognition/parseq/data_test/"
    dataset_type = ""
    convert_mdb_to_image_and_label(data_dir, out_dir, dataset_type)


if __name__ == "__main__":
    main()
