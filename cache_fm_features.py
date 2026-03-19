import argparse
import os
import numpy as np
import torch
from tqdm.auto import tqdm
import skimage.io as io

from accelerate.logging import get_logger
from torchvision.transforms import Normalize

logger = get_logger(__name__)


def preprocess_raw_image(x):
    assert x.shape[-1] == 256
    x = x / 255.0
    x = Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143))(x)
    return x


def extract_feature_db(encoder, image_dir, device):
    file_list = sorted(os.listdir(image_dir))
    feat_db = {}
    for file_name in tqdm(file_list, desc=f"Extract {os.path.basename(image_dir)}"):
        file_path = os.path.join(image_dir, file_name)
        image = io.imread(file_path)[:, :, :3]
        with torch.no_grad():
            image = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2)
            raw_image = preprocess_raw_image(image).to(device)
            z = encoder.forward_features(raw_image)["x_norm_patchtokens"]
            feat = z.squeeze(0).cpu().numpy()
        feat_db[file_name] = feat
    return feat_db


def main(args):
    device = args.device
    encoder = torch.hub.load(args.repo_path, args.model, source="local", weights=args.weight_path)
    encoder = encoder.to(device)
    encoder.eval()

    output_dir = os.path.join(args.output_root, "Vaihingen", "dinov3_vitl16_sat493m")
    os.makedirs(output_dir, exist_ok=True)

    image_dir = os.path.join(args.data_path, args.img_type)
    feat_db = extract_feature_db(encoder, image_dir, device)

    output_path = os.path.join(output_dir, f"{args.img_type}_latent.npy")
    np.save(output_path, feat_db)

    print(f"saved features to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="cache foundation model features")
    parser.add_argument("--data-path", type=str, default="/224010098/DATASET/DACR/Vaihingen")
    parser.add_argument("--img-type", type=str, default="thin", choices=["cloudy", "clear", "thin", "thick"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.repo_path = "/224010098/workspace/DINOV3/DINOV3-pth/dinov3"
    args.model = "dinov3_vitl16"
    args.weight_path = "/224010098/workspace/DINOV3/DINOV3-pth/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
    args.output_root = "./fm_feature"
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
