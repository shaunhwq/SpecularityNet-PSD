import os
import argparse
import math

import cv2
import torch
import numpy as np
from tqdm import tqdm

import models.arch


def pre_process(image: np.array, device: str, input_size: int = 256) -> torch.tensor:
    """
    Pre-process and sending image to device before feeding it to the model.

    :param image: BGR image read with OpenCV's cv2.imread()
    :param device: device name e.g. 'cpu', 'cuda:0', 'cuda:1'
    :returns: An input tensor used by RefinedNet
    """
    # Get final shape, while maintaining aspect ratio (e.g. shortside resize shape)
    if m_img.shape[0] < m_img.shape[1]:
        size = (int(input_size * m_img.shape[1] / m_img.shape[0]), input_size)
    else:
        size = (input_size, int(input_size * m_img.shape[0] / m_img.shape[1]))

    # Slightly more efficient image resize by using PyrDown? But has gaussian blurring sooo...
    if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]):
        scale = int(math.log2(min(m_img.shape[0] / size[1], m_img.shape[1] / size[0])))
        for _ in range(0, scale):
            m_img = cv2.pyrDown(m_img)
        if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]):
            m_img = cv2.resize(m_img, size, cv2.INTER_AREA)

    # Conversion to RGB and then to tensor
    rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    formatted_rgb_image = np.ascontiguousarray(rgb_image.astype(np.float32) / 255.0).transpose((2, 0, 1))
    input_tensor = torch.from_numpy(formatted_rgb_image).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    return input_tensor


def post_process(model_output: dict) -> np.array:
    """
    Process the model's output (RefinedNet). We only need 'refined' though

    :param model_output: A dictionary with keys 'refined', 'coarse', 'detect'
    :returns: Output numpy array image with the color format B, G, R
    """
    image_tensor = model_output['refined'].squeeze()
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    bgr_image = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
    print(bgr_image)
    return bgr_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Path to input folder containing images")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path to output folder")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Device to use e.g. 'cuda:0', 'cuda:1', 'cpu'")
    parser.add_argument("-w", "--weights", type=str, help="Path to weights", default="")
    parser.add_argument("-s", "--model_input_size", type=int, default=256, help="Shortside size to resize to before feeding into model.")
    args = parser.parse_args()

    # Prepare output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Check that model weights exist
    is_file = os.path.isfile(args.weights)
    is_pt_file = os.path.splitext(args.weights)[-1] in [".pt", ".pth"]
    assert is_file and is_pt_file, f"Weights provided is invalid or is not a file {args.weights}"

    # Load model weights
    model = models.arch.refined(in_channels=3, out_channels=3)
    state_dict = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state_dict['icnn'])
    model.to(args.device)
    model.eval()

    image_paths = [os.path.join(args.input_dir, file_path) for file_path in os.listdir(args.input_dir) if file_path[0] != "."]

    for image_path in tqdm(image_paths, total=len(image_paths), desc="Processing Spec-Net refined..."):
        in_image = cv2.imread(image_path)
        assert len(in_image.shape) == 3, "Should be a color image"

        # Process the image
        in_tensor = pre_process(in_image, args.device, args.model_input_size)
        with torch.no_grad():
            out_tensor = model(in_tensor)

        out_image = post_process(out_tensor)

        # TODO: Need to resize back to original image's size.
        #   do we want to use cv2 resize only? or do we wanna do pyrUp (following their example)

        # Write image? but might affect image quality if we write to jpg instead of png
        cv2.imshow("frame", out_image)
        key = cv2.waitKey(0)
        if key & 255 == 27:
            break

    cv2.destroyAllWindows()
