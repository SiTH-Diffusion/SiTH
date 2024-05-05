"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import os
import numpy as np
import PIL.Image as Image
import argparse
import cv2


def main(args):

    # Make sure the output path exists
    os.makedirs(args.output_path, exist_ok=True)

    # Get the list of images
    img_list = [os.path.join(args.input_path, x) for x in sorted(os.listdir(args.input_path)) if x.endswith(('.png'))]

    for img_path in img_list:

        # We assue the image is RGBA
        try:
            img = Image.open(img_path)
            assert img.mode == 'RGBA'
        except:
            print(f"Error: {img_path} is not RGBA image")
            continue

        canvas = Image.new('RGBA', (args.size, args.size))

        img_width, img_height = img.size
        target_height = int( args.size * args.ratio)

        resize_ratio = target_height / img_height
        target_width = int(img_width * resize_ratio)

        img_resized = img.resize((target_width, target_height))

        rgb = np.asarray(img_resized)[..., :3]
        mask = np.asarray(img_resized)[..., 3]

        # Erode the mask to remove the noisy borders
        new_mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=1)

        img_resized = Image.fromarray(np.concatenate([rgb, new_mask[..., None]], axis=-1))

        #place resized image on the center
        x_offset = int((args.size - target_width) // 2)
        y_offset = int((args.size - target_height) // 2)

        canvas.paste(img_resized, (x_offset, y_offset))


        canvas.save(os.path.join(args.output_path, os.path.basename(img_path)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a square images given RGBA images')

    parser.add_argument("-i", "--input-path", default='./data/examples/rgba', type=str, help="Input RGBA path")
    parser.add_argument("-o", "--output-path", default='./data/examples/images', type=str, help="Output path")
    parser.add_argument("--ratio", default=0.85, type=float, help="Ratio of the image height to the canvas height")
    parser.add_argument("--size", default=1024, type=int, help="Canvas size")

    main(parser.parse_args())

