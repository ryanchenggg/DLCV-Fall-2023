import os
from PIL import Image

def concatenate_images(image_paths, output_path):
    """
    Concatenates a list of images into a single image.

    Args:
    image_paths (list of str): List of image file paths to concatenate.
    output_path (str): Path to save the concatenated image.
    """
    # Load images with name called depth_xxx.png
    depth_images = [os.path.join(image_paths, f) for f in os.listdir(image_paths) if f.startswith('depth_') and f.endswith('.png')]

    rows = 4

    # Load depth images
    images = []
    for path in depth_images:
        try:
            img = Image.open(path)
            images.append(img)
        except IOError:
            print(f"Could not open image file {path}")

     # Calculate the number of images per row
    images_per_row = len(images) // rows
    if len(images) % rows != 0:
        images_per_row += 1

    # Ensure all images are the same size
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)

    # Create a new blank image with the correct size
    total_width = max_width * images_per_row
    total_height = max_height * rows
    new_im = Image.new('RGB', (total_width, total_height))

    # Paste each image into the new image
    x_offset = 0
    y_offset = 0
    for i, im in enumerate(images):
        if i > 0 and i % images_per_row == 0:
            x_offset = 0
            y_offset += max_height

        new_im.paste(im, (x_offset, y_offset))
        x_offset += max_width

    # Save the concatenated image
    new_im.save(output_path)

    print(f"Saved concatenated image to {output_path}")

# Example usage
# Assuming you have a list of image paths named image_paths
# image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', ..., 'path/to/image10.jpg']
# concatenate_images(image_paths, 'concatenated_output.jpg')
# Note: Replace 'path/to/imageX.jpg' with actual file paths and ensure you have 10 images in the list.
if __name__ == '__main__':
    image_paths = '/home/ryan0309/dlcv/dlcv-fall-2023-hw4-ryanchenggg/nerf_pl/results/klevr/test'
    output_path = '/home/ryan0309/dlcv/dlcv-fall-2023-hw4-ryanchenggg/nerf_pl/results/klevr/test/depth_concat.png'
    concatenate_images(image_paths, output_path)