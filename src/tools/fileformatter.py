import os
import glob
import argparse
from tqdm import tqdm

def rename_images(directory: str,
                  extension: str):

    images = sorted(glob.glob(os.path.join(directory, extension)))

    for i, img in tqdm(enumerate(images), desc='Renaming images'):
        new_name = os.path.join(directory, f"{i:04d}{os.path.splitext(img)[1]}")
        os.rename(img, new_name)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for renaming images in a directory.')
    parser.add_argument('--directory', type=str, help='The directory containing the images.')
    parser.add_argument('--extension', type=str, default='*.jpg', help='The extension of the images to rename.')
    
    directory = parser.parse_args().directory
    extension = parser.parse_args().extension
    print(f'Renaming images in {directory} with extension {extension}...')
    rename_images(directory, extension)