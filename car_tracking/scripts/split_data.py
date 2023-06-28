import argparse
import random
import shutil
from pathlib import Path

IMG_EXTENSIONS = ("png", "jpg", "jpeg")
random.seed(0)


def save_samples(img_list: list[Path], indexes: list[int], save_dir: Path) -> None:
    """
    Save given samples from the list of images

    Attributes:
        img_list: list[Path] - list of paths to images
        indexes:  list[int] - list of indexes of samples to save
        save_dir: Path - saving directory
    """
    for idx in indexes:
        shutil.copyfile(img_list[idx], str(save_dir / "images" / img_list[idx].name))
        src_path = str(img_list[idx]).replace('images', 'labels')
        extension = img_list[idx].suffix
        src_path = src_path.replace(extension, '.txt')
        shutil.copyfile(src_path, str(save_dir / "labels" / (img_list[idx].stem + ".txt")))


def parse_args() -> argparse.Namespace:
    """
    Parse arguments (paths to the original and resulting datasets)

    Returns:
        argparse.Namespace object with parsed arguments
    """
    parser = argparse.ArgumentParser(description="Description of your script")

    parser.add_argument(
        'path',
        type=Path,
        help="path to the original dataset"
    )
    parser.add_argument(
        'save_dir',
        type=Path,
        help="path to the resulting dataset"
    )
    return parser.parse_args()


def main():
    """
    Randomly split the data into training, validation and testing sets
    70% - training dataset, 15% - validation dataset, 15% - testing dataset
    """
    args = parse_args()

    img_list = [path for path in sorted(args.path.glob('images/*')) if path.suffix[1:].lower() in IMG_EXTENSIONS]
    indexes = list(range(len(img_list)))
    random.shuffle(indexes)
    train_idx = [i for i in indexes[:int(0.7 * len(img_list))]]
    val_idx = [i for i in indexes[int(0.7 * len(img_list)):int(0.85 * len(img_list))]]
    test_idx = [i for i in indexes[int(0.85 * len(img_list)):]]

    Path.mkdir(args.save_dir, exist_ok=True)
    Path.mkdir(args.save_dir / 'train', exist_ok=True)
    Path.mkdir(args.save_dir / 'train' / 'labels', exist_ok=True)
    Path.mkdir(args.save_dir / 'train' / 'images', exist_ok=True)
    Path.mkdir(args.save_dir / 'val', exist_ok=True)
    Path.mkdir(args.save_dir / 'val' / 'labels', exist_ok=True)
    Path.mkdir(args.save_dir / 'val' / 'images', exist_ok=True)
    Path.mkdir(args.save_dir / 'test', exist_ok=True)
    Path.mkdir(args.save_dir / 'test' / 'labels', exist_ok=True)
    Path.mkdir(args.save_dir / 'test' / 'images', exist_ok=True)
    save_samples(img_list, train_idx, args.save_dir / 'train')
    save_samples(img_list, val_idx, args.save_dir / 'val')
    save_samples(img_list, test_idx, args.save_dir / 'test')


if __name__ == '__main__':
    main()
