import argparse
import urllib.request
import shutil
from pathlib import Path
from zipfile import ZipFile

SHAPENET_PART_URL = "https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip"


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    urllib.request.urlretrieve(url, target)


def extract(archive_path: Path, output_dir: Path) -> None:
    with ZipFile(archive_path, "r") as zf:
        zf.extractall(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ShapeNet Part for HRC-Net.")
    parser.add_argument("--root", type=Path, default=Path("datasets/shapenetpart"))
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    archive_path = args.root / "shapenet_part_seg_hdf5_data.zip"
    download(SHAPENET_PART_URL, archive_path)
    extract(archive_path, args.root)

    extracted = args.root / "hdf5_data"
    if extracted.exists():
        for item in extracted.iterdir():
            destination = args.root / item.name
            if destination.exists():
                continue
            shutil.move(str(item), str(destination))

    print(f"ShapeNet Part prepared at: {args.root.resolve()}")


if __name__ == "__main__":
    main()