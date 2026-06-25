import argparse
import textwrap
from pathlib import Path


INSTRUCTIONS = """
ScanObjectNN does not provide a stable single official archive URL across mirrors.

Recommended steps:
1. Download the PB_T50_RS split from the official project page:
   https://hkust-vgd.github.io/scanobjectnn/
2. Unpack the archive under the target root directory.
3. Ensure the train/test files are available as `.h5`, `.npz`, or `.pt`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ScanObjectNN for HRC-Net.")
    parser.add_argument("--root", type=Path, default=Path("datasets/scanobjectnn"))
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    readme_path = args.root / "README.txt"
    readme_path.write_text(textwrap.dedent(INSTRUCTIONS).strip() + "\n", encoding="utf-8")
    print(f"ScanObjectNN target folder created at: {args.root.resolve()}")
    print(f"Manual preparation instructions written to: {readme_path.resolve()}")


if __name__ == "__main__":
    main()
