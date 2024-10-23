
from typing import Any, Optional, Callable, Tuple

from PIL import Image

from dinov2.data.datasets.extended import ExtendedVisionDataset


class RecursiveImageDataset(ExtendedVisionDataset):
    def __init__(self,
                 root: list[str],  # Change type to List[str]
                 verify_images: bool = False,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:

        super().__init__(root, transforms, transform, target_transform)

        image_paths = []

        # Iterate over each file path in the root list
        with open(root, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            image_paths.append(line)

        invalid_images = set()
        if verify_images:
            print("Verifying images. This ran at ~100 images/sec/cpu for me. Probably depends heavily on disk perf.")
            invalid_images = set(verify_images(image_paths))
            print("Skipping invalid images:", invalid_images)

        self.image_paths = [p for p in image_paths if p not in invalid_images]
        print(f"Total images: {len(self.image_paths)}")

    def get_image_data(self, index: int) -> bytes:  # should return an image as an array

        image_path = self.image_paths[index]
        img = Image.open(image_path).convert(mode="RGB")

        return img

    def get_target(self, index: int) -> Any:
        return 0

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_paths)