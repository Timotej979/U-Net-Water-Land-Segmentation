import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from numpy import asarray, newaxis
from cv2 import cvtColor, COLOR_BGR2GRAY

from PIL import Image

from torchvision.datasets.vision import VisionDataset

# IMAGE LOADERS

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)



def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

# Findclasses for the ImageLoader class. Not used.
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

# Makedataset for the ImageLoader class. Not used.
def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        image_only_transform (callable, optional): A function/transform that takes
            in the image and transforms it, not transforming the mask. The 'image_only_transform'
            is performed before 'transform'.
        transform (callable, optional): A albumentations transform function/transform 
            that takes in a sample and returns a transformed version. Applies the transform 
            to both image and provided mask. This is where ToTensor() should be applied.
            E.g, ``albumentations.RandomCrop`` for images and masks.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any] = default_loader,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        image_only_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=image_only_transform)

        samples = self.make_dataset(self.root, extensions, is_valid_file)

        self.image_only_transform = image_only_transform

        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        # self.targets = [s[1] for s in samples]
        self.masks = [s[1] for s in samples]

        # Use accimage for better performance

    @staticmethod
    def make_dataset(

        directory: str,
        # class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, str]]:
        """Generates a list of samples of a form (path_to_sample, path_to_mask).

        This function is modified to work with the laboro_tomato dataset.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, str]]: samples of a form (path_to_sample, path_to_mask)
        """
        # if class_to_idx is None:
        #     # prevent potential bug since make_dataset() would use the class_to_idx logic of the
        #     # find_classes() function, instead of using that of the find_classes() method, which
        #     # is potentially overridden and thus could have a different logic.
        #     raise ValueError("The class_to_idx parameter cannot be None.")
        # return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

        img_dir = os.path.join(directory, 'images')
        mask_dir = os.path.join(directory, 'masks')

        # if (extensions == None) ^ (is_valid_file == None):
        #     raise ValueError("both extensions and is_valid_file should not be passed")

        img_paths = []

        # Get list of images
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                img_path = os.path.relpath(os.path.join(root, file), img_dir)
                img_paths.append(img_path)
                
        samples = []

        # For each image, find its corresponding mask
        for img in img_paths:
            img_path = os.path.join(img_dir, img)
            mask_path = os.path.join(mask_dir, img.replace('.jpg', '.png'))

            # Check if file exists
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Cant find {img_path}")
            elif not os.path.isfile(mask_path):
                raise FileNotFoundError(f"Cant find {mask_path}, based on image path {img_dir}")

            # Check for file extensions
            if not is_image_file(img_path):
                raise ValueError(f"File not valid: {img_path}")
            elif not is_image_file(mask_path):
                raise ValueError(f"File not valid: {mask_path}")
            
            samples.append((img_path, mask_path))

        # If list still empty, raise error
        if len(samples) == 0:
            raise ValueError(f"No images found")

        return samples

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img, mask, img_path, mask_path) where target is class_index of the target class.
        """
        # path, target = self.samples[index]
        img_path, mask_path = self.samples[index]

        # Read image and convert to numpy array
        img = asarray(self.loader(img_path))
        mask = asarray(self.loader(mask_path))

        # Convert mask to single channel grayscale float [0. or 1.]
        mask = cvtColor(mask, COLOR_BGR2GRAY)/255
        mask = mask[..., newaxis]


        if self.image_only_transform is not None:
            transformed = self.image_only_transform(image=img)
            img = transformed['image']
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     mask = self.target_transform(mask)

        return img, mask, img_path, mask_path

    def __len__(self) -> int:
        return len(self.samples)




class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
