import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


class CIFAR10Policy_autocontrast(object):
    def __init__(self, fillcolor=(128, 128, 128), magnitude=1):
        self.policies = [SubPolicy(1.0, "autocontrast", magnitude)]
        self.magnitude = magnitude

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class CIFAR10Policy_brightness(object):
    def __init__(self, fillcolor=(128, 128, 128), magnitude=1):
        self.policies = [SubPolicy(1.0, "brightness", magnitude)]
        self.magnitude = magnitude

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class CIFAR10Policy_color(object):
    def __init__(self, fillcolor=(128, 128, 128), magnitude=1):
        self.policies = [SubPolicy(1.0, "color", magnitude)]
        self.magnitude = magnitude

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class CIFAR10Policy_contrast(object):
    def __init__(self, fillcolor=(128, 128, 128), magnitude=1):
        self.policies = [SubPolicy(1.0, "contrast", magnitude)]
        self.magnitude = magnitude

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class CIFAR10Policy_equalize(object):
    def __init__(self, fillcolor=(128, 128, 128), magnitude=1):
        self.policies = [SubPolicy(1.0, "equalize", magnitude)]
        self.magnitude = magnitude

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class CIFAR10Policy_invert(object):
    def __init__(self, fillcolor=(128, 128, 128), magnitude=1):
        self.policies = [SubPolicy(1.0, "invert", magnitude)]
        self.magnitude = magnitude

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class CIFAR10Policy_posterize(object):
    def __init__(self, fillcolor=(128, 128, 128), magnitude=1):
        self.policies = [SubPolicy(1.0, "posterize", magnitude)]
        self.magnitude = magnitude

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class CIFAR10Policy_rotate(object):
    def __init__(self, fillcolor=(128, 128, 128), magnitude=1):
        self.policies = [SubPolicy(1.0, "rotate", magnitude)]
        self.magnitude = magnitude

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class CIFAR10Policy_sharpness(object):
    def __init__(self, fillcolor=(128, 128, 128), magnitude=1):
        self.policies = [SubPolicy(1.0, "sharpness", magnitude)]
        self.magnitude = magnitude

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class CIFAR10Policy_shearX(object):
    def __init__(self, fillcolor=(128, 128, 128), magnitude=1):
        self.policies = [SubPolicy(1.0, "shearX", magnitude)]
        self.magnitude = magnitude

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class CIFAR10Policy_shearY(object):
    def __init__(self, fillcolor=(128, 128, 128), magnitude=1):
        self.policies = [SubPolicy(1.0, "shearY", magnitude)]
        self.magnitude = magnitude

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class CIFAR10Policy_solarize(object):
    def __init__(self, fillcolor=(128, 128, 128), magnitude=1):
        self.policies = [SubPolicy(1.0, "solarize", magnitude)]
        self.magnitude = magnitude

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class CIFAR10Policy_translateX(object):
    def __init__(self, fillcolor=(128, 128, 128), magnitude=1):
        self.policies = [SubPolicy(1.0, "translateX", magnitude)]
        self.magnitude = magnitude

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class CIFAR10Policy_translateY(object):
    def __init__(self, fillcolor=(128, 128, 128), magnitude=1):
        self.policies = [SubPolicy(1.0, "translateY", magnitude)]
        self.magnitude = magnitude

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SubPolicy(object):
    def __init__(
            self,
            p1,
            operation1,
            magnitude_idx1,
            # p2,
            # operation2,
            # magnitude_idx2,
            fillcolor=(128, 128, 128),
    ):
        ranges = {
            "shearx": np.linspace(0, 0.3, 10),
            "sheary": np.linspace(0, 0.3, 10),
            "translatex": np.linspace(0, 150 / 331, 10),
            "translatey": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            # "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "posterize": np.array([8, 8, 7, 6, 5, 4, 3, 2, 1, 1]),  # newly added to make difference
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(
                rot, Image.new("RGBA", rot.size, (128,) * 4), rot
            ).convert(img.mode)

        func = {
            "shearx": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "sheary": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "translatex": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor,
            ),
            "translatey": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor,
            ),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img),
        }

        self.p1 = p1
        self._operation1_name = operation1
        self.operation1 = func[operation1.lower()]
        self.magnitude1 = ranges[operation1.lower()][magnitude_idx1]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)

        return img

    def __repr__(self):
        return f"{self._operation1_name} with p:{self.p1} and magnitude:{self.magnitude1} \t" \
            # f"{self._operation2_name} with p:{self.p2} and magnitude:{self.magnitude2} \n"
