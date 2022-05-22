# Reference: autoaugment pytorch repo
# https://github.com/DeepVoltaire/AutoAugment

import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

class CIFAR10Policy_autocontrast(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [SubPolicy(1.0, "autocontrast", 5)]
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"

class CIFAR10Policy_brightness(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [SubPolicy(1.0, "brightness", 5)]
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"

class CIFAR10Policy_color(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [SubPolicy(1.0, "color", 5)]
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"

class CIFAR10Policy_contrast(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [SubPolicy(1.0, "contrast", 5)]
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"

class CIFAR10Policy_equalize(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [SubPolicy(1.0, "equalize", 5)]
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"

class CIFAR10Policy_invert(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [SubPolicy(1.0, "invert", 5)]
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"
    
class CIFAR10Policy_posterize(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [SubPolicy(1.0, "posterize", 9)]
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"
    
class CIFAR10Policy_rotate(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [SubPolicy(1.0, "rotate", 5)]
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"
   
class CIFAR10Policy_sharpness(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [SubPolicy(1.0, "sharpness", 5)]
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"
    
class CIFAR10Policy_shearX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [SubPolicy(1.0, "shearX", 5)]
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"
    
class CIFAR10Policy_shearY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [SubPolicy(1.0, "shearY", 5)]
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"

class CIFAR10Policy_solarize(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [SubPolicy(1.0, "solarize", 5)]
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"

class CIFAR10Policy_translateX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [SubPolicy(1.0, "translateX", 5)]
    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)
    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"
    
class CIFAR10Policy_translateY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [SubPolicy(1.0, "translateY", 5)]
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
            "posterize": np.array([8,8,7,6,5,4,3,2,1,1]), #newly added to make difference
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
        # self.p2 = p2
        # self._operation2_name = operation2
        # self.operation2 = func[operation2.lower()]
        # self.magnitude2 = ranges[operation2.lower()][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        # if random.random() < self.p2:
        #     img = self.operation2(img, self.magnitude2)
        return img

    def __repr__(self):
        return f"{self._operation1_name} with p:{self.p1} and magnitude:{self.magnitude1} \t" \
               # f"{self._operation2_name} with p:{self.p2} and magnitude:{self.magnitude2} \n"


class RandAugment:
    """
    # randaugment is adaptived from UDA tensorflow implementation:
    # https://github.com/jizongFox/uda
    """

    @classmethod
    def get_trans_list(cls):
        trans_list = [
            'Invert', 'Sharpness', 'AutoContrast', 'Posterize',
            'ShearX', 'TranslateX', 'TranslateY', 'ShearY', 'Rotate',
            'Equalize', 'Contrast', 'Color', 'Solarize', 'Brightness']
        return trans_list

    @classmethod
    def get_rand_policies(cls):
        op_list = []
        for trans in cls.get_trans_list():
            for magnitude in range(1, 10):
                op_list += [(0.5, trans, magnitude)]
        policies = []
        for op_1 in op_list:
            for op_2 in op_list:
                policies += [[op_1, op_2]]
        return policies

    def __init__(self) -> None:
        super().__init__()
        self._policies = self.get_rand_policies()

    def __call__(self, img):
        randomly_chosen_policy = self._policies[random.randint(0, len(self._policies) - 1)]
        policy = SubPolicy(*randomly_chosen_policy[0], *randomly_chosen_policy[1])
        return policy(img)

    def __repr__(self):
        return "Random Augment Policy"
