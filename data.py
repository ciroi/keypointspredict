import numpy as np
import cv2
import math
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import itertools

train_boarder = 112
LABEL_PATH = "label/"
IMAGE_PATH = "image/"


def channel_norm(img):
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels, mean, std


def parse_line(line):
    line_parts = line.strip().split()
    img_name = line_parts[0]
    rect = list(map(int, list(map(float, line_parts[1:5]))))
    landmarks = list(map(float, line_parts[5: len(line_parts)]))
    return img_name, rect, landmarks


def adjustPt(landmarks, preShape, newShape):
    x_rate = newShape[1]/preShape[1]
    y_rate = newShape[0]/preShape[0]
    rate = np.array([[x_rate, y_rate]], dtype=np.float32)
    length = len(landmarks)
    landmarks = landmarks.reshape(-1, 2)
    landmarks = landmarks * rate
    return landmarks.reshape(length)

class Normalize(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image_resize = np.asarray(
                            image.resize((train_boarder, train_boarder), Image.BILINEAR),
                            dtype=np.float32)       # Image.ANTIALIAS)
        landmarks = adjustPt(landmarks, image.size, image_resize.shape)
        image, mean, std = channel_norm(image_resize)
        return {'image': image,
                'landmarks': landmarks,
                'mean':mean,
                'std':std
                }


class ToTensor(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        image = np.expand_dims(image, axis=0)
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks),
                'mean': sample['mean'],
                'std': sample['std']
                }

def random_affine(img, targets=(), degrees=90, translate=.1, scale=.1, shear=10, border=(0, 0)):
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    random = np.random
    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[1] + border[1]  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[0] + border[0]  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        num = n // 2
        xy = np.ones((num, 3))
        xy[:, :2] = targets.reshape(num, 2)
        xy = (xy @ M.T)[:, :2].reshape(num*2)

    return img.astype(np.float32), xy.astype(np.float32)

class FaceLandmarksDataset(Dataset):
    def __init__(self, src_lines, imagePath, transform=None):
        self.lines = src_lines
        self.transform = transform
        self.imagePath = imagePath

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_name, rect, landmarks = parse_line(self.lines[idx])
        clss = landmarks[-1]
        landmarks = landmarks[:-1]
        
        img = Image.open(self.imagePath+img_name).convert('L')     
        img_crop = img.crop(tuple(rect))            
        landmarks = np.array(landmarks).astype(np.float32)
        landmarks1 = landmarks.reshape(-1, 2)
        rect = np.array(rect[:2]).reshape(-1, 2).astype(np.float32)
        landmarks1 -= rect
        landmarks = landmarks1.reshape(len(landmarks))
		
        sample = {'image': img_crop, 'landmarks': landmarks}
        sample = self.transform(sample)

        if np.random.random() < 0.5:
            sample['image'], sample['landmarks'] = random_affine(np.squeeze(sample['image'].numpy()), sample['landmarks'])
            sample = ToTensor()(sample)

        sample['landmarks'] = torch.cat([sample['landmarks'], torch.tensor([clss])], 0)
        return sample


def load_data(phase, labelPath, imagePath):
    data_file = phase + '.txt'
    with open(labelPath+data_file) as f:
        lines = f.readlines()
    if phase == 'Train' or phase == 'train':
        tsfm = transforms.Compose([
            Normalize(),               
            ToTensor()]                 
        )
    else:
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor()
        ])
    data_set = FaceLandmarksDataset(lines, imagePath, transform=tsfm)
    return data_set


def get_train_test_set(labelPath=LABEL_PATH, imagePath=IMAGE_PATH):
    train_set = load_data('train', labelPath, imagePath)
    valid_set = load_data('test', labelPath, imagePath)
    return train_set, valid_set


if __name__ == '__main__':
    train_set = load_data('train', labelPath=LABEL_PATH, imagePath=IMAGE_PATH)
    idx = np.random.randint(1,len(train_set))
    for idx in range(len(train_set)):
        sample = train_set[idx]
        landmarks = sample['landmarks']
        image = sample['image'] * sample['std'] +sample['mean']
        image = np.squeeze(image.numpy().astype(np.uint8))
        for pt in [landmarks[i:i+2]for i in range(0, len(landmarks), 2)]:
                cv2.circle(image, (int(float(pt[0])), int(float(pt[1]))), radius=2, color=(255, 255, 0), thickness=-2)
        cv2.imshow("src1", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()