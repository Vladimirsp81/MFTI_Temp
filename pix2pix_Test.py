import argparse
import os
import random
import torch
import torchvision
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from PIL import Image
from Generator_Descriminator import Generator

parser = argparse.ArgumentParser(description='Pix2Pix for edges-shoes')

# Общие параметры
parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders train, val, etc)')
parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')

# Предобработка данных
parser.add_argument('--no_resize_or_crop', action='store_true', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

# Гипер параметры обучения
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batchSize', type=int, default=1, help='test Batch size')

# Разное
parser.add_argument('--model_path', type=str, default='./MFTI_Temp/models')
parser.add_argument('--sample_path', type=str, default='./MFTI_Temp/test_results')

##### Вспомогательная функция для загрузки данных и их предобработки
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

class ImageFolder(data.Dataset):
    def __init__(self, opt):
        # os.listdir функция, возвращающая перечень папок
        self.root = opt.dataroot
        self.no_resize_or_crop = opt.no_resize_or_crop
        self.no_flip = opt.no_flip
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5),
                                                                  (0.5, 0.5, 0.5))])
        self.dir_AB = os.path.join(opt.dataroot, 'train')  # ./edges2shoes/train
        self.image_paths = list(map(lambda x: os.path.join(self.dir_AB, x), os.listdir(self.dir_AB)))

    def __getitem__(self, index):
        AB_path = self.image_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        AB = self.transform(AB)
        w_total = AB.size(2)
        w = int(w_total / 2)

        A = AB[:, :256, :256]
        B = AB[:, :256, w:w + 256]

        return {'A': A, 'B': B}

    def __len__(self):
        return len(self.image_paths)

##### Вспомогательная функция для тренировки на GPU
def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

##### Вспомогательная функция для Math
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

######################### Функция для обучения
def main():
    # Объявление загрузчика
    cudnn.benchmark = True
    global args
    args = parser.parse_args()
    print(args)

    dataset = ImageFolder(args)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=args.batchSize,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)

    g_path = os.path.join(args.model_path, 'generator-%d.pkl' % (args.num_epochs))

    # Объявляем сеть и загружаем веса предобученной модели Генератора
    generator = Generator(args.batchSize)
    generator.load_state_dict(torch.load(g_path))
    generator.eval()

    if torch.cuda.is_available():
        generator = generator.cuda()

    total_step = len(data_loader) # Для принта лога
    for i, sample in enumerate(data_loader):
        AtoB = args.which_direction == 'AtoB'
        input_A = sample['A']
        input_B = sample['B']

        real_A = to_variable(input_A)
        fake_B = generator(real_A)
        real_B = to_variable(input_B)

        # принт лога
        print('Validation[%d/%d]' % (i + 1, total_step))
        # сохраняем полученные изображения
        res = torch.cat((torch.cat((real_A, fake_B), dim=3), real_B), dim=3)
        torchvision.utils.save_image(denorm(res.data), os.path.join(args.sample_path, 'Generated-%d.png' % (i + 1)))

if __name__ == "__main__":
    main()