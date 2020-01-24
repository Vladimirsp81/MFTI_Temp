import argparse
import os
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch import optim
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from PIL import Image
from Generator_Descriminator import *

parser = argparse.ArgumentParser(description='Pix2Pix for edges-shoes')

# Общие параметры
parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders train, val, etc)')
parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')

# Предобработка данных
parser.add_argument('--no_resize_or_crop', action='store_true', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

# Гипер параметры обучения и оптимизатора
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5)  # начальное значение моментума для Adam
parser.add_argument('--beta2', type=float, default=0.999)  # верхнее значение моментума для Adam
parser.add_argument('--lambda_A', type=float, default=100.0)

# Разное
parser.add_argument('--model_path', type=str, default='./MFTI_Temp/models')  # путь временного сохранения весов модели
parser.add_argument('--sample_path', type=str, default='./MFTI_Temp/results')  # Путь сохранения образцов изображений 
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--sample_step', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=2)

##### Вспомогательная функция для загрузки данных и их предобработки
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

##### Вспомогательная функция для обучения на GPU
def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

##### Вспомогательная функция для Math
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

##### Вспомогательная функция для определения GAN Loss
def GAN_Loss(input, target, criterion):
    if target == True:
        tmp_tensor = torch.FloatTensor(input.size()).fill_(1.0)
        labels = Variable(tmp_tensor, requires_grad=False)
    else:
        tmp_tensor = torch.FloatTensor(input.size()).fill_(0.0)
        labels = Variable(tmp_tensor, requires_grad=False)

    if torch.cuda.is_available():
        labels = labels.cuda()

    return criterion(input, labels)

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
                                  num_workers=args.num_workers,
                                  drop_last=True)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)

    # Объявление сетей
    generator = Generator(args.batchSize)
    #generator.load_state_dict(torch.load(g_path))
    #generator.eval()

    discriminator = Discriminator(args.batchSize)

    # Выбор метрик
    criterionGAN = nn.BCELoss()
    criterionL1 = nn.L1Loss()

    # Задание оптимизаторов для сетей
    g_optimizer = optim.Adam(generator.parameters(), args.lr, [args.beta1, args.beta2])
    d_optimizer = optim.Adam(discriminator.parameters(), args.lr, [args.beta1, args.beta2])

    if torch.cuda.is_available():
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    """Train generator and discriminator."""
    total_step = len(data_loader) # Для принта лога
    for epoch in range(args.num_epochs):
        for i, sample in enumerate(data_loader):

            AtoB = args.which_direction == 'AtoB'
            input_A = sample['A']
            input_B = sample['B']

            # ===================== Обучение Дескриминатора =====================#
            discriminator.zero_grad()

            real_A = to_variable(input_A)
            fake_B = generator(real_A)
            real_B = to_variable(input_B)

            # d_optimizer.zero_grad()

            pred_fake = discriminator(real_A, fake_B)
            loss_D_fake = GAN_Loss(pred_fake, False, criterionGAN)

            pred_real = discriminator(real_A, real_B)
            loss_D_real = GAN_Loss(pred_real, True, criterionGAN)

            # Определяем общий loss
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward(retain_graph=True)
            d_optimizer.step()

            # ===================== Обучение Генератора =====================#
            generator.zero_grad()

            pred_fake = discriminator(real_A, fake_B)
            loss_G_GAN = GAN_Loss(pred_fake, True, criterionGAN)

            loss_G_L1 = criterionL1(fake_B, real_B)

            loss_G = loss_G_GAN + loss_G_L1 * args.lambda_A
            loss_G.backward()
            g_optimizer.step()

            # принт лога обучения
            if (i + 1) % args.log_step == 0:
                print('Epoch [%d/%d], BatchStep[%d/%d], D_Real_loss: %.4f, D_Fake_loss: %.4f, G_loss: %.4f, G_L1_loss: %.4f'
                      % (epoch + 1, args.num_epochs, i + 1, total_step, loss_D_real.data, loss_D_fake.data, loss_G_GAN.data, loss_G_L1.data))

            # Сохранение изображений
            if (i + 1) % args.sample_step == 0:
                res = torch.cat((torch.cat((real_A, fake_B), dim=3), real_B), dim=3)
                torchvision.utils.save_image(denorm(res.data), os.path.join(args.sample_path, 'Generated-%d-%d.png' % (epoch + 1, i + 1)))

        # Сохранение весов моделей на каждой эпохе
        g_path = os.path.join(args.model_path, 'generator-%d.pkl' % (epoch + 1))
        torch.save(generator.state_dict(), "/content/drive/My Drive/Colab_Notebooks/Temp")
        g_path_D = os.path.join(args.model_path, 'discriminator-%d.pkl' % (epoch + 1))
        torch.save(discriminator.state_dict(), "/content/drive/My Drive/Colab_Notebooks/Temp")

if __name__ == "__main__":
    main()