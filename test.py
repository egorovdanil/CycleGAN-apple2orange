"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML

print("\nCheck result:")

from PIL import Image
import imagehash
from math import log10, sqrt
import cv2
import numpy as np

class bcolors:
    WARNING = '\033[93m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

ref_path = "./results/apple2orange_cyclegan/test_latest/images/"
res_path = "./results_ref/apple2orange_cyclegan/test_latest/images/"

imaeges = ['apple2orange_fake_A.png', 'apple2orange_fake_B.png']

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

class CompareImage(object):

    def __init__(self, image_1_path, image_2_path):
        self.minimum_commutative_image_diff = 1
        self.image_1_path = image_1_path
        self.image_2_path = image_2_path

    def compare_image(self):
        image_1 = cv2.imread(self.image_1_path, 0)
        image_2 = cv2.imread(self.image_2_path, 0)
        commutative_image_diff = self.get_image_difference(image_1, image_2)

        if commutative_image_diff < self.minimum_commutative_image_diff:
            #print("Matched")
            return commutative_image_diff
        return 10000 #random failure value

    @staticmethod
    def get_image_difference(image_1, image_2):
        first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
        second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])

        img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
        img_template_probability_match = cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
        img_template_diff = 1 - img_template_probability_match

        # taking only 10% of histogram diff, since it's less accurate than template method
        commutative_image_diff = (img_hist_diff / 10) + img_template_diff
        return commutative_image_diff

Passed = True

print(bcolors.WARNING + '[TEST 1]' + bcolors.ENDC + ' Compare resulting images with reference by histogram difference:')
for image in imaeges:
    compare_image = CompareImage(res_path + image, ref_path + image)
    image_difference = compare_image.compare_image()
    if image_difference != 10000:
        print('Check ' + image + ': ' + f'images are similar. Difference: {image_difference.round(4)}')
    else:
        print('Check ' + image + ': ' + 'images are not similar.')
        Passed = False

if (Passed):
     print(bcolors.OKGREEN + 'Test PASSED' + bcolors.ENDC)
else:
     print(bcolors.FAIL + 'Test FAILED' + bcolors.ENDC)


Passed = True

print(bcolors.WARNING + '[TEST 2]' + bcolors.ENDC + ' Compare resulting images with reference by hash:')
for image in imaeges:
    hash0 = imagehash.average_hash(Image.open(res_path + image))
    hash1 = imagehash.average_hash(Image.open(ref_path + image))
    cutoff = 5  # maximum bits that could be different between the hashes.
    hash_diff = hash0 - hash1
    if hash_diff < cutoff:
        print('Check ' + image + ': ' + f'images are similar. Difference: {hash_diff}')
    else:
        print('Check ' + image + ': ' + f'images are not similar. Difference: {hash_diff}')
        Passed = False

if (Passed):
     print(bcolors.OKGREEN + 'Test PASSED' + bcolors.ENDC)
else:
     print(bcolors.FAIL + 'Test FAILED' + bcolors.ENDC)


Passed = True

real_images = ['apple2orange_real_A.png', 'apple2orange_real_B.png']
reconstructed_images = ['apple2orange_rec_A.png', 'apple2orange_rec_B.png']

threshold = 27.0

print(bcolors.WARNING + '[TEST 3]' + bcolors.ENDC + ' Compare reconstructed images with real by PSNR:')
for i in range(len(real_images)):
    original = cv2.imread(res_path + real_images[i])
    compressed = cv2.imread(res_path + reconstructed_images[i], 1)
    value = PSNR(original, compressed)
    print('Check ' + reconstructed_images[i] + ':', f"PSNR value is {value} dB")
    if threshold < value:
        print(f'More than threshold {threshold} dB')
    else:
        print(f'Less than threshold {threshold} dB')
        Passed = False

if (Passed):
     print(bcolors.OKGREEN + 'Test PASSED' + bcolors.ENDC)
else:
     print(bcolors.FAIL + 'Test FAILED' + bcolors.ENDC)