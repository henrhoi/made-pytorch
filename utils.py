import pickle
from os.path import join
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# Mixture of logistics
import torch


def mixture_of_logistics_sample_data_1():
    count = 1000
    rand = np.random.RandomState(0)
    samples = 0.4 + 0.1 * rand.randn(count)
    data = np.digitize(samples, np.linspace(0.0, 1.0, 20))
    split = int(0.8 * len(data))
    train_data, test_data = data[:split], data[split:]
    return train_data, test_data


def mixture_of_logistics_sample_data_2():
    count = 10000
    rand = np.random.RandomState(0)
    a = 0.3 + 0.1 * rand.randn(count)
    b = 0.8 + 0.05 * rand.randn(count)
    mask = rand.rand(count) < 0.5
    samples = np.clip(a * mask + b * (1 - mask), 0.0, 1.0)
    data = np.digitize(samples, np.linspace(0.0, 1.0, 100))
    split = int(0.8 * len(data))
    train_data, test_data = data[:split], data[split:]
    return train_data, test_data


def show_training_plot(train_losses, test_losses, title):
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, test_losses, label='test loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    plt.show()


def show_distribution_1d(data, distribution, title):
    d = len(distribution)

    plt.figure()
    plt.hist(data, bins=np.arange(d) - 0.5, label='train data', density=True)

    x = np.linspace(-0.5, d - 0.5, 1000)
    y = distribution.repeat(1000 // d)
    plt.plot(x, y, label='learned distribution')

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()


def show_results(dset_type, fn):
    if dset_type == 1:
        train_data, test_data = mixture_of_logistics_sample_data_1()
        d = 20
    elif dset_type == 2:
        train_data, test_data = mixture_of_logistics_sample_data_2()
        d = 100
    else:
        raise Exception('Invalid dset_type:', dset_type)

    train_losses, test_losses, distribution = fn(train_data, test_data, d, dset_type)
    assert np.allclose(np.sum(distribution), 1), f'Distribution sums to {np.sum(distribution)} != 1'

    print(f'Final Test Loss: {test_losses[-1]:.4f}')

    show_training_plot(train_losses, test_losses, f'Dataset {dset_type} Train Plot')
    show_distribution_1d(train_data, distribution, f'Dataset {dset_type} Learned Distribution')


# MADE
def load_pickled_data(fname, include_labels=False):
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    train_data, test_data = data['train'], data['test']
    if 'mnist.pkl' in fname or 'shapes.pkl' in fname:
        # Binarize MNIST and shapes dataset
        train_data = (train_data > 127.5).astype('uint8')
        test_data = (test_data > 127.5).astype('uint8')
    if 'celeb.pkl' in fname:
        train_data = train_data[:, :, :, [2, 1, 0]]
        test_data = test_data[:, :, :, [2, 1, 0]]
    if include_labels:
        return train_data, test_data, data['train_labels'], data['test_labels']
    return train_data, test_data


# Question 2
def sample_2d_data(image_file, n, d):
    from PIL import Image
    import itertools

    im = Image.open(image_file).resize((d, d)).convert('L')
    im = np.array(im).astype('float32')
    dist = im / im.sum()

    pairs = list(itertools.product(range(d), range(d)))
    idxs = np.random.choice(len(pairs), size=n, replace=True, p=dist.reshape(-1))
    samples = [pairs[i] for i in idxs]

    return dist, np.array(samples)


def show_distribution_2d(true_dist, learned_dist):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.imshow(true_dist)
    ax1.set_title('True Distribution')
    ax1.axis('off')
    ax2.imshow(learned_dist)
    ax2.set_title('Learned Distribution')
    ax2.axis('off')
    plt.show()


def show_samples(samples, nrow=10, title='Samples'):
    samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()


def show_results_made(dset_type, part, fn):
    data_dir = "data"
    if part == 'a':
        if dset_type == 1:
            n, d = 10000, 25
            true_dist, data = sample_2d_data(join(data_dir, 'smiley.jpg'), n, d)
        elif dset_type == 2:
            n, d = 100000, 200
            true_dist, data = sample_2d_data(join(data_dir, 'geoffrey-hinton.jpg'), n, d)
        else:
            raise Exception('Invalid dset_type:', dset_type)
        split = int(0.8 * len(data))
        train_data, test_data = data[:split], data[split:]
    elif part == 'b':
        if dset_type == 1:
            train_data, test_data = load_pickled_data(join(data_dir, 'shapes.pkl'))
            img_shape = (20, 20)
        elif dset_type == 2:
            train_data, test_data = load_pickled_data(join(data_dir, 'mnist.pkl'))
            img_shape = (28, 28)
        else:
            raise Exception('Invalid dset type:', dset_type)
    else:
        raise Exception('Invalid part', part)

    if part == 'a':
        train_losses, test_losses, distribution = fn(train_data, test_data, d, dset_type)
        assert np.allclose(np.sum(distribution), 1), f'Distribution sums to {np.sum(distribution)} != 1'

        print(f'Final Test Loss: {test_losses[-1]:.4f}')

        show_training_plot(train_losses, test_losses, f'Dataset {dset_type} Train Plot)')
        show_distribution_2d(true_dist, distribution)

    elif part == 'b':
        train_losses, test_losses, samples = fn(train_data, test_data, img_shape, dset_type)
        samples = samples.astype('float32') * 255
        print(f'Final Test Loss: {test_losses[-1]:.4f}')
        show_training_plot(train_losses, test_losses, f'Dataset {dset_type} Train Plot')
        show_samples(samples)
