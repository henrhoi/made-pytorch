import torch.nn as nn
from linear_masked import LinearMasked
from utils import *


class MADE(nn.Module):
    """
    Class implementing MADE for 1 channel image-input based on "MADE: Masked Autoencoder for Distribution Estimation" by Germain et. al.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        self.net = nn.Sequential(
            LinearMasked(input_dim, input_dim),
            nn.ReLU(),
            LinearMasked(input_dim, input_dim),
            nn.ReLU(),
            LinearMasked(input_dim, input_dim)
        )
        self.apply_masks()

    def forward(self, x):
        return self.net(x)

    def apply_masks(self):
        # Set order of masks, i.e. who can make which edges
        ordering = np.arange(self.input_dim)

        # Create masks
        masks = []
        masks.append(ordering[:, None] <= ordering[None, :])
        masks.append(ordering[:, None] <= ordering[None, :])
        masks.append(ordering[:, None] < ordering[None, :])

        # Set the masks in all LinearMasked layers
        layers = [layer for layer in self.net.modules() if isinstance(layer, LinearMasked)]
        for i in range(len(layers)):
            layers[i].set_mask(masks[i])


def train_made(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    test_data: An (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    image_shape: (H, W), height and width of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
            used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in {0, 1}
    """

    use_cuda = False
    device = torch.device('cuda') if use_cuda else None

    train_data = torch.from_numpy(
        train_data.reshape((train_data.shape[0], train_data.shape[1] * train_data.shape[2]))).float().to(device)
    test_data = torch.from_numpy(
        test_data.reshape((test_data.shape[0], test_data.shape[1] * test_data.shape[2]))).float().to(device)

    def nll_loss(batch, output):
        return torch.nn.functional.binary_cross_entropy(torch.sigmoid(output), batch)

    H, W = image_shape[0], image_shape[1]
    input_dim = H * W

    dataset_params = {
        'batch_size': 32,
        'shuffle': True
    }

    made = MADE(input_dim).cuda() if use_cuda else MADE(input_dim)
    n_epochs = 10
    lr = 0.003

    train_loader = torch.utils.data.DataLoader(train_data, **dataset_params)
    optimizer = torch.optim.Adam(made.parameters(), lr=lr)

    init_test_loss = nll_loss(test_data, made(test_data))

    train_losses = []
    test_losses = [init_test_loss.item()]

    for epoch in range(n_epochs):
        for batch_x in train_loader:
            optimizer.zero_grad()
            output = made(batch_x)
            loss = nll_loss(batch_x, output)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        test_loss = nll_loss(test_data, made(test_data))
        test_losses.append(test_loss.item())
        print(f"[{100*(epoch+1)/n_epochs:.2f}%] Epoch {epoch + 1}")

    if use_cuda:
        torch.cuda.empty_cache()

    samples = torch.zeros(size=(100, H * W)).to(device)
    made.eval()
    with torch.no_grad():
        for i in range(H * W):
            out = made(samples)
            proba = torch.sigmoid(out)
            torch.bernoulli(proba[:, i], out=samples[:, i])

    return np.array(train_losses), np.array(test_losses), samples.reshape((100, H, W, 1)).detach().cpu().numpy()


def train_and_show_results_shapes_data():
    """
    Trains MADE and displays samples and training plot for Shapes dataset
    """
    show_results_made(1, 'b', train_made)


def train_and_show_results_mnist_data():
    """
    Trains MADE and displays samples and training plot for MNIST dataset
    """
    show_results_made(2, 'b', train_made)
