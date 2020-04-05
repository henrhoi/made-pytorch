import torch.nn as nn
from linear_masked import LinearMasked
from utils import *


class MADE2D(nn.Module):
    """
    Class implementing MADE for 2D input based on "MADE: Masked Autoencoder for Distribution Estimation" by Germain et. al.
    """

    def __init__(self, input_dim, d):
        super().__init__()
        self.d = d
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
        order_0 = np.concatenate((np.repeat(1, self.d), np.repeat(2, self.d)))
        order_1 = np.repeat(1, self.input_dim)
        order_2 = np.repeat(1, self.input_dim)

        # Create masks
        masks = []
        masks.append(order_0[:, None] <= order_1[None, :])
        masks.append(order_1[:, None] <= order_2[None, :])
        masks.append(order_2[:, None] < order_0[None, :])

        # Set the masks in all LinearMasked layers
        layers = [layer for layer in self.net.modules() if isinstance(layer, LinearMasked)]
        for i in range(len(layers)):
            layers[i].set_mask(masks[i])


def train_made(train_data, test_data, d, dset_id):
    """
    train_data: An (n_train, 2) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test, 2) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for each random variable x1 and x2
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
            used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d, d) of probabilities (the learned joint distribution)
    """

    def make_one_hot(data, num_classes=d):
        one_hot = np.zeros((len(data), 2 * d))
        one_hot[np.arange(data.shape[0]), data[:, 0]] = 1
        one_hot[np.arange(data.shape[0]), d + data[:, 1]] = 1
        return torch.from_numpy(one_hot).float()

    use_cuda = True
    device = torch.device('cuda') if use_cuda else None

    train_data = make_one_hot(train_data).to(device)
    test_data = make_one_hot(test_data).to(device)

    def get_proba(batch, output, d):
        output_1 = torch.softmax(output[:, :d], dim=1)
        output_2 = torch.softmax(output[:, d:], dim=1)

        indices_1 = torch.nonzero(batch[:, :d])
        indices_2 = torch.nonzero(batch[:, d:])

        return torch.gather(input=output_1, dim=1, index=indices_1[:, [1]]) \
               * torch.gather(input=output_2, dim=1, index=indices_2[:, [1]])

    def nll_loss(batch, output, d):
        proba = get_proba(batch, output, d)
        return -torch.mean(torch.log(proba + 1e-8))

    made = MADE2D(2 * d, d).cuda() if use_cuda else MADE2D(2 * d, d)
    dataset_params = {
        'batch_size': 32,
        'shuffle': True
    }

    n_epochs = 50 if dset_id == 2 else 10
    lr = 0.0025 if dset_id == 2 else 0.01
    train_loader = torch.utils.data.DataLoader(train_data, **dataset_params)
    optimizer = torch.optim.Adam(made.parameters(), lr=lr)

    init_test_loss = nll_loss(test_data, made(test_data), d)

    train_losses = []
    test_losses = [init_test_loss.item()]

    for epoch in range(n_epochs):
        for batch_x in train_loader:
            optimizer.zero_grad()
            output = made(batch_x)
            loss = nll_loss(batch_x, output, d)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        test_loss = nll_loss(torch.tensor(test_data), made(torch.tensor(test_data)), d)
        test_losses.append(test_loss.item())
        print(f"[{100*(epoch+1)/n_epochs:.2f}%] Epoch {epoch + 1}") if epoch % 2 else None

    x_s = make_one_hot(torch.tensor([[i, j] for i in range(d) for j in range(d)])).to(device)

    proba = get_proba(x_s, made(x_s), d).reshape(d, d)

    return np.array(train_losses), np.array(test_losses), proba.detach().cpu().numpy()


def train_and_show_results_smiley_data():
    """
    Trains 2D MADE and displays samples and training plot for smiley distribution
    """
    show_results_made(1, 'a', train_made)


def train_and_show_results_geoffrey_hinton_data():
    """
    Trains 2D MADE and displays samples and training plot for Geoffrey Hinton distribution
    """
    show_results_made(2, 'a', train_made)
