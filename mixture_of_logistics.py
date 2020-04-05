from utils import *


def generative_model_histogram(train_data, test_data, d, dset_id):
    """
    train_data: An (n_train,) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test,) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for random variable x
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
                used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d,) of model probabilities
    """

    def nll_loss(x, theta):
        model = torch.exp(torch.gather(theta, dim=0, index=x)) / torch.exp(theta).sum()
        return torch.mean(-torch.log(model))

    dataset_params = {
        'batch_size': 32,
        'shuffle': True
    }

    theta = torch.zeros(d, dtype=torch.float32, requires_grad=True)
    n_epochs = 40
    train_loader = torch.utils.data.DataLoader(train_data, **dataset_params)
    optimizer = torch.optim.Adam([theta], lr=0.01)

    train_losses = []
    test_losses = [nll_loss(torch.tensor(test_data), theta).item()]

    for epoch in range(n_epochs):
        for batch_x in train_loader:
            optimizer.zero_grad()
            loss = nll_loss(batch_x, theta)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        test_loss = nll_loss(torch.tensor(test_data), theta)
        test_losses.append(test_loss.item())

    return np.array(train_losses), np.array(test_losses), torch.softmax(theta, dim=0).detach().numpy()


def discretized_mixture_of_logistics(train_data, test_data, d, dset_id):
    """
    train_data: An (n_train,) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test,) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for random variable x
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
            used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d,) of model probabilities
    """

    def model(x, pis, means, stds):
        x = x.reshape((x.shape[0], 1))
        x_plus = x + 0.5
        x_minus = x - 0.5

        # Handeling edge cases
        x_plus[x_plus > d - 1] = torch.tensor(float('inf'), requires_grad=True)
        x_minus[x_minus < 0] = torch.tensor(float('-inf'), requires_grad=True)

        unweighted_mixtures = torch.sigmoid((x_plus - means) / torch.exp(stds)) - torch.sigmoid(
            (x_minus - means) / torch.exp(stds))
        model_proba = (unweighted_mixtures * torch.softmax(pis, dim=1)).sum(dim=1, keepdim=False)

        return model_proba

    def nll_loss(x, pis, means, stds):
        model_proba = model(x, pis, means, stds)
        return torch.mean(-torch.log(model_proba))

    dataset_params = {
        'batch_size': 32,
        'shuffle': True
    }

    if dset_id == 2:
        pis = torch.tensor([[1., 2., 3., 4.]], requires_grad=True)
        means = torch.tensor([[20., 30., 80., 90.]], requires_grad=True)
        stds = torch.tensor([[1., 2., 3., 4.]], requires_grad=True)
    else:
        pis = torch.tensor([[1., 2., 3., 4.]], requires_grad=True)
        means = torch.tensor([[1., 2., 3., 4.]], requires_grad=True)
        stds = torch.tensor([[1., 2., 3., 4.]], requires_grad=True)

    n_epochs = 20
    train_loader = torch.utils.data.DataLoader(train_data, **dataset_params)
    optimizer = torch.optim.Adam([pis, means, stds], lr=0.01)

    train_losses = []
    test_losses = [nll_loss(torch.tensor(test_data), pis, means, stds).item()]

    for epoch in range(n_epochs):
        for batch_x in train_loader:
            optimizer.zero_grad()
            loss = nll_loss(batch_x, pis, means, stds)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        test_loss = nll_loss(torch.tensor(test_data), pis, means, stds)
        test_losses.append(test_loss.item())

    x_s = torch.tensor([i for i in range(d)])
    model_probabilities = model(x_s, pis, means, stds)
    return np.array(train_losses), np.array(test_losses), model_probabilities.detach().numpy()


if __name__ == '__main__':
    show_results(dset_type=1, fn=generative_model_histogram)
    show_results(dset_type=2, fn=discretized_mixture_of_logistics)


