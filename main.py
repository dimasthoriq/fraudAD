import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.manifold import TSNE

from utils import get_loaders, get_features
from models import SSLNet, DeepSAD
from trainers import train
from evals import evaluate

config = {
    'method': 'sad',    # 'sad-maha', 'ssd', 'sad'
    'eta': 10.,

    # Training
    'epochs': 1000,
    'lr': 1e-4,
    'weight_decay': 1e-6,
    'patience': 20,
    'min_delta': 1e-8,
    'sched_patience': 10,
    'sched_factor': 0.5,

    # Network
    'dims': [30, 16, 16, 8],
    'drop': None,
    'norm': True,
    'activation': 'LeakyReLU',

    # Dataset
    'val_split': 0.1,
    'test_split': 0.1,
    'batch_size': 1024,

    # Utility
    'seed': 15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_path': './Data/creditcard.csv',
    'print_freq': 20,
}

if __name__ == '__main__':
    hyperparam_str = str(config['temperature']) + 'temp' if config['method'] == 'ssd' else str(config['eta']) + 'eta'
    train_loader, val_loader, test_loader = get_loaders(config['data_path'],
                                                        config['val_split'],
                                                        config['test_split'],
                                                        config['seed'],
                                                        config['batch_size'],
                                                        config['method']
                                                        )

    if config['method'] == 'ssd':
        model = SSLNet(config).to(config['device'])
    else:
        model = DeepSAD(config).to(config['device'])

    model, train_losses, val_losses, _, _, _, _ = train(model, train_loader, val_loader, config)
    time_stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

    # Save the learning curves
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"./Experiments/{config['method']}_{hyperparam_str}_{time_stamp}_learning_curves.png")

    # Evaluate the model
    train_ap, train_fpr, train_roc = evaluate(model, train_loader, train_loader, config['method'])
    val_ap, val_fpr, val_roc = evaluate(model, train_loader, val_loader, config['method'])
    test_ap, test_fpr, test_roc = evaluate(model, train_loader, test_loader, config['method'])

    print(f'Train FPR@95: {train_fpr:.4f}, Val FPR@95: {val_fpr:.4f}, Test FPR@95: {test_fpr:.4f}')
    print(f'Train AUPRC: {train_ap:.4f}, Val AUPRC: {val_ap:.4f}, Test AUPRC: {test_ap:.4f}')
    print(f'Train ROC: {train_roc:.4f}, Val ROC: {val_roc:.4f}, Test ROC: {test_roc:.4f}')

    # Get embeddings
    z_train, y_train = get_features(model, train_loader)
    z_val, y_val = get_features(model, val_loader)
    z_test, y_test = get_features(model, test_loader)

    # Sample only 1000 points from normal, but all frauds
    n_train = pd.DataFrame(z_train[y_train == 0])
    n_val = pd.DataFrame(z_val[y_val == 0])
    n_test = pd.DataFrame(z_test[y_test == 0])

    n_train = n_train.sample(1000, random_state=15)
    n_val = n_val.sample(1000, random_state=15)
    n_test = n_test.sample(1000, random_state=15)

    z_train = pd.concat([n_train, pd.DataFrame(z_train[y_train == 1])], axis=0).values
    z_val = pd.concat([n_val, pd.DataFrame(z_val[y_val == 1])], axis=0).values
    z_test = pd.concat([n_test, pd.DataFrame(z_test[y_test == 1])], axis=0).values

    y_train = np.concatenate([np.zeros(1000), np.ones(z_train.shape[0] - 1000)])
    y_val = np.concatenate([np.zeros(1000), np.ones(z_val.shape[0] - 1000)])
    y_test = np.concatenate([np.zeros(1000), np.ones(z_test.shape[0] - 1000)])

    # TSNE
    z_train_2d = TSNE(n_components=2).fit_transform(z_train)
    z_val_2d = TSNE(n_components=2).fit_transform(z_val)
    z_test_2d = TSNE(n_components=2).fit_transform(z_test)

    # Save the TSNE plot
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.scatter(z_train_2d[y_train == 0, 0], z_train_2d[y_train == 0, 1], s=10, label='Normal')
    plt.scatter(z_train_2d[y_train == 1, 0], z_train_2d[y_train == 1, 1], s=10, label='Fraud')
    plt.title(f'Train FPR@95: {train_fpr:.4f}')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.scatter(z_val_2d[y_val == 0, 0], z_val_2d[y_val == 0, 1], s=10, label='Normal')
    plt.scatter(z_val_2d[y_val == 1, 0], z_val_2d[y_val == 1, 1], s=10, label='Fraud')
    plt.title(f'Val FPR@95: {val_fpr:.4f}')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.scatter(z_test_2d[y_test == 0, 0], z_test_2d[y_test == 0, 1], s=10, label='Normal')
    plt.scatter(z_test_2d[y_test == 1, 0], z_test_2d[y_test == 1, 1], s=10, label='Fraud')
    plt.title(f'Test FPR@95: {test_fpr:.4f}')
    plt.legend()
    plt.savefig(f"./Experiments/{config['method']}_{hyperparam_str}_{time_stamp}_tsne.png")
