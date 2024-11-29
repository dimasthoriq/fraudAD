import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split


def get_loaders(data_path, val_split, test_split, seed, batch_size, method='ssd'):
    df = pd.read_csv(data_path)

    for i in ['Time', 'Amount']:
        df.loc[:, i] = (df[i] - df[i].mean()) / df[i].std()

    train, test = train_test_split(df, test_size=test_split, stratify=df['Class'],
                                   random_state=seed)
    train, val = train_test_split(train, test_size=val_split, stratify=train['Class'],
                                  random_state=seed)

    print(train.shape, val.shape, test.shape)

    x = train.iloc[:, :-1].values
    y = train.iloc[:, -1].values

    x_val = val.iloc[:, :-1].values
    y_val = val.iloc[:, -1].values

    x_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    if method == 'ssd':
        x_pos = x + np.random.normal(0, 0.05, x.shape)
        x_pos_val = x_val + np.random.normal(0, 0.05, x_val.shape)

        train = torch.utils.data.TensorDataset(torch.tensor(x, dtype=torch.float32),
                                               torch.tensor(x_pos, dtype=torch.float32),
                                               torch.tensor(y, dtype=torch.float32)
                                               )

        val = torch.utils.data.TensorDataset(torch.tensor(x_val, dtype=torch.float32),
                                             torch.tensor(x_pos_val, dtype=torch.float32),
                                             torch.tensor(y_val, dtype=torch.float32)
                                             )

    else:
        train = torch.utils.data.TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        val = torch.utils.data.TensorDataset(torch.tensor(x_val, dtype=torch.float32),
                                             torch.tensor(y_val, dtype=torch.float32))
    test = torch.utils.data.TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                          torch.tensor(y_test, dtype=torch.float32))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_features(model, dataloader):
    features, labels = [], []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    for i, batch in enumerate(dataloader):
        if len(batch) == 3:
            x, x_pos, label = batch
        else:
            x, label = batch

        x, label = x.to(device), label.to(device)

        features.append(model(x).data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    return features, labels
