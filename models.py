import torch


class SSLNet(torch.nn.Module):
    def __init__(self, config):
        super(SSLNet, self).__init__()
        self.drop = config['drop']
        self.norm = config['norm']
        self.dims = config['dims']
        self.activation = config['activation']
        self._build_network()

    def _build_network(self):
        layers = []
        for i in range(len(self.dims)-1):
            # adds linear layer
            layers.append(torch.nn.Linear(self.dims[i], self.dims[i+1]))

            if i < len(self.dims) - 2:
                if self.norm:  # adds batch normalization layer
                    layers.append(torch.nn.BatchNorm1d(self.dims[i+1]))

                # adds activation layer
                if self.activation == 'ReLU':
                    layers.append(torch.nn.ReLU())
                elif self.activation == 'LeakyReLU':
                    layers.append(torch.nn.LeakyReLU())
                elif self.activation == 'ELU':
                    layers.append(torch.nn.ELU())
                elif self.activation == 'Sigmoid':
                    layers.append(torch.nn.Sigmoid())
                elif self.activation == 'Tanh':
                    layers.append(torch.nn.Tanh())

                if self.drop is not None:  # adds dropout layer
                    layers.append(torch.nn.Dropout(self.drop))

        self.embeddings = torch.nn.Sequential(*layers)

    def forward(self, x):
        z = self.embeddings(x)
        return torch.nn.functional.normalize(z, dim=-1)


class DeepSAD(torch.nn.Module):
    def __init__(self, config):
        super(DeepSAD, self).__init__()
        self.drop = config['drop']
        self.norm = config['norm']
        self.dims = config['dims']
        self.activation = config['activation']
        self._build_network()

    def _build_network(self):
        layers = []
        for i in range(len(self.dims)-1):
            # adds linear layer
            layers.append(torch.nn.Linear(self.dims[i], self.dims[i+1], bias=False))

            if i < len(self.dims) - 2:
                if self.norm:  # adds batch normalization layer
                    layers.append(torch.nn.BatchNorm1d(self.dims[i+1], affine=False))

                # adds activation layer
                if self.activation == 'ReLU':
                    layers.append(torch.nn.ReLU())
                elif self.activation == 'LeakyReLU':
                    layers.append(torch.nn.LeakyReLU())
                elif self.activation == 'ELU':
                    layers.append(torch.nn.ELU())
                elif self.activation == 'Sigmoid':
                    layers.append(torch.nn.Sigmoid())
                elif self.activation == 'Tanh':
                    layers.append(torch.nn.Tanh())

                if self.drop is not None:  # adds dropout layer
                    layers.append(torch.nn.Dropout(self.drop))

        self.embeddings = torch.nn.Sequential(*layers)

    def forward(self, x):
        z = self.embeddings(x)
        return z
