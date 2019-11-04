from torch import nn

class CriticNetwork(nn.Module):

    def __init__(
        self,
        encoder_class,
        input_dim,
        embedding_dim,
        hidden_dim,
        n_layers,
        encoder_normalization
    ):
        super(CriticNetwork, self).__init__()

        self.encoder_class = encoder_class
        self.hidden_dim = hidden_dim

        self.encoder = self.encoder_class(
            node_dim=input_dim,
            n_heads=8,
            embed_dim=embedding_dim,
            n_layers=n_layers,
            normalization=encoder_normalization
        )

        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        """

        :param inputs: (batch_size, graph_size, input_dim)
        :return:
        """
        graph_embeddings = self.encoder(inputs)['graph_embs']
        return self.value_head(graph_embeddings)
