import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MeanAbsoluteError, MeanSquaredError


class LSTMPredictor(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        """
        LSTMPredictor is the LSTM architecture with Pytorch Lightning to predict the remaining time
        @param input_dim: input size of the LSTM
        @param hidden_dim: hidden size of the LSTM / input size of the linear layer
        @param num_layers: number of layers in the LSTM
        @param output_dim: output size of the linear layer (always 1 fr regression problem)
        @param dropout: probability with which nodes in neural network will be zeroed
        """

        super(LSTMPredictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                            dropout=self.dropout, batch_first=True)
        self.linear_block = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, int(self.hidden_dim * 0.3)),
            nn.BatchNorm1d(int(self.hidden_dim * 0.3)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(self.hidden_dim * 0.3), self.output_dim),
            nn.Tanh())

    def init_hidden(self,):
        h0 = torch.zeros(self.num_layers, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, self.hidden_dim)
        return h0, c0

    def forward(self, x):
        """
        the forward pass of the neural network
        @param x: input tensor to neural network
        @return: output of neural network after all layers
        """

        h0, c0 = self.init_hidden()
        output, _ = self.lstm(x, (h0, c0))
        x = self.linear_block(output)
        return x


class DFNNPredictor(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        """
        DFNNPredictor is the DFNN architecture with Pytorch Lightning to predict the remaining time
        @param input_dim: input size of the DFNN
        @param hidden_dim: hidden size of the DFNN
        @param output_dim: output size of the linear layer (always 1 fr regression problem)
        @param dropout: probability with which nodes in neural network will be zeroed
        """

        super(DFNNPredictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.dfnn = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_dim, int(self.hidden_dim * 1.2)),
            nn.BatchNorm1d(int(self.hidden_dim * 1.2)),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(int(self.hidden_dim * 1.2), int(self.hidden_dim * 0.3)),
            nn.BatchNorm1d(int(self.hidden_dim * 0.3)),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(int(self.hidden_dim * 0.3), self.output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        the forward pass of the neural network
        @param x: input tensor to neural network
        @return: output of neural network after all layers
        """

        x = self.dfnn(x)
        return x


class DeepLearningPredictor(pl.LightningModule):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, learning_rate, network_architecture,
                 scaler):
        """
        DeepLearningPredictor is the neural network implemented with Pytorch Lightning to predict the remaining time
        @param input_dim: input size of the LSTM
        @param hidden_dim: hidden size of the LSTM / input size of the linear layer (only affecting LSTM architecture)
        @param num_layers: number of layers in the LSTM block
        @param output_dim: output size of the linear layer (always 1 fr regression problem)
        @param dropout: probability with which nodes in neural network will be zeroed
        @param learning_rate:
        """

        super(DeepLearningPredictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.scaler = scaler
        self.test_mae = MeanAbsoluteError()
        self.test_rmse = MeanSquaredError(squared=True)

        if network_architecture == "LSTM":
            self.model = LSTMPredictor(input_dim=self.input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers,
                                       output_dim=self.output_dim, dropout=self.dropout)
        else:
            self.model = DFNNPredictor(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim,
                                       dropout=self.dropout)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        complete training step of the neural network
        @param batch: current batch for training
        @param batch_idx: current batch index for training
        @return: training loss
        """

        x, y = batch
        y_hat = self.forward(x)
        y = y.unsqueeze(1)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)

        total = len(y)
        logs = {"train_loss": loss}
        batch_dictionary = {
            "loss": loss,
            "logs": logs,
            "total": total
        }
        return batch_dictionary

    def validation_step(self, batch, batch_idx):
        """
        complete validation step of the neural network
        @param batch: current batch for validation
        @param batch_idx: current batch index for validation
        @return: validation loss
        """

        x, y = batch
        y_hat = self.forward(x)
        y = y.unsqueeze(1)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)

        total = len(y)
        logs = {"val_loss": loss}
        batch_dictionary = {
            "loss": loss,
            "logs": logs,
            "total": total
        }
        return batch_dictionary

    def test_step(self, batch, batch_idx):
        """
        complete test step of the neural network
        @param batch: current batch for testing
        @param batch_idx: current batch index for testing
        """

        x, y = batch
        y_hat = self.forward(x)
        y = y.unsqueeze(1)

        loss = nn.functional.mse_loss(y_hat, y)

        self.log("test_loss", loss)

        self.test_rmse.update(y_hat, y)
        y_hat = self.scaler.inverse_transform(np.array(y_hat).reshape(-1, 1))
        y = self.scaler.inverse_transform(np.array(y).reshape(-1, 1))
        self.test_mae.update(torch.from_numpy(y_hat), torch.from_numpy(y))

        self.log("test_rmse", self.test_rmse)
        self.log("test_mae", self.test_mae)

        total = len(y)
        logs = {"test_loss": loss}
        batch_dictionary = {
            "loss": loss,
            "logs": logs,
            "total": total
        }
        return batch_dictionary

    def configure_optimizers(self):
        """
        configuration of Adam optimizer
        @return: configured optimizer
        """

        optimizer = optim.Adam(self.parameters(), self.learning_rate)
        return optimizer
