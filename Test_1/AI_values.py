import torch
import torch.nn as nn
import torch.optim as optim

class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

class MachineLearningMain():
    def __init__(self):
        self.model = SimpleModel(input_size=4, output_size=1)  # Beispiel: 4 Eingangsmerkmale, 1 Ausgangsmerkmal
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def train(self, inputs, targets, epochs=100):
        for epoch in range(epochs):
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
            targets_tensor = torch.tensor(targets, dtype=torch.float32)

            self.optimizer.zero_grad()
            outputs = self.model(inputs_tensor)
            loss = self.criterion(outputs, targets_tensor)
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

        return self.model

# Beispiel für die Verwendung der ML-Klasse
if __name__ == "__main__":
    ml_instance = MachineLearningMain()
    # Beispielinputs und -targets
    train_inputs = torch.rand((100, 4))
    train_targets = torch.rand((100, 1))

    trained_model = ml_instance.train(train_inputs, train_targets)
