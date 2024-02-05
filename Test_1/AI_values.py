import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

class MachineLearningMain():
    def __init__(self):
        self.model = SimpleModel(input_size=4, output_size=1)  
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def train_from_file(self, file_path, epochs=100, comment_prefix='values'):
        with open(file_path, 'r') as file:
            num_comment_lines = sum(1 for line in file if line.startswith(comment_prefix))
            data = np.loadtxt(file_path, skiprows=num_comment_lines)
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            if data.shape[1] >= 5:
                inputs = data[:, :4]
                targets = data[:, 4].reshape(-1, 1)

                for epoch in range(epochs):
                    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
                    targets_tensor = torch.tensor(targets, dtype=torch.float32)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs_tensor)

                    if inputs_tensor.dim() == 1:
                        inputs_tensor = inputs_tensor.unsqueeze(0)

                    loss = self.criterion(outputs, targets_tensor)
                    loss.backward()
                    self.optimizer.step()

                    if epoch % 10 == 0:
                        print(f'Epoch {epoch}, Loss: {loss.item()}')
            else:
                print('Nicht genügend Spalten in den Daten vorhanden.')

        return self.model

# Beispiel für die Verwendung der ML-Klasse
if __name__ == "__main__":
    ml_instance = MachineLearningMain()
    file_path = r'C:\Users\lukas\OneDrive\Dokumente\Python_test-projk\example.txt'

    trained_model = ml_instance.train_from_file(file_path, epochs=100, comment_prefix='values')
