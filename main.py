import torch
from torch.utils.data import DataLoader
from data.preprocessing import preprocess_data, split_data
from data.dataset import FunctionCallDataset
from model.model import FunctionCallModel
from model.train import train_model, evaluate_model

try:
    # Hyperparameters
    num_epochs = 3
    batch_size = 16
    learning_rate = 0.001
    dataset_path = 'data\\advanced_function_calls.csv'

    # Load and preprocess the data
    inputs, labels, label_encoder = preprocess_data(dataset_path)
    X_train, X_test, y_train, y_test = split_data(inputs, labels)

    # Create DataLoader instances for train and test sets
    train_dataset = FunctionCallDataset(X_train, y_train)
    test_dataset = FunctionCallDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize the model, loss function, and optimizer
    model = FunctionCallModel(num_classes=len(label_encoder.classes_))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Evaluate the model
    accuracy = evaluate_model(model, test_loader)
    print(f'Accuracy on test set: {accuracy}')
except Exception as e:
    # Handle exceptions gracefully
    print("An error occurred:", e)
