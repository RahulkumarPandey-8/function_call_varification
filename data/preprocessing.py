import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import GPT2Tokenizer

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to the EOS token

def preprocess_data(file_path):
    # Load data from CSV file
    data = pd.read_csv(file_path)

    # Tokenize the contexts
    inputs = tokenizer(data['context'].tolist(), padding=True, truncation=True, return_tensors="pt")

    # Encode the function calls
    label_encoder = LabelEncoder()
    labels = torch.tensor(label_encoder.fit_transform(data['function_call'].tolist()))

    return inputs, labels, label_encoder

def split_data(inputs, labels, test_size=0.2):
    input_ids_train, input_ids_test, labels_train, labels_test = train_test_split(
        inputs['input_ids'], labels, test_size=test_size, random_state=42)
    attention_mask_train, attention_mask_test = train_test_split(
        inputs['attention_mask'], test_size=test_size, random_state=42)

    train_inputs = {'input_ids': input_ids_train, 'attention_mask': attention_mask_train}
    test_inputs = {'input_ids': input_ids_test, 'attention_mask': attention_mask_test}

    return train_inputs, test_inputs, labels_train, labels_test
