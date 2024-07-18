import json
import numpy as np
import random
import torch
import torch.nn as nn
import nltk
from nltk.stem.porter import PorterStemmer
from torch.utils.data import Dataset, DataLoader
from torchviz import make_dot

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('all')

# Define utility functions
def write_to_json_file(path, data):
    filepathname = f'./{path}/data.json'
    with open(filepathname, 'w') as fp:
        json.dump(data, fp)

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

# Read and process intents data
with open('D:\\AI PRO\\Chat-Bot---Financial-digital-Assistant\\data.json', 'r') as f:
    intents = json.load(f)

stemmer = PorterStemmer()
all_words = []
tags = []
pair = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        pair.append((w, tag))

ignore_words = ['?', '.', '!', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(pair), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Create training data
X_train = []
y_train = []

for (pattern_sentence, tag) in pair:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Define hyperparameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

print("input size =", input_size)
print("output size =", output_size)

# Define dataset class
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Define neural network model
class NeuralNetModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetModel, self).__init__()
        self.linearlayer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.linearlayer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.linearlayer3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.linearlayer1(x)
        output = self.relu(output)
        output = self.linearlayer2(output)
        output = self.relu(output)
        output = self.linearlayer3(output)
        return output

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetModel(input_size, hidden_size, output_size).to(device)
make_dot(torch.randn(47, 209).mean(), params=dict(model.named_parameters()))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of parameters: {count_parameters(model)}")

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    total = 0
    correct = 0
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scores = model(words)
        _, pred = scores.max(1)
        total += len(words)
        correct += (pred == labels).sum()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {float(correct) / float(total) * 100:.2f}%')

print(f'Final Accuracy: {float(correct) / float(total) * 100:.2f}%')
print(f'Final loss: {loss.item():.4f}')

# Save the trained model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}
FILE = "DATA.pth"
torch.save(data, FILE)
print(f'Training complete. File saved to {FILE}')

# Load the trained model
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNetModel(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def bargain(value):
    tokens = nltk.word_tokenize(value)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)
    numbers = {str(i): i for i in range(1, 31)}
    OptionsToReply = ["Sorry, this is of latest fashion, Can you raise the amount a little bit",
                      "This is a very special thing, we can't give you at this much less cost",
                      "Oh no sorry. Please raise a little bit"]

    for word, wordType in entities:
        word = stemmer.stem(word)
        if wordType in ['CD'] and word in numbers:
            if numbers[word] > 20:
                print("FinBot: Yes agreed! Now, you can buy the ribbon at this price")
            else:
                print(f"FinBot: {random.choice(OptionsToReply)}")

# Chat with the bot
bot_name = "FinBot"
name = input("Enter Your Name: ")
print("FinBot: Hey, Let's chat! (type 'quit' to exit) Also, when you start bargaining give digits")

while True:
    sent = input(f"{name}: ")
    if sent == "quit":
        break
    if sent in [str(i) for i in range(1, 31)]:
        bargain(sent)
    else:
        sent = tokenize(sent)
        X = bag_of_words(sent, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: I do not understand...")
