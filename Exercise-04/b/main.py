import torch
import torch.nn as nn
import numpy as np


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size, emoji_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, emoji_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        # Shape: (number of layers, batch size, state size)
        zero_state = torch.zeros(1, 1, 128)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(
            x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


emoji_arr = {
    'dog': '\U0001F436',
    'frog': '\U0001F438',
    'fog': '\U0001F32B',
    'hedgehog': '\U0001F987',
    'egg': '\U0001F95A',
    'eswatini': '\U0001F1F8',
    'god': '\U0001F436',
}

index_to_char = [' ', 'a', 'd', 'e', 'f', 'g',
                 'h', 'i', 'o', 'r', 's', 't', 'w', 'n']
char_encodings = np.eye(len(index_to_char))
index_to_emoji = [emoji_arr['dog'], emoji_arr['frog'], emoji_arr['fog'],
                  emoji_arr['hedgehog'], emoji_arr['egg'], emoji_arr['eswatini'], emoji_arr['dog']]
emoji_encodings = np.eye(len(index_to_emoji))
encoding_size = len(char_encodings)


x_train = torch.tensor([
    [[char_encodings[2]], [char_encodings[8]], [char_encodings[5]], [char_encodings[0]], [
        char_encodings[0]], [char_encodings[0]], [char_encodings[0]], [char_encodings[0]]],  # Dog
    [[char_encodings[4]], [char_encodings[9]], [char_encodings[8]], [char_encodings[5]], [
        char_encodings[0]], [char_encodings[0]], [char_encodings[0]], [char_encodings[0]]],  # Frog
    [[char_encodings[4]], [char_encodings[8]], [char_encodings[5]], [char_encodings[0]], [
        char_encodings[0]], [char_encodings[0]], [char_encodings[0]], [char_encodings[0]]],  # Fog
    [[char_encodings[6]], [char_encodings[3]], [char_encodings[2]], [char_encodings[5]], [
        char_encodings[3]], [char_encodings[6]], [char_encodings[8]], [char_encodings[5]]],  # Hedgehog
    [[char_encodings[3]], [char_encodings[5]], [char_encodings[5]], [char_encodings[0]], [
        char_encodings[0]], [char_encodings[0]], [char_encodings[0]], [char_encodings[0]]],  # Egg
    [[char_encodings[3]], [char_encodings[10]], [char_encodings[12]], [char_encodings[1]], [
        char_encodings[11]], [char_encodings[7]], [char_encodings[13]], [char_encodings[7]]],  # Eswatini
    [[char_encodings[5]], [char_encodings[8]], [char_encodings[2]], [char_encodings[0]], [
        char_encodings[0]], [char_encodings[0]], [char_encodings[0]], [char_encodings[0]]],  # God
], dtype=torch.float)  # ' hello'
y_train = torch.tensor([
    [emoji_encodings[0], emoji_encodings[0], emoji_encodings[0], emoji_encodings[0],
        emoji_encodings[0], emoji_encodings[0], emoji_encodings[0], emoji_encodings[0]],  # dog
    [emoji_encodings[1], emoji_encodings[1], emoji_encodings[1], emoji_encodings[1],
        emoji_encodings[1], emoji_encodings[1], emoji_encodings[1], emoji_encodings[1]],  # dog
    [emoji_encodings[2], emoji_encodings[2], emoji_encodings[2], emoji_encodings[2],
        emoji_encodings[2], emoji_encodings[2], emoji_encodings[2], emoji_encodings[2]],  # dog
    [emoji_encodings[3], emoji_encodings[3], emoji_encodings[3], emoji_encodings[3],
        emoji_encodings[3], emoji_encodings[3], emoji_encodings[3], emoji_encodings[3]],  # dog
    [emoji_encodings[4], emoji_encodings[4], emoji_encodings[4], emoji_encodings[4],
        emoji_encodings[4], emoji_encodings[4], emoji_encodings[4], emoji_encodings[4]],  # dog
    [emoji_encodings[5], emoji_encodings[5], emoji_encodings[5], emoji_encodings[5],
        emoji_encodings[5], emoji_encodings[5], emoji_encodings[5], emoji_encodings[5]],
    [emoji_encodings[6], emoji_encodings[6], emoji_encodings[6], emoji_encodings[6],
        emoji_encodings[6], emoji_encodings[6], emoji_encodings[6], emoji_encodings[6]],   # dog
], dtype=torch.float)  # 'hello '

model = LongShortTermMemoryModel(encoding_size, len(emoji_encodings))


def generate(string):
    model.reset()
    for i in range(len(string)):
        char_index = index_to_char.index(string[i])
        y = model.f(torch.tensor(
            [[char_encodings[char_index]]], dtype=torch.float))
        if i == len(string) - 1:
            print(index_to_emoji[y.argmax(1)], ": ", string)


optimizer = torch.optim.RMSprop(model.parameters(), 0.0001)
for epoch in range(500):
    for i in range(x_train.size()[0]):
        model.reset()
        model.loss(x_train[i], y_train[i]).backward()
        optimizer.step()
        optimizer.zero_grad()


generate("eswatini")
