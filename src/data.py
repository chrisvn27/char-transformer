import pathlib
import torch


path = pathlib.Path(__file__).resolve().parent.parent / "data" / "input.txt"

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

vocabulary = sorted(set(text))

mapping = {letter: int_pos for int_pos, letter in enumerate(vocabulary)}

def encoder(string):
    list_integer = [mapping[char] for char in string]

    return list_integer

def decoder(list_integer):
    list_chars = [vocabulary[integer] for integer in list_integer]
    decoded_string = "".join(list_chars)

    return decoded_string

def get_batch(batch_size = 64, train = True):
    if train:
        N = len(train_data)
    else:
        N = len(val_data)

    rand_idxs = torch.randint(N- window_size, (batch_size,))

    return rand_idxs



encoded_text = encoder(text)
tensor_text = torch.tensor(encoded_text)

train_data_fraction = 0.9
total_indices = len(tensor_text)
train_indices = int(train_data_fraction*total_indices)
train_data = tensor_text[:train_indices]
val_data = tensor_text[train_indices:]

batch_size = 64
window_size = 4



if __name__ == "__main__":

    print(f"Length of text: {len(text):,} characters")
    print(f"Unique characters in text: {len(vocabulary)}")
    print(f"Sorted vocabulary: {vocabulary}")
    
    print(f"Dictionary: {mapping}")
    print(f"Testing letter to int mapping[d]: {mapping['d']}")
    print(f"Testing int to letter such as vocabulary[42] (should be d): {vocabulary[42]}")

    word = 'hello'
    print(f"Encoding {word} to numbers")

    list_integer = encoder(word)
    print(f"Hello in list of numbers is: {list_integer}")

    print(f"Decoding list of numbers to string")
    decoded_word = decoder(list_integer)

    assert word == decoded_word, f"word should have been {word}"

    print(f"Text into tensor checking shape/type: {tensor_text.shape}, {tensor_text.dtype}")

    print(f"Checking training data and validation data add up to the total")
    print(f"Train data = {len(train_data)}, Validation data = {len(val_data)}")
    print(f"Train and validation data add to {len(train_data) + len(val_data)}")
    assert len(train_data) + len(val_data) == len(text)

    print(f"Testing function get_batch to get indixs for train or val")

    print(f"Train testing with batch size of 64")
    idxs_train = get_batch()
    assert len(idxs_train) == batch_size, f"length should be {batch_size}"
    assert idxs_train.max() < len(train_data) - window_size, f"Max idxs should be less than N_train -window_size -1 which is {len(train_data) - window_size -1}"
    
    print(f"Validation testing testing with batch size of 64")
    idxs_val = get_batch(batch_size, train=False)
    assert len(idxs_val) == batch_size, f"length should be {batch_size}"
    assert idxs_val.max() < len(val_data) - window_size, f"Max idxs should be less than N_val -window_size -1  which is {len(val_data) - window_size -1}"

    assert idxs_val.max() < 100, f"Max idxs should be less than N_val -window_size -1  which is {len(val_data) - window_size -1}"
    
    
    
    


