import pathlib

path = pathlib.Path(__file__).resolve().parent.parent / "data" / "input.txt"

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

if __name__ == "__main__":

    print(f"Length of text: {len(text):,} characters")

    vocabulary = set(text)
    print(f"Unique characters in text: {len(vocabulary)}")
    vocabulary = sorted(vocabulary)
    print(f"Sorted vocabulary: {vocabulary}")
    

    mapping = {letter: int_pos for int_pos, letter in enumerate(vocabulary)}
    print(f"Dictionary: {mapping}")
    print(f"Testing letter to int mapping[d]: {mapping['d']}")
    print(f"Testing int to letter such as vocabulary[42] (should be d): {vocabulary[42]}")

    print(f"Converting 'hello' into numbers")
    word = 'hello'

    list_integer = []
    for char in word:
        list_integer.append(mapping[char])

    print(f"Hello in list of numbers is: {list_integer}")

    print(f"Converting list of numbers back to words")

    decoded_word = [ vocabulary[integer] for integer in list_integer]
    decoded_word = "".join(decoded_word)
    

    assert word == decoded_word, f"word should have been {word}"



