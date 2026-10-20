import urllib.request
import pathlib

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if __name__ == "__main__":
    path = pathlib.Path(__file__).resolve().parent.parent / "data" / "input.txt"
    if path.is_file():
        print("Already exists, skipping")
    else :
        path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading Tiny ShakesPeare")
        urllib.request.urlretrieve(url, path)
        print("Done")



