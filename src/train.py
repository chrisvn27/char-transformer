import torch
from data import get_batch, vocabulary, encoder, decoder
from model import BigramLanguageModel


device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)

def train_model(model, optim, batch_size=64):
    model.train()
    optim.zero_grad()
    data_batch = get_batch(batch_size, train=True)
    x_batch , y_batch = data_batch
    _, loss = model(x_batch,y_batch)
    loss.backward()
    optim.step()

    return loss.item()


@torch.no_grad()
def estimate_loss(model,batch_size=64, n_batches=100):
    model.eval()
    batches_train = []
    batches_val = []
    for _ in range(n_batches):
        x_batch_train, y_batch_train = get_batch(batch_size,train=True)
        _, loss_train = model(x_batch_train, y_batch_train)
        batches_train.append(loss_train.item())
        x_batch_val, y_batch_val = get_batch(batch_size, train=False)
        _, loss_val = model(x_batch_val, y_batch_val)
        batches_val.append(loss_val.item())

    loss_batch_train = torch.tensor(batches_train).mean().item()
    loss_batch_val = torch.tensor(batches_val).mean().item()

    model.train()

    return  loss_batch_train, loss_batch_val


if __name__ == "__main__":

    model = BigramLanguageModel(len(vocabulary))
    optim = torch.optim.AdamW(model.parameters(), lr=0.01)

    steps = 2000
    loss_train = []
    loss_val = []

    init_train , init_val = estimate_loss(model)
    print(f"Initial: train loss = {init_train:.4f}, val loss = {init_val:.4f}")

    for s in range(steps):
        _ = train_model(model, optim, batch_size=64)

        if s % 100 == 0:
            train_avg_loss, val_avg_loss = estimate_loss(model)
            loss_train.append(train_avg_loss)
            loss_val.append(val_avg_loss)

            print(f"Step {s}: train loss = {train_avg_loss:.4f}, val loss = {val_avg_loss:.4f}")

    torch.save(model.state_dict(), 'bigram_128windowsize.pth')

    state_dict = torch.load('bigram_128windowsize.pth', weights_only=True)
    model.load_state_dict(state_dict)

    context = "There was a day"
    ctx_enc = encoder(context)
    ctx_enc = torch.tensor(ctx_enc, dtype=torch.int64).unsqueeze(0)
    pred = model.generate(ctx_enc,120)
    print(decoder(pred[0].tolist()))



