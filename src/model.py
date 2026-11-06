import torch
import torch.nn as nn
import torch.nn.functional as F



class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        self.vocab_size = vocab_size


    def forward(self, x, y=None):
        logits = self.embedding(x)
        if y is None:
            loss = None
        else: 
            flat_logits = logits.view(-1,self.vocab_size)
            flat_y = y.view(-1)
            loss = F.cross_entropy(flat_logits, flat_y)

        return logits, loss

    @torch.no_grad()
    def generate(self, context, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(context)
            last = logits[:,-1,:]
            probs = F.softmax(last, dim=-1)
            next_char = torch.multinomial(probs,1)
            context = torch.cat((context, next_char), dim = 1)

        return context



if __name__ == "__main__":
    from data import vocabulary, get_batch
    model = BigramLanguageModel(len(vocabulary))
    x, y = get_batch(4,True)
    logits, loss = model(x, y)
    print(logits.shape, loss.item())

    from data import decoder
    ctx = torch.zeros((1,1), dtype=torch.long)
    out = model.generate(ctx,100)
    print(decoder(out[0].tolist()))