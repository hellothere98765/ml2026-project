import torch 
def take_high_weights(cooc_path, output_path):
    cooc = torch.load(cooc_path).float()

    row_sums = cooc.sum(dim=1, keepdim = True)
    normalized = cooc/row_sums.clamp(min=1)

    total = normalized.sum(dim = 0)
    result = normalized - 1/16000 * (total - normalized)

    topv, topi = torch.topk(result, k=3, dim=1)

    torch.save({"top_values": topv, "top_indices": topi}, output_path)

take_high_weights("cooc.pt", "high_weights.pt")