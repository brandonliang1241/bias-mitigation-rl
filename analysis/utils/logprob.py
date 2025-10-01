import torch

def batch_sentence_logprob(model, tokenizer, sentences, device, batch_size=32, max_length=512):
    """
    Compute average token log-probabilities for a batch of sentences.
    """
    results = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            input_ids = enc.input_ids
            attention_mask = enc.attention_mask
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1).to(torch.float32)
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            label_mask = attention_mask[:, 1:].contiguous()
            token_log_probs = token_log_probs * label_mask
            sum_logprob = token_log_probs.sum(dim=1).cpu().numpy()
            token_counts = label_mask.sum(dim=1).cpu().numpy()
            avg_logprob = (sum_logprob / (token_counts + 1e-12)).tolist()
            results.extend(avg_logprob)
    return results

