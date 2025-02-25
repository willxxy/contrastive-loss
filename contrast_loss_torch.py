import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalContrastiveLoss(nn.Module):
    def __init__(self, similar_target=0.9, dissimilar_target=0.1):

        super().__init__()
        self.similar_target = similar_target
        self.dissimilar_target = dissimilar_target
        self.similar_types = {"synonym_noun", "synonym_verb", "out_set"}

    def forward(self, image_embeddings, pos_text_embeddings, neg_text_embeddings, transform_types):

        image_embeddings = torch.mean(image_embeddings, dim=1)  # -> (B, D)
        pos_text_embeddings = torch.mean(pos_text_embeddings, dim=1)  # -> (B, D)
        neg_text_embeddings = torch.mean(neg_text_embeddings, dim=1)  # -> (B, D)

        s_pos = F.cosine_similarity(image_embeddings, pos_text_embeddings, dim=-1)
        s_neg = F.cosine_similarity(image_embeddings, neg_text_embeddings, dim=-1)

        loss = 0.0
        for i, t in enumerate(transform_types):
            loss_pos = (1.0 - s_pos[i]) ** 2

            if t in self.similar_types:
                loss_neg = (self.similar_target - s_neg[i]) ** 2
            else:
                loss_neg = (s_neg[i] - self.dissimilar_target) ** 2

            loss += loss_pos + loss_neg

        return loss / (2 * len(transform_types))

def main():
    torch.manual_seed(42)
    
    batch_size = 4
    img_seq_len = 10
    text_seq_len = 8
    embed_dim = 256
    
    image_embeddings = torch.randn(batch_size, img_seq_len, embed_dim)
    pos_text_embeddings = torch.randn(batch_size, text_seq_len, embed_dim)
    neg_text_embeddings = torch.randn(batch_size, text_seq_len, embed_dim)
    
    transform_types = ["synonym_noun", "antonym", "negation", "out_set"]
    
    criterion = ConditionalContrastiveLoss(similar_target=0.9, dissimilar_target=0.1)
    
    loss = criterion(image_embeddings, pos_text_embeddings, neg_text_embeddings, transform_types)
    
    print(f"Batch size: {batch_size}")
    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Positive text embeddings shape: {pos_text_embeddings.shape}")
    print(f"Negative text embeddings shape: {neg_text_embeddings.shape}")
    print(f"Transform types: {transform_types}")
    print(f"Loss: {loss.item():.6f}")
    
    with torch.no_grad():
        img_emb = torch.mean(image_embeddings, dim=1)
        pos_emb = torch.mean(pos_text_embeddings, dim=1)
        neg_emb = torch.mean(neg_text_embeddings, dim=1)
        
        pos_sims = F.cosine_similarity(img_emb, pos_emb, dim=-1)
        neg_sims = F.cosine_similarity(img_emb, neg_emb, dim=-1)
        
        print("\nSimilarities:")
        for i, t in enumerate(transform_types):
            target = criterion.similar_target if t in criterion.similar_types else criterion.dissimilar_target
            print(f"Sample {i} ({t}):")
            print(f"  Positive similarity: {pos_sims[i]:.4f} (target: 1.0000)")
            print(f"  Negative similarity: {neg_sims[i]:.4f} (target: {target:.4f})")


if __name__ == "__main__":
    main()