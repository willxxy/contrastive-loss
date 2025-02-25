import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

class ConditionalContrastiveLoss(nn.Module):
    similar_target: float = 0.9
    dissimilar_target: float = 0.1
    
    def __call__(self, image_embeddings, pos_text_embeddings, neg_text_embeddings, transform_indices):
        image_embeddings = jnp.mean(image_embeddings, axis=1)  # -> (B, D)
        pos_text_embeddings = jnp.mean(pos_text_embeddings, axis=1)  # -> (B, D)
        neg_text_embeddings = jnp.mean(neg_text_embeddings, axis=1)  # -> (B, D)

        image_norm = jnp.linalg.norm(image_embeddings, axis=-1, keepdims=True)
        pos_text_norm = jnp.linalg.norm(pos_text_embeddings, axis=-1, keepdims=True)
        neg_text_norm = jnp.linalg.norm(neg_text_embeddings, axis=-1, keepdims=True)
        
        image_embeddings_norm = image_embeddings / jnp.maximum(image_norm, 1e-8)
        pos_text_embeddings_norm = pos_text_embeddings / jnp.maximum(pos_text_norm, 1e-8)
        neg_text_embeddings_norm = neg_text_embeddings / jnp.maximum(neg_text_norm, 1e-8)

        s_pos = jnp.sum(image_embeddings_norm * pos_text_embeddings_norm, axis=-1)
        s_neg = jnp.sum(image_embeddings_norm * neg_text_embeddings_norm, axis=-1)
        
        similar_indices = jnp.array([1, 2, 5])
        
        is_similar = jnp.sum(
            jnp.expand_dims(transform_indices, -1) == jnp.expand_dims(similar_indices, 0),
            axis=-1
        ).astype(jnp.float32)
        
        loss_pos = (1.0 - s_pos) ** 2
        
        targets = jnp.where(is_similar > 0, 
                            self.similar_target, 
                            self.dissimilar_target)
        
        loss_neg = jnp.where(
            is_similar > 0,
            (targets - s_neg) ** 2,
            (s_neg - targets) ** 2
        )
        
        sample_losses = loss_pos + loss_neg
        
        return jnp.mean(sample_losses)

def main():
    np.random.seed(42)
    key = jax.random.PRNGKey(42)
    
    batch_size = 4
    img_seq_len = 10
    text_seq_len = 8
    embed_dim = 16
    
    image_embeddings = jax.random.normal(key, (batch_size, img_seq_len, embed_dim))
    key, subkey = jax.random.split(key)
    pos_text_embeddings = jax.random.normal(subkey, (batch_size, text_seq_len, embed_dim))
    key, subkey = jax.random.split(key)
    neg_text_embeddings = jax.random.normal(subkey, (batch_size, text_seq_len, embed_dim))
    
    transform_indices = [0, 1, 2, 3]
    
    loss_module = ConditionalContrastiveLoss()
    
    params = {}
    
    loss_value = loss_module.apply(params, image_embeddings, pos_text_embeddings, 
                                  neg_text_embeddings, transform_indices)
    
    print(f"Computed loss value: {loss_value}")
    
    all_similar = [1, 2, 5, 5]
    loss_similar = loss_module.apply(params, image_embeddings, pos_text_embeddings, 
                                    neg_text_embeddings, all_similar)
    
    print(f"Loss with all similar transformations: {loss_similar}")
    
    all_dissimilar = [3, 4, 6, 7]
    loss_dissimilar = loss_module.apply(params, image_embeddings, pos_text_embeddings, 
                                       neg_text_embeddings, all_dissimilar)
    
    print(f"Loss with all dissimilar transformations: {loss_dissimilar}")

if __name__ == "__main__":
    main()
