use candle_nn::{embedding, Embedding, VarBuilder, VarMap};
use candle_core::Module;
use std::rc::Rc;
use candle_core::{Device, DType, IndexOp};
use anyhow::Result;
use candle_core::Tensor;
use rand::Rng;

pub struct Bigram {
    embedding: Embedding,
    vocab_size: usize,
    device: Rc<Device>,
}

impl Bigram {
    pub fn new(vocab_size: usize, device: &Rc<Device>) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device.clone());
        let token_embedding = embedding(vocab_size, vocab_size, vb)?;
        Ok(Self { embedding: token_embedding, vocab_size, device: device.clone()})
    }

    pub fn forward(&self, x: &Tensor, target: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits = self.embedding.forward(x)?;
        let batch_size  = logits.dim(0)?;
        let time_steps = logits.dim(1)?;
        let vocab_size = logits.dim(2)?;
        let logits = logits.reshape(&[batch_size * time_steps, vocab_size])?;
        let target = target.reshape(&[batch_size * time_steps])?;
        let loss = candle_nn::loss::cross_entropy(&logits, &target)?;
        Ok((logits, loss))
    }

    pub fn generate(&self, x: &Tensor, max_new_tokens: usize) -> Result<Tensor> {
        let mut x = x.clone();
        for _ in 0..max_new_tokens {
            let logits = self.embedding.forward(&x)?;
            let last_index = logits.dim(1)? - 1;
            let logits = logits.i((.., last_index, ..))?.squeeze(1)?;
            
            // Ensure the tensor is contiguous before applying softmax
            let logits = logits.contiguous()?;
            let probs = candle_nn::ops::softmax_last_dim(&logits)?;
            
            // Sample the next token using categorical sampling
            let next_token = self.sample_from_probs(&probs)?;
            
            // Reshape next_token to have the same batch dimension as x
            let next_token = next_token.unsqueeze(1)?;
            
            // Concatenate along the sequence dimension (dim 1)
            x = Tensor::cat(&[&x, &next_token], 1)?;
        }
        Ok(x)
    }

    fn sample_from_probs(&self, probs: &Tensor) -> Result<Tensor> {
        // Handle batch dimension - sample for each batch item
        let batch_size = probs.dim(0)?;
        let vocab_size = probs.dim(1)?;
        
        let mut sampled_tokens = Vec::new();
        let mut rng = rand::thread_rng();
        
        for batch_idx in 0..batch_size {
            // Get probabilities for this batch item
            let batch_probs = probs.i(batch_idx)?;
            let probs_vec = batch_probs.to_vec1::<f32>()?;
            
            // Generate a random number between 0 and 1
            let random_val: f32 = rng.gen();
            
            // Find the index where cumulative probability exceeds random_val
            let mut cumulative = 0.0;
            let mut sampled_idx = vocab_size - 1; // fallback
            
            for (i, &prob) in probs_vec.iter().enumerate() {
                cumulative += prob;
                if cumulative >= random_val {
                    sampled_idx = i;
                    break;
                }
            }
            
            sampled_tokens.push(sampled_idx as u32);
        }
        
        // Create tensor with sampled tokens
        let token_ids = Tensor::from_slice(&sampled_tokens, (batch_size,), probs.device())?;
        Ok(token_ids)
    }
}
