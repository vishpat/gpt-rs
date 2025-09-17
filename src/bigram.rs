use candle_nn::{embedding, linear, Embedding, VarBuilder, VarMap, Linear};
use candle_core::Module;
use std::rc::Rc;
use candle_core::{Device, DType, IndexOp};
use anyhow::Result;
use candle_core::Tensor;
use rand::Rng;
use log::debug;

pub struct Bigram {
    embedding: Embedding,
    linear: Linear,
    vocab_size: usize,
}

impl Bigram {
    pub fn new(vocab_size: usize, vb: &VarBuilder) -> Result<Self> {
        let token_embedding = embedding(vocab_size, 2*vocab_size, vb.pp("embedding"))?;
        let token_linear = linear(2*vocab_size, vocab_size, vb.pp("linear"))?;
        Ok(Self { embedding: token_embedding, linear: token_linear, vocab_size})
    }

    pub fn forward(&self, x: &Tensor, target: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits = self.embedding.forward(x)?;
        let logits = self.linear.forward(&logits)?;
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
            let logits = self.linear.forward(&logits)?;
            let last_index = logits.dim(1)? - 1;
            let logits = logits.i((.., last_index, ..))?.squeeze(1)?;
            let logits = logits.contiguous()?;
            let probs = candle_nn::ops::softmax_last_dim(&logits)?;
            let next_token = self.sample_from_probs(&probs)?;
            let next_token = next_token.unsqueeze(1)?;
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
