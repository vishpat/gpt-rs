use candle_nn::{embedding, Embedding, VarBuilder, VarMap};
use candle_core::Module;
use std::rc::Rc;
use candle_core::{Device, DType};
use anyhow::Result;
use candle_core::Tensor;

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
}
