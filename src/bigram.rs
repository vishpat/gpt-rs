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

    pub fn forward(&self, x: &Tensor, _target: Option<&Tensor>) -> Result<Tensor> {
        let logits = self.embedding.forward(x)?;
        Ok(logits)
    }
}
