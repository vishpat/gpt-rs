mod vocab;
mod dataset;
mod bigram;

use anyhow::Result;
use candle_core::Device;
use std::rc::Rc;

use vocab::Vocab;
use dataset::{Dataset, DatasetType};
use bigram::Bigram;

const BLOCK_SIZE: usize = 8;
const BATCH_SIZE: usize = 4;

fn main() -> Result<()> {
    let device = Rc::new(Device::cuda_if_available(0)?);
    let vocab = Vocab::new("input.txt", &device)?;
    let dataset = Dataset::new("input.txt", &vocab, &device)?;
    let (x, y) = dataset.get_batch(DatasetType::Train)?;
    println!("X shape: {:?}", x.shape());
    println!("Y shape: {:?}", y.shape());
    println!("X: {}", x);
    println!("Y: {}", y);
    let bigram = Bigram::new(vocab.len(), &device)?;
    let logits = bigram.forward(&x, Some(&y))?;
    println!("Logits shape: {:?}", logits.shape());
    println!("Logits: {}", logits);
    Ok(())
}
