mod vocab;
mod dataset;
mod bigram;

use anyhow::Result;
use candle_core::{Device, IndexOp};
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
    let bigram = Bigram::new(vocab.len(), &device)?;
    let (_logits, _loss) = bigram.forward(&x, &y)?;
    let generated = bigram.generate(&x, 100)?;
    
    // Decode the generated tokens to text for better readability
    for i in 0..generated.dim(0)? {
        let batch_tokens = generated.i(i)?;
        let text = vocab.decode(&batch_tokens)?;
        println!("Generated text (batch {}): {}", i, text);
    }
    Ok(())
}
