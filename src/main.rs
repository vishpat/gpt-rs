mod vocab;
mod dataset;
mod bigram;

use anyhow::Result;
use candle_core::Device;
use std::rc::Rc;
use log::{info, debug};

use vocab::Vocab;
use dataset::{Dataset, DatasetType};
use bigram::Bigram;

const BLOCK_SIZE: usize = 8;
const BATCH_SIZE: usize = 4;

fn main() -> Result<()> {

    env_logger::init();
    info!("Starting the program");
    let device = Rc::new(Device::cuda_if_available(0)?);
    let vocab = Vocab::new("input.txt", &device)?;
    let dataset = Dataset::new("input.txt", &vocab, &device)?;
    let (x, y) = dataset.get_batch(DatasetType::Train)?;
    let bigram = Bigram::new(vocab.len(), &device)?;
    let (_logits, _loss) = bigram.forward(&x, &y)?;
    
    let x = vocab.encode("\n")?.reshape(&[1, 1])?;
    debug!("X: {}", x);
    let generated = bigram.generate(&x, 100)?;
    let text = vocab.decode(&generated.squeeze(0)?)?;
    info!("Generated text: {}", text);
   
    Ok(())
}
