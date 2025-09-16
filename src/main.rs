mod vocab;
mod dataset;

use anyhow::Result;
use candle_core::Device;
use std::rc::Rc;

use vocab::Vocab;
use dataset::{Dataset, DatasetType};

const BLOCK_SIZE: usize = 8;
const BATCH_SIZE: usize = 4;

fn main() -> Result<()> {
    let device = Rc::new(Device::cuda_if_available(0)?);
    let vocab = Vocab::new("input.txt", &device)?;
    let dataset = Dataset::new("input.txt", &vocab, &device)?;
    let (x, y) = dataset.get_batch(DatasetType::Train)?;
    println!("Train data: {:?}", x.shape());
    println!("Test data: {:?}", y.shape());

    Ok(())
}
