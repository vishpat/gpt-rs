mod vocab;
mod dataset;

use anyhow::Result;
use candle_core::Device;
use std::rc::Rc;

use vocab::Vocab;
use dataset::Dataset;

const BLOCK_SIZE: usize = 8;
const BATCH_SIZE: usize = 4;

fn main() -> Result<()> {
    let device = Rc::new(Device::cuda_if_available(0)?);
    let vocab = Vocab::new("input.txt", &device)?;
    let dataset = Dataset::new("input.txt", &vocab)?;
    println!("Train data: {:?}", dataset.train_data().shape());
    println!("Test data: {:?}", dataset.test_data().shape());

    Ok(())
}
