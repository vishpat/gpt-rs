mod vocab;
mod dataset;
mod bigram;

use anyhow::Result;
use candle_core::{Device };
use candle_nn::{Optimizer, AdamW, ParamsAdamW, VarBuilder, VarMap, SGD};
use candle_core::DType;
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

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device.clone());

    let bigram = Bigram::new(vocab.len(), &vb)?;
    let mut sgd = SGD::new(varmap.all_vars(), 0.01)?;
    
    for _ in 0..100 {
        let (x, y) = dataset.get_batch(DatasetType::Train)?;
        let (_logits, loss) = bigram.forward(&x, &y)?;
        sgd.backward_step(&loss)?;
        info!("Loss: {}", loss);
    }    

    let x = vocab.encode("\n")?.reshape(&[1, 1])?;
    debug!("X: {}", x);
    let generated = bigram.generate(&x, 100)?;
    let text = vocab.decode(&generated.squeeze(0)?)?;
    info!("Generated text: {}", text);
   
    Ok(())
}
