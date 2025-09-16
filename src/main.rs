mod vocab;

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::fs::File;
use std::io::Read;
use std::rc::Rc;

use vocab::Vocab;

const BLOCK_SIZE: usize = 8;
const BATCH_SIZE: usize = 4;

fn read_data(path: &str, vocab: &Vocab) -> Result<Tensor> {
    let mut file = File::open(path)?;
    let mut data = String::new();
    file.read_to_string(&mut data)?;
    Ok(vocab.encode(&data)?)
}

fn main() -> Result<()> {
    let device = Rc::new(Device::cuda_if_available(0)?);
    let vocab = Vocab::new("input.txt", &device)?;

    let data = read_data("input.txt", &vocab)?;
    println!("Data: {:?}", data);
    let size = data.dim(0)?;
    let train_size = (size as f32*0.9) as usize;
    let train_size = train_size - (train_size % BLOCK_SIZE);
    let test_size = size - train_size;
    let test_size = test_size - (test_size % BLOCK_SIZE);
    println!("Train size: {:?} {}", train_size, train_size % BLOCK_SIZE);
    println!("Test size: {:?} {}", test_size, test_size % BLOCK_SIZE);
    println!("Block size: {}", BLOCK_SIZE);
    let train_data = data.narrow(0, 0, train_size)?;
    let train_data = train_data.reshape(&[train_size / BLOCK_SIZE, BLOCK_SIZE])?;
    let test_data = data.narrow(0, train_size, test_size)?;
    let test_data = test_data.reshape(&[test_size / BLOCK_SIZE, BLOCK_SIZE])?;
    println!("Train data: {:?}", train_data.shape());
    println!("Test data: {:?}", test_data.shape());
    Ok(())
}
