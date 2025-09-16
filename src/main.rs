mod vocab;

use anyhow::Result;
use candle_core::Device;
use std::fs::File;
use std::io::Read;
use std::rc::Rc;

use vocab::Vocab;

const BLOCK_SIZE: usize = 8;
const BATCH_SIZE: usize = 4;

fn read_data(path: &str) -> Result<String> {
    let mut file = File::open(path)?;
    let mut data = String::new();
    file.read_to_string(&mut data)?;
    Ok(data)
}

fn main() -> Result<()> {
    let device = Rc::new(Device::cuda_if_available(0)?);
    let vocab = Vocab::new("input.txt", &device)?;

    let data = vocab.encode(&read_data("input.txt")?)?;
    println!("{:?}", data);
    Ok(())
}
