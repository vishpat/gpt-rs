mod vocab;

use anyhow::Result;
use candle_core::Device;
use std::rc::Rc;

use vocab::Vocab;

fn main() -> Result<()> {
    let device = Rc::new(Device::cuda_if_available(0)?);
    let vocab = Vocab::new("input.txt", &device)?;
    let text = "Hello, world!";
    let encoded = vocab.encode(text)?;
    println!("Encoded: {:?}", encoded);
    let decoded = vocab.decode(&encoded)?;
    println!("Decoded: {:?}", decoded);
    Ok(())
}
