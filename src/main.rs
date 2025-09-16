use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use std::fs::File;
use std::io::{BufReader, BufRead};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;


struct Vocab {
    device: Rc<Device>,
    map: HashMap<char, usize>,
    rev_map: HashMap<usize, char>,
}

impl Vocab {
    fn new(path: &str, device: &Rc<Device>) -> Result<Self> {
        let map = get_vocab_map(path)?;
        let rev_map = map.iter().map(|(k, v)| (*v, *k)).collect();
        Ok(Self {device: device.clone(), map, rev_map})
    }

    fn encode(&self, text: &str) -> Result<Tensor> {
        let mut vec = Vec::new();
        for char in text.chars() {
            vec.push(self.map[&char] as f32);
        }
        let tensor = Tensor::from_slice(&vec, (text.len(), 1), &self.device)?;
        Ok(tensor)
    }

    fn decode(&self, tensor: &Tensor) -> Result<String> {
        let mut text = String::new();
        
        let vec = tensor.to_vec1::<f32>()?;
        for index in vec.iter() {
            text.push(self.rev_map[&(*index as usize)]);
        }
        Ok(text)
    }
}



fn get_vocab_map(path: &str) -> Result<HashMap<char, usize>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut set = HashSet::new();
    for line in reader.lines() {
        let line = line?;
        for char in line.chars() {
            set.insert(char);
        }
    }
    let mut map = HashMap::new();
    for (i, char) in set.iter().enumerate() {
        map.insert(*char, i);
    }
    Ok(map)
}

fn main() -> Result<()> {
    let device = Rc::new(Device::cuda_if_available(0)?);
    let vocab = Vocab::new("input.txt", &device)?;
    let text = "Hello, world!";
    let encoded = vocab.encode(text)?;
    println!("{:?}", encoded);
    let decoded = vocab.decode(&encoded)?;
    println!("{:?}", decoded);
    Ok(())
}
