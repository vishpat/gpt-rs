use anyhow::Result;
use candle_core::{Device, Tensor};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::rc::Rc;

#[derive(Debug)]
pub struct Vocab {
    device: Rc<Device>,
    map: HashMap<char, usize>,
    rev_map: HashMap<usize, char>,
}

impl Vocab {
    pub fn new(path: &str, device: &Rc<Device>) -> Result<Self> {
        let map = get_vocab_map(path)?;
        let rev_map = map.iter().map(|(k, v)| (*v, *k)).collect();
        Ok(Self {
            device: device.clone(),
            map,
            rev_map,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Tensor> {
        let mut vec = Vec::new();
        for char in text.chars() {
            vec.push(self.map[&char] as u32);
        }
        let tensor = Tensor::from_slice(&vec, (text.len(), ), &self.device)?;
        Ok(tensor)
    }

    pub fn decode(&self, tensor: &Tensor) -> Result<String> {
        let mut text = String::new();

        let vec = tensor.to_vec1::<u32>()?;
        for index in vec.iter() {
            text.push(self.rev_map[&(*index as usize)]);
        }
        Ok(text)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
}

fn get_vocab_map(path: &str) -> Result<HashMap<char, usize>> {
    
    let mut file = File::open(path)?;
    let mut data = String::new();
    file.read_to_string(&mut data)?;

    let mut set = HashSet::new();
    for char in data.chars() {
        set.insert(char);
    }
    let mut map = HashMap::new();
    for (i, char) in set.iter().enumerate() {
        map.insert(*char, i);
    }
    Ok(map)
}
