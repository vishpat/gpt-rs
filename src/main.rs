use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};
use std::fs::File;
use std::io::{BufReader, BufRead};
use std::collections::{HashMap, HashSet};


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
    let vocab_map = get_vocab_map("input.txt")?;
    println!("{:?}", vocab_map);
    Ok(())
}
