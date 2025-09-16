use candle_core::Tensor;
use anyhow::Result;
use std::fs::File;
use std::io::Read;

use crate::vocab::Vocab;
use crate::BLOCK_SIZE;
use crate::BATCH_SIZE;
use std::rc::Rc;
use candle_core::Device;
use rand::Rng;

fn read_data(data_file: &str, vocab: &Vocab) -> Result<Tensor> {
    let mut file = File::open(data_file)?;
    let mut data = String::new();
    file.read_to_string(&mut data)?;
    vocab.encode(&data)
}

pub enum DatasetType {
    Train,
    Test,
}

pub struct Dataset {
    device: Rc<Device>,
    train_data: Tensor,
    test_data: Tensor,
}

impl Dataset {
    pub fn new(data_file: &str, vocab: &Vocab, device: &Rc<Device>) -> Result<Self> {
        let data = read_data(data_file, vocab)?;
        let size = data.dim(0)?;
        let train_size = ((size as f32 * 0.9) as usize / BLOCK_SIZE) * BLOCK_SIZE;
        let test_size = ((size - train_size) / BLOCK_SIZE) * BLOCK_SIZE;
        let train_data = data.narrow(0, 0, train_size)?;
        let test_data = data.narrow(0, train_size, test_size)?;
        Ok(Self { device: device.clone(), train_data, test_data })
    }

    pub fn get_batch(&self, dataset_type: DatasetType) -> Result<(Tensor, Tensor)> {
        let data = match dataset_type {
            DatasetType::Train => &self.train_data,
            DatasetType::Test => &self.test_data,
        };

        let mut x = vec![];
        let mut y = vec![];
        for i in 0..BATCH_SIZE {
            let random_index = rand::thread_rng().gen_range(0..data.dim(0)? - BLOCK_SIZE  - 1);
            let xi = data.narrow(0, random_index, BLOCK_SIZE)?;
            let yi = data.narrow(0, random_index + 1, BLOCK_SIZE)?;
            x.push(xi);
            y.push(yi);
        }
        Ok((Tensor::stack(&x, 0)?.to_device(&self.device)?, Tensor::stack(&y, 0)?.to_device(&self.device)?))
    }
}