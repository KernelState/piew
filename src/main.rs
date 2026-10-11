use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::{self as nn, Linear, Module, VarBuilder, VarMap};
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{Encoding, Tokenizer, TokenizerBuilder};

const VOCAB: &'static str = "vocab.json";
const VOCAB_SIZE: usize = 4096;
/// window size in tokens
const WINDOW_SIZE: usize = 12;
const DATA: &'static str = "verdict.txt";

fn main() -> Result<()> {
    let build = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            NFC.into(),
        ])))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()));
    let tokenizer = if Path::new(VOCAB).exists() {
        Tokenizer::from_file(VOCAB).unwrap()
    } else {
        let mut tokenizer = build.build().unwrap();

        let mut trainer = BpeTrainerBuilder::new()
            .show_progress(true)
            .vocab_size(VOCAB_SIZE)
            .min_frequency(0)
            .build();

        tokenizer
            .train_from_files(&mut trainer, vec![DATA.to_string()])
            .unwrap();
        tokenizer.save(VOCAB, false).unwrap();

        Tokenizer::from_file(VOCAB).unwrap()
    };

    let toks = tokenizer
        .encode(std::fs::read_to_string(DATA)?, true)
        .unwrap();
    println!("Got {} tokens in the verdict", toks.len());

    let mut model = Model::new(1, VOCAB_SIZE, vec![], &Device::Cpu)?;

    let final_loss = train_model(&mut model, toks.get_ids().to_vec());

    Ok(())
}

struct Model<'a> {
    pub dev: Device,
    vb: VarBuilder<'a>,
    varmap: VarMap,
    pub layers: Vec<nn::Linear>,
}

impl<'a> Model<'a> {
    pub fn new(i: usize, o: usize, h: Vec<(usize, usize)>, dev: &'a Device) -> Result<Self> {
        assert!(i != 0, "Cannot have 0 input length");
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let mut ls = Vec::new();
        ls.push(nn::linear(
            i,
            if h.len() != 0 { h[0].0 } else { o },
            vb.pp("input"),
        )?);
        for (n, (i_, o_)) in (0..h.len()).zip(h.iter()) {
            ls.push(nn::linear(
                *i_,
                if h.len() > n + 1 { h[n + 1].0 } else { *o_ },
                vb.pp(format!("layer{n}")),
            )?);
        }
        Ok(Self {
            layers: ls,
            varmap,
            vb,
            dev: dev.clone(),
        })
    }
    pub fn forward(&self, i: Tensor) -> Tensor {
        let mut x = i;
        self.layers.iter().for_each(|i| x = i.forward(&x).unwrap());
        x
    }
}

fn relu(x: &Tensor) -> Result<Tensor> {
    Ok(x.clamp(0.0, f64::MAX)?)
}

fn single_word(index: usize, vocab_size: usize, device: &Device) -> Result<Tensor> {
    let mut values = vec![0.0f32; vocab_size];
    values[index] = 1.0;
    Ok(Tensor::from_vec(values, vocab_size, device)?)
}

fn train_model(m: &mut Model, data: Vec<u32>) -> f64 {
    // TODO: make it so you can iterate, back propagate and optimize
    // TODO: have an actual embedding system for the model to comprehend text
    //   from the tokens (attention)
    assert!(
        data.len() > WINDOW_SIZE,
        "Window size cannot be larger than data length"
    );
    let mut loss = data
        .iter()
        .map(|x| *x as f64)
        .collect::<Vec<f64>>()
        .windows(WINDOW_SIZE + 1)
        .map(|w| {
            nn::loss::mse(
                &single_word(w[WINDOW_SIZE] as usize, VOCAB_SIZE, &m.dev).unwrap(),
                &(relu(&m.forward(
                    Tensor::from_slice(&w[0..WINDOW_SIZE - 1], vec![1, 1], &m.dev).unwrap(),
                ))
                .unwrap()),
            )
            .unwrap()
            .to_vec1::<f64>()
            .unwrap()
            .iter()
            .sum::<f64>()
        })
        .sum();
    loss = loss / (data.len() - WINDOW_SIZE) as f64;
    loss
}
