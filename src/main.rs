use std::path::Path;

use anyhow::Result;
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{Tokenizer, TokenizerBuilder};

const VOCAB: &'static str = "vocab.json";
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
            .vocab_size(4096)
            .min_frequency(0)
            .build();

        tokenizer.train_from_files(&mut trainer, vec![DATA.to_string()]).unwrap();
        tokenizer.save(VOCAB, false).unwrap();

        Tokenizer::from_file(VOCAB).unwrap()
    };

    let toks = tokenizer.encode(std::fs::read_to_string(DATA)?, true);
    println!("Got {} tokens in the verdict", toks.iter().len());

    Ok(())
}
