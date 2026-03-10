use std::path::Path;

use anyhow::Result;
use sentencepiece::SentencePieceProcessor;

pub struct Tokenizer {
    inner: SentencePieceProcessor,
}

impl Tokenizer {
    pub fn open(path: &Path) -> Result<Self> {
        let inner = SentencePieceProcessor::open(path)?;
        Ok(Self { inner })
    }

    pub fn blank_id(&self) -> u32 {
        self.inner.len() as u32
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        Ok(self.inner.decode_piece_ids(token_ids)?.trim().to_owned())
    }
}
