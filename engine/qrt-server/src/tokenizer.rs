use std::collections::HashSet;
use std::path::{Path, PathBuf};

use thiserror::Error;
use tokenizers::Tokenizer;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("tokenizer file was not found: {0}")]
    NotFound(PathBuf),
    #[error("could not load tokenizer: {0}")]
    Load(String),
    #[error("tokenization failed: {0}")]
    Encode(String),
    #[error("detokenization failed: {0}")]
    Decode(String),
}

pub trait TokenCodec: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError>;
    fn decode(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, TokenizerError>;
    fn token_string(&self, token_id: u32) -> Option<String>;
    fn is_eos(&self, token_id: u32) -> bool;
    fn vocab_size(&self) -> usize;
}

pub struct QwenTokenizer {
    tokenizer: Tokenizer,
    eos_token_ids: HashSet<u32>,
}

impl QwenTokenizer {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let path = model_dir.as_ref().join("tokenizer.json");
        if !path.is_file() {
            return Err(TokenizerError::NotFound(path));
        }
        let tokenizer =
            Tokenizer::from_file(&path).map_err(|error| TokenizerError::Load(error.to_string()))?;
        let eos_token_ids = ["<|im_end|>", "<|endoftext|>"]
            .into_iter()
            .filter_map(|token| tokenizer.token_to_id(token))
            .collect();
        Ok(Self {
            tokenizer,
            eos_token_ids,
        })
    }

    pub fn tokenizer_path(model_dir: impl AsRef<Path>) -> PathBuf {
        model_dir.as_ref().join("tokenizer.json")
    }
}

impl TokenCodec for QwenTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| TokenizerError::Encode(error.to_string()))
    }

    fn decode(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, TokenizerError> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| TokenizerError::Decode(error.to_string()))
    }

    fn token_string(&self, token_id: u32) -> Option<String> {
        self.tokenizer.id_to_token(token_id)
    }

    fn is_eos(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id)
    }

    fn vocab_size(&self) -> usize {
        // The model's generation vocabulary includes added special tokens.
        // `false` reports only the base model vocabulary and incorrectly
        // rejects valid IDs such as Qwen's <|im_end|> at the HTTP boundary.
        self.tokenizer.get_vocab_size(true)
    }
}

#[cfg(test)]
pub struct ByteCodec;

#[cfg(test)]
impl TokenCodec for ByteCodec {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        Ok(text
            .as_bytes()
            .iter()
            .map(|value| u32::from(*value))
            .collect())
    }

    fn decode(
        &self,
        token_ids: &[u32],
        _skip_special_tokens: bool,
    ) -> Result<String, TokenizerError> {
        let bytes: Vec<u8> = token_ids.iter().map(|value| *value as u8).collect();
        String::from_utf8(bytes).map_err(|error| TokenizerError::Decode(error.to_string()))
    }

    fn token_string(&self, token_id: u32) -> Option<String> {
        char::from_u32(token_id).map(|value| value.to_string())
    }

    fn is_eos(&self, token_id: u32) -> bool {
        token_id == 0
    }

    fn vocab_size(&self) -> usize {
        256
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::{models::wordlevel::WordLevel, AddedToken};

    #[test]
    fn vocabulary_size_and_decode_include_added_special_tokens() {
        let vocab = [("A".to_owned(), 0), ("<unk>".to_owned(), 1)]
            .into_iter()
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_owned())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        assert_eq!(
            tokenizer.add_special_tokens(&[AddedToken::from("<|im_end|>", true)]),
            1
        );
        let eos_token_id = tokenizer.token_to_id("<|im_end|>").unwrap();
        let codec = QwenTokenizer {
            tokenizer,
            eos_token_ids: HashSet::from([eos_token_id]),
        };

        assert_eq!(codec.vocab_size(), 3);
        assert_eq!(codec.decode(&[eos_token_id], true).unwrap(), "");
        assert_eq!(codec.decode(&[eos_token_id], false).unwrap(), "<|im_end|>");
        assert!(codec.is_eos(eos_token_id));
    }
}
