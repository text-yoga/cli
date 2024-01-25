#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

pub mod jina_bert;
use anyhow::Error as E;
use candle_core::{
    utils::{cuda_is_available, metal_is_available},
    DType, Device, Module, Tensor,
};
use candle_nn::VarBuilder;
use hf_hub::api::tokio::ApiBuilder;
use itertools::{Either, Itertools};
use jina_bert::BertModel;
use jina_bert::Config;
use tokenizers::Tokenizer;

pub async fn build_model_and_tokenizer(
    model: Option<String>,
    tokenizer: Option<String>,
    cpu: bool,
) -> anyhow::Result<(BertModel, tokenizers::Tokenizer)> {
    use hf_hub::{Repo, RepoType};

    let api = ApiBuilder::new().with_progress(true).build()?;
    let model = match &model {
        Some(model_file) => std::path::PathBuf::from(model_file),
        None => {
            api.repo(Repo::new(
                "jinaai/jina-embeddings-v2-small-en".to_string(),
                RepoType::Model,
            ))
            .get("model.safetensors")
            .await?
        }
    };
    let tokenizer = match tokenizer {
        Some(file) => std::path::PathBuf::from(file),
        None => {
            api.repo(Repo::new(
                "jinaai/jina-embeddings-v2-small-en".to_string(),
                RepoType::Model,
            ))
            .get("tokenizer.json")
            .await?
        }
    };
    let device = device(cpu)?;
    let mut config = Config::v2_base();
    config.hidden_size = 512;
    config.intermediate_size = 2048;
    config.num_attention_heads = 8;
    config.num_hidden_layers = 4;
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)? };
    println!("Initializing model...");
    let model = BertModel::new(vb, &config)?;
    println!("...done.");
    Ok((model, tokenizer))
}

pub fn device(cpu: bool) -> anyhow::Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        println!("Using metal acceleration");
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

pub(crate) fn embed(
    model: &BertModel,
    tokenizer: &mut Tokenizer,
    normalize_embeddings: bool,
    sentences: Vec<String>,
) -> Result<(Vec<(String, Vec<f32>)>, Vec<(String, candle_core::Error)>), anyhow::Error> {
    let device = &model.device;
    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    } else {
        let pp = tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }

    println!("tokenizing texts");
    let tokens = tokenizer
        .encode_batch(sentences.clone(), true)
        .map_err(E::msg)?;
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Tensor::new(tokens.as_slice(), device)
        })
        .collect::<candle_core::Result<Vec<_>>>()?;

    let token_ids = Tensor::stack(&token_ids, 0)?;
    println!("running inference on batch {:?}", token_ids.shape());
    let embeddings = model.forward(&token_ids)?;
    println!("generated embeddings {:?}", embeddings.shape());
    // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
    let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
    let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
    let embeddings = if normalize_embeddings {
        normalize_l2(&embeddings)?
    } else {
        embeddings
    };
    println!("pooled embeddings {:?}", embeddings.shape());

    let result: (Vec<(String, Vec<f32>)>, Vec<(String, candle_core::Error)>) = sentences
        .into_iter()
        .map(|s| s.clone())
        .enumerate()
        .partition_map(|(i, sentence)| {
            match embeddings
                .get(i)
                .and_then(|embedding| embedding.to_vec1::<f32>())
            {
                Ok(embedding) => Either::Left((sentence, embedding)),
                Err(err) => Either::Right((sentence, err)),
            }
        });
    Ok(result)
}

pub fn normalize_l2(v: &Tensor) -> candle_core::Result<Tensor> {
    v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
}
