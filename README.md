# text.yoga cli

⚠️ WIP - not yet ready for use.

A cli for automatic generation of embeddings for text documents.


## Features

- 100% Rust - No python required
- Hardware-accelerated (uses [candle](https://github.com/huggingface/candle))
- Supported models
    - jinaai/jina-embeddings-v2-small-en
- Supported file types
    - markdown

## Usage

```
cargo run --features metal --release -- --input-folder=<folder-with-markdown-files>
```


## License

Licensed under [MIT](./LICENSE).
