use anyhow;
use markdown;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProcessError {
    #[error("Failed to parse markdown: {0}")]
    InvalidMarkdown(String),
}

//[TODO] Implement this
pub fn markdown_splitter(markdown: &String) -> anyhow::Result<Vec<String>> {
    let mdast = markdown::to_mdast(markdown, &markdown::ParseOptions::default())
        .map_err(ProcessError::InvalidMarkdown)?;

    Ok(vec![])
}

#[cfg(test)]
mod test {
    use crate::markdown_parser;

    #[test]
    fn test_markdown_splitter() -> anyhow::Result<()> {
        assert_eq!(
            markdown_parser::markdown_splitter(&String::from("# Some markdown"))?,
            vec![] as Vec<String>
        );
        Ok(())
    }
}
