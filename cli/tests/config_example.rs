//! Smoke test: the shipped `config.example.toml` must parse with the
//! current `CallConfig` schema. Lives here (CLI crate) because the
//! example file ships alongside the CLI; the schema itself lives in
//! `sapphire-call-core`.

use sapphire_call_core::config::CallConfig;

#[test]
fn shipped_example_parses() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config.example.toml");
    let _ = CallConfig::load(&path).expect("config.example.toml should parse");
}
