//! Provider registry — bundles all configured `Provider` implementations
//! together with the routing logic that picks one per `room_id`.
//!
//! Constructed once at startup from the parsed `Config`. The Anthropic
//! provider is always present under the name [`ANTHROPIC_PROVIDER_NAME`];
//! additional providers come from `[providers.<name>]` config entries.

use crate::config::{ANTHROPIC_PROVIDER_NAME, Config, ProviderConfig};
use crate::provider::Provider;
use crate::provider::anthropic::AnthropicProvider;
use crate::provider::openai_compatible::OpenAICompatibleProvider;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;

pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn Provider>>,
}

impl ProviderRegistry {
    /// Build a registry from the parsed config.
    ///
    /// Always installs the Anthropic provider under `"anthropic"`. Each
    /// `[providers.<name>]` entry adds another provider keyed by its
    /// configured name. Profile validation is performed up front and
    /// returned as an error so misconfigurations fail at startup rather
    /// than mid-conversation.
    pub fn from_config(config: &Config) -> Result<Self> {
        let errors = config.validate_profiles();
        if !errors.is_empty() {
            anyhow::bail!("invalid profile configuration:\n  - {}", errors.join("\n  - "));
        }

        let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        providers.insert(
            ANTHROPIC_PROVIDER_NAME.to_string(),
            Arc::new(AnthropicProvider::new(&config.anthropic)),
        );

        for (name, pcfg) in &config.providers {
            if name == ANTHROPIC_PROVIDER_NAME {
                anyhow::bail!(
                    "provider name '{ANTHROPIC_PROVIDER_NAME}' is reserved for the built-in"
                );
            }
            let provider: Arc<dyn Provider> = match pcfg {
                ProviderConfig::OpenAiCompatible(c) => {
                    let mut cfg = c.clone();
                    if cfg.provider_name.is_none() {
                        cfg.provider_name = Some(name.clone());
                    }
                    Arc::new(OpenAICompatibleProvider::new(&cfg))
                }
            };
            providers.insert(name.clone(), provider);
        }

        Ok(Self { providers })
    }

    /// Look up a provider by name. Returns `None` if no such provider was
    /// registered.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Provider>> {
        self.providers.get(name).cloned()
    }

    /// The built-in Anthropic provider — guaranteed to be present.
    pub fn anthropic(&self) -> Arc<dyn Provider> {
        self.providers
            .get(ANTHROPIC_PROVIDER_NAME)
            .cloned()
            .expect("anthropic provider is always registered")
    }

    /// Resolve the provider that should handle a chat in `room_id`.
    ///
    /// Falls back to the Anthropic provider if (a) no profile is configured
    /// for the room, (b) the resolved profile names a provider that isn't
    /// registered (validation should have caught this — defensive only).
    pub fn for_room(&self, config: &Config, room_id: &str) -> Arc<dyn Provider> {
        let provider_name = config
            .profile_for(room_id)
            .and_then(|p| config.provider_for_profile(p))
            .unwrap_or(ANTHROPIC_PROVIDER_NAME);
        self.get(provider_name).unwrap_or_else(|| self.anthropic())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Config {
        toml::from_str(s).expect("config should parse")
    }

    #[test]
    fn registry_always_has_anthropic() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"
"#,
        );
        let reg = ProviderRegistry::from_config(&cfg).unwrap();
        assert!(reg.get(ANTHROPIC_PROVIDER_NAME).is_some());
    }

    #[test]
    fn registry_loads_openai_compatible() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[providers.local]
type = "openai_compatible"
base_url = "http://127.0.0.1:8080/v1"
model = "gemma-4-31b-it"
"#,
        );
        let reg = ProviderRegistry::from_config(&cfg).unwrap();
        let local = reg.get("local").expect("local provider registered");
        // The provider's name should default to its registry key.
        assert_eq!(local.name(), "local");
    }

    #[test]
    fn from_config_rejects_invalid_profiles() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[profiles.default]
provider = "ghost"
"#,
        );
        let err = ProviderRegistry::from_config(&cfg).err()
            .expect("expected an error");
        assert!(format!("{err:#}").contains("ghost"));
    }

    #[test]
    fn from_config_rejects_reserved_name() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[providers.anthropic]
type = "openai_compatible"
base_url = "http://x/v1"
model = "y"
"#,
        );
        let err = ProviderRegistry::from_config(&cfg).err()
            .expect("expected an error");
        assert!(format!("{err:#}").contains("reserved"));
    }

    #[test]
    fn for_room_falls_back_to_anthropic() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"
"#,
        );
        let reg = ProviderRegistry::from_config(&cfg).unwrap();
        let p = reg.for_room(&cfg, "!any:srv");
        assert_eq!(p.name(), "anthropic");
    }

    #[test]
    fn for_room_routes_to_configured_profile() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[providers.local]
type = "openai_compatible"
base_url = "http://127.0.0.1:8080/v1"
model = "gemma-4-31b-it"

[profiles.nsfw]
provider = "local"

[rooms."!nsfw:srv"]
profile = "nsfw"
"#,
        );
        let reg = ProviderRegistry::from_config(&cfg).unwrap();
        assert_eq!(reg.for_room(&cfg, "!nsfw:srv").name(), "local");
        assert_eq!(reg.for_room(&cfg, "!other:srv").name(), "anthropic");
    }
}
