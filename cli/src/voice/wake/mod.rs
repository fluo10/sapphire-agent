//! Wake-word detection — currently a hole with the right shape in it.
//!
//! The satellite used to run openWakeWord here, on `ort`. That meant a second
//! ONNX Runtime beside the one `sherpa-onnx` links for VAD, and the two vendor
//! their own copies of re2: linking both into one binary gives duplicate
//! symbols on Linux and LNK2038 on Windows. Since Cargo unifies features
//! additively, one crate pulling `ort` forced every sherpa-linking crate in the
//! workspace onto *shared* sherpa linking — and a shared-linked binary does not
//! start once it is copied away from `target/`, which is how the published
//! agent binary came to fail on every install (#182).
//!
//! So detection is gone from the client until the server does it (#183). The
//! plumbing around it deliberately is not: [`Detector`] is an empty enum, so
//! the listen loop keeps its wake branches, statically dead, and #183 restores
//! the behaviour by making this type real again rather than by re-threading the
//! state machine.
//!
//! A satellite that connects to a server with `[voice].wake_word_model` set
//! warns and runs VAD-only.

mod disabled;
pub use disabled::Detector;
