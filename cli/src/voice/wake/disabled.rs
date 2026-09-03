//! Stand-in for [`super::Detector`] in builds without `wake-oww`.
//!
//! An empty enum, so `Option<Detector>` in the listen loop can only ever
//! be `None` and every wake branch there is statically unreachable. That
//! is the whole point: the loop's wake handling stays one code path
//! across both feature settings instead of being cut up by `cfg`s. The
//! methods below exist only to type-check those unreachable branches —
//! each one destructures a value that cannot exist.

pub enum Detector {}

impl Detector {
    pub fn feed(&mut self, _samples: &[i16]) -> anyhow::Result<Option<String>> {
        match *self {}
    }

    pub fn reset(&mut self) {
        match *self {}
    }
}
