// Correctness and logic
#![warn(clippy::unit_cmp)] // Detects comparing unit types
#![warn(clippy::match_same_arms)] // Duplicate match arms
#![warn(clippy::unreachable)] // Detects unreachable code

// Performance-focused
#![warn(clippy::inefficient_to_string)] // `format!("{}", x)` vs `x.to_string()`
#![warn(clippy::map_clone)] // Cloning inside `map()` unnecessarily
#![warn(clippy::unnecessary_to_owned)] // Detects redundant `.to_owned()` or `.clone()`
#![warn(clippy::large_stack_arrays)] // Helps avoid stack overflows
#![warn(clippy::box_collection)] // Warns on boxed `Vec`, `String`, etc.
#![warn(clippy::vec_box)] // Avoids using `Vec<Box<T>>` when unnecessary
#![warn(clippy::needless_collect)] // Avoids `.collect().iter()` chains

// Style and idiomatic Rust
#![warn(clippy::redundant_clone)] // Detects unnecessary `.clone()`
#![warn(clippy::identity_op)] // e.g., `x + 0`, `x * 1`
#![warn(clippy::needless_return)] // Avoids `return` at the end of functions
#![warn(clippy::let_unit_value)] // Avoids binding `()` to variables
#![warn(clippy::manual_map)] // Use `.map()` instead of manual `match`

// Maintainability
#![warn(clippy::missing_panics_doc)] // Docs for functions that might panic
#![warn(clippy::missing_safety_doc)] // Docs for `unsafe` functions
#![warn(clippy::missing_const_for_fn)] // Suggests making eligible functions `const`

//! # Audio Sample Conversion Library
//!
//! This crate provides functionality for working with audio samples and
//! converting between different audio sample formats. It focuses on correctness,
//! performance, and ease of use.
//!
//! ## Supported Sample Types
//!
//! - `i16`: 16-bit signed integer samples - Common in WAV files and CD-quality audio
//! - `i24`: 24-bit signed integer samples - From the [i24] crate. In-between PCM_16 and PCM_32 in
//!     terms of quality and space on disk.
//! - `i32`: 32-bit signed integer samples (high-resolution audio)
//! - `f32`: 32-bit floating point samples (common in audio processing)
//! - `f64`: 64-bit floating point samples (high-precision audio processing)
//!
//! ## Features
//!
//! - **Type Safety**: Using Rust's type system to ensure correct conversions
//! - **High Performance**: Simple code that enables the compiler to produce fast code. The
//!     [AudioSample] trait simply enables working with some primitives within the context of audio
//!     processing - floats between -1.0 and 1.0 etc.
//!
//! ## Usage Examples
//!
//! ### Basic Sample Conversion
//!
//! ```rust
//! use audio_sample::{AudioSample, ConvertTo};
//!
//! // Convert an i16 sample to floating point
//! let i16_sample: i16 = i16::MAX / 2; // 50% of max amplitude
//! let f32_sample: f32 = i16_sample.convert_to();
//! assert!((f32_sample - 0.5).abs() < 0.0001);
//!
//! // Convert a floating point sample to i16
//! let f32_sample: f32 = -0.75;
//! let i16_sample: i16 = f32_sample.convert_to();
//! assert_eq!(i16_sample, -24575); // -75% of max amplitude
//! ```
//!
//! ### Converting Buffers of Samples
//!
//! ```rust
//! use audio_sample::ConvertSlice;
//!
//! // Using ConvertSlice trait for Box<[T]>
//! let i16_buffer: Box<[i16]> = vec![0, 16384, -16384, 32767].into_boxed_slice();
//! let f32_buffer: Box<[f32]> = i16_buffer.convert_slice();
//! ```
//! ## Implementation Details
//!
//! ### Integer Scaling
//!
//! When converting between integer formats of different bit depths:
//!
//! - **Widening conversions** (e.g., i16 to i32): The samples are shifted left to preserve amplitude.
//! - **Narrowing conversions** (e.g., i32 to i16): The samples are shifted right, which may lose precision.
//!
//! ### Float to Integer Conversion
//!
//! - Floating-point samples are assumed to be in the range -1.0 to 1.0.
//! - They are scaled to the full range of the target integer type.
//! - Values are rounded to the nearest integer rather than truncated.
//! - Values outside the target range are clamped to prevent overflow.
//!
//! ### Integer to Float Conversion
//!
//! - Integer samples are scaled to the range -1.0 to 1.0.
//! - The maximum positive integer value maps to 1.0.
//! - The minimum negative integer value maps to -1.0.
//!
//!
//! ## Bugs / Issues
//! Report them on the [Github Page](<https://www.github.com/jmg049/audio_sample>) and I will try and get to it as soon as I can :)
//!

use bytemuck::{Pod, Zeroable};
use i24::i24;
use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};
/// Marker trait used to signify a valid audio sample type.
pub trait AudioSample:
    Clone + Copy + Debug + Display + Pod + Zeroable + PartialEq + PartialOrd + Send + Sync
{
}

impl AudioSample for i16 {}
impl AudioSample for i24 {}
impl AudioSample for i32 {}
impl AudioSample for f32 {}
impl AudioSample for f64 {}

/// ConvertTo<To> expresses the ability of one [AudioSample] type to convert into another sample type (the “To” type).
pub trait ConvertTo<To: AudioSample> {
    fn convert_to(&self) -> To;
}

/// ConvertSlice expresses the ability to convert a slice of type [AudioSample] into a box of some
/// other [AudioSample] ``To``.
pub trait ConvertSlice<To: AudioSample> {
    fn convert_slice(&self) -> Box<[To]>;
}

impl<From, To> ConvertSlice<To> for Box<[From]>
where
    From: AudioSample + ConvertTo<To>,
    To: AudioSample,
{
    fn convert_slice(&self) -> Box<[To]> {
        self.iter().map(|sample| sample.convert_to()).collect()
    }
}

pub struct Samples<T: AudioSample> {
    samples: Box<[T]>,
}

impl<T: AudioSample> Deref for Samples<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<T: AudioSample> DerefMut for Samples<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

impl<T: AudioSample> AsRef<[T]> for Samples<T> {
    fn as_ref(&self) -> &[T] {
        self.samples.as_ref()
    }
}

impl<T: AudioSample> AsMut<[T]> for Samples<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.samples.as_mut()
    }
}

impl<T: AudioSample> Samples<T> {
    pub const fn new(samples: Box<[T]>) -> Self {
        Self { samples }
    }

    pub fn len(&self) -> usize {
        self.as_ref().len()
    }

    pub fn is_empty(&self) -> bool {
        self.as_ref().is_empty()
    }

    pub fn to_owned<S: AudioSample>(&self) -> Vec<S>
    where
        T: ConvertTo<S>,
        Box<[T]>: ConvertSlice<S>,
    {
        // The boxed slice conversion is done by our blanket impl.
        Vec::from(self.samples.convert_slice())
    }
}

// i16 //
impl ConvertTo<Self> for i16 {
    fn convert_to(&self) -> Self {
        *self
    }
}

impl ConvertTo<i24> for i16 {
    fn convert_to(&self) -> i24 {
        i24::from_i32((i32::from(*self)) << 8)
    }
}

impl ConvertTo<i32> for i16 {
    fn convert_to(&self) -> i32 {
        (i32::from(*self)) << 16
    }
}

impl ConvertTo<f32> for i16 {
    fn convert_to(&self) -> f32 {
        (f32::from(*self) / (f32::from(Self::MAX))).clamp(-1.0, 1.0)
    }
}

impl ConvertTo<f64> for i16 {
    fn convert_to(&self) -> f64 {
        (f64::from(*self) / f64::from(Self::MAX)).clamp(-1.0, 1.0)
    }
}

// i24 //
impl ConvertTo<i16> for i24 {
    #[allow(clippy::cast_possible_truncation)]
    fn convert_to(&self) -> i16 {
        (self.to_i32() >> 8) as i16
    }
}

impl ConvertTo<Self> for i24 {
    fn convert_to(&self) -> Self {
        *self
    }
}

impl ConvertTo<i32> for i24 {
    fn convert_to(&self) -> i32 {
        self.to_i32() << 8
    }
}

impl ConvertTo<f32> for i24 {
    fn convert_to(&self) -> f32 {
        (self.to_i32() as f32) / (i32::MAX as f32)
    }
}

impl ConvertTo<f64> for i24 {
    fn convert_to(&self) -> f64 {
        f64::from(self.to_i32()) / f64::from(i32::MAX)
    }
}

// i32 //
impl ConvertTo<i16> for i32 {
    fn convert_to(&self) -> i16 {
        (*self >> 16) as i16
    }
}

impl ConvertTo<i24> for i32 {
    fn convert_to(&self) -> i24 {
        i24::from_i32(*self >> 8)
    }
}

impl ConvertTo<Self> for i32 {
    fn convert_to(&self) -> Self {
        *self
    }
}

impl ConvertTo<f32> for i32 {
    fn convert_to(&self) -> f32 {
        (*self as f32 / Self::MAX as f32).clamp(-1.0, 1.0)
    }
}

impl ConvertTo<f64> for i32 {
    fn convert_to(&self) -> f64 {
        (f64::from(*self) / f64::from(Self::MAX)).clamp(-1.0, 1.0)
    }
}

// f32 //
impl ConvertTo<i16> for f32 {
    #[allow(clippy::cast_possible_truncation)]
    fn convert_to(&self) -> i16 {
        (*self * Self::from(i16::MAX))
            .clamp(Self::from(i16::MIN), Self::from(i16::MAX))
            .round() as i16
    }
}

impl ConvertTo<i24> for f32 {
    fn convert_to(&self) -> i24 {
        i24::from_i32(
            ((*self * (i32::MAX as f32)).clamp(i32::MIN as f32, i32::MAX as f32)).round() as i32,
        )
    }
}

impl ConvertTo<i32> for f32 {
    fn convert_to(&self) -> i32 {
        ((*self * (i32::MAX as Self)).clamp(i32::MIN as Self, i32::MAX as Self)).round() as i32
    }
}

impl ConvertTo<Self> for f32 {
    fn convert_to(&self) -> Self {
        *self
    }
}

impl ConvertTo<f64> for f32 {
    fn convert_to(&self) -> f64 {
        f64::from(*self)
    }
}

// f64 //
impl ConvertTo<i16> for f64 {
    #[allow(clippy::cast_possible_truncation)]
    fn convert_to(&self) -> i16 {
        ((*self * Self::from(i16::MAX)).clamp(Self::from(i16::MIN), Self::from(i16::MAX))).round()
            as i16
    }
}

impl ConvertTo<i24> for f64 {
    #[allow(clippy::cast_possible_truncation)]
    fn convert_to(&self) -> i24 {
        i24::from_i32(
            ((*self * Self::from(i32::MAX)).clamp(Self::from(i32::MIN), Self::from(i32::MAX)))
                .round() as i32,
        )
    }
}

impl ConvertTo<i32> for f64 {
    #[allow(clippy::cast_possible_truncation)]
    fn convert_to(&self) -> i32 {
        ((*self * Self::from(i32::MAX)).clamp(Self::from(i32::MIN), Self::from(i32::MAX))).round()
            as i32
    }
}

impl ConvertTo<f32> for f64 {
    #[allow(clippy::cast_possible_truncation)]
    fn convert_to(&self) -> f32 {
        *self as f32
    }
}

impl ConvertTo<Self> for f64 {
    fn convert_to(&self) -> Self {
        *self
    }
}

#[cfg(test)]
mod conversion_tests {

    use super::*;
    use std::fs::File;
    use std::io::BufRead;
    use std::path::Path;
    use std::str::FromStr;

    use approx_eq::assert_approx_eq;

    #[test]
    fn i16_to_f32() {
        let i16_samples: Vec<i16> =
            read_text_to_vec(Path::new("./test_resources/one_channel_i16.txt")).unwrap();
        let i16_samples: &[i16] = &i16_samples;

        let f32_samples: Vec<f32> =
            read_text_to_vec(Path::new("./test_resources/one_channel_f32.txt")).unwrap();
        let f32_samples: &[f32] = &f32_samples;
        for (expected_sample, actual_sample) in f32_samples.iter().zip(i16_samples) {
            let actual_sample: f32 = actual_sample.convert_to();
            assert_approx_eq!(*expected_sample as f64, actual_sample as f64, 1e-4);
        }
    }

    #[test]
    fn i16_to_f32_slice() {
        let i16_samples: Vec<i16> =
            read_text_to_vec(Path::new("./test_resources/one_channel_i16.txt")).unwrap();
        let i16_samples: Box<[i16]> = i16_samples.into_boxed_slice();
        let f32_samples: Vec<f32> =
            read_text_to_vec(Path::new("./test_resources/one_channel_f32.txt")).unwrap();

        let f32_samples: &[f32] = &f32_samples;
        let converted_i16_samples: Box<[f32]> = i16_samples.convert_slice();

        for (expected_sample, actual_sample) in converted_i16_samples.iter().zip(f32_samples) {
            assert_approx_eq!(*expected_sample as f64, *actual_sample as f64, 1e-4);
        }
    }

    #[test]
    fn f32_to_i16() {
        let i16_samples: Vec<i16> =
            read_text_to_vec(Path::new("./test_resources/one_channel_i16.txt")).unwrap();
        let i16_samples: &[i16] = &i16_samples;

        let f32_samples: Vec<f32> =
            read_text_to_vec(Path::new("./test_resources/one_channel_f32.txt")).unwrap();

        let f32_samples: &[f32] = &f32_samples;
        for (expected_sample, actual_sample) in i16_samples.iter().zip(f32_samples) {
            let converted_sample: i16 = actual_sample.convert_to();
            assert_eq!(
                *expected_sample, converted_sample,
                "Failed to convert sample {} to i16",
                actual_sample
            );
        }
    }

    #[cfg(test)]
    fn read_lines<P>(filename: P) -> std::io::Result<std::io::Lines<std::io::BufReader<File>>>
    where
        P: AsRef<Path>,
    {
        let file = File::open(filename)?;
        Ok(std::io::BufReader::new(file).lines())
    }

    #[cfg(test)]
    fn read_text_to_vec<T: FromStr>(fp: &Path) -> Result<Vec<T>, Box<dyn std::error::Error>>
    where
        <T as FromStr>::Err: std::error::Error + 'static,
    {
        let mut data = Vec::new();
        let lines = read_lines(fp)?;
        for line in lines {
            let line = line?;
            for sample in line.split(" ") {
                let parsed_sample: T = match sample.trim().parse::<T>() {
                    Ok(num) => num,
                    Err(err) => {
                        eprintln!("Failed to parse {}", sample);
                        panic!("{}", err)
                    }
                };
                data.push(parsed_sample);
            }
        }
        Ok(data)
    }
}
