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

use i24::i24;
use std::{fmt::Debug, ops::Index};

// A read-only reference to some data of type T: AudioSample, which is also needs to be iterable.
pub struct Samples<'a, T: AudioSample> {
    samples: &'a [T],
}

impl<'a, T: AudioSample> Samples<'a, T> {
    pub const fn new(samples: &'a [T]) -> Self {
        Self { samples }
    }

    pub const fn len(&'_ self) -> usize {
        self.samples.len()
    }

    pub const fn is_empty(&'_ self) -> bool {
        self.samples.is_empty()
    }

    /// Converts the struct into an owned
    pub fn to_owned<S: AudioSample>(&'_ self) -> Vec<S>
    where
        T: ConvertTo<S>,
    {
        Vec::from(self.samples.convert_slice())
    }

    pub const fn iter(&'a self) -> SamplesIter<'a, T> {
        SamplesIter {
            idx: 0,
            samples: self,
        }
    }
}

impl<'a, T: AudioSample> AsRef<[T]> for Samples<'a, T> {
    fn as_ref(&self) -> &'a [T] {
        self.samples
    }
}

impl<T: AudioSample> Index<usize> for Samples<'_, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.samples[index]
    }
}

pub struct SamplesIter<'a, T: AudioSample> {
    idx: usize,
    samples: &'a Samples<'a, T>,
}

impl<T: AudioSample> Iterator for SamplesIter<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.samples[self.idx];

        self.idx += 1;
        if self.idx >= self.samples.len() {
            return None;
        }

        Some(current)
    }
}

pub trait AudioSample:
    Clone
    + Copy
    + ConvertTo<i16>
    + ConvertTo<i24>
    + ConvertTo<i32>
    + ConvertTo<f32>
    + ConvertTo<f64>
    + Send
    + Sync
    + Debug
{
}

/// Trait for converting between audio sample types
/// The type ``T`` must implement the ``AudioSample`` trait
pub trait ConvertTo<T: AudioSample> {
    fn convert_to(&self) -> T
    where
        Self: Sized + AudioSample;
}

/// Trait for converting between audio sample types in a slice
/// The type ``T`` must implement the ``AudioSample`` trait
pub trait ConvertSlice<T: AudioSample> {
    fn convert_slice(&self) -> Box<[T]>;
}

impl AudioSample for i16 {}
impl AudioSample for i24 {}
impl AudioSample for i32 {}
impl AudioSample for f32 {}
impl AudioSample for f64 {}

impl<T: AudioSample> ConvertSlice<T> for Box<[i16]>
where
    i16: ConvertTo<T>,
{
    fn convert_slice(&self) -> Box<[T]> {
        self.iter()
            .map(|sample| sample.convert_to())
            .collect::<Vec<T>>()
            .into_boxed_slice()
    }
}

// Similar implementations for other types
impl<T: AudioSample> ConvertSlice<T> for Box<[i24]>
where
    i24: ConvertTo<T>,
{
    fn convert_slice(&self) -> Box<[T]> {
        self.iter()
            .map(|sample| sample.convert_to())
            .collect::<Vec<T>>()
            .into_boxed_slice()
    }
}

impl<T: AudioSample> ConvertSlice<T> for Box<[i32]>
where
    i32: ConvertTo<T>,
{
    fn convert_slice(&self) -> Box<[T]> {
        self.iter()
            .map(|sample| sample.convert_to())
            .collect::<Vec<T>>()
            .into_boxed_slice()
    }
}

impl<T: AudioSample> ConvertSlice<T> for Box<[f32]>
where
    f32: ConvertTo<T>,
{
    fn convert_slice(&self) -> Box<[T]> {
        self.iter()
            .map(|sample| sample.convert_to())
            .collect::<Vec<T>>()
            .into_boxed_slice()
    }
}

impl<T: AudioSample, S: AudioSample> ConvertSlice<T> for &'_ [S]
where
    S: ConvertTo<T>,
{
    fn convert_slice(&self) -> Box<[T]> {
        self.iter()
            .map(|sample| sample.convert_to())
            .collect::<Vec<T>>()
            .into_boxed_slice()
    }
}

// Add an extension trait for efficient batch operations
pub trait AudioBatchConversion<T: AudioSample> {
    /// Perform an optimized batch conversion
    fn convert_batch(&self) -> Vec<T>;
}

impl<T: AudioSample> ConvertSlice<T> for Box<[f64]>
where
    f64: ConvertTo<T>,
{
    fn convert_slice(&self) -> Box<[T]> {
        self.iter()
            .map(|sample| sample.convert_to())
            .collect::<Vec<T>>()
            .into_boxed_slice()
    }
}

/// Trait that bundles all the audio conversion constraints
pub trait AudioConversionCapable: AudioSample
where
    i16: ConvertTo<Self>,
    i24: ConvertTo<Self>,
    i32: ConvertTo<Self>,
    f32: ConvertTo<Self>,
    f64: ConvertTo<Self>,
    Box<[i16]>: ConvertSlice<Self>,
    Box<[i24]>: ConvertSlice<Self>,
    Box<[i32]>: ConvertSlice<Self>,
    Box<[f32]>: ConvertSlice<Self>,
    Box<[f64]>: ConvertSlice<Self>,
{
}

// Blanket implementation for all types that satisfy the constraints
impl<T: AudioSample> AudioConversionCapable for T
where
    i16: ConvertTo<T>,
    i24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    Box<[i16]>: ConvertSlice<T>,
    Box<[i24]>: ConvertSlice<T>,
    Box<[i32]>: ConvertSlice<T>,
    Box<[f32]>: ConvertSlice<T>,
    Box<[f64]>: ConvertSlice<T>,
{
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
