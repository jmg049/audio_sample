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
//! use audio_sample::{ConvertSequence, Samples};
//!
//! // Using ConvertSlice trait for Box<[T]>
//! let i16_buffer: Box<[i16]> = vec![0, 16384, -16384, 32767].into_boxed_slice();
//! let f32_buffer: Samples<f32> = i16_buffer.convert_sequence();
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

use std::alloc::Layout;
use std::any::TypeId;
use std::convert::{AsMut, AsRef};
use std::fmt::{Debug, Display};
use std::ops::{Deref, DerefMut};

use bytemuck::{Pod, cast_slice};
use i24::i24;

/// Allocates an exact sized heap buffer for samples.
pub(crate) fn alloc_sample_buffer<T>(len: usize) -> Box<[T]>
where
    T: AudioSample + Copy + Debug,
{
    if len == 0 {
        return <Box<[T]>>::default();
    }

    let layout = match Layout::array::<T>(len) {
        Ok(layout) => layout,
        Err(_) => panic!("Failed to allocate buffer of size {}", len),
    };

    let ptr = unsafe { std::alloc::alloc(layout) as *mut T };
    let slice_ptr: *mut [T] = core::ptr::slice_from_raw_parts_mut(ptr, len);
    unsafe { Box::from_raw(slice_ptr) }
}

/// Marker trait for audio sample types.
pub trait AudioSample:
    Copy
    + Pod
    + ConvertTo<i16>
    + ConvertTo<i32>
    + ConvertTo<i24>
    + ConvertTo<f32>
    + ConvertTo<f64>
    + Sync
    + Send
    + Debug
{
}

impl AudioSample for i16 {}
impl AudioSample for i24 {}
impl AudioSample for i32 {}
impl AudioSample for f32 {}
impl AudioSample for f64 {}

/// Trait for converting one sample type to another.
pub trait ConvertTo<T: AudioSample> {
    fn convert_to(&self) -> T;
}

pub trait ConvertSequence<T: AudioSample> {
    fn convert_sequence(self) -> Box<[T]>;
}

impl<T: AudioSample, F> ConvertSequence<T> for Box<[F]>
where
    F: AudioSample + ConvertTo<T>,
{
    fn convert_sequence(self) -> Box<[T]> {
        let mut out: Box<[T]> = alloc_sample_buffer(self.len());
        for i in 0..self.len() {
            out[i] = self[i].convert_to();
        }
        out
    }
}

// ========================
// Conversion implementations
// ========================

// i16 //
impl ConvertTo<i16> for i16 {
    #[inline(always)]
    fn convert_to(&self) -> i16 {
        *self
    }
}

impl ConvertTo<i24> for i16 {
    #[inline(always)]
    fn convert_to(&self) -> i24 {
        i24::from_i32((*self as i32) << 8)
    }
}

impl ConvertTo<i32> for i16 {
    #[inline(always)]
    fn convert_to(&self) -> i32 {
        (*self as i32) << 16
    }
}

impl ConvertTo<f32> for i16 {
    #[inline(always)]
    fn convert_to(&self) -> f32 {
        ((*self as f32) / (i16::MAX as f32)).clamp(-1.0, 1.0)
    }
}

impl ConvertTo<f64> for i16 {
    #[inline(always)]
    fn convert_to(&self) -> f64 {
        ((*self as f64) / (i16::MAX as f64)).clamp(-1.0, 1.0)
    }
}

// i24 //
impl ConvertTo<i16> for i24 {
    #[inline(always)]
    fn convert_to(&self) -> i16 {
        (self.to_i32() >> 8) as i16
    }
}

impl ConvertTo<i24> for i24 {
    #[inline(always)]
    fn convert_to(&self) -> i24 {
        *self
    }
}

impl ConvertTo<i32> for i24 {
    #[inline(always)]
    fn convert_to(&self) -> i32 {
        self.to_i32() << 8
    }
}

impl ConvertTo<f32> for i24 {
    #[inline(always)]
    fn convert_to(&self) -> f32 {
        (self.to_i32() as f32) / (i32::MAX as f32)
    }
}

impl ConvertTo<f64> for i24 {
    #[inline(always)]
    fn convert_to(&self) -> f64 {
        (self.to_i32() as f64) / (i32::MAX as f64)
    }
}

// i32 //
impl ConvertTo<i16> for i32 {
    #[inline(always)]
    fn convert_to(&self) -> i16 {
        (*self >> 16) as i16
    }
}

impl ConvertTo<i24> for i32 {
    #[inline(always)]
    fn convert_to(&self) -> i24 {
        i24::from_i32(*self >> 8)
    }
}

impl ConvertTo<i32> for i32 {
    #[inline(always)]
    fn convert_to(&self) -> i32 {
        *self
    }
}

impl ConvertTo<f32> for i32 {
    #[inline(always)]
    fn convert_to(&self) -> f32 {
        ((*self as f32) / (i32::MAX as f32)).clamp(-1.0, 1.0)
    }
}

impl ConvertTo<f64> for i32 {
    #[inline(always)]
    fn convert_to(&self) -> f64 {
        ((*self as f64) / (i32::MAX as f64)).clamp(-1.0, 1.0)
    }
}

// f32 //
impl ConvertTo<i16> for f32 {
    #[inline(always)]
    fn convert_to(&self) -> i16 {
        ((*self * (i16::MAX as f32)).clamp(i16::MIN as f32, i16::MAX as f32)).round() as i16
    }
}

impl ConvertTo<i24> for f32 {
    #[inline(always)]
    fn convert_to(&self) -> i24 {
        i24::from_i32(
            ((*self * (i32::MAX as f32)).clamp(i32::MIN as f32, i32::MAX as f32)).round() as i32,
        )
    }
}

impl ConvertTo<i32> for f32 {
    #[inline(always)]
    fn convert_to(&self) -> i32 {
        ((*self * (i32::MAX as f32)).clamp(i32::MIN as f32, i32::MAX as f32)).round() as i32
    }
}

impl ConvertTo<f32> for f32 {
    #[inline(always)]
    fn convert_to(&self) -> f32 {
        *self
    }
}

impl ConvertTo<f64> for f32 {
    #[inline(always)]
    fn convert_to(&self) -> f64 {
        *self as f64
    }
}

// f64 //
impl ConvertTo<i16> for f64 {
    #[inline(always)]
    fn convert_to(&self) -> i16 {
        ((*self * (i16::MAX as f64)).clamp(i16::MIN as f64, i16::MAX as f64)).round() as i16
    }
}

impl ConvertTo<i24> for f64 {
    #[inline(always)]
    fn convert_to(&self) -> i24 {
        i24::from_i32(
            ((*self * (i32::MAX as f64)).clamp(i32::MIN as f64, i32::MAX as f64)).round() as i32,
        )
    }
}

impl ConvertTo<i32> for f64 {
    #[inline(always)]
    fn convert_to(&self) -> i32 {
        ((*self * (i32::MAX as f64)).clamp(i32::MIN as f64, i32::MAX as f64)).round() as i32
    }
}

impl ConvertTo<f32> for f64 {
    #[inline(always)]
    fn convert_to(&self) -> f32 {
        *self as f32
    }
}

impl ConvertTo<f64> for f64 {
    #[inline(always)]
    fn convert_to(&self) -> f64 {
        *self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Samples<T>
where
    T: AudioSample,
{
    pub(crate) samples: Box<[T]>,
}

impl<T> AsRef<[T]> for Samples<T>
where
    T: AudioSample,
{
    fn as_ref(&self) -> &[T] {
        &self.samples
    }
}

impl<T> AsMut<[T]> for Samples<T>
where
    T: AudioSample,
{
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.samples
    }
}

impl<T> Deref for Samples<T>
where
    T: AudioSample,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.samples
    }
}

impl<T> DerefMut for Samples<T>
where
    T: AudioSample,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.samples
    }
}

impl<T> From<Vec<T>> for Samples<T>
where
    T: AudioSample,
{
    fn from(samples: Vec<T>) -> Self {
        Samples {
            samples: samples.into_boxed_slice(),
        }
    }
}

impl<T> From<&[T]> for Samples<T>
where
    T: AudioSample,
{
    fn from(samples: &[T]) -> Self {
        Samples {
            samples: Box::from(samples),
        }
    }
}

impl<T> From<Box<[T]>> for Samples<T>
where
    T: AudioSample,
{
    fn from(samples: Box<[T]>) -> Self {
        Samples { samples }
    }
}

impl<T> From<&[u8]> for Samples<T>
where
    T: AudioSample,
{
    fn from(bytes: &[u8]) -> Self {
        let casted_samples: &[T] = cast_slice::<u8, T>(bytes);
        Samples {
            samples: Box::from(casted_samples),
        }
    }
}

impl<T> Display for Samples<T>
where
    T: AudioSample + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.samples)
    }
}

impl<T> Samples<T>
where
    T: AudioSample,
{
    /// Constructs a new Samples struct from a boxed slice of audio samples.
    pub const fn new(samples: Box<[T]>) -> Self {
        Self { samples }
    }

    #[inline(always)]
    pub fn convert<F: AudioSample>(self) -> Samples<F>
    where
        T: ConvertTo<F>,
        Box<[T]>: ConvertSequence<F>,
    {
        if TypeId::of::<T>() == TypeId::of::<F>() {
            let data: Box<[T]> = self.samples;
            return Samples {
                samples: Box::from(cast_slice::<T, F>(&data)),
            };
        }
        let converted_samples = self.samples.convert_sequence();
        Samples {
            samples: converted_samples,
        }
    }

    /// Converts the boxed slice of samples to the corresponding bytes.
    pub fn as_bytes(&self) -> &[u8] {
        cast_slice::<T, u8>(&self.samples)
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
        let converted_i16_samples: Box<[f32]> = i16_samples.convert_sequence();

        for (_, (expected_sample, actual_sample)) in
            converted_i16_samples.iter().zip(f32_samples).enumerate()
        {
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
