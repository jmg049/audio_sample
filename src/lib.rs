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
//! - `i24`: 24-bit signed integer samples - From the [i24 crate](https://github.com/jmg049/i24). In-between PCM_16 and PCM_32 in
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
//! let f32_buffer: Samples<f32> = i16_buffer.convert_sequence().into();
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

pub mod samples;
pub use samples::Samples;
use std::alloc::Layout;
use std::fmt::Debug;

use bytemuck::Pod;
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
        i24::saturating_from_i32((*self as i32) << 8)
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
        (self.to_i32() as f32) / (i24::MAX.to_i32() as f32)
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
        i24::try_from_i32(*self >> 8).unwrap()
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
        i24::saturating_from_i32(
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
        i24::saturating_from_i32(
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
#[cfg(test)]
mod sample_conversion_tests {
    use super::*;
    use ::i24::i24;

    /// Helper function for approximate float comparison
    fn approx_eq_f32(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }
    
    /// Helper function for approximate float comparison
    fn approx_eq_f64(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    /// Tests for identity conversions (same type to same type)
    /// These should be exact with no loss of precision
    #[test]
    fn test_identity_conversions() {
        // Integer identity
        let i16_val: i16 = 12345;
        let i16_id_val: i16 = i16_val.convert_to();
        assert_eq!(i16_id_val, i16_val);

        let i24_val = i24::saturating_from_i32(1234567);
        let converted: i24 = i24_val.convert_to();
        assert_eq!(converted.to_i32(), i24_val.to_i32());
        
        let i32_val: i32 = 1234567890;
        let i32_id_val: i32 = i32_val.convert_to();
        assert_eq!(i32_id_val, i32_val);

        // Float identity
        let f32_val: f32 = 0.12345;
        let f32_id_val: f32 = f32_val.convert_to();
        assert_eq!(f32_id_val, f32_val);
        
        let f64_val: f64 = 0.12345678901234;
        let f64_id_val: f64 = f64_val.convert_to();
        assert_eq!(f64_id_val, f64_val);
    }

    /// Tests for converting between integer types
    /// Conversions between different bit depths should scale appropriately
    #[test]
    fn test_integer_to_integer_conversion() {
        // i16 to i32 (widening)
        let i16_val: i16 = 12345;
        let expected_i32 = (i16_val as i32) << 16;
        let i32_val: i32 = i16_val.convert_to();
        assert_eq!(i32_val, expected_i32);

        // i32 to i16 (narrowing)
        let i32_val: i32 = 12345 << 16;
        let expected_i16 = 12345;
        let i16_val: i16 = i32_val.convert_to();
        assert_eq!(i16_val, expected_i16);

        // i16 to i24 (widening)
        let i16_val: i16 = 12345;
        let expected_i24 = i24::saturating_from_i32((i16_val as i32) << 8);
        let i24_val: i24 = i16_val.convert_to();
        assert_eq!(i24_val.to_i32(), expected_i24.to_i32());

        // i24 to i16 (narrowing)
        let i32_for_i24 = 12345 << 8;
        let i24_val = i24::saturating_from_i32(i32_for_i24);
        let expected_i16 = (i32_for_i24 >> 8) as i16;
        let i16_val: i16 = i24_val.convert_to();
        assert_eq!(i16_val, expected_i16);

        // i24 to i32 (widening)
        let i24_val = i24::saturating_from_i32(12345);
        let expected_i32 = 12345 << 8;
        let i32_val: i32 = i24_val.convert_to();
        assert_eq!(i32_val, expected_i32);

        // i32 to i24 (narrowing)
        let i32_val: i32 = 12345 << 8;
        let expected_i24 = i24::saturating_from_i32(i32_val >> 8);
        let i24_val: i24 = i32_val.convert_to();
        assert_eq!(i24_val.to_i32(), expected_i24.to_i32());
    }

    /// Tests for converting between float types
    /// Should maintain precision within float capabilities
    #[test]
    fn test_float_to_float_conversion() {
        // f32 to f64 (widening)
        let f32_val: f32 = 0.12345;
        let f64_val: f64 = f32_val.convert_to();
        assert!(approx_eq_f64(f64_val, f32_val as f64, 1e-6));

        // f64 to f32 (narrowing)
        let f64_val: f64 = 0.12345678901234;
        let f32_val: f32 = f64_val.convert_to();
        assert!(approx_eq_f64(f32_val as f64, f64_val, 1e-6));
    }

    /// Tests for converting from integer to float
    /// Integer values should map to the -1.0 to 1.0 float range
    #[test]
    fn test_integer_to_float_conversion() {
        // i16 to f32
        let i16_val: i16 = i16::MAX / 2; // 50% of maximum
        let f32_val: f32 = i16_val.convert_to();
        assert!(approx_eq_f32(f32_val, 0.5, 1e-4), "{} != {}", f32_val, 0.5);

        // i16 to f64
        let i16_val: i16 = i16::MAX / 4; // 25% of maximum
        let f64_val: f64 = i16_val.convert_to();
        assert!(approx_eq_f64(f64_val, 0.25, 1e-4), "{} != {}", f64_val, 0.25);

        // i24 to f32
        let i24_val: i24 = i24::MAX / i24!(8); // ~12.5% of maximum
        let f32_val: f32 = i24_val.convert_to();
        let expected = 1.0 / 8.0;

        assert!(approx_eq_f32(f32_val, expected as f32, 1e-4), "{} != {}", f32_val, expected as f32);

        // i32 to f32
        let i32_val: i32 = i32::MAX / 10; // 10% of maximum
        let f32_val: f32 = i32_val.convert_to();
        assert!(approx_eq_f32(f32_val, 0.1, 1e-4));

        // Negative values
        let i16_val: i16 = -(i16::MAX / 2); // -50% (approx)
        let f32_val: f32 = i16_val.convert_to();
        assert!(approx_eq_f32(f32_val, -0.5, 1e-4));
    }

    /// Tests for converting from float to integer
    /// Float range -1.0 to 1.0 should map to full integer range
    #[test]
    fn test_float_to_integer_conversion() {
        // f32 to i16
        let f32_val: f32 = 0.5;  // 50%
        let i16_val: i16 = f32_val.convert_to();
        assert_eq!(i16_val, 16384); // approx i16::MAX/2

        // f32 to i24s
        let f32_val: f32 = 0.25;  // 25%

        let i24_val: i24 = f32_val.convert_to();
        let expected = i24::saturating_from_i32((i32::MAX as f64 * 0.25).round() as i32);

        let diff = (i24_val.to_i32() - expected.to_i32()).abs();
        assert!(diff <= 1, "F32 to I24 conversion failed: {} != {} with diff = {}", i24_val.to_i32(), expected.to_i32(), diff);

        // f32 to i32
        let f32_val: f32 = 0.75;  // 75%
        let i32_val: i32 = f32_val.convert_to();

        let diff = (i32_val - (i32::MAX as f64 * 0.75).round() as i32).abs();
        assert!(diff <= 1, "F32 to I32 conversion failed: {} != {} with diff = {}", i32_val, (i32::MAX as f64 * 0.75).round() as i32, diff);

        // f64 to i16
        let f64_val: f64 = -0.5;  // -50%
        let i16_val: i16 = f64_val.convert_to();
        assert_eq!(i16_val, -16384); // approx -i16::MAX/2
    }

    /// Tests for edge cases - maximum/minimum values
    #[test]
    fn test_edge_cases_max_min() {
        // Maximum values
        let i16_max = i16::MAX;
        let f32_val: f32 = i16_max.convert_to();
        assert!(approx_eq_f32(f32_val, 1.0, 1e-6));
        
        let i32_val: i32 = i16_max.convert_to();
        assert_eq!(i32_val, i32::MAX & 0xFFFF0000u32 as i32);

        let i32_max = i32::MAX;
        let f32_val: f32 = i32_max.convert_to();
        assert!(approx_eq_f32(f32_val, 1.0, 1e-6));
        
        let i16_val: i16 = i32_max.convert_to();
        assert_eq!(i16_val, i16::MAX);


        let f32_max: f32 = 1.0;
        let i16_val: i16 = f32_max.convert_to();
        let diff = (i16_val as f32 - i16_max as f32).abs();
        assert!(diff <= 1.0, "F32 to I16 conversion failed: {} != {} with diff = {}", i16_val, i16_max, diff);
        
        let i32_val: i32 = f32_max.convert_to();
        let diff = (i32_val as f32 - i32_max as f32).abs();
        assert!(diff <= 1.0, "F32 to I32 conversion failed: {} != {} with diff = {}", i32_val, i32_max, diff);

        // Minimum values
        let i16_min = i16::MIN;
        let f32_val: f32 = i16_min.convert_to();
        assert!(approx_eq_f32(f32_val, -1.0, 1e-6));
        
        let i32_min = i32::MIN;
        let f32_val: f32 = i32_min.convert_to();
        assert!(approx_eq_f32(f32_val, -1.0, 1e-6));
        
        let i16_val: i16 = i32_min.convert_to();
        assert_eq!(i16_val, i16::MIN);

        let f32_min: f32 = -1.0;
        let i16_val: i16 = f32_min.convert_to();
        let diff = (i16_val as f32 - i16_min as f32).abs();
        assert!(diff <= 1.0, "F32 to I16 conversion failed: {} != {} with diff = {}", i16_val, i16_min, diff);
        
        let i32_val: i32 = f32_min.convert_to();
        assert_eq!(i32_val, i32::MIN);
    }

    /// Tests for clamping behavior on out-of-range values
    #[test]
    fn test_clamping_behavior() {
        // Floats outside -1.0 to 1.0 should be clamped
        let f32_over: f32 = 1.5;
        let i16_val: i16 = f32_over.convert_to();
        assert_eq!(i16_val, i16::MAX);
        
        let f32_under: f32 = -1.5;
        let i16_val: i16 = f32_under.convert_to();
        assert_eq!(i16_val, i16::MIN);

        let f64_over: f64 = 2.0;
        let i32_val: i32 = f64_over.convert_to();
        assert_eq!(i32_val, i32::MAX);
        
        let f64_under: f64 = -2.0;
        let i32_val: i32 = f64_under.convert_to();
        assert_eq!(i32_val, i32::MIN);
    }

    /// Tests for round-trip conversions (should have minimal loss)
    #[test]
    fn test_round_trip_conversions() {
        // i16 -> f32 -> i16
        for val in [-16384, -8192, -1, 0, 1, 8192, 16384, 32767] {
            let original: i16 = val;
            let f32_val: f32 = original.convert_to();
            let round_trip: i16 = f32_val.convert_to();
            
            // For most values, round-trip should be exact
            // For some edge cases, allow tiny differences due to float precision
            let diff = (original as i32 - round_trip as i32).abs();
            assert!(diff <= 1, "Round trip i16->f32->i16 failed for {}: {} != {}", original, original, round_trip);
        }

        // i32 -> f32 -> i32
        // Here we expect some precision loss due to f32 not having enough bits
        for val in [0, 1, -1, i32::MAX/4, -i32::MAX/4] {
            let original: i32 = val;
            let f32_val: f32 = original.convert_to();
            let round_trip: i32 = f32_val.convert_to();
            
            // Allow larger differences for i32 due to precision limitations
            let diff_ratio = (original as f64 - round_trip as f64).abs() / (original as f64).abs().max(1.0);
            assert!(diff_ratio < 0.0001, "Round trip i32->f32->i32 failed for {}: {} != {}", original, original, round_trip);
        }

        // f32 -> i16 -> f32
        for val in [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0] {
            let original: f32 = val;
            let i16_val: i16 = original.convert_to();
            let round_trip: f32 = i16_val.convert_to();
            
            // Expect some quantization error due to i16's limited resolution
            assert!(approx_eq_f32(round_trip, original, 0.0004));
        }
    }

    /// Tests for sequence conversions
    #[test]
    fn test_sequence_conversion() {
        // Empty sequence
        let empty: Box<[i16]> = Box::new([]);
        let converted: Box<[f32]> = empty.convert_sequence();
        assert_eq!(converted.len(), 0);

        // Typical sequence
        let i16_samples: Box<[i16]> = Box::new([0, 16384, -16384, 32767, -32768]);
        let i16_samples_len = i16_samples.len();
        let f32_samples: Box<[f32]> = i16_samples.convert_sequence();
        
        assert_eq!(f32_samples.len(), i16_samples_len, "Length mismatch {} != {}", i16_samples_len, f32_samples.len());
        assert!(approx_eq_f32(f32_samples[0], 0.0, 1e-4));
        assert!(approx_eq_f32(f32_samples[1], 0.5, 1e-4));
        assert!(approx_eq_f32(f32_samples[2], -0.5, 1e-4));
        assert!(approx_eq_f32(f32_samples[3], 1.0, 1e-4));
        assert!(approx_eq_f32(f32_samples[4], -1.0, 1e-4));
    }
}

#[cfg(test)]
mod samples_storage_tests {
    use crate::samples::Samples;
    use memmap2::MmapOptions;
    use std::io::Write;
    use std::sync::Arc;
    use tempfile::tempfile;

    /// Tests for creating Samples with owned storage
    #[test]
    fn test_owned_creation() {
        let data = Box::new([1i16, 2, 3, 4, 5]);
        let samples = Samples::new(data);
        
        assert_eq!(samples.len(), 5);
        assert_eq!(samples[0], 1);
        assert_eq!(samples[4], 5);
    }

    /// Tests for creating Samples with borrowed storage
    #[test]
    fn test_borrowed_creation() {
        // Using 'static lifetime data
        static DATA: [i16; 5] = [1, 2, 3, 4, 5];
        
        let samples = Samples::from_slice(&DATA);
        
        assert_eq!(samples.len(), 5);
        assert_eq!(samples[0], 1);
        assert_eq!(samples[4], 5);
    }

    /// Tests for creating Samples with shared storage
    #[test]
    fn test_shared_creation() {
        let data: Arc<[i16]> = Arc::new([1, 2, 3, 4, 5]);
        let samples = Samples::from_shared(data.clone());
        
        assert_eq!(samples.len(), 5);
        assert_eq!(samples[0], 1);
        assert_eq!(samples[4], 5);
        
        // Verify it's actually shared
        assert_eq!(Arc::strong_count(&data), 2);
    }

    /// Tests for creating Samples with memory-mapped storage
    #[test]
    fn test_memory_mapped_creation() -> Result<(), Box<dyn std::error::Error>> {
        // Create a temporary file with some i16 samples
        let mut temp_file = tempfile()?;
        let samples_data: [i16; 5] = [1, 2, 3, 4, 5];
        
        // Write the raw bytes
        let bytes = bytemuck::cast_slice(&samples_data);
        temp_file.write_all(bytes)?;
        temp_file.flush()?;
        
        // Memory map the file (non-mutable)
        let mmap = unsafe { MmapOptions::new().map(&temp_file)? };
        let mmap_arc = Arc::new(mmap);
        
        // Create samples from the memory-mapped file
        let samples = Samples::<i16>::from_mmap(mmap_arc, 0, 5);
        
        assert_eq!(samples.len(), 5);
        assert_eq!(samples[0], 1);
        assert_eq!(samples[4], 5);
        
        Ok(())
    }

    /// Tests for accessing samples using as_slice
    #[test]
    fn test_as_slice() {
        // Test all storage types
        let owned_data = Box::new([1i16, 2, 3]);
        let owned_samples = Samples::new(owned_data);
        
        static BORROWED_DATA: [i16; 3] = [4, 5, 6];
        let borrowed_samples = Samples::from_slice(&BORROWED_DATA);
        
        let shared_data: Arc<[i16]> = Arc::new([7, 8, 9]);
        let shared_samples = Samples::from_shared(shared_data);
        
        // Verify as_slice returns correct data
        assert_eq!(owned_samples.as_slice(), &[1, 2, 3]);
        assert_eq!(borrowed_samples.as_slice(), &[4, 5, 6]);
        assert_eq!(shared_samples.as_slice(), &[7, 8, 9]);
    }

    /// Tests for as_mut_slice which may convert storage
    #[test]
    fn test_as_mut_slice() {
        // Start with borrowed samples (which should be converted)
        static BORROWED_DATA: [i16; 3] = [1, 2, 3];
        let mut borrowed_samples = Samples::from_slice(&BORROWED_DATA);
        
        // Get mutable slice (should convert to owned)
        let slice = borrowed_samples.as_mut_slice();
        
        // Modify the data
        slice[0] = 10;
        slice[1] = 20;
        slice[2] = 30;
        
        // Verify changes took effect
        assert_eq!(borrowed_samples.as_slice(), &[10, 20, 30]);
        
        // Shared data should also convert to owned if there are multiple refs
        let shared_data: Arc<[i16]> = Arc::new([4, 5, 6]);
        let shared_data_clone = shared_data.clone();
        let mut shared_samples = Samples::from_shared(shared_data);
        
        // Get mutable slice (should convert to owned since Arc has multiple refs)
        let slice = shared_samples.as_mut_slice();
        slice[0] = 40;
        
        // Verify our change took effect but the original is unchanged
        assert_eq!(shared_samples.as_slice(), &[40, 5, 6]);
        assert_eq!(&*shared_data_clone, &[4, 5, 6]);
    }

    /// Tests for as_bytes
    #[test]
    fn test_as_bytes() {
        let data = Box::new([0x1234i16, 0x5678]);
        let samples = Samples::new(data);
        
        let bytes = samples.as_bytes();
        
        // The exact byte representation depends on endianness
        // For little-endian systems:
        if cfg!(target_endian = "little") {
            assert_eq!(bytes, &[0x34, 0x12, 0x78, 0x56]);
        } else {
            // Big-endian
            assert_eq!(bytes, &[0x12, 0x34, 0x56, 0x78]);
        }
    }

    /// Tests for len and is_empty
    #[test]
    fn test_len_and_is_empty() {
        // Empty samples
        let empty_samples = Samples::<i16>::new(Box::new([]));
        assert_eq!(empty_samples.len(), 0);
        assert!(empty_samples.is_empty());
        
        // Non-empty samples
        let samples = Samples::<i16>::new(Box::new([1, 2, 3]));
        assert_eq!(samples.len(), 3);
        assert!(!samples.is_empty());
    }

    /// Tests for to_owned
    #[test]
    fn test_to_owned() {
        // Test with borrowed samples
        static BORROWED_DATA: [i16; 3] = [1, 2, 3];
        let borrowed_samples = Samples::from_slice(&BORROWED_DATA);
        
        let mut owned_samples = borrowed_samples.to_owned();
        
        // Data should be the same
        assert_eq!(owned_samples.as_slice(), &[1, 2, 3]);
        
        // We can modify the owned version without affecting the original
        let mut_slice = owned_samples.as_mut_slice();
        mut_slice[0] = 10;
        
        assert_eq!(owned_samples.as_slice(), &[10, 2, 3]);
        assert_eq!(borrowed_samples.as_slice(), &[1, 2, 3]);
    }

}
#[cfg(test)]
mod samples_processing_tests {
    use crate::samples::{Samples, WindowType};

    // Helper function for approximate float comparison
    fn approx_eq_f32(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    /// Tests for same-type conversion (potentially zero-copy)
    #[test]
    fn test_same_type_conversion() {
        // Create a regular Samples instead of memory-mapped
        let samples = Samples::<i16>::new(Box::new([0x1234, 0x5678]));
        
        // Convert to same type
        let converted = samples.convert::<i16>();
        
        // Data should be the same
        assert_eq!(converted.as_slice(), &[0x1234, 0x5678]);
    }

    /// Tests for different-type conversion
    #[test]
    fn test_different_type_conversion() {
        let i16_samples = Samples::<i16>::new(Box::new([0, 16384, -16384, 32767, -32768]));
        
        // Convert to f32
        let f32_samples = i16_samples.convert::<f32>();
        
        // Verify conversion
        assert_eq!(f32_samples.len(), i16_samples.len());
        assert!(approx_eq_f32(f32_samples[0], 0.0, 1e-4));
        assert!(approx_eq_f32(f32_samples[1], 0.5, 1e-4));
        assert!(approx_eq_f32(f32_samples[2], -0.5, 1e-4));
        assert!(approx_eq_f32(f32_samples[3], 1.0, 1e-4));
        assert!(approx_eq_f32(f32_samples[4], -1.0, 1e-4));
    }

    /// Tests for the map method
    #[test]
    fn test_map() {
        let samples = Samples::<i16>::new(Box::new([1, 2, 3, 4, 5]));
        
        // Simple mapping - double each value
        let doubled = samples.map(|x| x * 2);
        
        assert_eq!(doubled.as_slice(), &[2, 4, 6, 8, 10]);
        
        // More complex mapping - modulo
        let mod3 = samples.map(|x| x % 3);
        
        assert_eq!(mod3.as_slice(), &[1, 2, 0, 1, 2]);
        
        // Map with float samples
        let float_samples = Samples::<f32>::new(Box::new([0.1, 0.2, 0.3]));
        let scaled = float_samples.map(|x| x * 10.0);
        
        assert!(approx_eq_f32(scaled[0], 1.0, 1e-6));
        assert!(approx_eq_f32(scaled[1], 2.0, 1e-6));
        assert!(approx_eq_f32(scaled[2], 3.0, 1e-6));
    }

    /// Tests for extract_channel
    #[test]
    fn test_extract_channel() {
        // Create interleaved stereo data [L,R,L,R,L,R]
        let interleaved = Samples::<i16>::new(Box::new([10, 20, 30, 40, 50, 60]));
        
        // Extract left channel (index 0)
        let left = interleaved.extract_channel(0, 2);
        assert_eq!(left.as_slice(), &[10, 30, 50]);
        
        // Extract right channel (index 1)
        let right = interleaved.extract_channel(1, 2);
        assert_eq!(right.as_slice(), &[20, 40, 60]);
        
        // Try to extract non-existent channel
        let invalid = interleaved.extract_channel(2, 2);
        assert!(invalid.is_empty());
        
        // Test with surround sound (5.1)
        let surround = Samples::<i16>::new(
            Box::new([1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16])
        );
        
        let center = surround.extract_channel(2, 6);
        assert_eq!(center.as_slice(), &[3, 13]);
    }

    /// Tests for window functions
    #[test]
    fn test_window_functions() {
        // Test rectangular window (no change)
        let samples = Samples::<f32>::new(Box::new([0.5, 0.5, 0.5, 0.5]));
        let rectangular = samples.window(WindowType::Rectangular);
        
        // Should be identical
        for (orig, windowed) in samples.iter().zip(rectangular.iter()) {
            assert!(approx_eq_f32(*orig, *windowed, 1e-6));
        }
        
        // Test Hann window
        let hann = samples.window(WindowType::Hann);
        
        // Expected Hann window values for 4 samples
        let expected_hann = [
            0.0, // 0.5 * (1 - cos(0))
            0.5 * 0.75, // 0.5 * (1 - cos(π/3))
            0.5 * 0.75, // 0.5 * (1 - cos(2π/3))
            0.0, // 0.5 * (1 - cos(π))
        ];
        
        for (i, (windowed, expected)) in hann.iter().zip(expected_hann.iter()).enumerate() {
            assert!(approx_eq_f32(*windowed, *expected, 1e-2), 
                "Hann window sample {} incorrect", i);
        }
        
        // Test Hamming window
        let hamming = samples.window(WindowType::Hamming);
        
        // Test Blackman window
        let blackman = samples.window(WindowType::Blackman);
        
        // Just verify they're different from the original
        // (Full validation would require specific expected values)
        assert!(hamming.iter().zip(samples.iter()).any(|(h, s)| (h - s).abs() > 0.01));
        assert!(blackman.iter().zip(samples.iter()).any(|(b, s)| (b - s).abs() > 0.01));
    }
}

#[cfg(test)]
mod samples_trait_tests {
    use crate::samples::Samples;
    use std::fmt::Write;

    /// Tests for Deref trait (slice-like access)
    #[test]
    fn test_deref() {
        let samples = Samples::<i16>::new(Box::new([1, 2, 3, 4, 5]));
        
        // Access via index (using Deref)
        assert_eq!(samples[0], 1);
        assert_eq!(samples[4], 5);
        
        // Use slice methods
        assert_eq!(samples.len(), 5);
        assert_eq!(samples.iter().sum::<i16>(), 15);
        
        // Slice syntax
        assert_eq!(&samples[1..4], &[2, 3, 4]);
    }
    
    /// Tests for DerefMut trait (mutable slice-like access)
    #[test]
    fn test_deref_mut() {
        let mut samples = Samples::<i16>::new(Box::new([1, 2, 3, 4, 5]));
        
        // Modify via index (using DerefMut)
        samples[0] = 10;
        samples[4] = 50;
        
        assert_eq!(samples.as_slice(), &[10, 2, 3, 4, 50]);
        
        // Use mutable slice methods
        samples.swap(1, 3);
        
        assert_eq!(samples.as_slice(), &[10, 4, 3, 2, 50]);
    }
    

    /// Tests for Display trait
    #[test]
    fn test_display() {
        let samples = Samples::<i16>::new(Box::new([1, 2, 3]));
        
        let mut output = String::new();
        write!(output, "{}", samples).unwrap();
        
        // Check that the output contains the type and length
        assert!(output.contains("Samples"));
        assert!(output.contains("i16"));
        assert!(output.contains("3 samples"));
    }
}

#[cfg(test)]
mod misc_tests {
    use super::*;
    
    /// Tests for allocation of sample buffers
    #[test]
    fn test_alloc_sample_buffer() {
        // Allocate empty buffer
        let empty: Box<[i16]> = alloc_sample_buffer(0);
        assert_eq!(empty.len(), 0);
        
        // Allocate typical buffer
        let typical: Box<[i16]> = alloc_sample_buffer(100);
        assert_eq!(typical.len(), 100);
        
        // Allocate large buffer
        let large: Box<[i16]> = alloc_sample_buffer(10000);
        assert_eq!(large.len(), 10000);
    }
    
    /// Test for potential panic conditions
    #[test]
    #[should_panic]
    fn test_alloc_sample_buffer_overflow() {
        // Try to allocate a buffer that's too large
        // This should panic with a clear error message
        let _huge: Box<[i16]> = alloc_sample_buffer(usize::MAX / 2);
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
