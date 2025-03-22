# Audio Sample Conversion Library

This crate provides functionality for working with audio samples and
converting between different audio sample formats. It focuses on correctness,
performance, and ease of use.

## Supported Sample Types

- `i16`: 16-bit signed integer samples - Common in WAV files and CD-quality audio
- `i24`: 24-bit signed integer samples - From the [i24] crate. In-between PCM_16 and PCM_32 in
    terms of quality and space on disk.
- `i32`: 32-bit signed integer samples (high-resolution audio)
- `f32`: 32-bit floating point samples (common in audio processing)
- `f64`: 64-bit floating point samples (high-precision audio processing)

## Features

- **Type Safety**: Using Rust's type system to ensure correct conversions
- **High Performance**: Simple code that enables the compiler to produce fast code. The
    [AudioSample] trait simply enables working with some primitives within the context of audio
    processing - floats between -1.0 and 1.0 etc.

## Usage Examples

### Basic Sample Conversion

```rust
use audio_sample::{AudioSample, ConvertTo};

// Convert an i16 sample to floating point
let i16_sample: i16 = i16::MAX / 2; // 50% of max amplitude
let f32_sample: f32 = i16_sample.convert_to();
assert!((f32_sample - 0.5).abs() < 0.0001);

// Convert a floating point sample to i16
let f32_sample: f32 = -0.75;
let i16_sample: i16 = f32_sample.convert_to();
assert_eq!(i16_sample, -24575); // -75% of max amplitude
```

### Converting Buffers of Samples

```rust
use audio_sample::ConvertSlice;

// Using ConvertSlice trait for Box<[T]>
let i16_buffer: Box<[i16]> = vec![0, 16384, -16384, 32767].into_boxed_slice();
let f32_buffer: Box<[f32]> = i16_buffer.convert_slice();
```
## Implementation Details

### Integer Scaling

When converting between integer formats of different bit depths:

- **Widening conversions** (e.g., i16 to i32): The samples are shifted left to preserve amplitude.
- **Narrowing conversions** (e.g., i32 to i16): The samples are shifted right, which may lose precision.

### Float to Integer Conversion

- Floating-point samples are assumed to be in the range -1.0 to 1.0.
- They are scaled to the full range of the target integer type.
- Values are rounded to the nearest integer rather than truncated.
- Values outside the target range are clamped to prevent overflow.

### Integer to Float Conversion

- Integer samples are scaled to the range -1.0 to 1.0.
- The maximum positive integer value maps to 1.0.
- The minimum negative integer value maps to -1.0.


## Bugs / Issues
Report them on the [Github Page](<https://www.github.com/jmg049/audio_sample>) and I will try and get to it as soon as I can :)

