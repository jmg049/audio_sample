// src/bin/audio_demo.rs
//
// A demonstration binary that showcases the visualization features
// of the audio_sample crate.

use std::env;
use std::f64::consts::PI;
use audio_sample::{AudioSample, ConvertTo, Samples};

#[cfg(feature = "visualization")]
use audio_sample::visualization::{
    waveform::{WaveformOptions, VisualizationStyle},
    multichannel::MultichannelOptions,
    spectral::SpectralOptions,
    inspect::InspectionOptions,
    terminal::{TerminalOptions, color},
};

fn main() {
    #[cfg(not(feature = "visualization"))]
    {
        println!("This demo requires the 'visualization' feature.");
        println!("Recompile with: cargo run --bin audio_demo --features visualization");
        return;
    }

    #[cfg(feature = "visualization")]
    {
        let args: Vec<String> = env::args().collect();
        let demo_type = if args.len() > 1 { &args[1] } else { "all" };

        match demo_type {
            "sine" => demo_sine_wave(),
            "stereo" => demo_stereo(),
            "shapes" => demo_waveform_shapes(),
            "spectrum" => demo_spectrum(),
            "terminal" => demo_terminal_colors(),
            "multichannel" => demo_multi_channel(),
            "all" | _ => demo_all(),
        }
    }
}

#[cfg(feature = "visualization")]
fn demo_all() {
    println!("{}Audio Sample Visualization Demo{}", color::BOLD, color::RESET);
    println!("==================================\n");

    demo_sine_wave();
    demo_stereo();
    demo_waveform_shapes();
    demo_spectrum();
    demo_terminal_colors();
    demo_multi_channel();
}

#[cfg(feature = "visualization")]
fn demo_sine_wave() {
    println!("{}Sine Wave Demo{}", color::BOLD, color::RESET);
    println!("=============\n");

    // Generate a sine wave
    let sine_samples = generate_sine_wave(440.0, 44100, 1000);
    
    // Display with default formatting
    println!("{}", sine_samples);
    
    // Display with custom options
    let options = WaveformOptions::new()
        .width(100)
        .height(15)
        .style(VisualizationStyle::Detailed)
        .show_axis(true);
        
    println!("\nCustom waveform view:");
    println!("{}", sine_samples.format_with_options(&options));
    
    println!("\nCompact view:");
    println!("{}", sine_samples.format_compact(100));
    
    println!("\n");
}

#[cfg(feature = "visualization")]
fn demo_stereo() {
    println!("{}Stereo Audio Demo{}", color::BOLD, color::RESET);
    println!("================\n");

    // Generate stereo audio (left: 440Hz, right: 880Hz)
    let stereo_samples = generate_stereo_audio(440.0, 880.0, 44100, 1000);
    
    // Display as interleaved multi-channel
    let options = MultichannelOptions::new()
        .width(100)
        .height(7)
        .style(VisualizationStyle::Detailed)
        .channel_labels(vec!["Left Channel".to_string(), "Right Channel".to_string()]);
        
    println!("{}", stereo_samples.format_multichannel(2, &options));
    
    // Extract and display individual channels
    println!("\nExtracted left channel:");
    let left_channel = stereo_samples.extract_channel(0, 2);
    println!("{}", left_channel.format_compact(100));
    
    println!("\nExtracted right channel:");
    let right_channel = stereo_samples.extract_channel(1, 2);
    println!("{}", right_channel.format_compact(100));
    
    println!("\n");
}

#[cfg(feature = "visualization")]
fn demo_waveform_shapes() {
    println!("{}Waveform Shapes Demo{}", color::BOLD, color::RESET);
    println!("===================\n");

    // Generate different waveform shapes
    let sine_samples = generate_sine_wave(440.0, 44100, 1000);
    let square_samples = generate_square_wave(440.0, 44100, 1000);
    let sawtooth_samples = generate_sawtooth_wave(440.0, 44100, 1000);
    let triangle_samples = generate_triangle_wave(440.0, 44100, 1000);
    
    // Display all waveforms with the same options
    let options = WaveformOptions::new()
        .width(100)
        .height(7)
        .style(VisualizationStyle::Detailed);
        
    println!("Sine wave:");
    println!("{}", sine_samples.format_with_options(&options));
    
    println!("\nSquare wave:");
    println!("{}", square_samples.format_with_options(&options));
    
    println!("\nSawtooth wave:");
    println!("{}", sawtooth_samples.format_with_options(&options));
    
    println!("\nTriangle wave:");
    println!("{}", triangle_samples.format_with_options(&options));
    
    // Compare all waveforms in compact view
    println!("\nCompact comparison:");
    println!("Sine:     {}", sine_samples.format_compact(100));
    println!("Square:   {}", square_samples.format_compact(100));
    println!("Sawtooth: {}", sawtooth_samples.format_compact(100));
    println!("Triangle: {}", triangle_samples.format_compact(100));
    
    println!("\n");
}

#[cfg(feature = "visualization")]
fn demo_spectrum() {
    println!("{}Spectrum Demo{}", color::BOLD, color::RESET);
    println!("=============\n");

    // Generate a complex tone (fundamental + harmonics)
    let complex_tone = generate_complex_tone(440.0, 44100, 1000);
    
    // Display waveform
    let wf_options = WaveformOptions::new()
        .width(100)
        .height(7)
        .style(VisualizationStyle::Detailed);
        
    println!("Complex tone waveform:");
    println!("{}", complex_tone.format_with_options(&wf_options));
    
    // Display spectrum
    let spec_options = SpectralOptions::new()
        .width(100)
        .height(20)
        .sample_rate(44100)
        .use_db_scale(true);
        
    println!("\nSpectrum visualization:");
    println!("{}", complex_tone.format_spectrum(&spec_options));
    
    println!("\n");
}

#[cfg(feature = "visualization")]
fn demo_terminal_colors() {
    println!("{}Terminal Colors Demo{}", color::BOLD, color::RESET);
    println!("===================\n");

    // Generate a sine wave
    let sine_samples = generate_sine_wave(440.0, 44100, 1000);
    
    // Display with colored terminal output
    let term_options = TerminalOptions::new()
        .width(100)
        .height(15)
        .use_colors(true)
        .unicode_support(true);
        
    println!("Colored waveform:");
    println!("{}", audio_sample::visualization::terminal::format_colored_waveform(&sine_samples, &term_options));
    
    // Display different waveforms with color
    println!("\nColored waveform shapes:");
    let square_samples = generate_square_wave(440.0, 44100, 1000);
    println!("{}", audio_sample::visualization::terminal::format_colored_waveform(&square_samples, &term_options));
    
    // Display without Unicode
    let ascii_options = TerminalOptions::new()
        .width(100)
        .height(15)
        .use_colors(true)
        .unicode_support(false);
        
    println!("\nASCII-only waveform (with colors):");
    println!("{}", audio_sample::visualization::terminal::format_colored_waveform(&sine_samples, &ascii_options));
    
    println!("\n");
}

#[cfg(feature = "visualization")]
fn demo_multi_channel() {
    println!("{}Multi-Channel Audio Demo{}", color::BOLD, color::RESET);
    println!("======================\n");

    // Generate 4-channel audio (simulated ambisonic B-format)
    let multichannel_samples = generate_ambisonic_audio(44100, 1000);
    
    // Display with custom options
    let options = MultichannelOptions::new()
        .width(100)
        .height(5)
        .style(VisualizationStyle::Detailed)
        .channel_labels(vec![
            "W (Omni)".to_string(),
            "X (Front-Back)".to_string(),
            "Y (Left-Right)".to_string(),
            "Z (Up-Down)".to_string()
        ]);
        
    println!("{}", multichannel_samples.format_multichannel(4, &options));
    
    // Display as compact
    let compact_options = MultichannelOptions::new()
        .width(100)
        .height(5)
        .style(VisualizationStyle::Compact)
        .channel_labels(vec![
            "W (Omni)".to_string(),
            "X (Front-Back)".to_string(),
            "Y (Left-Right)".to_string(),
            "Z (Up-Down)".to_string()
        ]);
        
    println!("\nCompact multi-channel view:");
    println!("{}", multichannel_samples.format_multichannel(4, &compact_options));
    
    println!("\n");
}

#[cfg(feature = "visualization")]
fn generate_sine_wave(freq: f64, sample_rate: u32, num_samples: usize) -> Samples<f32> {
    let mut samples = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let t = i as f64 / sample_rate as f64;
        let amplitude = (2.0 * PI * freq * t).sin() as f32;
        samples.push(amplitude);
    }
    
    Samples::from(samples)
}

#[cfg(feature = "visualization")]
fn generate_square_wave(freq: f64, sample_rate: u32, num_samples: usize) -> Samples<f32> {
    let mut samples = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let t = i as f64 / sample_rate as f64;
        let cycle_position = (t * freq) % 1.0;
        let amplitude = if cycle_position < 0.5 { 1.0 } else { -1.0 };
        samples.push(amplitude as f32);
    }
    
    Samples::from(samples)
}

#[cfg(feature = "visualization")]
fn generate_sawtooth_wave(freq: f64, sample_rate: u32, num_samples: usize) -> Samples<f32> {
    let mut samples = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let t = i as f64 / sample_rate as f64;
        let cycle_position = (t * freq) % 1.0;
        let amplitude = (2.0 * cycle_position - 1.0) as f32;
        samples.push(amplitude);
    }
    
    Samples::from(samples)
}

#[cfg(feature = "visualization")]
fn generate_triangle_wave(freq: f64, sample_rate: u32, num_samples: usize) -> Samples<f32> {
    let mut samples = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let t = i as f64 / sample_rate as f64;
        let cycle_position = (t * freq) % 1.0;
        let amplitude = if cycle_position < 0.5 {
            // Rising edge (0 to 1)
            (4.0 * cycle_position - 1.0) as f32
        } else {
            // Falling edge (1 to 0)
            (3.0 - 4.0 * cycle_position) as f32
        };
        samples.push(amplitude);
    }
    
    Samples::from(samples)
}

#[cfg(feature = "visualization")]
fn generate_complex_tone(fundamental: f64, sample_rate: u32, num_samples: usize) -> Samples<f32> {
    let mut samples = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let t = i as f64 / sample_rate as f64;
        
        // Fundamental
        let mut amplitude = (2.0 * PI * fundamental * t).sin();
        
        // 2nd harmonic (half amplitude)
        amplitude += 0.5 * (2.0 * PI * (2.0 * fundamental) * t).sin();
        
        // 3rd harmonic (third amplitude)
        amplitude += 0.33 * (2.0 * PI * (3.0 * fundamental) * t).sin();
        
        // 4th harmonic (quarter amplitude)
        amplitude += 0.25 * (2.0 * PI * (4.0 * fundamental) * t).sin();
        
        // Normalize amplitude to avoid clipping
        amplitude *= 0.5;
        
        samples.push(amplitude as f32);
    }
    
    Samples::from(samples)
}

#[cfg(feature = "visualization")]
fn generate_stereo_audio(freq_left: f64, freq_right: f64, sample_rate: u32, num_frames: usize) -> Samples<f32> {
    let mut samples = Vec::with_capacity(num_frames * 2);
    
    for i in 0..num_frames {
        let t = i as f64 / sample_rate as f64;
        
        // Left channel
        let left = (2.0 * PI * freq_left * t).sin() as f32;
        samples.push(left);
        
        // Right channel
        let right = (2.0 * PI * freq_right * t).sin() as f32;
        samples.push(right);
    }
    
    Samples::from(samples)
}

#[cfg(feature = "visualization")]
fn generate_ambisonic_audio(sample_rate: u32, num_frames: usize) -> Samples<f32> {
    let mut samples = Vec::with_capacity(num_frames * 4);
    
    for i in 0..num_frames {
        let t = i as f64 / sample_rate as f64;
        
        // W channel (omnidirectional)
        let w = 0.7 * (2.0 * PI * 300.0 * t).sin() as f32;
        samples.push(w);
        
        // X channel (front-back)
        let x = 0.5 * (2.0 * PI * 400.0 * t).sin() as f32;
        samples.push(x);
        
        // Y channel (left-right)
        let y = 0.3 * (2.0 * PI * 500.0 * t).sin() as f32;
        samples.push(y);
        
        // Z channel (up-down)
        let z = 0.2 * (2.0 * PI * 600.0 * t).sin() as f32;
        samples.push(z);
    }
    
    Samples::from(samples)
}