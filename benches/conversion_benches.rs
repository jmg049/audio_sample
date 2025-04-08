use criterion::{AxisScale, Criterion, PlotConfiguration, criterion_group, criterion_main};

use rand::Rng;

use audio_sample::samples::Samples;
use i24::i24;
use std::time::Duration;

macro_rules! bench_sample_conversions {
    ($c:expr, $($from_type:ty => [$($to_type:ty),+]),+) => {
        const DURATIONS: [Duration; 3] = [
            Duration::from_secs(1),
            Duration::from_secs(10),
            Duration::from_secs(60),
        ];
        const SAMPLE_RATES: [usize; 3] = [8000, 16000, 44100];
        const N_CHANNELS: usize = 1;

        $(
            $(
                for duration in &DURATIONS {
                    for sample_rate in SAMPLE_RATES {
                        let group_name = format!(
                            "Samples conversion {} to {} - {}s - {}Hz - {}ch",
                            stringify!($from_type),
                            stringify!($to_type),
                            duration.as_secs(),
                            sample_rate,
                            N_CHANNELS
                        );

                        let warm_up_time = std::cmp::max(3, duration.as_secs() / 10);
                        let measurement_time = std::cmp::max(15, duration.as_secs() / 2);
                        let sample_size = if duration.as_secs() > 600 { 50 } else { 100 };

                        let mut group = $c.benchmark_group(&group_name);
                        group
                            .sample_size(sample_size)
                            .warm_up_time(std::time::Duration::from_secs(warm_up_time))
                            .measurement_time(std::time::Duration::from_secs(measurement_time))
                            .noise_threshold(0.05)
                            .significance_level(0.1)
                            .confidence_level(0.95);

                        group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

                        let samples: Samples<$from_type> = noise(*duration, sample_rate, N_CHANNELS).convert();

                        group.bench_function(&group_name, |b| {
                            b.iter(|| {
                                let _samples = samples.convert::<$to_type>();
                            });
                        });
                    }
                }
            )+
        )+
    };
}

fn bench_samples_conversion(c: &mut Criterion) {
    bench_sample_conversions!(
        c,
        i16 => [i24, i32, f32, f64],
        i24 => [i16, i32, f32, f64],
        i32 => [i16, i24, f32, f64],
        f32 => [i16, i24, i32, f64],
        f64 => [i16, i24, i32, f32]
    );
}

fn noise(duration_sec: Duration, sample_rate: usize, n_channels: usize) -> Samples<f32> {
    let n_samples = duration_sec.as_secs_f32() * sample_rate as f32 * n_channels as f32;
    let n_samples = n_samples.ceil() as usize;

    let mut rng = rand::rng();
    let data = (0..n_samples)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect::<Vec<f32>>();

    Samples::from(data.into_boxed_slice())
}

criterion_group!(benches, bench_samples_conversion,);
criterion_main!(benches);
