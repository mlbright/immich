use anyhow::{Context, Result, bail};
use jpegxl_rs::decode::{Data, Pixels};
use jpegxl_rs::decoder_builder;
use jpegxl_rs::parallel::threads_runner::ThreadsRunner;

const CONTAINER_SIGNATURE: &[u8] = b"\x00\x00\x00\x0c\x4a\x58\x4c\x20\x0d\x0a\x87\x0a";

pub enum JxlImage {
    Jpeg(Vec<u8>),
    Raw {
        mode: &'static str,
        width: u32,
        height: u32,
        data: Vec<u8>,
    },
}

pub fn accept(data: &[u8]) -> bool {
    data.starts_with(b"\xff\x0a")
        || data.starts_with(CONTAINER_SIGNATURE)
        || data.get(4..7) == Some(b"JXL")
}

fn mode(
    num_channels: u32,
    has_alpha_channel: bool,
    pixels: Option<&Pixels>,
) -> Result<&'static str> {
    let mode = match (num_channels, has_alpha_channel) {
        (1, false) => "L",
        (1, true) => "LA",
        (3, false) => "RGB",
        (3, true) => "RGBA",
        _ => bail!("Unsupported color mode"),
    };
    Ok(match (mode, pixels) {
        ("L", Some(Pixels::Uint16(_))) => "I;16",
        ("L", Some(Pixels::Float(_) | Pixels::Float16(_))) => "F",
        _ => mode,
    })
}

fn pixels_to_bytes_8bit(pixels: Pixels) -> Vec<u8> {
    match pixels {
        Pixels::Uint8(pixels) => pixels,
        Pixels::Uint16(pixels) => pixels.into_iter().map(|pixel| (pixel >> 8) as u8).collect(),
        Pixels::Float(pixels) => pixels
            .into_iter()
            .map(|pixel| (pixel * 255.0) as u8)
            .collect(),
        Pixels::Float16(pixels) => pixels
            .into_iter()
            .map(|pixel| (pixel.to_f32() * 255.0) as u8)
            .collect(),
    }
}

fn pixels_to_bytes(pixels: Pixels) -> Vec<u8> {
    match pixels {
        Pixels::Uint8(pixels) => pixels,
        Pixels::Uint16(pixels) => pixels
            .iter()
            .flat_map(|pixel| pixel.to_ne_bytes())
            .collect(),
        Pixels::Float(pixels) => pixels
            .iter()
            .flat_map(|pixel| pixel.to_ne_bytes())
            .collect(),
        Pixels::Float16(pixels) => pixels
            .iter()
            .flat_map(|pixel| pixel.to_f32().to_ne_bytes())
            .collect(),
    }
}

pub fn open(data: &[u8], num_threads: isize) -> Result<JxlImage> {
    let parallel_runner = ThreadsRunner::new(None, usize::try_from(num_threads).ok())
        .context("Could not create JxlThreadsRunner")?;
    let decoder = decoder_builder()
        .icc_profile(true)
        .parallel_runner(&parallel_runner)
        .build()?;
    let (info, image) = decoder.reconstruct(data)?;
    let pixels = match &image {
        Data::Pixels(pixels) => Some(pixels),
        Data::Jpeg(_) => None,
    };
    let mode = mode(info.num_color_channels, info.has_alpha_channel, pixels)?;
    Ok(match image {
        Data::Jpeg(jpeg) => JxlImage::Jpeg(jpeg),
        Data::Pixels(pixels) => JxlImage::Raw {
            mode,
            width: info.width,
            height: info.height,
            data: match info.num_color_channels {
                1 => pixels_to_bytes(pixels),
                3 => pixels_to_bytes_8bit(pixels),
                _ => bail!("image color channels must be 1 or 3"),
            },
        },
    })
}
