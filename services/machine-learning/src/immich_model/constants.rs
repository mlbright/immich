pub const FACE_DETECTION_SIZE: i64 = 640;
pub const OCR_DETECTION_RATIOS: [f64; 4] = [1.0, 4.0 / 3.0, 3.0 / 2.0, 2.0];
pub const OCR_DETECTION_SIZES: [i64; 3] = [736, 1088, 1440];
pub const OCR_RECOGNITION_WIDTHS: [i64; 6] = [224, 320, 448, 640, 1280, 2048];

pub fn ocr_canvases(size: i64) -> Vec<(i64, i64)> {
    let mut canvases: Vec<(i64, i64)> = Vec::new();
    for ratio in OCR_DETECTION_RATIOS {
        let long = (ratio * size as f64 / 32.0).ceil() as i64 * 32;
        for canvas in [(long, size), (size, long)] {
            if !canvases.contains(&canvas) {
                canvases.push(canvas);
            }
        }
    }
    canvases
}
