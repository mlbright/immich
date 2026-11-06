use anyhow::Result;
use ndarray::Array2;
use opencv::core::{
    BORDER_CONSTANT, Mat, Point, Point2f, Rect, RotatedRect, Scalar, Size2f, Vector,
};
use opencv::imgproc;
use opencv::prelude::*;

pub type Quad = [[f32; 2]; 4];

pub fn order_points_clockwise(points: &Quad) -> Quad {
    let mut by_x = *points;
    by_x.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
    let (left_first, left_second) = (by_x[0], by_x[1]);
    let (right_first, right_second) = (by_x[2], by_x[3]);
    let (top_left, bottom_left) = if left_first[1] > left_second[1] {
        (left_second, left_first)
    } else {
        (left_first, left_second)
    };
    let (top_right, bottom_right) = if right_first[1] > right_second[1] {
        (right_second, right_first)
    } else {
        (right_first, right_second)
    };
    [top_left, top_right, bottom_right, bottom_left]
}

fn span(values: &[f32], limit: i64) -> (i64, i64) {
    let lo = values.iter().copied().fold(f32::INFINITY, f32::min).floor() as i64;
    let hi = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil() as i64;
    (lo.max(0).min(limit - 1), hi.max(0).min(limit - 1))
}

fn quad(points: &Mat) -> Result<Quad> {
    let mut quad = [[0.0f32; 2]; 4];
    for (row, point) in quad.iter_mut().enumerate() {
        point[0] = *points.at_2d::<f32>(row as i32, 0)?;
        point[1] = *points.at_2d::<f32>(row as i32, 1)?;
    }
    Ok(quad)
}

pub struct DBPostProcess {
    thresh: f32,
    max_candidates: usize,
    unclip_ratio: f64,
    min_size: f64,
    dilation_kernel: Option<Mat>,
}

impl DBPostProcess {
    pub fn new(
        thresh: f64,
        max_candidates: usize,
        unclip_ratio: f64,
        use_dilation: bool,
    ) -> Result<Self> {
        Ok(Self {
            thresh: thresh as f32,
            max_candidates,
            unclip_ratio,
            min_size: 3.0,
            dilation_kernel: if use_dilation {
                Some(Mat::new_rows_cols_with_default(
                    2,
                    2,
                    opencv::core::CV_8UC1,
                    Scalar::all(1.0),
                )?)
            } else {
                None
            },
        })
    }

    pub fn call(
        &self,
        probs: &Array2<f32>,
        ori_shape: (usize, usize),
        box_thresh: f64,
    ) -> Result<(Vec<Quad>, Vec<f32>)> {
        let (height, width) = probs.dim();
        let mask: Vec<u8> = probs
            .iter()
            .map(|value| u8::from(*value > self.thresh))
            .collect();
        let mut bitmap =
            Mat::new_rows_cols_with_data(height as i32, width as i32, &mask)?.try_clone()?;
        if let Some(kernel) = &self.dilation_kernel {
            let mut dilated = Mat::default();
            imgproc::dilate(
                &bitmap,
                &mut dilated,
                kernel,
                Point::new(-1, -1),
                1,
                BORDER_CONSTANT,
                imgproc::morphology_default_border_value()?,
            )?;
            bitmap = dilated;
        }
        let pred_values: Vec<f32> = probs.iter().copied().collect();
        let pred =
            Mat::new_rows_cols_with_data(height as i32, width as i32, &pred_values)?.try_clone()?;
        let (boxes, scores) =
            self.boxes_from_bitmap(&pred, &bitmap, ori_shape.1, ori_shape.0, box_thresh)?;
        Ok(Self::filter_det_res(
            boxes,
            scores,
            ori_shape.0,
            ori_shape.1,
        ))
    }

    fn boxes_from_bitmap(
        &self,
        pred: &Mat,
        bitmap: &Mat,
        dest_width: usize,
        dest_height: usize,
        box_thresh: f64,
    ) -> Result<(Vec<Quad>, Vec<f32>)> {
        let size = bitmap.size()?;
        let (height, width) = (size.height as f64, size.width as f64);
        let mut contours: Vector<Vector<Point>> = Vector::new();
        imgproc::find_contours(
            bitmap,
            &mut contours,
            imgproc::RETR_LIST,
            imgproc::CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        )?;

        let mut boxes = Vec::new();
        let mut scores = Vec::new();
        for contour in contours.iter().take(self.max_candidates) {
            let rect = imgproc::min_area_rect(&contour)?;
            if f64::from(rect.size.width.min(rect.size.height)) < self.min_size {
                continue;
            }
            let mut points = Mat::default();
            imgproc::box_points(rect, &mut points)?;
            let score = Self::box_score(pred, &quad(&points)?)?;
            if box_thresh > score {
                continue;
            }
            let (unclipped, sside) = self.unclip(&rect)?;
            if sside < self.min_size + 2.0 {
                continue;
            }
            boxes.push(unclipped);
            scores.push(score as f32);
        }
        let factors = [dest_width as f64 / width, dest_height as f64 / height];
        let limits = [dest_width as f32, dest_height as f32];
        for quad in &mut boxes {
            for point in quad.iter_mut() {
                for axis in 0..2 {
                    let scaled = (f64::from(point[axis]) * factors[axis]) as f32;
                    point[axis] = scaled.round_ties_even().clamp(0.0, limits[axis]);
                }
            }
        }
        Ok((boxes, scores))
    }

    pub fn box_score(probs: &Mat, poly: &Quad) -> Result<f64> {
        let size = probs.size()?;
        let xs: Vec<f32> = poly.iter().map(|point| point[0]).collect();
        let ys: Vec<f32> = poly.iter().map(|point| point[1]).collect();
        let (xmin, xmax) = span(&xs, i64::from(size.width));
        let (ymin, ymax) = span(&ys, i64::from(size.height));
        let mut mask = Mat::zeros(
            (ymax - ymin + 1) as i32,
            (xmax - xmin + 1) as i32,
            opencv::core::CV_8UC1,
        )?
        .to_mat()?;
        let points: Vector<Point> = poly
            .iter()
            .map(|point| Point::new(point[0] as i32, point[1] as i32))
            .collect();
        let polygons: Vector<Vector<Point>> = Vector::from_iter([points]);
        imgproc::fill_poly(
            &mut mask,
            &polygons,
            Scalar::all(1.0),
            imgproc::LINE_8,
            0,
            Point::new(-xmin as i32, -ymin as i32),
        )?;
        let region = Mat::roi(
            probs,
            Rect::new(
                xmin as i32,
                ymin as i32,
                (xmax - xmin + 1) as i32,
                (ymax - ymin + 1) as i32,
            ),
        )?;
        Ok(opencv::core::mean(&region, &mask)?[0])
    }

    fn unclip(&self, rect: &RotatedRect) -> Result<(Quad, f64)> {
        let width = f64::from(rect.size.width);
        let height = f64::from(rect.size.height);
        let grow = width * height * self.unclip_ratio / (width + height);
        let grown = RotatedRect::new(
            Point2f::new(rect.center.x, rect.center.y),
            Size2f::new((width + grow) as f32, (height + grow) as f32),
            rect.angle,
        )?;
        let mut points = Mat::default();
        imgproc::box_points(grown, &mut points)?;
        Ok((quad(&points)?, width.min(height) + grow))
    }

    fn filter_det_res(
        boxes: Vec<Quad>,
        scores: Vec<f32>,
        img_height: usize,
        img_width: usize,
    ) -> (Vec<Quad>, Vec<f32>) {
        let limits = [(img_width as f32) - 1.0, (img_height as f32) - 1.0];
        let mut kept_boxes = Vec::new();
        let mut kept_scores = Vec::new();
        for (quad, score) in boxes.into_iter().zip(scores) {
            let mut ordered = order_points_clockwise(&quad);
            for point in &mut ordered {
                point[0] = point[0].clamp(0.0, limits[0]);
                point[1] = point[1].clamp(0.0, limits[1]);
            }
            let side = |point: [f32; 2]| {
                let dx = point[0] - ordered[0][0];
                let dy = point[1] - ordered[0][1];
                0.0 + (dx * dx + dy * dy)
            };
            if side(ordered[1]).min(side(ordered[3])) >= 16.0 {
                kept_boxes.push(ordered);
                kept_scores.push(score);
            }
        }
        (kept_boxes, kept_scores)
    }
}
