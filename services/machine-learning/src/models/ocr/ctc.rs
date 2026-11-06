use std::path::Path;

use anyhow::{Context, Result, anyhow};
use ndarray::{ArrayD, Ix2, Ix3};

use crate::numpy::{argmax, pairwise_sum, sum};

pub fn py_splitlines(text: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                lines.push(std::mem::take(&mut current));
            }
            '\n' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}'
            | '\u{2029}' => {
                lines.push(std::mem::take(&mut current));
            }
            c => current.push(c),
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

pub struct Greedy {
    pub indices: Vec<Vec<i32>>,
    pub confidence: Vec<Vec<f32>>,
}

pub fn kept(indices: &[i32]) -> Vec<bool> {
    indices
        .iter()
        .enumerate()
        .map(|(step, index)| *index != 0 && (step == 0 || indices[step - 1] != *index))
        .collect()
}

pub fn picked(outputs: &[ArrayD<f32>]) -> Result<Greedy> {
    let indices = outputs
        .first()
        .context("no indices")?
        .clone()
        .into_dimensionality::<Ix2>()?;
    let confidence = outputs
        .get(1)
        .context("no confidence")?
        .clone()
        .into_dimensionality::<Ix2>()?;
    Ok(Greedy {
        indices: indices
            .outer_iter()
            .map(|row| row.iter().map(|value| *value as i32).collect())
            .collect(),
        confidence: confidence.outer_iter().map(|row| row.to_vec()).collect(),
    })
}

pub fn logits(outputs: &[ArrayD<f32>]) -> Result<Greedy> {
    let raw = outputs.first().context("no logits")?;
    let shape = raw.shape();
    let (batch, steps, classes) = (shape[0], shape[1], shape[shape.len() - 1]);
    let flat: Vec<f32> = raw.iter().copied().collect();
    let mut greedy = Greedy {
        indices: Vec::with_capacity(batch),
        confidence: Vec::with_capacity(batch),
    };
    for b in 0..batch {
        let rows: Vec<&[f32]> = (0..steps)
            .map(|t| &flat[(b * steps + t) * classes..(b * steps + t + 1) * classes])
            .collect();
        let indices: Vec<i32> = rows.iter().map(|row| argmax(row) as i32).collect();
        let keep = kept(&indices);
        let confidence = rows
            .iter()
            .zip(indices.iter())
            .zip(keep.iter())
            .map(|((row, index), keep)| {
                if !keep {
                    return 0.0;
                }
                let best = row[*index as usize];
                let weights: Vec<f32> = row
                    .iter()
                    .filter(|value| **value > best - 16.0)
                    .map(|value| (value - best).exp())
                    .collect();
                let total = weights[0] + pairwise_sum(&weights[1..]);
                1.0 / total
            })
            .collect();
        greedy.indices.push(indices);
        greedy.confidence.push(confidence);
    }
    Ok(greedy)
}

pub fn probabilities(outputs: &[ArrayD<f32>]) -> Result<Greedy> {
    let probs = outputs
        .first()
        .context("no probabilities")?
        .as_standard_layout();
    let probs = probs.view().into_dimensionality::<Ix3>()?;
    let mut greedy = Greedy {
        indices: Vec::new(),
        confidence: Vec::new(),
    };
    for row in probs.outer_iter() {
        let mut indices = Vec::new();
        let mut confidence = Vec::new();
        for step in row.outer_iter() {
            let step = step.as_slice().context("probabilities are contiguous")?;
            let index = argmax(step);
            indices.push(index as i32);
            confidence.push(step[index]);
        }
        greedy.indices.push(indices);
        greedy.confidence.push(confidence);
    }
    Ok(greedy)
}

pub type GreedyFn = fn(&[ArrayD<f32>]) -> Result<Greedy>;

pub fn greedy_for(output_name: &str) -> GreedyFn {
    match output_name {
        "ctc_indices" => picked,
        "ctc_logits" => logits,
        _ => probabilities,
    }
}

pub struct CtcDecoder {
    pub charset: Vec<String>,
    greedy: GreedyFn,
}

impl CtcDecoder {
    pub fn new(charset: Vec<String>, greedy: GreedyFn) -> Self {
        let mut full = Vec::with_capacity(charset.len() + 2);
        full.push(String::new());
        full.extend(charset);
        full.push(" ".to_owned());
        Self {
            charset: full,
            greedy,
        }
    }

    pub fn from_file(charset_path: &Path, greedy: GreedyFn) -> Result<Self> {
        if !charset_path.is_file() {
            return Err(anyhow!(
                "Recognition charset not found: {}",
                charset_path.display()
            ));
        }
        Ok(Self::new(
            py_splitlines(&std::fs::read_to_string(charset_path)?),
            greedy,
        ))
    }

    pub fn len(&self) -> usize {
        self.charset.len()
    }

    pub fn is_empty(&self) -> bool {
        self.charset.is_empty()
    }

    pub fn call(&self, outputs: &[ArrayD<f32>]) -> Result<(Vec<String>, Vec<f32>)> {
        let greedy = (self.greedy)(outputs)?;
        self.decode(&greedy)
    }

    pub fn decode(&self, greedy: &Greedy) -> Result<(Vec<String>, Vec<f32>)> {
        let mut texts = Vec::with_capacity(greedy.indices.len());
        let mut scores = Vec::with_capacity(greedy.indices.len());
        for (indices, probs) in greedy.indices.iter().zip(greedy.confidence.iter()) {
            let keep = kept(indices);
            let masked: Vec<f32> = probs
                .iter()
                .zip(keep.iter())
                .map(|(prob, keep)| if *keep { *prob } else { 0.0 })
                .collect();
            let count = keep.iter().filter(|keep| **keep).count().max(1);
            scores.push((f64::from(sum(&masked)) / count as f64) as f32);
            let text = indices
                .iter()
                .zip(keep.iter())
                .filter(|(_, keep)| **keep)
                .map(|(index, _)| {
                    self.charset
                        .get(*index as usize)
                        .map(String::as_str)
                        .context("index out of range")
                })
                .collect::<Result<String>>()?;
            texts.push(text);
        }
        Ok((texts, scores))
    }
}
