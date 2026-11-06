use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use tracing::error;

use super::ort::{GraphSpec, Manifest, UNREADABLE, fresh};
use crate::config::settings;
use crate::immich_model::runtime::apply_rewrites;
use crate::onnx::{self, DecodeError};

pub fn main() -> ExitCode {
    let spec: GraphSpec = match serde_json::from_reader(io::stdin()) {
        Ok(spec) => spec,
        Err(error) => {
            error!("Cannot read the graph to prepare: {error}");
            return ExitCode::FAILURE;
        }
    };
    match locked(&spec) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) if unreadable(&error) => ExitCode::from(UNREADABLE as u8),
        Err(error) => {
            error!("{error:?}");
            ExitCode::FAILURE
        }
    }
}

fn locked(spec: &GraphSpec) -> Result<()> {
    let lock = File::create(spec.model_path.with_extension("lock"))?;
    lock.lock()?;
    if fresh(spec)?.is_none() {
        prepare(spec)?;
    }
    Ok(())
}

fn unreadable(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause.is::<DecodeError>()
            || cause.downcast_ref::<ort::Error>().is_some_and(|error| {
                matches!(
                    error.code(),
                    ort::ErrorCode::NoSuchFile | ort::ErrorCode::InvalidProtobuf
                )
            })
    })
}

pub fn prepare(spec: &GraphSpec) -> Result<()> {
    let directory = spec.directory()?;
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory)?;
    let mut graph = rewritten(spec, narrowed(spec)?)?;
    if spec.provider() == "CPUExecutionProvider" {
        graph = prepacked(spec, &graph)?;
    }
    let parent = spec.model_path.parent().unwrap_or(Path::new(""));
    let manifest = Manifest {
        facts: spec.facts()?,
        graph: graph
            .strip_prefix(parent)
            .with_context(|| format!("{} is outside {}", graph.display(), parent.display()))?
            .to_string_lossy()
            .into_owned(),
    };
    fs::write(spec.manifest()?, serde_json::to_string(&manifest)?)?;
    Ok(())
}

fn narrowed(spec: &GraphSpec) -> Result<PathBuf> {
    if !spec.half() {
        return Ok(spec.model_path.clone());
    }
    bail!("Deriving a float16 graph is not ported yet")
}

fn rewritten(spec: &GraphSpec, graph: PathBuf) -> Result<PathBuf> {
    if settings().legacy_models() {
        return Ok(graph);
    }
    apply_rewrites(&graph, &spec.plan())
}

fn prepacked(spec: &GraphSpec, graph: &Path) -> Result<PathBuf> {
    let destination = spec.directory()?.join("model.onnx");
    let options = spec
        .sess_options()?
        .with_optimized_model_path(&destination)
        .map_err(ort::Error::<()>::from)?
        .with_config_entry(
            "session.optimized_model_external_initializers_file_name",
            "model.data",
        )
        .map_err(ort::Error::<()>::from)?
        .with_config_entry("session.save_external_prepacked_constant_initializers", "1")
        .map_err(ort::Error::<()>::from)?
        .with_config_entry(
            "session.optimized_model_external_initializers_min_size_in_bytes",
            "1024",
        )
        .map_err(ort::Error::<()>::from)?;
    drop(spec.session(graph, Some(options))?);
    align(&destination)?;
    Ok(destination)
}

struct Pack {
    tensor: usize,
    entry: usize,
    blob_key: String,
    triples: Vec<[String; 3]>,
}

fn int(value: &str) -> Result<u64> {
    value
        .parse()
        .with_context(|| format!("invalid literal for int() with base 10: '{value}'"))
}

fn align(destination: &Path) -> Result<()> {
    let mut model = onnx::load(destination)?;
    let mut length_at: BTreeMap<u64, u64> = BTreeMap::new();
    let mut offsets: Vec<(usize, usize)> = Vec::new();
    let mut packs: Vec<Pack> = Vec::new();
    for (tensor_index, tensor) in model.graph.initializer.iter().enumerate() {
        let mut keys: Vec<&str> = Vec::new();
        let mut last: HashMap<&str, usize> = HashMap::new();
        for (entry_index, entry) in tensor.external_data.iter().enumerate() {
            if last.insert(&entry.key, entry_index).is_none() {
                keys.push(&entry.key);
            }
        }
        for key in keys {
            let entry_index = last[key];
            let value = &tensor.external_data[entry_index].value;
            if key == "offset" {
                let length = last.get("length").context("KeyError: 'length'")?;
                length_at.insert(int(value)?, int(&tensor.external_data[*length].value)?);
                offsets.push((tensor_index, entry_index));
            } else if key.starts_with("prepacked_") {
                let mut items = value.split('|');
                let blob_key = items.next().unwrap_or_default().to_owned();
                let triples = items
                    .map(|item| {
                        let parts: Vec<String> = item.split(';').map(str::to_owned).collect();
                        <[String; 3]>::try_from(parts).map_err(|parts| {
                            anyhow::anyhow!("expected 3 values to unpack, got {}", parts.len())
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                for [offset, length, _] in &triples {
                    length_at.insert(int(offset)?, int(length)?);
                }
                packs.push(Pack {
                    tensor: tensor_index,
                    entry: entry_index,
                    blob_key,
                    triples,
                });
            }
        }
    }

    let mut moved: HashMap<u64, u64> = HashMap::new();
    let mut cursor = 0u64;
    for (offset, length) in &length_at {
        cursor = cursor.next_multiple_of(if *length > 1 << 20 { 4096 } else { 64 });
        moved.insert(*offset, cursor);
        cursor += length;
    }

    let data = destination.with_extension("data");
    let aligned = data.with_extension("aligned");
    {
        let mut source = File::open(&data)?;
        let mut target = File::create(&aligned)?;
        for (offset, length) in &length_at {
            source.seek(SeekFrom::Start(*offset))?;
            target.seek(SeekFrom::Start(moved[offset]))?;
            io::copy(&mut (&mut source).take(*length), &mut target)?;
        }
    }
    fs::rename(&aligned, &data)?;

    for (tensor, entry) in offsets {
        let entry = &mut model.graph.initializer[tensor].external_data[entry];
        entry.value = moved[&int(&entry.value)?].to_string();
    }
    for pack in packs {
        let mut parts = vec![pack.blob_key];
        for [offset, length, checksum] in pack.triples {
            parts.push(format!("{};{length};{checksum}", moved[&int(&offset)?]));
        }
        model.graph.initializer[pack.tensor].external_data[pack.entry].value = parts.join("|");
    }
    onnx::save(&model, destination)
}
