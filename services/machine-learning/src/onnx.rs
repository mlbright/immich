use std::ops::Range;
use std::path::Path;

use anyhow::{Context, Result};

const MODEL_GRAPH: u64 = 7;
const GRAPH_INITIALIZER: u64 = 5;
const TENSOR_EXTERNAL_DATA: u64 = 13;
const ENTRY_KEY: u64 = 1;
const ENTRY_VALUE: u64 = 2;
pub(crate) const LENGTH_DELIMITED: u8 = 2;

#[derive(Debug, thiserror::Error)]
#[error("Error parsing message")]
pub struct DecodeError;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StringStringEntryProto {
    pub key: String,
    pub value: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TensorProto {
    pub external_data: Vec<StringStringEntryProto>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraphProto {
    pub initializer: Vec<TensorProto>,
}

#[derive(Clone, Debug)]
pub struct ModelProto {
    encoded: Vec<u8>,
    pub graph: GraphProto,
}

pub(crate) struct Field {
    pub number: u64,
    pub wire: u8,
    pub head: Range<usize>,
    pub body: Range<usize>,
}

impl Field {
    fn holds(&self, number: u64) -> bool {
        self.number == number && self.wire == LENGTH_DELIMITED
    }
}

pub(crate) fn varint(bytes: &[u8], at: &mut usize) -> Result<u64, DecodeError> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *bytes.get(*at).ok_or(DecodeError)?;
        *at += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(DecodeError)
}

pub(crate) fn encode_varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

pub(crate) fn fields(message: &[u8]) -> Result<Vec<Field>, DecodeError> {
    let mut fields = Vec::new();
    let mut at = 0;
    while at < message.len() {
        let start = at;
        let tag = varint(message, &mut at)?;
        let (number, wire) = (tag >> 3, (tag & 7) as u8);
        if number == 0 {
            return Err(DecodeError);
        }
        let width = match wire {
            0 => {
                let mut end = at;
                varint(message, &mut end)?;
                end - at
            }
            1 => 8,
            LENGTH_DELIMITED => {
                usize::try_from(varint(message, &mut at)?).map_err(|_| DecodeError)?
            }
            5 => 4,
            _ => return Err(DecodeError),
        };
        let end = at.checked_add(width).ok_or(DecodeError)?;
        if end > message.len() {
            return Err(DecodeError);
        }
        fields.push(Field {
            number,
            wire,
            head: start..at,
            body: at..end,
        });
        at = end;
    }
    Ok(fields)
}

fn submessages(message: &[u8], number: u64) -> Result<Vec<&[u8]>, DecodeError> {
    Ok(fields(message)?
        .into_iter()
        .filter(|field| field.holds(number))
        .map(|field| &message[field.body])
        .collect())
}

pub(crate) fn string(bytes: &[u8]) -> Result<String, DecodeError> {
    String::from_utf8(bytes.to_vec()).map_err(|_| DecodeError)
}

fn rebuild(
    message: &[u8],
    number: u64,
    mut replace: impl FnMut(&[u8]) -> Result<Vec<u8>>,
) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(message.len());
    for field in fields(message)? {
        if field.holds(number) {
            let body = replace(&message[field.body])?;
            let tag_end = field.head.start
                + message[field.head.clone()]
                    .iter()
                    .position(|byte| byte & 0x80 == 0)
                    .ok_or(DecodeError)?
                + 1;
            out.extend_from_slice(&message[field.head.start..tag_end]);
            encode_varint(body.len() as u64, &mut out);
            out.extend_from_slice(&body);
        } else {
            out.extend_from_slice(&message[field.head.start..field.body.end]);
        }
    }
    Ok(out)
}

fn parse_entry(entry: &[u8]) -> Result<StringStringEntryProto, DecodeError> {
    let mut parsed = StringStringEntryProto::default();
    for field in fields(entry)? {
        if field.holds(ENTRY_KEY) {
            parsed.key = string(&entry[field.body])?;
        } else if field.holds(ENTRY_VALUE) {
            parsed.value = string(&entry[field.body])?;
        }
    }
    Ok(parsed)
}

fn parse_tensor(tensor: &[u8]) -> Result<TensorProto, DecodeError> {
    Ok(TensorProto {
        external_data: submessages(tensor, TENSOR_EXTERNAL_DATA)?
            .into_iter()
            .map(parse_entry)
            .collect::<Result<_, _>>()?,
    })
}

fn parse_graph(encoded: &[u8]) -> Result<GraphProto, DecodeError> {
    let mut graph = GraphProto::default();
    for message in submessages(encoded, MODEL_GRAPH)? {
        for tensor in submessages(message, GRAPH_INITIALIZER)? {
            graph.initializer.push(parse_tensor(tensor)?);
        }
    }
    Ok(graph)
}

pub fn load(path: &Path) -> Result<ModelProto> {
    let encoded = std::fs::read(path)?;
    let graph = parse_graph(&encoded)?;
    Ok(ModelProto { encoded, graph })
}

impl ModelProto {
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let mut tensors = self.graph.initializer.iter();
        rebuild(&self.encoded, MODEL_GRAPH, |graph| {
            rebuild(graph, GRAPH_INITIALIZER, |tensor| {
                let mut entries = tensors
                    .next()
                    .context("initializers changed while saving")?
                    .external_data
                    .iter();
                rebuild(tensor, TENSOR_EXTERNAL_DATA, |entry| {
                    let value = &entries
                        .next()
                        .context("external data changed while saving")?
                        .value;
                    rebuild(entry, ENTRY_VALUE, |_| Ok(value.as_bytes().to_vec()))
                })
            })
        })
    }
}

pub fn save(model: &ModelProto, path: &Path) -> Result<()> {
    std::fs::write(path, model.serialize()?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(fields: &[(u64, Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (number, body) in fields {
            encode_varint(number << 3 | u64::from(LENGTH_DELIMITED), &mut out);
            encode_varint(body.len() as u64, &mut out);
            out.extend_from_slice(body);
        }
        out
    }

    fn entry(key: &str, value: &str) -> Vec<u8> {
        message(&[(ENTRY_KEY, key.into()), (ENTRY_VALUE, value.into())])
    }

    fn model(values: &[&str]) -> Vec<u8> {
        let mut tensor = message(&[(8, b"weight".to_vec())]);
        tensor.extend(message(&[(
            TENSOR_EXTERNAL_DATA,
            entry("location", "model.data"),
        )]));
        for value in values {
            tensor.extend(message(&[(TENSOR_EXTERNAL_DATA, entry("offset", value))]));
        }
        tensor.extend([0x70, 0x01]);
        let graph = message(&[(1, b"node".to_vec()), (GRAPH_INITIALIZER, tensor)]);
        let mut model = vec![0x08, 0x08];
        model.extend(message(&[(MODEL_GRAPH, graph)]));
        model.extend([0x15, 1, 2, 3, 4]);
        model
    }

    fn proto(encoded: Vec<u8>) -> ModelProto {
        let graph = parse_graph(&encoded).unwrap();
        ModelProto { encoded, graph }
    }

    #[test]
    fn reads_external_data() {
        let model = proto(model(&["4096"]));
        assert_eq!(
            model.graph.initializer[0].external_data,
            [
                StringStringEntryProto {
                    key: "location".into(),
                    value: "model.data".into()
                },
                StringStringEntryProto {
                    key: "offset".into(),
                    value: "4096".into()
                },
            ]
        );
    }

    #[test]
    fn unchanged_model_serializes_to_the_same_bytes() {
        let encoded = model(&["7", "64"]);
        assert_eq!(proto(encoded.clone()).serialize().unwrap(), encoded);
    }

    #[test]
    fn rewritten_values_resize_every_enclosing_message() {
        let mut model_proto = proto(model(&["7"]));
        model_proto.graph.initializer[0].external_data[1].value = "1".repeat(200);
        assert_eq!(model_proto.serialize().unwrap(), model(&[&"1".repeat(200)]));
    }

    #[test]
    fn truncated_message_is_a_decode_error() {
        let encoded = model(&["7"]);
        assert!(parse_graph(&encoded[..encoded.len() - 8]).is_err());
    }
}
