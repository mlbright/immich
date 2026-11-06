use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::onnx::{DecodeError, Field, LENGTH_DELIMITED, encode_varint, fields, string, varint};

pub mod data_type {
    pub const FLOAT: i64 = 1;
    pub const UINT8: i64 = 2;
    pub const INT8: i64 = 3;
    pub const UINT16: i64 = 4;
    pub const INT16: i64 = 5;
    pub const INT32: i64 = 6;
    pub const INT64: i64 = 7;
    pub const STRING: i64 = 8;
    pub const BOOL: i64 = 9;
    pub const FLOAT16: i64 = 10;
    pub const DOUBLE: i64 = 11;
    pub const UINT32: i64 = 12;
    pub const UINT64: i64 = 13;
    pub const BFLOAT16: i64 = 16;
}

const ATTRIBUTE_FLOAT: u64 = 1;
const ATTRIBUTE_INT: u64 = 2;
const ATTRIBUTE_FLOATS: u64 = 6;
const ATTRIBUTE_INTS: u64 = 7;

#[derive(Clone, Debug)]
struct Part {
    number: u64,
    bytes: Vec<u8>,
}

fn tag(number: u64, wire: u8, out: &mut Vec<u8>) {
    encode_varint(number << 3 | u64::from(wire), out);
}

pub(crate) fn put_bytes(out: &mut Vec<u8>, number: u64, body: &[u8]) {
    tag(number, LENGTH_DELIMITED, out);
    encode_varint(body.len() as u64, out);
    out.extend_from_slice(body);
}

fn put_varint(out: &mut Vec<u8>, number: u64, value: u64) {
    tag(number, 0, out);
    encode_varint(value, out);
}

fn put_fixed32(out: &mut Vec<u8>, number: u64, value: u32) {
    tag(number, 5, out);
    out.extend_from_slice(&value.to_le_bytes());
}

fn part(message: &[u8], field: &Field) -> Part {
    Part {
        number: field.number,
        bytes: message[field.head.start..field.body.end].to_vec(),
    }
}

fn wrapped(number: u64, body: &[u8]) -> Part {
    let mut bytes = Vec::with_capacity(body.len() + 6);
    put_bytes(&mut bytes, number, body);
    Part { number, bytes }
}

fn read_varint(message: &[u8], field: &Field) -> Result<u64, DecodeError> {
    let mut at = field.body.start;
    varint(message, &mut at)
}

fn packed_varints(message: &[u8], field: &Field) -> Result<Vec<u64>, DecodeError> {
    if field.wire != LENGTH_DELIMITED {
        return Ok(vec![read_varint(message, field)?]);
    }
    let mut values = Vec::new();
    let mut at = field.body.start;
    while at < field.body.end {
        values.push(varint(message, &mut at)?);
    }
    Ok(values)
}

fn packed_fixed<const N: usize>(message: &[u8], field: &Field) -> Vec<[u8; N]> {
    message[field.body.clone()].as_chunks::<N>().0.to_vec()
}

fn assemble(mut parts: Vec<Part>) -> Vec<u8> {
    parts.sort_by_key(|part| part.number);
    parts.into_iter().flat_map(|part| part.bytes).collect()
}

fn string_string_entry(key: &str, value: &str) -> Vec<u8> {
    let mut body = Vec::new();
    put_bytes(&mut body, 1, key.as_bytes());
    put_bytes(&mut body, 2, value.as_bytes());
    body
}

fn parse_entry(body: &[u8]) -> Result<(String, String), DecodeError> {
    let (mut key, mut value) = (String::new(), String::new());
    for field in fields(body)? {
        match (field.number, field.wire) {
            (1, LENGTH_DELIMITED) => key = string(&body[field.body])?,
            (2, LENGTH_DELIMITED) => value = string(&body[field.body])?,
            _ => {}
        }
    }
    Ok((key, value))
}

fn insert_entry(entries: &mut Vec<(String, String)>, key: String, value: String) {
    match entries.iter_mut().find(|(existing, _)| *existing == key) {
        Some((_, slot)) => *slot = value,
        None => entries.push((key, value)),
    }
}

#[derive(Clone, Debug)]
pub struct ValueInfo {
    pub name: String,
    body: Vec<u8>,
    pub elem_type: Option<i64>,
    pub rank: Option<usize>,
    typed: bool,
    annotated: bool,
}

impl ValueInfo {
    fn parse(body: &[u8]) -> Result<Self, DecodeError> {
        let mut info = ValueInfo {
            name: String::new(),
            body: body.to_vec(),
            elem_type: None,
            rank: None,
            typed: false,
            annotated: false,
        };
        for field in fields(body)? {
            match (field.number, field.wire) {
                (1, LENGTH_DELIMITED) => info.name = string(&body[field.body])?,
                (2, LENGTH_DELIMITED) => {
                    info.typed = true;
                    let type_proto = &body[field.body];
                    for kind in fields(type_proto)? {
                        if kind.number != 1 || kind.wire != LENGTH_DELIMITED {
                            continue;
                        }
                        let tensor = &type_proto[kind.body];
                        for entry in fields(tensor)? {
                            match (entry.number, entry.wire) {
                                (1, 0) => {
                                    info.elem_type = Some(read_varint(tensor, &entry)? as i64)
                                }
                                (2, LENGTH_DELIMITED) => {
                                    let shape = &tensor[entry.body];
                                    info.rank = Some(
                                        fields(shape)?.iter().filter(|dim| dim.number == 1).count(),
                                    );
                                }
                                _ => {}
                            }
                        }
                    }
                }
                (3 | 4, _) => info.annotated = true,
                _ => {}
            }
        }
        Ok(info)
    }

    fn should_create(&self) -> bool {
        !self.name.is_empty() && (self.typed || self.annotated)
    }

    fn bare(&self) -> Result<Self, DecodeError> {
        let mut body = Vec::new();
        for field in fields(&self.body)? {
            if field.number <= 2 {
                body.extend_from_slice(&self.body[field.head.start..field.body.end]);
            }
        }
        let mut info = ValueInfo::parse(&body)?;
        info.annotated = false;
        Ok(info)
    }

    fn metadata(&self) -> Result<Vec<(String, String)>, DecodeError> {
        let mut entries = Vec::new();
        for field in fields(&self.body)? {
            if field.number == 4 && field.wire == LENGTH_DELIMITED {
                let (key, value) = parse_entry(&self.body[field.body])?;
                insert_entry(&mut entries, key, value);
            }
        }
        Ok(entries)
    }

    fn synthesize(
        name: &str,
        elem_type: i64,
        dims: &[i64],
        metadata: &[(String, String)],
    ) -> Result<Self, DecodeError> {
        let mut shape = Vec::new();
        for dim in dims {
            let mut dimension = Vec::new();
            put_varint(&mut dimension, 1, *dim as u64);
            put_bytes(&mut shape, 1, &dimension);
        }
        let mut tensor = Vec::new();
        put_varint(&mut tensor, 1, elem_type as u64);
        put_bytes(&mut tensor, 2, &shape);
        let mut type_proto = Vec::new();
        put_bytes(&mut type_proto, 1, &tensor);
        let mut body = Vec::new();
        put_bytes(&mut body, 1, name.as_bytes());
        put_bytes(&mut body, 2, &type_proto);
        let mut sorted = metadata.to_vec();
        sorted.sort();
        for (key, value) in sorted {
            put_bytes(&mut body, 4, &string_string_entry(&key, &value));
        }
        ValueInfo::parse(&body)
    }
}

#[derive(Clone, Debug)]
pub struct Attribute {
    pub name: String,
    field: Vec<u8>,
    pub kind: u64,
    pub f: Option<f32>,
    pub i: Option<i64>,
    pub t: Option<Vec<u8>>,
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
    pub subgraph: bool,
}

impl Attribute {
    fn parse(field: Vec<u8>, body: &[u8]) -> Result<Self, DecodeError> {
        let mut attribute = Attribute {
            name: String::new(),
            field,
            kind: 0,
            f: None,
            i: None,
            t: None,
            floats: Vec::new(),
            ints: Vec::new(),
            subgraph: false,
        };
        for entry in fields(body)? {
            match (entry.number, entry.wire) {
                (1, LENGTH_DELIMITED) => attribute.name = string(&body[entry.body])?,
                (2, 5) => {
                    let bytes: [u8; 4] = body[entry.body].try_into().map_err(|_| DecodeError)?;
                    attribute.f = Some(f32::from_le_bytes(bytes));
                }
                (3, 0) => attribute.i = Some(read_varint(body, &entry)? as i64),
                (5, LENGTH_DELIMITED) => attribute.t = Some(body[entry.body].to_vec()),
                (6 | 11, LENGTH_DELIMITED) => attribute.subgraph = true,
                (7, 5 | LENGTH_DELIMITED) => attribute.floats.extend(
                    packed_fixed::<4>(body, &entry)
                        .into_iter()
                        .map(f32::from_le_bytes),
                ),
                (8, 0 | LENGTH_DELIMITED) => attribute.ints.extend(
                    packed_varints(body, &entry)?
                        .into_iter()
                        .map(|value| value as i64),
                ),
                (20, 0) => attribute.kind = read_varint(body, &entry)?,
                _ => {}
            }
        }
        Ok(attribute)
    }

    pub fn float(name: &str, value: f32) -> Vec<u8> {
        let mut body = Vec::new();
        put_bytes(&mut body, 1, name.as_bytes());
        put_fixed32(&mut body, 2, value.to_bits());
        put_varint(&mut body, 20, ATTRIBUTE_FLOAT);
        body
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    original: Option<Vec<u8>>,
    rest: Vec<Part>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub attributes: Vec<Attribute>,
    pub metadata: Vec<(String, String)>,
}

impl Node {
    fn parse(body: &[u8]) -> Result<Self, DecodeError> {
        let mut node = Node {
            original: Some(body.to_vec()),
            rest: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            name: String::new(),
            op_type: String::new(),
            domain: String::new(),
            attributes: Vec::new(),
            metadata: Vec::new(),
        };
        for field in fields(body)? {
            let wire = field.wire;
            match (field.number, wire) {
                (1, LENGTH_DELIMITED) => node.inputs.push(string(&body[field.body])?),
                (2, LENGTH_DELIMITED) => node.outputs.push(string(&body[field.body])?),
                _ => {
                    let raw = part(body, &field);
                    match (field.number, wire) {
                        (3, LENGTH_DELIMITED) => node.name = string(&body[field.body.clone()])?,
                        (4, LENGTH_DELIMITED) => node.op_type = string(&body[field.body.clone()])?,
                        (5, LENGTH_DELIMITED) => node.attributes.push(Attribute::parse(
                            raw.bytes.clone(),
                            &body[field.body.clone()],
                        )?),
                        (7, LENGTH_DELIMITED) => node.domain = string(&body[field.body.clone()])?,
                        (9, LENGTH_DELIMITED) => {
                            let (key, value) = parse_entry(&body[field.body.clone()])?;
                            insert_entry(&mut node.metadata, key, value);
                        }
                        _ => {}
                    }
                    node.rest.push(raw);
                }
            }
        }
        Ok(node)
    }

    pub fn new(
        op_type: &str,
        domain: &str,
        name: &str,
        inputs: Vec<String>,
        outputs: Vec<String>,
        attributes: Vec<Vec<u8>>,
        metadata: Vec<(String, String)>,
    ) -> Result<Self> {
        let mut body = Vec::new();
        for input in &inputs {
            put_bytes(&mut body, 1, input.as_bytes());
        }
        for output in &outputs {
            put_bytes(&mut body, 2, output.as_bytes());
        }
        if !name.is_empty() {
            put_bytes(&mut body, 3, name.as_bytes());
        }
        put_bytes(&mut body, 4, op_type.as_bytes());
        for attribute in &attributes {
            put_bytes(&mut body, 5, attribute);
        }
        if !domain.is_empty() {
            put_bytes(&mut body, 7, domain.as_bytes());
        }
        let mut sorted = metadata;
        sorted.sort();
        for (key, value) in &sorted {
            put_bytes(&mut body, 9, &string_string_entry(key, value));
        }
        let mut node = Node::parse(&body)?;
        node.original = None;
        Ok(node)
    }

    fn encode(&self) -> Vec<u8> {
        if let Some(original) = &self.original {
            return original.clone();
        }
        let mut body = Vec::new();
        for input in &self.inputs {
            put_bytes(&mut body, 1, input.as_bytes());
        }
        let kept = self.outputs.len()
            - self
                .outputs
                .iter()
                .rev()
                .take_while(|output| output.is_empty())
                .count();
        for output in &self.outputs[..kept] {
            put_bytes(&mut body, 2, output.as_bytes());
        }
        for raw in &self.rest {
            body.extend_from_slice(&raw.bytes);
        }
        body
    }

    pub fn touch(&mut self) {
        self.original = None;
    }

    pub fn attribute(&self, name: &str) -> Option<&Attribute> {
        self.attributes
            .iter()
            .find(|attribute| attribute.name == name)
    }

    fn set_name(&mut self, name: String) {
        self.rest.retain(|raw| raw.number != 3);
        let position = self
            .rest
            .iter()
            .position(|raw| raw.number > 3)
            .unwrap_or(self.rest.len());
        self.rest.insert(position, wrapped(3, name.as_bytes()));
        self.name = name;
        self.touch();
    }

    fn remove_attribute(&mut self, name: &str) {
        if let Some(index) = self.attributes.iter().position(|a| a.name == name) {
            let removed = self.attributes.remove(index);
            if let Some(position) = self.rest.iter().position(|raw| raw.bytes == removed.field) {
                self.rest.remove(position);
            }
            self.touch();
        }
    }
}

#[derive(Clone, Debug)]
pub struct Initializer {
    pub name: String,
    body: Vec<u8>,
    pub data_type: i64,
    pub dims: Vec<i64>,
}

impl Initializer {
    fn parse(body: &[u8]) -> Result<Self, DecodeError> {
        let mut initializer = Initializer {
            name: String::new(),
            body: body.to_vec(),
            data_type: 0,
            dims: Vec::new(),
        };
        for field in fields(body)? {
            match (field.number, field.wire) {
                (1, 0 | LENGTH_DELIMITED) => initializer.dims.extend(
                    packed_varints(body, &field)?
                        .into_iter()
                        .map(|dim| dim as i64),
                ),
                (2, 0) => initializer.data_type = read_varint(body, &field)? as i64,
                (8, LENGTH_DELIMITED) => initializer.name = string(&body[field.body])?,
                _ => {}
            }
        }
        Ok(initializer)
    }

    fn size(&self) -> i64 {
        self.dims.iter().product()
    }

    fn value_info(&self) -> Result<ValueInfo, DecodeError> {
        ValueInfo::synthesize(&self.name, self.data_type, &self.dims, &[])
    }

    fn renamed(body: &[u8], name: &str) -> Result<Vec<u8>, DecodeError> {
        let mut parts: Vec<Part> = fields(body)?
            .iter()
            .filter(|field| field.number != 8)
            .map(|field| part(body, field))
            .collect();
        parts.push(wrapped(8, name.as_bytes()));
        Ok(assemble(parts))
    }

    fn item_size(&self) -> Option<usize> {
        Some(match self.data_type {
            data_type::UINT8 | data_type::INT8 | data_type::BOOL => 1,
            data_type::UINT16 | data_type::INT16 | data_type::FLOAT16 | data_type::BFLOAT16 => 2,
            data_type::FLOAT | data_type::INT32 | data_type::UINT32 => 4,
            data_type::INT64 | data_type::UINT64 | data_type::DOUBLE => 8,
            _ => return None,
        })
    }

    fn bytes(&self, directory: &Path) -> Result<Option<Vec<u8>>> {
        let body = &self.body;
        let mut external: HashMap<String, String> = HashMap::new();
        let mut typed: Vec<u8> = Vec::new();
        let item = self.item_size();
        for field in fields(body)? {
            match (field.number, field.wire) {
                (9, LENGTH_DELIMITED) => return Ok(Some(body[field.body].to_vec())),
                (13, LENGTH_DELIMITED) => {
                    let (key, value) = parse_entry(&body[field.body])?;
                    external.insert(key, value);
                }
                (4, 5 | LENGTH_DELIMITED) => {
                    for value in packed_fixed::<4>(body, &field) {
                        typed.extend_from_slice(&value);
                    }
                }
                (10, 1 | LENGTH_DELIMITED) => {
                    for value in packed_fixed::<8>(body, &field) {
                        typed.extend_from_slice(&value);
                    }
                }
                (5 | 7 | 11, 0 | LENGTH_DELIMITED) => {
                    let Some(width) = item else { return Ok(None) };
                    for value in packed_varints(body, &field)? {
                        typed.extend_from_slice(&value.to_le_bytes()[..width]);
                    }
                }
                (6, _) => return Ok(None),
                _ => {}
            }
        }
        if let Some(location) = external.get("location") {
            let offset: u64 = external.get("offset").map_or(Ok(0), |o| o.parse())?;
            let length = match external.get("length") {
                Some(length) => length.parse()?,
                None => {
                    let Some(width) = item else { return Ok(None) };
                    self.size() as u64 * width as u64
                }
            };
            let mut file = File::open(directory.join(location))?;
            file.seek(SeekFrom::Start(offset))?;
            let mut data = vec![0u8; length as usize];
            file.read_exact(&mut data)?;
            return Ok(Some(data));
        }
        Ok(Some(typed))
    }
}

#[derive(Clone, Debug)]
struct Opset {
    domain: String,
    version: i64,
    field: Option<Vec<u8>>,
}

#[derive(Clone, Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub initializers: Vec<Initializer>,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
    value_info: HashMap<String, ValueInfo>,
    rest: Vec<Part>,
    node_names: HashSet<String>,
    node_counter: usize,
}

impl Graph {
    fn parse(body: &[u8]) -> Result<Self, DecodeError> {
        let mut graph = Graph {
            nodes: Vec::new(),
            initializers: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            value_info: HashMap::new(),
            rest: Vec::new(),
            node_names: HashSet::new(),
            node_counter: 0,
        };
        for field in fields(body)? {
            let content = &body[field.body.clone()];
            match (field.number, field.wire) {
                (1, LENGTH_DELIMITED) => graph.nodes.push(Node::parse(content)?),
                (5, LENGTH_DELIMITED) => graph.initializers.push(Initializer::parse(content)?),
                (11, LENGTH_DELIMITED) => graph.inputs.push(ValueInfo::parse(content)?),
                (12, LENGTH_DELIMITED) => graph.outputs.push(ValueInfo::parse(content)?),
                (13, LENGTH_DELIMITED) => {
                    let info = ValueInfo::parse(content)?;
                    graph.value_info.insert(info.name.clone(), info);
                }
                _ => graph.rest.push(part(body, &field)),
            }
        }
        graph.node_names = graph.nodes.iter().map(|node| node.name.clone()).collect();
        Ok(graph)
    }

    fn encode(&self) -> Result<Vec<u8>> {
        let mut parts = self.rest.clone();
        for node in &self.nodes {
            parts.push(wrapped(1, &node.encode()));
        }
        for initializer in &self.initializers {
            parts.push(wrapped(5, &initializer.body));
        }
        for input in &self.inputs {
            parts.push(wrapped(11, &input.body));
        }
        for output in &self.outputs {
            parts.push(wrapped(12, &output.body));
        }
        let input_names: HashSet<&str> = self.inputs.iter().map(|v| v.name.as_str()).collect();
        let output_names: HashSet<&str> = self.outputs.iter().map(|v| v.name.as_str()).collect();
        for initializer in &self.initializers {
            if input_names.contains(initializer.name.as_str()) {
                continue;
            }
            let info = match self.value_info.get(&initializer.name) {
                Some(info) => info.clone(),
                None => initializer.value_info()?,
            };
            parts.push(wrapped(13, &info.body));
        }
        for node in &self.nodes {
            for output in &node.outputs {
                if output.is_empty() || output_names.contains(output.as_str()) {
                    continue;
                }
                if let Some(info) = self.value_info.get(output)
                    && info.should_create()
                {
                    parts.push(wrapped(13, &info.body));
                }
            }
        }
        Ok(assemble(parts))
    }

    pub fn type_of(&self, name: &str) -> (Option<i64>, Option<usize>) {
        let declared = self
            .outputs
            .iter()
            .chain(self.inputs.iter())
            .find(|info| info.name == name)
            .or_else(|| self.value_info.get(name));
        if let Some(info) = declared
            && info.typed
        {
            return (info.elem_type, info.rank);
        }
        match self.initializers.iter().find(|i| i.name == name) {
            Some(initializer) => (Some(initializer.data_type), Some(initializer.dims.len())),
            None => (None, None),
        }
    }

    pub fn is_output(&self, name: &str) -> bool {
        self.outputs.iter().any(|output| output.name == name)
    }

    fn is_input(&self, name: &str) -> bool {
        self.inputs.iter().any(|input| input.name == name)
    }

    pub fn producers(&self) -> HashMap<&str, usize> {
        let mut producers = HashMap::new();
        for (index, node) in self.nodes.iter().enumerate() {
            for output in &node.outputs {
                if !output.is_empty() {
                    producers.insert(output.as_str(), index);
                }
            }
        }
        producers
    }

    pub fn uses(&self) -> HashMap<&str, Vec<usize>> {
        let mut uses: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, node) in self.nodes.iter().enumerate() {
            for input in &node.inputs {
                if !input.is_empty() {
                    uses.entry(input.as_str()).or_default().push(index);
                }
            }
        }
        uses
    }

    pub fn name_node(&mut self, op_type: &str) -> String {
        loop {
            let name = format!("node_{op_type}_{}", self.node_counter);
            self.node_counter += 1;
            if self.node_names.insert(name.clone()) {
                return name;
            }
        }
    }

    pub fn strip_value_info(&mut self, name: &str) -> Result<()> {
        if let Some(info) = self.value_info.get(name) {
            let bare = info.bare()?;
            self.value_info.insert(name.to_owned(), bare);
        }
        Ok(())
    }

    fn rename_uses(&mut self, from: &str, to: &str) {
        for node in &mut self.nodes {
            if node.inputs.iter().any(|input| input == from) {
                for input in &mut node.inputs {
                    if input == from {
                        *input = to.to_owned();
                    }
                }
                node.touch();
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    rest: Vec<Part>,
    opsets: Vec<Opset>,
    pub graph: Graph,
    directory: PathBuf,
}

impl Model {
    pub fn parse(body: &[u8], directory: &Path) -> Result<Self, DecodeError> {
        let mut rest = Vec::new();
        let mut opsets = Vec::new();
        let mut graph = None;
        for field in fields(body)? {
            let content = &body[field.body.clone()];
            match (field.number, field.wire) {
                (7, LENGTH_DELIMITED) => graph = Some(Graph::parse(content)?),
                (8, LENGTH_DELIMITED) => {
                    let mut opset = Opset {
                        domain: String::new(),
                        version: 0,
                        field: Some(part(body, &field).bytes),
                    };
                    for entry in fields(content)? {
                        match (entry.number, entry.wire) {
                            (1, LENGTH_DELIMITED) => opset.domain = string(&content[entry.body])?,
                            (2, 0) => opset.version = read_varint(content, &entry)? as i64,
                            _ => {}
                        }
                    }
                    opsets.push(opset);
                }
                _ => rest.push(part(body, &field)),
            }
        }
        Ok(Model {
            rest,
            opsets,
            graph: graph.ok_or(DecodeError)?,
            directory: directory.to_path_buf(),
        })
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut parts = self.rest.clone();
        parts.push(wrapped(7, &self.graph.encode()?));
        for opset in &self.opsets {
            parts.push(match &opset.field {
                Some(field) => Part {
                    number: 8,
                    bytes: field.clone(),
                },
                None => {
                    let mut body = Vec::new();
                    put_bytes(&mut body, 1, opset.domain.as_bytes());
                    put_varint(&mut body, 2, opset.version as u64);
                    wrapped(8, &body)
                }
            });
        }
        Ok(assemble(parts))
    }

    pub fn opset_version(&self, domain: &str) -> Option<i64> {
        self.opsets
            .iter()
            .find(|opset| opset.domain == domain)
            .map(|opset| opset.version)
    }

    pub fn import_opset(&mut self, domain: &str, version: i64) {
        if self.opset_version(domain).is_none() {
            self.opsets.push(Opset {
                domain: domain.to_owned(),
                version,
                field: None,
            });
        }
    }
}

pub fn load(path: &Path) -> Result<Model> {
    let bytes = std::fs::read(path)?;
    Ok(Model::parse(
        &bytes,
        path.parent().unwrap_or(Path::new("")),
    )?)
}

pub fn save(model: &Model, path: &Path) -> Result<()> {
    std::fs::write(path, model.encode()?)?;
    Ok(())
}

fn unique_name(
    preferred: &str,
    used: &mut HashSet<String>,
    counter: &mut HashMap<String, usize>,
) -> String {
    let mut name = preferred.to_owned();
    while used.contains(&name) {
        let count = counter.entry(preferred.to_owned()).or_default();
        *count += 1;
        name = format!("{preferred}_{count}");
    }
    used.insert(name.clone());
    name
}

pub fn name_fix(model: &mut Model) {
    let graph = &mut model.graph;
    let mut used_values: HashSet<String> = HashSet::new();
    let mut used_nodes: HashSet<String> = HashSet::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut value_counter = HashMap::new();
    let mut node_counter = HashMap::new();
    let declared: Vec<String> = graph
        .inputs
        .iter()
        .chain(graph.outputs.iter())
        .map(|info| info.name.clone())
        .chain(graph.initializers.iter().map(|i| i.name.clone()))
        .collect();
    for name in declared {
        if !name.is_empty() && seen.insert(name.clone()) {
            used_values.insert(name);
        }
    }
    for node in &mut graph.nodes {
        if node.name.is_empty() {
            let name = unique_name("node", &mut used_nodes, &mut node_counter);
            node.set_name(name);
        } else if used_nodes.contains(&node.name) {
            let name = unique_name(&node.name.clone(), &mut used_nodes, &mut node_counter);
            node.set_name(name);
        } else {
            used_nodes.insert(node.name.clone());
        }
        for input in &node.inputs {
            if !input.is_empty() && seen.insert(input.clone()) {
                used_values.insert(input.clone());
            }
        }
        let mut renamed = false;
        for output in &mut node.outputs {
            if output.is_empty() {
                *output = unique_name("v", &mut used_values, &mut value_counter);
                seen.insert(output.clone());
                renamed = true;
            } else if seen.insert(output.clone()) {
                used_values.insert(output.clone());
            }
        }
        if renamed {
            node.touch();
        }
    }
}

fn optional_outputs(op_type: &str) -> Option<&'static [bool]> {
    Some(match op_type {
        "Dropout" | "MaxPool" | "SoftmaxCrossEntropyLoss" => &[false, true],
        "LayerNormalization" => &[false, true, true],
        "GRU" | "RNN" => &[true, true],
        "LSTM" => &[true, true, true],
        "Unique" | "Attention" => &[false, true, true, true],
        _ => return None,
    })
}

const VARIADIC_OUTPUTS: [&str; 6] = [
    "Split",
    "If",
    "Loop",
    "Scan",
    "SequenceMap",
    "SplitToSequence",
];

pub fn remove_unused_nodes(model: &mut Model) -> bool {
    let standard_opset = model.opset_version("").is_some();
    let graph = &mut model.graph;
    let outputs: HashSet<String> = graph.outputs.iter().map(|o| o.name.clone()).collect();
    let mut uses: HashMap<String, usize> = HashMap::new();
    for node in &graph.nodes {
        for input in &node.inputs {
            if !input.is_empty() {
                *uses.entry(input.clone()).or_default() += 1;
            }
        }
    }
    let used = |name: &str, uses: &HashMap<String, usize>| {
        !name.is_empty() && (outputs.contains(name) || uses.get(name).copied().unwrap_or(0) > 0)
    };
    let mut count = 0;
    let mut index = graph.nodes.len();
    while index > 0 {
        index -= 1;
        let removable = !graph.nodes[index]
            .outputs
            .iter()
            .any(|output| used(output, &uses));
        if removable {
            let node = graph.nodes.remove(index);
            for input in node.inputs.iter().filter(|input| !input.is_empty()) {
                if let Some(count) = uses.get_mut(input) {
                    *count -= 1;
                }
            }
            count += 1;
            continue;
        }
        let node = &mut graph.nodes[index];
        let trailing = node
            .inputs
            .iter()
            .rev()
            .take_while(|input| input.is_empty())
            .count();
        if trailing > 0 {
            node.inputs.truncate(node.inputs.len() - trailing);
            node.touch();
        }
        if !standard_opset || !(node.domain.is_empty() || node.domain == "onnx.ai") {
            continue;
        }
        if node.op_type == "BatchNormalization" {
            let is_used = |i: usize| {
                node.outputs
                    .get(i)
                    .is_some_and(|output| used(output, &uses))
            };
            if is_used(1) || is_used(2) {
                continue;
            }
            for i in [1, 2] {
                if let Some(output) = node.outputs.get_mut(i) {
                    output.clear();
                }
            }
            node.remove_attribute("training_mode");
            node.touch();
            continue;
        }
        if VARIADIC_OUTPUTS.contains(&node.op_type.as_str()) {
            continue;
        }
        if let Some(optional) = optional_outputs(&node.op_type) {
            for (i, output) in node.outputs.iter_mut().enumerate() {
                if optional.get(i).copied().unwrap_or(false)
                    && !used(output, &uses)
                    && !output.is_empty()
                {
                    output.clear();
                    node.original = None;
                }
            }
        }
        let trailing = node
            .outputs
            .iter()
            .rev()
            .take_while(|output| output.is_empty())
            .count();
        if trailing > 0 {
            node.outputs.truncate(node.outputs.len() - trailing);
            node.touch();
        }
    }
    let before = graph.initializers.len();
    graph.initializers.retain(|initializer| {
        uses.get(&initializer.name).copied().unwrap_or(0) > 0
            || outputs.contains(&initializer.name)
            || graph
                .inputs
                .iter()
                .any(|input| input.name == initializer.name)
    });
    count += before - graph.initializers.len();
    count > 0
}

pub fn deduplicate_initializers(model: &mut Model, size_limit: i64) -> Result<bool> {
    let directory = model.directory.clone();
    let graph = &mut model.graph;
    let mut seen: HashMap<(i64, Vec<i64>, Vec<u8>), String> = HashMap::new();
    let mut renames: Vec<(String, String)> = Vec::new();
    for initializer in &graph.initializers {
        if graph.is_input(&initializer.name) || graph.is_output(&initializer.name) {
            continue;
        }
        if initializer.size() > size_limit {
            continue;
        }
        let Some(bytes) = initializer.bytes(&directory)? else {
            continue;
        };
        let key = (initializer.data_type, initializer.dims.clone(), bytes);
        match seen.get(&key) {
            Some(kept) => renames.push((initializer.name.clone(), kept.clone())),
            None => {
                seen.insert(key, initializer.name.clone());
            }
        }
    }
    for (duplicate, kept) in &renames {
        graph.rename_uses(duplicate, kept);
        graph
            .initializers
            .retain(|initializer| initializer.name != *duplicate);
    }
    Ok(!renames.is_empty())
}

pub fn topological_sort(model: &mut Model) -> Result<bool> {
    let graph = &mut model.graph;
    let count = graph.nodes.len();
    let producers: HashMap<String, usize> = graph
        .producers()
        .into_iter()
        .map(|(name, index)| (name.to_owned(), index))
        .collect();
    let mut depth = vec![0usize; count];
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); count];
    for (index, node) in graph.nodes.iter().enumerate() {
        if node.attributes.iter().any(|attribute| attribute.subgraph) {
            bail!("sorting a graph with subgraphs is not ported");
        }
        for input in node.inputs.iter().filter(|input| !input.is_empty()) {
            if let Some(&producer) = producers.get(input) {
                predecessors[index].push(producer);
                depth[producer] += 1;
            }
        }
    }
    let mut queue: BinaryHeap<usize> = (0..count).filter(|&index| depth[index] == 0).collect();
    let mut order = Vec::with_capacity(count);
    while let Some(index) = queue.pop() {
        order.push(index);
        for &predecessor in &predecessors[index] {
            depth[predecessor] -= 1;
            if depth[predecessor] == 0 {
                queue.push(predecessor);
            }
        }
    }
    if order.len() != count {
        bail!("Graph contains a cycle, topological sort is not possible.");
    }
    order.reverse();
    let modified = order
        .iter()
        .enumerate()
        .any(|(position, index)| position != *index);
    let mut nodes: Vec<Option<Node>> = graph.nodes.drain(..).map(Some).collect();
    graph.nodes = order
        .into_iter()
        .map(|index| nodes[index].take().context("node sorted twice"))
        .collect::<Result<_>>()?;
    Ok(modified)
}

fn constant_tensor(attribute: &Attribute, name: &str) -> Result<Option<Vec<u8>>> {
    let numeric = |data_type: i64, dims: &[i64], raw: Vec<u8>| {
        let mut body = Vec::new();
        for dim in dims {
            put_varint(&mut body, 1, *dim as u64);
        }
        put_varint(&mut body, 2, data_type as u64);
        put_bytes(&mut body, 8, name.as_bytes());
        put_bytes(&mut body, 9, &raw);
        body
    };
    Ok(Some(match attribute.name.as_str() {
        "value" => match &attribute.t {
            Some(tensor) => Initializer::renamed(tensor, name)?,
            None => return Ok(None),
        },
        "value_int" if attribute.kind == ATTRIBUTE_INT => numeric(
            data_type::INT64,
            &[],
            attribute.i.unwrap_or_default().to_le_bytes().to_vec(),
        ),
        "value_ints" if attribute.kind == ATTRIBUTE_INTS => numeric(
            data_type::INT64,
            &[attribute.ints.len() as i64],
            attribute
                .ints
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
        "value_float" if attribute.kind == ATTRIBUTE_FLOAT => numeric(
            data_type::FLOAT,
            &[],
            attribute.f.unwrap_or_default().to_le_bytes().to_vec(),
        ),
        "value_floats" if attribute.kind == ATTRIBUTE_FLOATS => numeric(
            data_type::FLOAT,
            &[attribute.floats.len() as i64],
            attribute
                .floats
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
        _ => return Ok(None),
    }))
}

pub fn lift_constants_to_initializers(model: &mut Model, size_limit: i64) -> Result<bool> {
    let graph = &mut model.graph;
    let mut index = 0;
    let mut modified = false;
    while index < graph.nodes.len() {
        let node = &graph.nodes[index];
        let lift = node.op_type == "Constant"
            && (node.domain.is_empty() || node.domain == "onnx.ai")
            && node.attributes.len() == 1
            && node
                .outputs
                .first()
                .is_some_and(|output| !graph.is_output(output));
        if !lift {
            index += 1;
            continue;
        }
        let name = node.outputs[0].clone();
        let Some(body) = constant_tensor(&node.attributes[0], &name)? else {
            index += 1;
            continue;
        };
        let initializer = Initializer::parse(&body)?;
        if initializer.size() < size_limit {
            index += 1;
            continue;
        }
        let metadata = match graph.value_info.get(&name) {
            Some(info) => info.metadata()?,
            None => Vec::new(),
        };
        let info =
            ValueInfo::synthesize(&name, initializer.data_type, &initializer.dims, &metadata)?;
        graph.value_info.insert(name, info);
        graph.initializers.push(initializer);
        graph.nodes.remove(index);
        modified = true;
    }
    Ok(modified)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value_info(name: &str, elem_type: i64, dims: &[i64]) -> Vec<u8> {
        ValueInfo::synthesize(name, elem_type, dims, &[])
            .unwrap()
            .body
    }

    fn model(nodes: &[Vec<u8>], extra: &[(u64, Vec<u8>)]) -> Vec<u8> {
        let mut graph = Vec::new();
        for node in nodes {
            put_bytes(&mut graph, 1, node);
        }
        put_bytes(&mut graph, 2, b"g");
        for (number, body) in extra {
            put_bytes(&mut graph, *number, body);
        }
        let mut body = Vec::new();
        put_varint(&mut body, 1, 10);
        put_bytes(&mut body, 7, &graph);
        let mut opset = Vec::new();
        put_bytes(&mut opset, 1, b"");
        put_varint(&mut opset, 2, 23);
        put_bytes(&mut body, 8, &opset);
        body
    }

    fn node(op: &str, inputs: &[&str], outputs: &[&str], name: &str) -> Vec<u8> {
        Node::new(
            op,
            "",
            name,
            inputs.iter().map(|s| s.to_string()).collect(),
            outputs.iter().map(|s| s.to_string()).collect(),
            Vec::new(),
            Vec::new(),
        )
        .unwrap()
        .encode()
    }

    #[test]
    fn a_canonical_model_saves_to_the_same_bytes() {
        let encoded = model(
            &[
                node("Relu", &["x"], &["y"], "n0"),
                node("Neg", &["y"], &["z"], "n1"),
            ],
            &[
                (11, value_info("x", 1, &[2])),
                (12, value_info("z", 1, &[2])),
                (13, value_info("y", 1, &[2])),
            ],
        );
        let parsed = Model::parse(&encoded, Path::new("")).unwrap();
        assert_eq!(parsed.encode().unwrap(), encoded);
    }

    #[test]
    fn stable_sort_keeps_a_sorted_graph_and_fixes_an_unsorted_one() {
        let sorted = model(
            &[
                node("Relu", &["x"], &["a"], "a"),
                node("Relu", &["x"], &["b"], "b"),
                node("Add", &["a", "b"], &["c"], "c"),
            ],
            &[
                (11, value_info("x", 1, &[1])),
                (12, value_info("c", 1, &[1])),
            ],
        );
        let mut parsed = Model::parse(&sorted, Path::new("")).unwrap();
        assert!(!topological_sort(&mut parsed).unwrap());
        let unsorted = model(
            &[
                node("Add", &["a", "b"], &["c"], "c"),
                node("Relu", &["x"], &["a"], "a"),
                node("Relu", &["x"], &["b"], "b"),
            ],
            &[
                (11, value_info("x", 1, &[1])),
                (12, value_info("c", 1, &[1])),
            ],
        );
        let mut parsed = Model::parse(&unsorted, Path::new("")).unwrap();
        assert!(topological_sort(&mut parsed).unwrap());
        let names: Vec<&str> = parsed.graph.nodes.iter().map(|n| n.name.as_str()).collect();
        assert_eq!(names, ["a", "b", "c"]);
    }

    #[test]
    fn unused_nodes_and_initializers_go() {
        let mut initializer = Vec::new();
        put_varint(&mut initializer, 2, 1);
        put_bytes(&mut initializer, 8, b"w");
        put_bytes(&mut initializer, 9, &[0, 0, 0, 0]);
        let encoded = model(
            &[
                node("Relu", &["w"], &["dead"], "d"),
                node("Neg", &["x"], &["z"], "n"),
            ],
            &[
                (5, initializer),
                (11, value_info("x", 1, &[1])),
                (12, value_info("z", 1, &[1])),
            ],
        );
        let mut parsed = Model::parse(&encoded, Path::new("")).unwrap();
        assert!(remove_unused_nodes(&mut parsed));
        assert_eq!(parsed.graph.nodes.len(), 1);
        assert!(parsed.graph.initializers.is_empty());
    }
}
