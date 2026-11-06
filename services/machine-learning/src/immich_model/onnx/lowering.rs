use anyhow::{Context, Result};

use crate::onnx_ir::{Attribute, Graph, Model, Node, data_type, name_fix};

const MS_DOMAIN: &str = "com.microsoft";
const RULE_NAME_TAG: &str = "pkg.onnxscript.rewriter.rule_name";
const FUSE_SKIP_LAYER_NORM: &str = "FuseSkipLayerNorm";
const UNNAMED: &str = "\u{0}";
const FUSED_SKIP_DTYPES: [i64; 2] = [data_type::FLOAT, data_type::FLOAT16];

struct Matched {
    add: usize,
    norm: usize,
    shared: bool,
}

fn named(values: &[String]) -> bool {
    values.iter().all(|value| !value.is_empty())
}

fn is_add(node: &Node) -> bool {
    node.op_type == "Add"
        && node.domain.is_empty()
        && node.inputs.len() == 2
        && named(&node.inputs)
        && node
            .outputs
            .first()
            .is_some_and(|output| !output.is_empty())
}

fn is_layer_norm(node: &Node) -> bool {
    node.op_type == "LayerNormalization"
        && node.domain.is_empty()
        && node.inputs.len() == 3
        && named(&node.inputs)
        && node
            .outputs
            .first()
            .is_some_and(|output| !output.is_empty())
}

fn valid_to_replace(graph: &Graph, matched: &[usize], outputs: &[&str]) -> bool {
    let uses = graph.uses();
    matched.iter().all(|&index| {
        graph.nodes[index].outputs.iter().all(|output| {
            output.is_empty()
                || outputs.contains(&output.as_str())
                || (!graph.is_output(output)
                    && uses
                        .get(output.as_str())
                        .is_none_or(|users| users.iter().all(|user| matched.contains(user))))
        })
    })
}

fn check(graph: &Graph, matched: &Matched) -> bool {
    let add = &graph.nodes[matched.add];
    let norm = &graph.nodes[matched.norm];
    let (x, skip, total) = (&add.inputs[0], &add.inputs[1], &add.outputs[0]);
    let (gamma, beta, normed) = (&norm.inputs[1], &norm.inputs[2], &norm.outputs[0]);
    if graph.is_output(normed) || graph.is_output(total) {
        return false;
    }
    let shared = graph
        .uses()
        .get(total.as_str())
        .is_some_and(|users| users.iter().any(|&user| user != matched.norm));
    if matched.shared != shared {
        return false;
    }
    let int = |name: &str, default: i64| {
        norm.attribute(name)
            .and_then(|attribute| attribute.i)
            .unwrap_or(default)
    };
    if norm.outputs.len() != 1 || int("stash_type", 1) != 1 {
        return false;
    }
    let (dtype, _) = graph.type_of(total);
    if ![-1, 2].contains(&int("axis", -1)) || !dtype.is_some_and(|d| FUSED_SKIP_DTYPES.contains(&d))
    {
        return false;
    }
    let rank3 = [x, skip, total].iter().all(|value| {
        let (value_dtype, rank) = graph.type_of(value);
        rank == Some(3) && value_dtype == dtype
    });
    rank3
        && [gamma, beta]
            .iter()
            .all(|value| graph.type_of(value).1 == Some(1))
}

fn match_unshared(graph: &Graph, index: usize) -> Option<Matched> {
    let norm = &graph.nodes[index];
    if !is_layer_norm(norm) {
        return None;
    }
    let add = *graph.producers().get(norm.inputs[0].as_str())?;
    if !is_add(&graph.nodes[add]) || graph.nodes[add].outputs[0] != norm.inputs[0] {
        return None;
    }
    if !valid_to_replace(graph, &[index, add], &[norm.outputs[0].as_str()]) {
        return None;
    }
    let matched = Matched {
        add,
        norm: index,
        shared: false,
    };
    check(graph, &matched).then_some(matched)
}

fn match_shared(graph: &Graph, index: usize) -> Option<Matched> {
    let add = &graph.nodes[index];
    if !is_add(add) {
        return None;
    }
    let total = add.outputs[0].as_str();
    for (candidate, norm) in graph.nodes.iter().enumerate() {
        if norm.op_type != "LayerNormalization" || !norm.domain.is_empty() {
            continue;
        }
        if !is_layer_norm(norm) || norm.inputs[0] != total {
            continue;
        }
        if !valid_to_replace(
            graph,
            &[index, candidate],
            &[total, norm.outputs[0].as_str()],
        ) {
            continue;
        }
        let matched = Matched {
            add: index,
            norm: candidate,
            shared: true,
        };
        return check(graph, &matched).then_some(matched);
    }
    None
}

fn merge_metadata(updated: &mut Vec<(String, String)>, updates: &[(String, String)]) {
    for (key, value) in updates {
        if value.is_empty() {
            continue;
        }
        match updated.iter_mut().find(|(existing, _)| existing == key) {
            Some((_, existing)) if !existing.is_empty() => {
                if key == RULE_NAME_TAG {
                    *existing = format!("{existing}, {value}");
                }
            }
            Some((_, existing)) => *existing = value.clone(),
            None => updated.push((key.clone(), value.clone())),
        }
    }
}

fn replace(model: &mut Model, matched: &Matched, anchor: usize) -> Result<usize> {
    let graph = &mut model.graph;
    let add = graph.nodes[matched.add].clone();
    let norm = graph.nodes[matched.norm].clone();
    let epsilon = norm
        .attribute("epsilon")
        .and_then(|attribute| attribute.f)
        .unwrap_or(1e-5f64 as f32);
    let mut metadata = vec![(RULE_NAME_TAG.to_owned(), FUSE_SKIP_LAYER_NORM.to_owned())];
    let order = if matched.shared {
        [&add, &norm]
    } else {
        [&norm, &add]
    };
    for node in order {
        merge_metadata(&mut metadata, &node.metadata);
    }
    let (total, normed) = (add.outputs[0].clone(), norm.outputs[0].clone());
    let outputs = if matched.shared {
        vec![
            normed.clone(),
            UNNAMED.to_owned(),
            UNNAMED.to_owned(),
            total.clone(),
        ]
    } else {
        vec![normed.clone()]
    };
    let name = graph.name_node("SkipLayerNormalization");
    let fused = Node::new(
        "SkipLayerNormalization",
        MS_DOMAIN,
        &name,
        vec![
            add.inputs[0].clone(),
            add.inputs[1].clone(),
            norm.inputs[1].clone(),
            norm.inputs[2].clone(),
        ],
        outputs,
        vec![Attribute::float("epsilon", epsilon)],
        metadata,
    )?;
    graph.strip_value_info(&normed)?;
    if matched.shared {
        graph.strip_value_info(&total)?;
    }
    let mut fused = Some(fused);
    let mut position = 0;
    let mut nodes = Vec::with_capacity(graph.nodes.len());
    for (index, node) in graph.nodes.drain(..).enumerate() {
        if index != matched.add && index != matched.norm {
            nodes.push(node);
        }
        if index == anchor {
            position = nodes.len();
            nodes.push(fused.take().context("fused node placed twice")?);
        }
    }
    graph.nodes = nodes;
    model.import_opset(MS_DOMAIN, 1);
    Ok(position)
}

pub fn fuse_skip_layer_norm(model: &mut Model) -> Result<bool> {
    let mut count = 0;
    let mut index = 0;
    while index < model.graph.nodes.len() {
        let matched =
            match_unshared(&model.graph, index).or_else(|| match_shared(&model.graph, index));
        if let Some(matched) = matched {
            index = replace(model, &matched, index)?;
            count += 1;
        }
        index += 1;
    }
    if count > 0 {
        name_fix(model);
    }
    for node in &mut model.graph.nodes {
        if node.op_type == "SkipLayerNormalization" && node.outputs.len() == 4 {
            node.outputs[1].clear();
            node.outputs[2].clear();
            node.touch();
        }
    }
    Ok(count > 0)
}
