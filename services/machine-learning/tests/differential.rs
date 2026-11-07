use std::path::PathBuf;

use reqwest::Method;
use reqwest::multipart::{Form, Part};
use serde_json::{Value, json};

const NEAR_TIE_FLIPS_PER_IMAGE: usize = 1;

type FormFactory = Box<dyn Fn() -> Form>;
type Header = Option<(&'static str, &'static str)>;

struct Target {
    reference: String,
    candidate: String,
    client: reqwest::Client,
}

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

async fn reply(response: reqwest::Response, base: &str, names: &[&str]) -> Reply {
    let authority = base
        .split_once("://")
        .map_or(base, |(_, authority)| authority);
    let headers = names
        .iter()
        .filter_map(|name| {
            let value = response.headers().get(*name)?.to_str().ok()?;
            let value = if *name == "allow" {
                let mut methods: Vec<&str> = value.split(", ").collect();
                methods.sort_unstable();
                methods.join(", ")
            } else {
                value.replace(authority, "{base}")
            };
            Some(((*name).to_owned(), value))
        })
        .collect();
    Reply {
        status: response.status().as_u16(),
        headers,
        body: response.text().await.expect("body"),
    }
}

impl Target {
    fn from_env() -> Option<Self> {
        Some(Self {
            reference: std::env::var("IMMICH_ML_REFERENCE_URL").ok()?,
            candidate: std::env::var("IMMICH_ML_CANDIDATE_URL").ok()?,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(600))
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .ok()?,
        })
    }

    async fn send(&self, base: &str, form: impl Fn() -> Form) -> Reply {
        let response = self
            .client
            .post(format!("{base}/predict"))
            .multipart(form())
            .send()
            .await
            .expect("request");
        reply(response, base, &["content-type"]).await
    }

    async fn request(&self, base: &str, method: &Method, path: &str, header: Header) -> Reply {
        let mut request = self.client.request(method.clone(), format!("{base}{path}"));
        if let Some((name, value)) = header {
            request = request.header(name, value);
        }
        let response = request.send().await.expect("request");
        reply(
            response,
            base,
            &["allow", "content-length", "content-type", "location"],
        )
        .await
    }

    async fn both(&self, form: impl Fn() -> Form) -> (Reply, Reply) {
        (
            self.send(&self.reference, &form).await,
            self.send(&self.candidate, &form).await,
        )
    }
}

#[derive(Default)]
struct Report {
    checks: usize,
    identical: usize,
    failures: Vec<String>,
    warnings: Vec<String>,
    differing: Vec<String>,
    worst: Vec<(String, f64)>,
}

struct Findings {
    problems: Vec<String>,
    warnings: Vec<String>,
    measures: Vec<(String, f64)>,
}

impl Report {
    fn record(
        &mut self,
        label: &str,
        reference: &Reply,
        candidate: &Reply,
        compare: impl FnOnce(&Value, &Value, &mut Findings),
    ) {
        self.checks += 1;
        if reference.status == candidate.status
            && reference.headers == candidate.headers
            && reference.body == candidate.body
        {
            self.identical += 1;
            return;
        }
        self.differing.push(label.to_owned());
        if reference.headers != candidate.headers {
            self.failures.push(format!(
                "{label}: headers differ\n  reference: {:?}\n  candidate: {:?}",
                reference.headers, candidate.headers
            ));
            return;
        }
        if reference.status != candidate.status {
            self.failures.push(format!(
                "{label}: status {} vs {}\n  reference: {}\n  candidate: {}",
                reference.status, candidate.status, reference.body, candidate.body
            ));
            return;
        }
        match (
            serde_json::from_str::<Value>(&reference.body),
            serde_json::from_str::<Value>(&candidate.body),
        ) {
            (Ok(expected), Ok(actual)) => {
                let mut findings = Findings {
                    problems: Vec::new(),
                    warnings: Vec::new(),
                    measures: Vec::new(),
                };
                compare(&expected, &actual, &mut findings);
                self.failures.extend(
                    findings
                        .problems
                        .into_iter()
                        .map(|problem| format!("{label}: {problem}")),
                );
                self.warnings.extend(
                    findings
                        .warnings
                        .into_iter()
                        .map(|warning| format!("{label}: {warning}")),
                );
                self.worst.extend(
                    findings
                        .measures
                        .into_iter()
                        .map(|(name, value)| (format!("{label} {name}"), value)),
                );
            }
            _ => self.failures.push(format!(
                "{label}: bodies differ\n  reference: {}\n  candidate: {}",
                reference.body, candidate.body
            )),
        }
    }
}

fn embedding(value: &Value) -> Vec<f64> {
    let text = value.as_str().expect("embedding string");
    serde_json::from_str::<Vec<f64>>(text).expect("embedding array")
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn floats(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .map(|items| items.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default()
}

fn compare_embedding(expected: &Value, actual: &Value, findings: &mut Findings, limit: f64) {
    let (e, a) = (embedding(expected), embedding(actual));
    if e.len() != a.len() {
        findings
            .problems
            .push(format!("embedding length {} vs {}", e.len(), a.len()));
        return;
    }
    let similarity = cosine(&e, &a);
    findings
        .measures
        .push(("cosine distance".to_owned(), 1.0 - similarity));
    if similarity < limit {
        findings.problems.push(format!(
            "embedding cosine similarity {similarity:.6} is below {limit}"
        ));
    }
}

fn same_size(expected: &Value, actual: &Value, findings: &mut Findings) {
    if expected.get("imageHeight") != actual.get("imageHeight")
        || expected.get("imageWidth") != actual.get("imageWidth")
    {
        findings.problems.push("image size differs".to_owned());
    }
}

fn compare_clip(expected: &Value, actual: &Value, findings: &mut Findings) {
    compare_embedding(&expected["clip"], &actual["clip"], findings, 0.9999);
    same_size(expected, actual, findings);
}

fn face_box(face: &Value) -> [f64; 4] {
    ["x1", "y1", "x2", "y2"].map(|corner| face["boundingBox"][corner].as_f64().unwrap_or(0.0))
}

fn iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let width = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let height = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let intersection = width * height;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

fn compare_faces(min_score: f64) -> impl Fn(&Value, &Value, &mut Findings) {
    move |expected, actual, findings| {
        let e = expected["facial-recognition"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let a = actual["facial-recognition"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let mut used = vec![false; a.len()];
        let borderline =
            |face: &Value| (face["score"].as_f64().unwrap_or(0.0) - min_score).abs() < 0.01;
        for ef in &e {
            let best = a
                .iter()
                .enumerate()
                .filter(|(index, _)| !used[*index])
                .map(|(index, af)| (index, iou(face_box(ef), face_box(af))))
                .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
            let Some((index, overlap)) = best.filter(|(_, overlap)| *overlap > 0.5) else {
                let message = format!(
                    "reference face at {:?} (score {}) has no match",
                    face_box(ef),
                    ef["score"]
                );
                if borderline(ef) {
                    findings.warnings.push(message)
                } else {
                    findings.problems.push(message)
                }
                continue;
            };
            used[index] = true;
            let af = &a[index];
            let shift = face_box(ef)
                .iter()
                .zip(face_box(af))
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max);
            findings
                .measures
                .push(("face box shift px".to_owned(), shift));
            let score =
                (ef["score"].as_f64().unwrap_or(0.0) - af["score"].as_f64().unwrap_or(0.0)).abs();
            findings
                .measures
                .push(("face score diff".to_owned(), score));
            let mut local = Findings {
                problems: Vec::new(),
                warnings: Vec::new(),
                measures: Vec::new(),
            };
            compare_embedding(&ef["embedding"], &af["embedding"], &mut local, 0.995);
            findings.measures.extend(local.measures);
            let mut issues = local.problems;
            if shift > 1.0 {
                issues.push(format!("face box moved {shift}px (overlap {overlap:.3})"));
            }
            if score > 1e-3 {
                issues.push(format!("face score differs by {score}"));
            }
            findings.problems.extend(issues);
        }
        for (index, af) in a.iter().enumerate().filter(|(index, _)| !used[*index]) {
            let message = format!(
                "candidate face {index} at {:?} (score {}) has no match",
                face_box(af),
                af["score"]
            );
            if borderline(af) {
                findings.warnings.push(message)
            } else {
                findings.problems.push(message)
            }
        }
        same_size(expected, actual, findings);
    }
}

fn quads(value: &Value) -> Vec<[f64; 8]> {
    floats(value).as_chunks::<8>().0.to_vec()
}

fn compare_ocr(expected: &Value, actual: &Value, findings: &mut Findings) {
    let (e, a) = (&expected["ocr"], &actual["ocr"]);
    let (et, at) = (
        e["text"].as_array().cloned().unwrap_or_default(),
        a["text"].as_array().cloned().unwrap_or_default(),
    );
    let (eb, ab) = (quads(&e["box"]), quads(&a["box"]));
    let (es, as_) = (floats(&e["textScore"]), floats(&a["textScore"]));
    let mut used = vec![false; ab.len()];
    let mut flips = 0;
    for (index, quad) in eb.iter().enumerate() {
        let matched = ab
            .iter()
            .enumerate()
            .filter(|(i, _)| !used[*i])
            .map(|(i, other)| (i, max_abs(quad, other)))
            .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
        match matched.filter(|(_, shift)| *shift < 0.01) {
            Some((other, shift)) => {
                used[other] = true;
                findings
                    .measures
                    .push(("box corner shift (fraction)".to_owned(), shift));
                findings
                    .measures
                    .push(("text score diff".to_owned(), (es[index] - as_[other]).abs()));
                if et[index] != at[other] {
                    flips += 1;
                    findings
                        .warnings
                        .push(format!("text {} vs {}", et[index], at[other]));
                }
            }
            None => findings.warnings.push(format!(
                "reference text {} has no matching box (score {})",
                et[index], es[index]
            )),
        }
    }
    let unmatched = used.iter().filter(|used| !**used).count();
    let missing = eb.len() - used.iter().filter(|used| **used).count();
    let allowed_flips = NEAR_TIE_FLIPS_PER_IMAGE.max(eb.len() / 20);
    if flips > allowed_flips || (missing + unmatched) * 10 > eb.len().max(1) {
        findings.problems.push(format!(
            "{flips} text flips and {} unmatched boxes out of {}",
            missing + unmatched,
            eb.len()
        ));
    }
}

fn exact(_: &Value, _: &Value, findings: &mut Findings) {
    findings.problems.push("responses differ".to_owned());
}

fn image_form(entries: Value, bytes: Vec<u8>) -> impl Fn() -> Form {
    move || {
        Form::new()
            .text("entries", entries.to_string())
            .part("image", Part::bytes(bytes.clone()).file_name("blob"))
    }
}

fn text_form(entries: Value, text: &'static str) -> impl Fn() -> Form {
    move || {
        Form::new()
            .text("entries", entries.to_string())
            .text("text", text)
    }
}

#[tokio::test]
#[ignore = "needs a running reference service and candidate service"]
async fn rust_matches_python() {
    let target =
        Target::from_env().expect("set IMMICH_ML_REFERENCE_URL and IMMICH_ML_CANDIDATE_URL");
    let images: PathBuf = std::env::var("IMMICH_ML_DIFFERENTIAL_IMAGES")
        .expect("set IMMICH_ML_DIFFERENTIAL_IMAGES")
        .into();
    let clip_model =
        std::env::var("IMMICH_ML_CLIP_MODEL").unwrap_or_else(|_| "ViT-B-32__openai".to_owned());
    let face_model =
        std::env::var("IMMICH_ML_FACE_MODEL").unwrap_or_else(|_| "buffalo_l".to_owned());
    let ocr_model =
        std::env::var("IMMICH_ML_OCR_MODEL").unwrap_or_else(|_| "PP-OCRv5_mobile".to_owned());
    let mut report = Report::default();

    let mut files: Vec<PathBuf> = std::fs::read_dir(&images)
        .expect("image directory")
        .filter_map(|entry| Some(entry.ok()?.path()))
        .collect();
    files.sort();
    for file in files {
        let name = file
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let bytes = std::fs::read(&file).expect("image");
        let clip = json!({"clip": {"visual": {"modelName": clip_model}}});
        let (reference, candidate) = target.both(image_form(clip, bytes.clone())).await;
        report.record(
            &format!("{name} clip"),
            &reference,
            &candidate,
            compare_clip,
        );

        for min_score in [0.7, 0.3] {
            let faces = json!({"facial-recognition": {"detection": {"modelName": face_model, "options": {"minScore": min_score}}, "recognition": {"modelName": face_model}}});
            let (reference, candidate) = target.both(image_form(faces, bytes.clone())).await;
            report.record(
                &format!("{name} faces@{min_score}"),
                &reference,
                &candidate,
                compare_faces(min_score),
            );
        }

        let ocr = json!({"ocr": {"detection": {"modelName": ocr_model, "options": {"minScore": 0.5, "maxResolution": 736}}, "recognition": {"modelName": ocr_model, "options": {"minScore": 0.8}}}});
        let (reference, candidate) = target.both(image_form(ocr, bytes.clone())).await;
        report.record(&format!("{name} ocr"), &reference, &candidate, compare_ocr);
    }

    for text in [
        "a photo of a dog",
        "Beach at SUNSET, with   palm trees!",
        "猫が寝ている",
        "żółta łódź na jeziorze",
        "\u{1c}spaces\u{1f}\ttabs\n",
        "🚗🚕🚙",
    ] {
        let entries =
            json!({"clip": {"textual": {"modelName": clip_model, "options": {"language": "en"}}}});
        let (reference, candidate) = target.both(text_form(entries, text)).await;
        report.record(
            &format!("text {text:?}"),
            &reference,
            &candidate,
            |e, a, findings| compare_embedding(&e["clip"], &a["clip"], findings, 0.99999),
        );
    }

    let cases: Vec<(&str, FormFactory)> = vec![
        ("no entries", Box::new(|| Form::new().text("text", "hello"))),
        (
            "empty entries",
            Box::new(|| Form::new().text("entries", "").text("text", "hello")),
        ),
        (
            "invalid json",
            Box::new(|| Form::new().text("entries", "{").text("text", "hello")),
        ),
        (
            "no task",
            Box::new(|| Form::new().text("entries", "{}").text("text", "hello")),
        ),
        (
            "unknown field",
            Box::new(|| {
                Form::new()
                    .text(
                        "entries",
                        r#"{"clip": {"visual": {"modelName": "ViT-B-32__openai", "extra": 1}}}"#,
                    )
                    .text("text", "hello")
            }),
        ),
        (
            "unknown model",
            Box::new(|| {
                Form::new()
                    .text(
                        "entries",
                        r#"{"clip": {"textual": {"modelName": "not-a-model"}}}"#,
                    )
                    .text("text", "hello")
            }),
        ),
        (
            "no payload",
            Box::new(|| {
                Form::new().text(
                    "entries",
                    r#"{"clip": {"textual": {"modelName": "ViT-B-32__openai"}}}"#,
                )
            }),
        ),
        (
            "empty text",
            Box::new(|| {
                Form::new()
                    .text(
                        "entries",
                        r#"{"clip": {"textual": {"modelName": "ViT-B-32__openai"}}}"#,
                    )
                    .text("text", "")
            }),
        ),
        (
            "missing dependency",
            Box::new(|| {
                Form::new()
                    .text(
                        "entries",
                        r#"{"facial-recognition": {"recognition": {"modelName": "buffalo_l"}}}"#,
                    )
                    .part("image", Part::bytes(vec![0u8; 4]).file_name("blob"))
            }),
        ),
        (
            "bad image",
            Box::new(|| {
                Form::new()
                    .text(
                        "entries",
                        r#"{"clip": {"visual": {"modelName": "ViT-B-32__openai"}}}"#,
                    )
                    .part(
                        "image",
                        Part::bytes(b"not an image".to_vec()).file_name("blob"),
                    )
            }),
        ),
    ];
    for (label, form) in cases {
        let (reference, candidate) = target.both(form).await;
        report.record(label, &reference, &candidate, exact);
    }

    let requests: [(Method, &str, Header); 18] = [
        (Method::GET, "/", None),
        (Method::GET, "/ping", None),
        (Method::GET, "/missing", None),
        (Method::GET, "/openapi.json", None),
        (Method::GET, "/docs", None),
        (Method::GET, "/docs/oauth2-redirect", None),
        (Method::GET, "/redoc", None),
        (Method::HEAD, "/openapi.json", None),
        (Method::HEAD, "/ping", None),
        (Method::POST, "/ping", None),
        (Method::DELETE, "/docs", None),
        (Method::GET, "/ping/", None),
        (Method::POST, "/predict/", None),
        (Method::HEAD, "/docs/", None),
        (Method::GET, "/p%69ng", None),
        (Method::GET, "/ping%2F?a=1&b=%20x", None),
        (Method::GET, "/docs/", Some(("x-forwarded-proto", "https"))),
        (Method::GET, "/docs/", Some(("host", "example.com:8080"))),
    ];
    for (method, path, header) in requests {
        let reference = target
            .request(&target.reference, &method, path, header)
            .await;
        let candidate = target
            .request(&target.candidate, &method, path, header)
            .await;
        let label = match header {
            Some((name, value)) => format!("{method} {path} {name}: {value}"),
            None => format!("{method} {path}"),
        };
        report.record(&label, &reference, &candidate, exact);
    }

    report
        .worst
        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "{} checks, {} byte-identical, {} warnings, {} failures",
        report.checks,
        report.identical,
        report.warnings.len(),
        report.failures.len()
    );
    println!("not byte-identical: {}", report.differing.join(", "));
    println!("largest differences:");
    for (name, value) in report.worst.iter().take(12) {
        println!("  {value:.3e}  {name}");
    }
    for warning in &report.warnings {
        println!("WARN {warning}");
    }
    for failure in &report.failures {
        println!("FAIL {failure}");
    }
    assert!(
        report.failures.is_empty(),
        "{} differential failures",
        report.failures.len()
    );
}
