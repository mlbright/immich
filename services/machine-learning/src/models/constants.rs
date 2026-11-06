use crate::config::clean_name;
use crate::schemas::ModelSource;

const OPENCLIP_MODELS: &[&str] = &[
    "RN101__openai",
    "RN101__yfcc15m",
    "RN50__cc12m",
    "RN50__openai",
    "RN50__yfcc15m",
    "RN50x16__openai",
    "RN50x4__openai",
    "RN50x64__openai",
    "ViT-B-16-SigLIP-256__webli",
    "ViT-B-16-SigLIP-384__webli",
    "ViT-B-16-SigLIP-512__webli",
    "ViT-B-16-SigLIP-i18n-256__webli",
    "ViT-B-16-SigLIP2__webli",
    "ViT-B-16-SigLIP__webli",
    "ViT-B-16-plus-240__laion400m_e31",
    "ViT-B-16-plus-240__laion400m_e32",
    "ViT-B-16__laion400m_e31",
    "ViT-B-16__laion400m_e32",
    "ViT-B-16__openai",
    "ViT-B-32-SigLIP2-256__webli",
    "ViT-B-32__laion2b-s34b-b79k",
    "ViT-B-32__laion2b_e16",
    "ViT-B-32__laion400m_e31",
    "ViT-B-32__laion400m_e32",
    "ViT-B-32__openai",
    "ViT-H-14-378-quickgelu__dfn5b",
    "ViT-H-14-quickgelu__dfn5b",
    "ViT-H-14__laion2b-s32b-b79k",
    "ViT-L-14-336__openai",
    "ViT-L-14-quickgelu__dfn2b",
    "ViT-L-14__laion2b-s32b-b82k",
    "ViT-L-14__laion400m_e31",
    "ViT-L-14__laion400m_e32",
    "ViT-L-14__openai",
    "ViT-L-16-SigLIP-256__webli",
    "ViT-L-16-SigLIP-384__webli",
    "ViT-L-16-SigLIP2-256__webli",
    "ViT-L-16-SigLIP2-384__webli",
    "ViT-L-16-SigLIP2-512__webli",
    "ViT-SO400M-14-SigLIP-384__webli",
    "ViT-SO400M-14-SigLIP2-378__webli",
    "ViT-SO400M-14-SigLIP2__webli",
    "ViT-SO400M-16-SigLIP2-256__webli",
    "ViT-SO400M-16-SigLIP2-384__webli",
    "ViT-SO400M-16-SigLIP2-512__webli",
    "ViT-g-14__laion2b-s12b-b42k",
    "ViT-gopt-16-SigLIP2-256__webli",
    "ViT-gopt-16-SigLIP2-384__webli",
    "XLM-Roberta-Base-ViT-B-32__laion5b_s13b_b90k",
    "XLM-Roberta-Large-ViT-H-14__frozen_laion5b_s13b_b90k",
    "nllb-clip-base-siglip__mrl",
    "nllb-clip-base-siglip__v1",
    "nllb-clip-large-siglip__mrl",
    "nllb-clip-large-siglip__v1",
];

const MCLIP_MODELS: &[&str] = &[
    "LABSE-Vit-L-14",
    "XLM-Roberta-Large-Vit-B-16Plus",
    "XLM-Roberta-Large-Vit-B-32",
    "XLM-Roberta-Large-Vit-L-14",
];

const INSIGHTFACE_MODELS: &[&str] = &["antelopev2", "buffalo_l", "buffalo_m", "buffalo_s"];

const PADDLE_MODELS: &[&str] = &[
    "CH__PP-OCRv5_mobile",
    "CH__PP-OCRv5_server",
    "EL__PP-OCRv5_mobile",
    "EN__PP-OCRv5_mobile",
    "ESLAV__PP-OCRv5_mobile",
    "KOREAN__PP-OCRv5_mobile",
    "LATIN__PP-OCRv5_mobile",
    "PP-OCRv5_mobile",
    "PP-OCRv5_server",
    "PP-OCRv6_medium",
    "PP-OCRv6_small",
    "PP-OCRv6_tiny",
    "TH__PP-OCRv5_mobile",
];

pub const SUPPORTED_PROVIDERS: &[&str] = &[
    "nv_tensorrt_rtx",
    "CUDAExecutionProvider",
    "MIGraphXExecutionProvider",
    "OpenVINOExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
];

pub const WEBLATE_TO_FLORES200: &[(&str, &str)] = &[
    ("af", "afr_Latn"),
    ("ar", "arb_Arab"),
    ("az", "azj_Latn"),
    ("be", "bel_Cyrl"),
    ("bg", "bul_Cyrl"),
    ("ca", "cat_Latn"),
    ("cs", "ces_Latn"),
    ("da", "dan_Latn"),
    ("de", "deu_Latn"),
    ("el", "ell_Grek"),
    ("en", "eng_Latn"),
    ("es", "spa_Latn"),
    ("et", "est_Latn"),
    ("fa", "pes_Arab"),
    ("fi", "fin_Latn"),
    ("fr", "fra_Latn"),
    ("he", "heb_Hebr"),
    ("hi", "hin_Deva"),
    ("hr", "hrv_Latn"),
    ("hu", "hun_Latn"),
    ("hy", "hye_Armn"),
    ("id", "ind_Latn"),
    ("it", "ita_Latn"),
    ("ja", "jpn_Hira"),
    ("kmr", "kmr_Latn"),
    ("ko", "kor_Hang"),
    ("lb", "ltz_Latn"),
    ("lt", "lit_Latn"),
    ("lv", "lav_Latn"),
    ("mfa", "zsm_Latn"),
    ("mk", "mkd_Cyrl"),
    ("mn", "khk_Cyrl"),
    ("mr", "mar_Deva"),
    ("ms", "zsm_Latn"),
    ("nb-NO", "nob_Latn"),
    ("nn", "nno_Latn"),
    ("nl", "nld_Latn"),
    ("pl", "pol_Latn"),
    ("pt-BR", "por_Latn"),
    ("pt", "por_Latn"),
    ("ro", "ron_Latn"),
    ("ru", "rus_Cyrl"),
    ("sk", "slk_Latn"),
    ("sl", "slv_Latn"),
    ("sr-Cyrl", "srp_Cyrl"),
    ("sv", "swe_Latn"),
    ("ta", "tam_Taml"),
    ("te", "tel_Telu"),
    ("th", "tha_Thai"),
    ("tr", "tur_Latn"),
    ("uk", "ukr_Cyrl"),
    ("ur", "urd_Arab"),
    ("vi", "vie_Latn"),
    ("zh-CN", "zho_Hans"),
    ("zh-Hans", "zho_Hans"),
    ("zh-TW", "zho_Hant"),
];

pub fn flores200(language: &str) -> Option<&'static str> {
    WEBLATE_TO_FLORES200
        .iter()
        .find(|(weblate, _)| *weblate == language)
        .map(|(_, flores)| *flores)
}

pub fn get_model_source(model_name: &str) -> Option<ModelSource> {
    let cleaned_name = clean_name(model_name);
    let cleaned_name = cleaned_name.as_str();

    if INSIGHTFACE_MODELS.contains(&cleaned_name) {
        return Some(ModelSource::Insightface);
    }

    if MCLIP_MODELS.contains(&cleaned_name) {
        return Some(ModelSource::Mclip);
    }

    if OPENCLIP_MODELS.contains(&cleaned_name) {
        return Some(ModelSource::Openclip);
    }

    if PADDLE_MODELS.contains(&cleaned_name) {
        return Some(ModelSource::Paddle);
    }

    None
}
