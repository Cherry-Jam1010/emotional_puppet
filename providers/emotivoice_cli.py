from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer
from yacs import config as CONFIG


MAX_WAV_VALUE = 32768.0
_MODELS = None
_G2P = None
_LEXICON = None


def _load_models(repo_root: Path):
    global _MODELS, _G2P, _LEXICON
    if _MODELS is not None:
        return _MODELS

    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    from config.joint.config import Config
    from frontend import G2p, g2p_cn_en, read_lexicon
    from models.prompt_tts_modified.jets import JETSGenerator
    from models.prompt_tts_modified.simbert import StyleEncoder

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config.model_config_path, "r", encoding="utf-8") as fin:
        conf = CONFIG.load_cfg(fin)
    conf.n_vocab = config.n_symbols
    conf.n_speaker = config.speaker_n_labels

    style_encoder = StyleEncoder(config)
    style_ckpt = torch.load(config.style_encoder_ckpt, map_location="cpu")
    style_state = {}
    for key, value in style_ckpt["model"].items():
        style_state[key[7:]] = value
    style_encoder.load_state_dict(style_state, strict=False)
    style_encoder.eval()

    generator = JETSGenerator(conf).to(device)
    generator_ckpt = torch.load(
        repo_root / "outputs" / "prompt_tts_open_source_joint" / "ckpt" / "g_00140000",
        map_location=device,
    )
    generator.load_state_dict(generator_ckpt["generator"])
    generator.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    with open(config.token_list_path, "r", encoding="utf-8") as f:
        token2id = {t.strip(): idx for idx, t in enumerate(f.readlines())}
    with open(config.speaker2id_path, "r", encoding="utf-8") as f:
        speaker2id = {t.strip(): idx for idx, t in enumerate(f.readlines())}

    _G2P = G2p()
    _LEXICON = read_lexicon(f"{repo_root}/lexicon/librispeech-lexicon.txt")
    _MODELS = {
        "config": config,
        "device": device,
        "style_encoder": style_encoder,
        "generator": generator,
        "tokenizer": tokenizer,
        "token2id": token2id,
        "speaker2id": speaker2id,
        "g2p_cn_en": g2p_cn_en,
    }
    return _MODELS


def _style_embedding(tokenizer, style_encoder, prompt: str) -> np.ndarray:
    encoded = tokenizer([prompt], return_tensors="pt")
    with torch.no_grad():
        output = style_encoder(
            input_ids=encoded["input_ids"],
            token_type_ids=encoded["token_type_ids"],
            attention_mask=encoded["attention_mask"],
        )
    return output["pooled_output"].cpu().squeeze().numpy()


def synthesize(repo_root: Path, text: str, prompt: str, speaker: str, output_path: Path) -> None:
    models = _load_models(repo_root)
    config = models["config"]
    device = models["device"]
    style_encoder = models["style_encoder"]
    generator = models["generator"]
    tokenizer = models["tokenizer"]
    token2id = models["token2id"]
    speaker2id = models["speaker2id"]
    g2p_cn_en = models["g2p_cn_en"]

    if speaker not in speaker2id:
        raise RuntimeError(f"Unknown EmotiVoice speaker: {speaker}")

    phonemes = g2p_cn_en(text, _G2P, _LEXICON)
    text_int = [token2id[ph] for ph in phonemes.split()]
    style_embedding = _style_embedding(tokenizer, style_encoder, prompt)
    content_embedding = _style_embedding(tokenizer, style_encoder, text)

    sequence = torch.from_numpy(np.array(text_int)).to(device).long().unsqueeze(0)
    sequence_len = torch.from_numpy(np.array([len(text_int)])).to(device)
    style_embedding = torch.from_numpy(style_embedding).to(device).unsqueeze(0)
    content_embedding = torch.from_numpy(content_embedding).to(device).unsqueeze(0)
    speaker_tensor = torch.from_numpy(np.array([speaker2id[speaker]])).to(device)

    with torch.no_grad():
        infer_output = generator(
            inputs_ling=sequence,
            inputs_style_embedding=style_embedding,
            input_lengths=sequence_len,
            inputs_content_embedding=content_embedding,
            inputs_speaker=speaker_tensor,
            alpha=1.0,
        )

    audio = infer_output["wav_predictions"].squeeze() * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype("int16")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(file=str(output_path), data=audio, samplerate=config.sampling_rate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    synthesize(
        repo_root=Path(args.repo_root).resolve(),
        text=payload["text"],
        prompt=payload["prompt"],
        speaker=payload["speaker"],
        output_path=Path(args.output).resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
