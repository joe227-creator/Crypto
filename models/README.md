# Local Model Assets

Model weights are local-only and ignored from Git. Adapters load these paths,
not shared Hugging Face cache paths, so experiments in this repository cannot
change model files used by other projects.

## Sources

- TimesFM 2.5 PyTorch: `https://github.com/google-research/timesfm`
  checkpoint `google/timesfm-2.5-200m-pytorch`
- Kronos base: `https://github.com/shiyu-coder/Kronos`
  checkpoint `NeoQuasar/Kronos-base`
- Kronos tokenizer: checkpoint `NeoQuasar/Kronos-Tokenizer-base`

## Files

- `models/timesfm-2.5-200m-pytorch/model.safetensors`
  SHA-256 `2F776EFE6245E42B24BC4153FFDF61810140210E4BD3B01FB21F7AA779AB6CE8`
- `models/kronos-base/model.safetensors`
  SHA-256 `ABFF193ACAB6DB1A0368E9773E75799D11403B6D054EE6D5F0A11AEABC5F4B83`
- `models/kronos-tokenizer-base/model.safetensors`
  SHA-256 `59D85F6AF76A2C3B8240EA06CB21DB4213B4EECA053F246B23E29CF832FC6BEE`

Source repositories are cloned under ignored `external/timesfm` and
`external/kronos`.
