# seed-socket-v1

This folder holds the public metadata for the Seed organ socket:

- `socket.json` — width, tokenizer id, rope, cell contract, and sha256 pins
- `calib_tokens.safetensors` — the small calibration set used when checking a transplant

The frozen token-embedding anchor, `anchor_emb.safetensors` (~6.0 MB, sha prefix `a8438384b2c1b973` in `socket.json`), is not shipped in git because this repository leaves out files larger than 5 MB.

- To **join the official v1 ecosystem** (so organs interchange with other v1 Seeds), you need that official anchor file placed beside these two files.
- To **start your own socket**, run `python parts_tool.py make-socket --from path/to/latest.pt --socket-dir socket/seed-socket-v1` from a trunk checkpoint. Organs built on a new anchor will not plug into official v1 hosts.

Do not edit `socket.json` by hand if you want compatibility. A new anchor or tokenizer is a new socket version.
