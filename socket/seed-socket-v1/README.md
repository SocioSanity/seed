# seed-socket-v1

This folder holds the official Seed organ socket:

- `socket.json` — width, tokenizer id, rope, cell contract, and sha256 pins
- `calib_tokens.safetensors` — the small calibration set used when checking a transplant
- `anchor_emb.safetensors` — the frozen token-embedding anchor (~6.0 MB, sha prefix `a8438384b2c1b973`)

The anchor is included so a checkout can join the official v1 ecosystem and organs can interchange with other v1 Seeds. Do not replace or edit it if you want compatibility.

To start your own socket instead, run `python parts_tool.py make-socket --from path/to/latest.pt --socket-dir socket/your-socket` from a trunk checkpoint. Organs built on a new anchor will not plug into official v1 hosts.

Do not edit `socket.json` by hand if you want compatibility. A new anchor or tokenizer is a new socket version.
