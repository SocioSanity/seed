# Seed parts: export, share, hot-install, hot-remove

A **part** is two files next to each other:

| file | what |
|---|---|
| `NAME.safetensors` | the weights (standard safetensors, fp32, readable by any safetensors library) |
| `NAME.json` | the manifest: `format: "seed-part/1"`, `kind`, `id`, `sha256` of the weights file, model signature, tokenizer, lineage, step, `protected` |

Kinds:

* **cell**: one cell (`q`, `proj`, `w12`, `down` weights + its `router` row). Tied to a layer slot.
* **organ**: one whole column, i.e. one cell in every layer (`L0.q … L{n-1}.router`). Needs a model with the same number of layers.
* **tree**: a whole Seed (full state dict + structure). You can chat with it, score parts against it, or start a run from it (`--init-tree`). It is never hot-installed into a running model.

Compatibility: a part installs only when `d`, `n_head`, `head_dim`, `mlp`, vocab, arch (`seed-v4-final`), RoPE style and the **tokenizer sha** all match. If any of these differ, the part is refused and moved to `rejected/` with a `.reason.txt` file. A part from another lineage (another run) is allowed, with a warning.

## Folders (default `parts`, set with `--parts-dir`)

```
inbox\       drop parts or command files here while training runs (checked every --inbox-every steps)
processed\   everything the trainer has handled (timestamped; nothing is deleted)
rejected\    refused parts or commands, each with NAME.reason.txt
exports\     parts the trainer exported
removed\     every organ or cell taken out of the model (hot-remove, immune, cell swap), archived as parts
scores\      JSON results from parts_tool.py score
```

## Hot install (while training)

Copy `NAME.safetensors` **first**, then `NAME.json`, into `inbox\`. The trainer reads the manifest only when the weights are present and the sha matches. A file that is still being copied is retried for up to 60 s.

* **organ**: added as a new column with alpha 0, so the model's output is exactly unchanged. It then fades in to alpha 1 over `--fade-steps` steps (default 300) while training continues. Its optimiser state starts fresh.
* **cell**: swaps into a slot, which keeps the model symmetric. The target is the manifest's `target` cell id if set; otherwise it replaces the least useful unprotected cell in the manifest's `layer`. The old cell fades out and is archived to `removed\`. The new cell is then loaded at alpha 0 and fades in.

Installs are refused when the model is hungry (`near_cap`) or when the part would push it past `--max-params`.

## Commands (`*.cmd.json` in the inbox)

```json
{"cmd": "export", "kind": "organ", "id": "auto"}       // most useful organ (or "o2"), EMA weights
{"cmd": "export", "kind": "cell",  "id": "c17"}        // or "auto" = most useful cell
{"cmd": "export", "kind": "tree"}                      // the whole model
{"cmd": "remove", "id": "o3"}                          // organ: fade out, archive to removed\, take out
{"cmd": "remove", "id": "c17"}                         // cell: fade out, archive, replace with a fresh cell (fades in)
{"cmd": "remove", "id": "last_installed"}
{"cmd": "remove", "id": "o1", "force": true}           // needed when the organ/cell is protected
{"cmd": "protect", "id": "o1"}   /   {"cmd": "unprotect", "id": "o1"}
```

`export` takes `"ema": false` to export the raw weights instead. **Protected** parts are never removed (unless forced), are never replaced by the immune system, and are never moved by the organizer. The tag is kept in exports and checkpoints. The last organ can't be removed.

## Offline tool

```
python parts_tool.py inspect  PART.json
python parts_tool.py validate PART.json --base runs/run1/latest.pt
python parts_tool.py export   --from runs/run1/latest.pt --kind organ --id o2 --out parts/exports
python parts_tool.py score    PART.json --base TREE.json [--device cuda]
python parts_tool.py protect  PART.json [--off]
```

`score` plugs the part into the base Seed at full strength, without training. It reports held-out TinyStories loss before and after (`gain` > 0 means it helps as-is) and writes `scores\score_<id>_<sha>_on_<lineage>.json`. For a fair comparison, use the same base tree for every part. A part from another lineage usually needs the fade-in plus training before it pays off, so a negative zero-shot score is normal for it.

## Chat with parts

```
python chat.py runs/run1/latest.pt --parts parts/exports\organ_o3_....json
```
