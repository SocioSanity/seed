# Seed tools

These are the tool runtimes the seeds were trained to call, and the ones `seed_pool.py --tools` actually runs. Each call is a JSON object inside `<tool_call>...</tool_call>`; the result is wrapped in `<tool_result>...</tool_result>`.

| tool | what it does | extra setup |
|---|---|---|
| `calculator` | exact arithmetic (`12*(3+4)`, `2^10`, …) | none |
| `wiki_search` | search a **local** Wikipedia-introductions index and return the best matches | set `SEED_WIKI` to a folder that contains `wiki_memory.py` and the index |
| `read_file` | read a text file from the workspace (paths cannot escape it) | default workspace is `./workspace`, or set `SEED_WORKSPACE` |
| `python_exec` | run a short Python snippet in an isolated process (`python -I`, temp dir, 10 s timeout) and return stdout/stderr, including tracebacks | none; this is the sandbox `fix1` was trained to use |

## Chat with tools

```bash
python seed_pool.py --dir pool5 chat --tools --prompt "What is 17 * 19?"
```

Without a local Wikipedia index, `wiki_search` returns a short error telling you to set `SEED_WIKI`. Calculator, `read_file`, and `python_exec` work with a fresh checkout.

## Call format

```json
{"name": "calculator", "arguments": {"expression": "17*19"}}
{"name": "wiki_search", "arguments": {"query": "Ada Lovelace"}}
{"name": "read_file", "arguments": {"path": "notes.txt"}}
{"name": "python_exec", "arguments": {"code": "print(sum(range(10)))"}}
```

`python_exec` is a local sandbox for the model's run-and-fix loop. It is not a general-purpose remote runner: there is no network setup, the process is isolated with `python -I`, and it is killed after 10 seconds.

The same helpers also live in `comparison/tiny100m/chat100m.py` for the 100M comparison model (`wiki_search`, `calculator`, `read_file`). `seed_pool.py` prefers this `tools/` package when it is present.
