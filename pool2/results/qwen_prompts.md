# Exact Qwen prompts (qwen2.5-coder:3b Q4_K_M, local Ollama GGUF blob, llama.cpp llama-server, greedy temp 0)

All prompts are raw ChatML: `<|im_start|>role\n...<|im_end|>\n`.

## fewshot (our format)
system:
```
You are a helpful assistant. You can call these tools:
{"name": "wiki_search", "description": "Search Wikipedia article introductions and return the best matches", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "what to look up"}}, "required": ["query"]}}
{"name": "calculator", "description": "Evaluate an arithmetic expression exactly", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "e.g. 12*(3+4)"}}, "required": ["expression"]}}
{"name": "read_file", "description": "Read a text file from the user's workspace", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "file name"}}, "required": ["path"]}}

Format rules: think briefly inside <think>...</think>. To call a tool write exactly <tool_call>{"name": "<tool>", "arguments": {...}}</tool_call> and stop; the tool result will be inserted as <tool_result>...</tool_result>. After the result, write a short final answer in plain text.

Example 1
User: Who was Ada Lovelace?
Assistant: <think>This is a factual question, I should search Wikipedia.</think><tool_call>{"name": "wiki_search", "arguments": {"query": "Ada Lovelace"}}</tool_call><tool_result>1. Ada Lovelace: Augusta Ada King, Countess of Lovelace (10 December 1815 – 27 November 1852) was an English mathematician and writer, chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine.</tool_result>Ada Lovelace was an English mathematician and writer known for her work on Charles Babbage's Analytical Engine.

Example 2
User: What is 48 * 25 + 7?
Assistant: <think>I will use the calculator to be exact.</think><tool_call>{"name": "calculator", "arguments": {"expression": "48 * 25 + 7"}}</tool_call><tool_result>result: 1207</tool_result>The answer is 1207.

Example 3
User: What is the Danube?
Assistant: <think>Let me look this up on Wikipedia.</think><tool_call>{"name": "wiki_search", "arguments": {"query": "Danube"}}</tool_call><tool_result>1. Danube: The Danube is the second-longest river in Europe, after the Volga in Russia. It flows through Central and Southeastern Europe, from the Black Forest south into the Black Sea.</tool_result>The Danube is the second-longest river in Europe; it flows from the Black Forest to the Black Sea.
```
user: `<question>`; the assistant turn is continued in place; stop at `</tool_call>` / `<|im_end|>`; tool output appended as `<tool_result>...</tool_result>`; up to 3 rounds.

## native (Qwen chat template)
system:
```
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "wiki_search", "description": "Search Wikipedia article introductions and return the best matches", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "what to look up"}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression exactly", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "e.g. 12*(3+4)"}}, "required": ["expression"]}}}
{"type": "function", "function": {"name": "read_file", "description": "Read a text file from the user's workspace", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "file name"}}, "required": ["path"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```
user: `<question>`. A call (tagged, or bare/fenced JSON parsed leniently) is re-rendered as `<tool_call>\n{json}\n</tool_call><|im_end|>`, the result is sent as `<|im_start|>user\n<tool_response>\n...\n</tool_response><|im_end|>` (Qwen2.5's tool-role rendering). Max 2 calls; if the cap is reached a final user turn is added:
```
Now answer my original question in one or two sentences, using the tool results above. Do not call any more tools.
```

## closed (no tools, reference)
system: `You are a helpful assistant.`, user: `<question>`.

Stories: plain prompt as raw text; chat prompt as a ChatML user turn with the closed system prompt.
