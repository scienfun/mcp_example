import asyncio, json, re, sys
import ollama
from fastmcp import Client

MCP_URL = "http://127.0.0.1:8001/mcp"
MODEL = "glm-4.7:cloud"

SYSTEM = """너는 로컬 워크스페이스 파일을 MCP tool로 읽어 분석하는 에이전트다.
너는 직접 파일을 읽을 수 없고, 반드시 아래 MCP tool 호출로만 정보를 얻어야 한다.

사용 가능한 MCP tool:
- list_dir(path=".")
- glob_files(pattern)
- read_text_file(path, max_chars=20000)

규칙:
- 응답은 반드시 JSON 하나만 출력한다. 다른 텍스트 금지.
- JSON 스키마:
  {"action":"call_tool","tool":"<tool_name>","args":{...}}
  또는
  {"action":"final","summary":"...","next_steps":[...]}
- 파일을 분석하려면 먼저 list_dir/glob_files로 경로를 확인한 뒤 read_text_file로 내용을 읽어라.
- 워크스페이스 밖 경로는 금지다.
"""

def extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"No JSON found. Model said:\n{text}")
    return json.loads(m.group(0))

async def call_tool(tool: str, args: dict):
    async with Client(MCP_URL) as c:
        return await c.call_tool(tool, args)

async def main():
    user_req = " ".join(sys.argv[1:]).strip() or input("요청> ").strip()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_req},
    ]

    for step in range(12):  # 무한루프 방지
        resp = ollama.chat(model=MODEL, messages=messages)
        content = resp["message"]["content"]
        plan = extract_json(content)

        if plan.get("action") == "final":
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return

        if plan.get("action") != "call_tool":
            raise ValueError(f"Unexpected action: {plan}")

        tool = plan["tool"]
        args = plan.get("args", {})
        result = await call_tool(tool, args)

        # tool 결과를 LLM에 다시 주입
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "tool", "content": json.dumps({"tool": tool, "result": result}, ensure_ascii=False)})

    raise RuntimeError("Too many steps (did not reach final).")

if __name__ == "__main__":
    asyncio.run(main())