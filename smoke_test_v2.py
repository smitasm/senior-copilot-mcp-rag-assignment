import asyncio
from orchestrator.llm import OllamaLLM
from orchestrator.mcp_client import ToolRegistry, connect_inprocess
from orchestrator.agent import investigate
from mcp_servers.alarm_server import build_alarm_server
from mcp_servers.ticketing_server import build_ticketing_server
from mcp_servers.knowledge_server import build_knowledge_server
from mcp_common.http_client import SimulatorClient

async def main():
    alarm_client = SimulatorClient(base_url="http://localhost:8000", token="demo-token")
    ticket_client = SimulatorClient(base_url="http://localhost:8001", token="demo-token")
    async with connect_inprocess(build_alarm_server(alarm_client)) as s1, \
               connect_inprocess(build_ticketing_server(ticket_client)) as s2, \
               connect_inprocess(build_knowledge_server()) as s3:
        reg = ToolRegistry()
        await reg.add(s1); await reg.add(s2); await reg.add(s3)
        llm = OllamaLLM(model="qwen2.5:3b")
        result = await investigate("Prepare an incident for the highest-priority active alarm in EastRefinery.", llm, reg)
        print("stopped_reason:", result.stopped_reason, "| turns:", result.turns_used)
        print("tools called:", [n for n, _ in result.tool_calls_made])
        print()
        print("=== priority_score calls and results (ground truth check) ===")
        for m in result.transcript:
            if m["role"] == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if tc["function"]["name"] == "priority_score":
                        print("  called with:", tc["function"]["arguments"])
            if m["role"] == "tool" and '"score"' in str(m.get("content", "")):
                print("  -> result:", m["content"][:200])
        print()
        if result.draft:
            d = result.draft
            print("=== FULL DRAFT ===")
            print("alarm_id:        ", d.alarm_id)
            print("alarm_name:      ", d.alarm_name)
            print("asset_id:        ", d.asset_id)
            print("asset_name:      ", d.asset_name)
            print("priority_band:   ", d.priority_band)
            print("likely_cause:    ", d.likely_cause)
            print("evidence:        ", d.evidence)
            print("recommended_actions:", d.recommended_actions)
            print("citations:       ", d.citations)
            print("ticket_priority: ", d.ticket_priority)
        else:
            print("NO DRAFT")
        await llm.aclose()

asyncio.run(main())
