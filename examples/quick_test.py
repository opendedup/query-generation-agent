"""
Quick Test Script

Quickly verify both MCP services are running and working correctly.
"""

import asyncio

import httpx


async def test_services() -> None:
    """Test both MCP services."""
    
    print("=" * 80)
    print("QUICK SERVICE TEST")
    print("=" * 80)
    print()
    
    discovery_url = "http://localhost:8080"
    query_gen_url = "http://localhost:8081"
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Test data discovery agent
        print("🔍 Testing Data Discovery Agent...")
        try:
            response = await client.get(f"{discovery_url}/health")
            if response.status_code == 200:
                print("   ✓ Service is healthy")
                
                # Try listing tools
                response = await client.get(f"{discovery_url}/mcp/tools")
                if response.status_code == 200:
                    tools = response.json()
                    print(f"   ✓ Found {len(tools.get('tools', []))} tools:")
                    for tool in tools.get("tools", []):
                        print(f"      - {tool['name']}")
                else:
                    print(f"   ⚠️  Could not list tools: {response.status_code}")
            else:
                print(f"   ✗ Service unhealthy: {response.status_code}")
        except Exception as e:
            print(f"   ✗ Cannot connect: {e}")
            print()
            print("   To start data-discovery-agent:")
            print("   cd /home/user/git/data-discovery-agent")
            print("   poetry run python -m data_discovery_agent.mcp")
        
        print()
        
        # Test query generation agent
        print("🤖 Testing Query Generation Agent...")
        try:
            response = await client.get(f"{query_gen_url}/health")
            if response.status_code == 200:
                print("   ✓ Service is healthy")
                
                # Try listing tools
                response = await client.get(f"{query_gen_url}/mcp/tools")
                if response.status_code == 200:
                    tools = response.json()
                    print(f"   ✓ Found {len(tools.get('tools', []))} tools:")
                    for tool in tools.get("tools", []):
                        print(f"      - {tool['name']}")
                else:
                    print(f"   ⚠️  Could not list tools: {response.status_code}")
            else:
                print(f"   ✗ Service unhealthy: {response.status_code}")
        except Exception as e:
            print(f"   ✗ Cannot connect: {e}")
            print()
            print("   To start query-generation-agent:")
            print("   cd /home/user/git/query-generation-agent")
            print("   poetry run python -m query_generation_agent.mcp")
        
        print()
    
    print("=" * 80)
    print()
    print("Next steps:")
    print("   - If both services are healthy, run:")
    print("     poetry run python examples/integrated_workflow_example.py")
    print()
    print("   - If any service is down, start it first (see commands above)")
    print()


if __name__ == "__main__":
    asyncio.run(test_services())

