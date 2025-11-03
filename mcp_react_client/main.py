"""
MCP ReAct Client - A ReAct-based client for MCP servers using LangChain

This client connects to an MCP Python interpreter server and allows users to interact
with Python environments through natural language commands.
"""

import asyncio
import os
import logging
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
import httpx

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from langchain_core.tools import tool
from .image_generator import generate_single_image, generate_comic
from .memory import MemoryManager


@tool
def create_image(prompt: str, filename: str = None, size: str = "1024x1024", quality: str = "standard") -> str:
    """
    Create an image using OpenAI DALL-E based on a text prompt.
    
    Args:
        prompt: Description of the image to generate
        filename: Optional filename for the saved image (defaults to auto-generated)
        size: Image size (1024x1024, 1792x1024, or 1024x1792)
        quality: Image quality (standard or hd)
    
    Returns:
        Path to the saved image file
    """
    try:
        return generate_single_image(prompt, filename, size, quality)
    except Exception as e:
        return f"Error generating image: {str(e)}"


@tool
def create_comic(topic: str) -> str:
    """
    Create a 4-panel comic based on a topic using OpenAI DALL-E.
    
    Args:
        topic: Topic or theme for the comic story
    
    Returns:
        Path to the generated comic image file
    """
    try:
        return generate_comic(topic)
    except Exception as e:
        return f"Error generating comic: {str(e)}"


# 글로벌 메모리 매니저 인스턴스
memory_manager = MemoryManager()


@tool
def search_memory(keyword: str) -> str:
    """
    Search through conversation history for a specific keyword.
    
    Args:
        keyword: The keyword to search for in previous conversations
    
    Returns:
        Search results from conversation history
    """
    try:
        current_memory = memory_manager.get_current_memory()
        if not current_memory:
            return "No active memory session. Start a conversation first."
        
        results = current_memory.search_memory(keyword)
        if not results:
            return f"No conversations found containing '{keyword}'"
        
        search_results = []
        for i, entry in enumerate(results[-5:], 1):  # 최근 5개만
            search_results.append(f"{i}. [{entry.timestamp}]")
            search_results.append(f"   User: {entry.user_input}")
            search_results.append(f"   Assistant: {entry.agent_response[:100]}...")
            if entry.tools_used:
                search_results.append(f"   Tools: {', '.join(entry.tools_used)}")
            search_results.append("")
        
        return "\n".join(search_results)
    except Exception as e:
        return f"Error searching memory: {str(e)}"


@tool
def show_memory_summary() -> str:
    """
    Show a summary of the current conversation session.
    
    Returns:
        Summary of conversation history including statistics
    """
    try:
        current_memory = memory_manager.get_current_memory()
        if not current_memory:
            return "No active memory session."
        
        summary = current_memory.get_summary()
        
        result = []
        result.append(f"📊 Memory Session Summary")
        result.append(f"Session ID: {summary['session_id']}")
        result.append(f"Total conversations: {summary['total_entries']}")
        
        if summary['first_interaction']:
            result.append(f"First interaction: {summary['first_interaction']}")
        if summary['last_interaction']:
            result.append(f"Last interaction: {summary['last_interaction']}")
        
        if summary['most_used_tools']:
            result.append("\nMost used tools:")
            for tool_name, count in summary['most_used_tools']:
                result.append(f"  • {tool_name}: {count} times")
        
        return "\n".join(result)
    except Exception as e:
        return f"Error getting memory summary: {str(e)}"


@tool
def get_conversation_context(count: int = 3) -> str:
    """
    Get recent conversation context.
    
    Args:
        count: Number of recent conversations to include (default: 3)
    
    Returns:
        Recent conversation history
    """
    try:
        current_memory = memory_manager.get_current_memory()
        if not current_memory:
            return "No active memory session."
        
        context = current_memory.get_conversation_context(count)
        return f"📝 Recent Conversation Context ({count} entries):\n\n{context}"
    except Exception as e:
        return f"Error getting conversation context: {str(e)}"


def load_mcp_config(config_path: str = None) -> Dict[str, Any]:
    """Load MCP configuration from mcp.json file"""
    if config_path is None:
        # Try common locations for mcp.json (project root first)
        possible_paths = [
            "mcp.json",  # Current project root
            "../mcp.json",  # Parent directory
            os.path.expanduser("~/.cursor/mcp.json")  # Cursor default location
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path is None:
            raise FileNotFoundError("Could not find mcp.json configuration file")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config.get('mcpServers', {})


async def create_mcp_sessions(mcp_config: Dict[str, Any]) -> List[tuple]:
    """Create MCP sessions for all configured servers"""
    sessions = []
    
    for server_name, server_config in mcp_config.items():
        try:
            print(f"🔌 Connecting to MCP server: {server_name}")
            
            if 'command' in server_config:
                # Stdio-based server
                server_params = StdioServerParameters(
                    command=server_config['command'],
                    args=server_config.get('args', []),
                    env=server_config.get('env', {})
                )
                
                # Store the server params and name for later use
                sessions.append((server_name, 'stdio', server_params))
                print(f"✅ Prepared connection to {server_name} (stdio)")
                
            elif 'url' in server_config:
                # URL-based server (SSE, HTTP, etc.)
                url = server_config['url']
                server_type = server_config.get('type', 'http')  # Default to HTTP
                headers = server_config.get('headers', {})
                
                # Store URL connection info
                sessions.append((server_name, server_type, {'url': url, 'headers': headers}))
                print(f"✅ Prepared connection to {server_name} ({server_type})")
                
        except Exception as e:
            print(f"❌ Failed to prepare connection to {server_name}: {e}")
            continue
    
    return sessions


async def load_all_tools_from_sessions(session_params: List[tuple]):
    """Load tools from all MCP sessions plus image generation and memory tools"""
    all_mcp_tools = []
    
    # Load MCP tools using the new connection-based approach
    for server_name, connection_type, connection_params in session_params:
        try:
            print(f"🔧 Loading tools from {server_name} ({connection_type})...")
            
            if connection_type == 'http':
                # HTTP-based connection using new StreamableHttpConnection
                url = connection_params['url']
                headers = connection_params.get('headers', {})
                
                connection = StreamableHttpConnection(url=url, headers=headers)
                # Add required transport key
                connection['transport'] = 'streamable_http'
                
                mcp_tools = await load_mcp_tools(
                    session=None, 
                    connection=connection,
                    server_name=server_name
                )
                all_mcp_tools.extend(mcp_tools)
                print(f"✅ Loaded {len(mcp_tools)} tools from {server_name}")
                
            elif connection_type == 'stdio':
                # Stdio-based connection (fallback to old method)
                async with stdio_client(connection_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        mcp_tools = await load_mcp_tools(session, server_name=server_name)
                        all_mcp_tools.extend(mcp_tools)
                        print(f"✅ Loaded {len(mcp_tools)} tools from {server_name}")
                        
            elif connection_type == 'sse':
                # SSE-based connection (fallback to old method)
                url = connection_params['url']
                headers = connection_params.get('headers', {})
                
                async with sse_client(url, headers=headers) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        mcp_tools = await load_mcp_tools(session, server_name=server_name)
                        all_mcp_tools.extend(mcp_tools)
                        print(f"✅ Loaded {len(mcp_tools)} tools from {server_name}")
                        
            else:
                print(f"⚠️  Warning: Unsupported connection type '{connection_type}' for {server_name}")
                
        except Exception as e:
            print(f"⚠️  Warning: Failed to load tools from {server_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Add image generation tools
    image_tools = [create_image, create_comic]
    
    # Add memory tools
    memory_tools = [search_memory, show_memory_summary, get_conversation_context]
    
    # Combine all tools
    all_tools = all_mcp_tools + image_tools + memory_tools
    
    return all_tools


def setup_logging(verbose=False):
    """Set up logging configuration for debugging ReAct agent."""
    if verbose:
        # Set up minimal logging - only show errors from other components
        logging.basicConfig(
            level=logging.ERROR,
            format='%(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()],
            force=True
        )
        
        # Suppress noisy loggers
        logging.getLogger("openai").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.ERROR)
        logging.getLogger("langchain").setLevel(logging.ERROR)
        logging.getLogger("langchain_core").setLevel(logging.ERROR)
        logging.getLogger("langgraph").setLevel(logging.ERROR)
        logging.getLogger("mcp").setLevel(logging.ERROR)
        logging.getLogger("langchain_mcp_adapters").setLevel(logging.ERROR)
    else:
        # Normal mode - suppress all debug logs
        logging.basicConfig(level=logging.WARNING, force=True)


def show_help():
    """Show help with example commands."""
    print("\n" + "=" * 60)
    print("📖 HELP - Example Commands")
    print("=" * 60)
    
    examples = [
        ("Python Environment Management", [
            "Show me all available Python environments",
            "List installed packages in default environment",
            "Install numpy package in system environment"
        ]),
        ("Python Code Execution", [
            "Run this Python code: print('Hello, World!')",
            "Execute this code: import math; print(math.pi)",
            "Run a calculation: 2 + 2 * 3"
        ]),
        ("File Operations", [
            "Create a Python file called 'test.py' with a hello function",
            "Read the contents of hello.py file",
            "List all files in the current directory"
        ]),
        ("Advanced Examples", [
            "Create a script that calculates fibonacci numbers",
            "Write a Python function to sort a list",
            "Run a data analysis script with pandas"
        ]),
        ("Complex Engineering Tasks", [
            "Optimize a hot-rolling steel process using Bayesian optimization",
            "Create a machine learning model for predictive maintenance",
            "Solve a multi-objective optimization problem with constraints"
        ]),
        ("Image Generation", [
            "Create an image of a sunset over mountains",
            "Generate a 4-panel comic about artificial intelligence",
            "Create a beautiful landscape with trees and rivers"
        ]),
        ("Memory & Context", [
            "What did we discuss about Python earlier?",
            "Search our conversation for 'optimization'",
            "Show me a summary of our session",
            "What tools have I used most often?"
        ])
    ]
    
    for category, commands in examples:
        print(f"\n🔸 {category}:")
        for cmd in commands:
            print(f"   • {cmd}")
    
    print("\n" + "=" * 60)


def show_tools(tools):
    """Show detailed information about available tools."""
    print("\n" + "=" * 60)
    print("🛠️  AVAILABLE TOOLS")
    print("=" * 60)
    
    for i, tool in enumerate(tools, 1):
        print(f"\n{i}. 🔧 {tool.name}")
        print(f"   📝 {tool.description}")
    
    print("\n" + "=" * 60)


async def interactive_mode(verbose=False):
    """Run the MCP ReAct client in interactive mode."""
    
    print("🚀 Starting MCP ReAct Client - Interactive Mode")
    if verbose:
        print("🔍 Verbose mode enabled - detailed logging active")
    print("=" * 50)
    print("💡 Type your commands and press Enter. Type 'quit', 'exit', or 'q' to stop.")
    print("🧠 Memory system active - your conversation history will be remembered!")
    print("=" * 50)
    
    # Setup logging
    setup_logging(verbose)
    
    # Start memory session
    memory = memory_manager.start_session()
    print(f"📝 Memory session started: {memory.session_id}")
    
    # Load MCP configuration
    try:
        mcp_config = load_mcp_config()
        print(f"📋 Loaded configuration for {len(mcp_config)} MCP servers")
    except Exception as e:
        print(f"❌ Failed to load MCP configuration: {e}")
        return
    
    try:
        # Prepare MCP server connections
        session_params = await create_mcp_sessions(mcp_config)
        
        if not session_params:
            print("❌ No MCP servers configured successfully")
            return
        
        print(f"✅ Prepared {len(session_params)} MCP server(s)")
        
        # Get all tools from all sessions
        tools = await load_all_tools_from_sessions(session_params)
        
        print(f"✅ Loaded {len(tools)} tools total")
        print("\n📚 Available tools:")
        for tool in tools:
            print(f"  • {tool.name}")
        
        print("\n💡 Special commands:")
        print("  • 'help' - Show example commands")
        print("  • 'tools' - List all available tools")
        print("  • 'verbose' - Toggle verbose logging on/off")
        print("  • 'memory' - Show memory summary")
        print("  • 'context' - Show recent conversation context")
        print("  • 'search <keyword>' - Search conversation history")
        print("  • 'clear_memory' - Clear conversation history")
        print("  • 'quit', 'exit', 'q' - Exit the program")
        
        print("\n" + "=" * 50)
        print("🎯 Ready for your commands!")
        
        # Interactive loop
        verbose_mode = verbose
        while True:
            try:
                # Get user input
                user_input = input("\n🤖 Enter your command: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'q', '']:
                    print("👋 Goodbye!")
                    break
                
                # Handle special commands
                if user_input.lower() == 'help':
                    show_help()
                    continue
                elif user_input.lower() == 'tools':
                    show_tools(tools)
                    continue
                elif user_input.lower() == 'verbose':
                    verbose_mode = not verbose_mode
                    setup_logging(verbose_mode)
                    status = "enabled" if verbose_mode else "disabled"
                    print(f"🔍 Verbose logging {status}")
                    continue
                elif user_input.lower() == 'memory':
                    summary = memory.get_summary()
                    print(f"\n📊 Memory Session Summary:")
                    print(f"Session ID: {summary['session_id']}")
                    print(f"Total conversations: {summary['total_entries']}")
                    if summary['most_used_tools']:
                        print("Most used tools:")
                        for tool_name, count in summary['most_used_tools'][:3]:
                            print(f"  • {tool_name}: {count} times")
                    continue
                elif user_input.lower() == 'context':
                    context = memory.get_conversation_context(3)
                    print(f"\n📝 Recent Conversation Context:")
                    print(context)
                    continue
                elif user_input.lower().startswith('search '):
                    keyword = user_input[7:].strip()
                    if keyword:
                        results = memory.search_memory(keyword)
                        if results:
                            print(f"\n🔍 Search results for '{keyword}':")
                            for i, entry in enumerate(results[-3:], 1):
                                print(f"{i}. [{entry.timestamp}]")
                                print(f"   User: {entry.user_input}")
                                print(f"   Assistant: {entry.agent_response[:100]}...")
                                print()
                        else:
                            print(f"No results found for '{keyword}'")
                    else:
                        print("Please provide a keyword to search for")
                    continue
                elif user_input.lower() == 'clear_memory':
                    memory.clear_memory()
                    print("🗑️ Memory cleared!")
                    continue
                
                print("\n" + "-" * 60)
                print(f"Processing: {user_input}")
                print("-" * 60)
                
                # Run the agent with user input
                response = await run_react_agent(tools, user_input, verbose=verbose_mode)
                
                # Extract tools used from response
                tools_used = []
                if response and "messages" in response:
                    for msg in response["messages"]:
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                tool_name = tool_call.get('function', {}).get('name', tool_call.get('name', 'unknown'))
                                if tool_name not in tools_used:
                                    tools_used.append(tool_name)
                
                # Get agent response for memory
                agent_response = ""
                if response and "messages" in response:
                    final_message = response["messages"][-1]
                    if hasattr(final_message, 'content'):
                        agent_response = final_message.content
                    else:
                        agent_response = str(final_message)
                
                # Save to memory
                memory.add_entry(user_input, agent_response, tools_used)
                
                print("\n" + "=" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 EOF detected. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error processing command: {e}")
                print("Please try again or type 'quit' to exit.")
                        
    except Exception as e:
        print(f"❌ Failed to set up MCP client: {e}")
        print("Make sure the MCP servers are available and properly configured.")
        return
    finally:
        # End memory session
        memory_manager.end_session()
        print(f"💾 Memory session saved and closed.")


async def demo_mode(verbose=False):
    """Run the MCP ReAct client in demo mode with predefined queries."""
    
    print("🚀 Starting MCP ReAct Client - Demo Mode")
    if verbose:
        print("🔍 Verbose mode enabled - detailed logging active")
    print("=" * 50)
    
    # Setup logging
    setup_logging(verbose)
    
    # Load MCP configuration
    try:
        mcp_config = load_mcp_config()
        print(f"📋 Loaded configuration for {len(mcp_config)} MCP servers")
    except Exception as e:
        print(f"❌ Failed to load MCP configuration: {e}")
        return
    
    # Test queries
    test_queries = [
        # "Show me all available Python environments on my system",
        # "Run this Python code: print('Hello, world!')",
        # "Create a new Python file called 'hello.py' with a function that says hello",
        "claude-skills 에 있는 스킬들을 알려줘",
        "claude-skills 에서 pptx 를 만들 수 있는 skill을 찾아줘",
        "computer-use mcp 를 이용하여 helloworld 를 찍는 js 파일을 만들어서 실행해봐",
        "claude-skills를 이용하여 간단히 python 역사에 대한 pptx를 만들어"
    ]
    
    try:
        # Prepare MCP server connections
        session_params = await create_mcp_sessions(mcp_config)
        
        if not session_params:
            print("❌ No MCP servers configured successfully")
            return
        
        print(f"✅ Prepared {len(session_params)} MCP server(s)")
        
        # Get all tools from all sessions
        tools = await load_all_tools_from_sessions(session_params)
        
        print(f"Loaded {len(tools)} tools total:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
        # Run each test query within the same session
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*20} Demo {i} {'='*20}")
            await run_react_agent(tools, query, verbose=verbose, include_memory_context=False)
            print("\n" + "="*50)
            
            # Add a small delay between queries
            await asyncio.sleep(2)
                
    except Exception as e:
        print(f"❌ Failed to set up MCP client: {e}")
        print("Make sure the MCP servers are available and properly configured.")
        return
    finally:
        pass  # No persistent sessions to close
    
    print("\n✅ All demos completed!")


async def main():
    """Main function to run the MCP ReAct client."""
    import sys
    
    # Parse command line arguments
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    # Remove verbose flags from argv for mode parsing
    clean_argv = [arg for arg in sys.argv if arg not in ["--verbose", "-v"]]
    
    # Check command line arguments
    if len(clean_argv) > 1:
        mode = clean_argv[1].lower()
        if mode == "demo":
            await demo_mode(verbose=verbose)
            return
        elif mode == "interactive":
            await interactive_mode(verbose=verbose)
            return
        else:
            print(f"❌ Unknown mode: {mode}")
            print("Usage: python -m mcp_react_client.main [interactive|demo] [--verbose|-v]")
            return
    
    # Default to interactive mode
    await interactive_mode(verbose=verbose)


async def run_react_agent(tools: List, query: str, verbose=False, include_memory_context=True):
    """Run the ReAct agent with the given query."""
    
    # Load environment variables
    load_dotenv()
    
    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it using: export OPENAI_API_KEY='your-api-key'") 
    
    # Prepare the query with memory context if requested
    enhanced_query = query
    if include_memory_context:
        current_memory = memory_manager.get_current_memory()
        if current_memory and len(current_memory.memory) > 0:
            # Check if the query is asking about previous conversation
            memory_keywords = ['뭐였지', '뭐라고', '어디', '누구', '언제', '무엇', '말했', '했다고', 'said', 'told', 'mentioned', 'remember', 'recall', 'previous', 'before', 'earlier', 'ago']
            is_memory_query = any(keyword in query.lower() for keyword in memory_keywords)
            
            if is_memory_query:
                # For memory-related queries, provide context and explicit instructions
                context = current_memory.get_conversation_context(5)
                enhanced_query = f"""This user is asking about something from our previous conversation. Here's our recent conversation history:

{context}

User's current question: {query}

Please analyze the conversation history above to answer the user's question. Look for the specific information they're asking about in the provided context. If you can find the answer in the conversation history, provide it directly. If not, let them know what information is available from our chat."""
            else:
                # For non-memory queries, just include recent context if relevant
                context = current_memory.get_conversation_context(2)
                if context and context != "No previous conversation history.":
                    enhanced_query = f"""Previous context (for reference only):
{context}

Current user query: {query}"""
    
    # Set up the OpenAI model
    model = ChatOpenAI(
        model="gpt-4o",  # Use gpt-4o for larger context window
        temperature=0,
        api_key=openai_api_key
    )
    
    # Create the ReAct agent
    agent = create_react_agent(model, tools)
    
    print(f"\n🤖 Processing query: {query}")
    print("-" * 50)
    
    if verbose:
        print("\n🔍 REACT AGENT TOOL TRACE:")
        print("=" * 50)
        print(f"🎯 Query: {query}")
        print(f"🛠️  Available tools: {[tool.name for tool in tools]}")
        print("=" * 50)
    
    try:
        # Run the agent
        response = await agent.ainvoke({"messages": [("user", enhanced_query)]})
        
        if verbose:
            print("\n📋 TOOL CALL SUMMARY:")
            print("-" * 30)
            tool_call_count = 0
            
            if response and "messages" in response:
                # Show only tool-related interactions
                for i, msg in enumerate(response["messages"]):
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        tool_call_count += len(msg.tool_calls)
                        print(f"\n🤖 Agent Decision #{i//2 + 1}:")
                        for j, tool_call in enumerate(msg.tool_calls, 1):
                            tool_name = tool_call.get('function', {}).get('name', tool_call.get('name', 'unknown'))
                            print(f"  {j}. 🔧 Tool: {tool_name}")
                            
                            # Show simplified args
                            args = tool_call.get('function', {}).get('arguments')
                            if args and isinstance(args, str):
                                try:
                                    import json
                                    parsed_args = json.loads(args)
                                    # Show only key parameters
                                    key_params = {}
                                    for key, value in parsed_args.items():
                                        if key in ['file_path', 'code', 'package_name', 'environment']:
                                            if isinstance(value, str) and len(value) > 50:
                                                key_params[key] = value[:50] + "..."
                                            else:
                                                key_params[key] = value
                                    if key_params:
                                        print(f"     📝 Key params: {key_params}")
                                except:
                                    pass
                    
                    # Show tool results
                    elif hasattr(msg, 'content') and hasattr(msg, 'role') and getattr(msg, 'role', None) == 'tool':
                        result_preview = str(msg.content)[:100] + "..." if len(str(msg.content)) > 100 else str(msg.content)
                        print(f"  ✅ Tool result: {result_preview}")
                
                print(f"\n📊 Total tool calls executed: {tool_call_count}")
            print("-" * 30)
        
        print("\n📋 Agent Response:")
        print("-" * 20)
        
        # Print the final message from the agent
        if response and "messages" in response:
            final_message = response["messages"][-1]
            if hasattr(final_message, 'content'):
                print(final_message.content)
            else:
                print(final_message)
        else:
            print(response)
            
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        if verbose:
            import traceback
            print("\n🔍 FULL ERROR TRACEBACK:")
            traceback.print_exc()
        return None
    
    return response


def cli_main():
    """Command line interface main function."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
