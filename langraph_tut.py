from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_aws import ChatBedrock
import boto3
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.llms.huggingface import HuggingFaceLLM


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class Assistant:
    def __init__(self, runnable: Runnable):
        # Initialize with the runnable that defines the process for interacting with the tools
        self.runnable = runnable

    def __call__(self, state: State):
        while True:
            # Invoke the runnable with the current state (messages and context)
            result = self.runnable.invoke(state)
            
            # If the tool fails to return valid output, re-prompt the user to clarify or retry
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                # Add a message to request a valid response
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                # Break the loop when valid output is obtained
                break

        # Return the final state after processing the runnable
        return {"messages": result}

@tool
def calculate_water_usage(daily_usage_liters: float) -> dict:
    """
    Tool to calculate water consumption metrics and costs based on daily usage.
    
    Args:
        daily_usage_liters (float): User's daily water consumption in liters
    
    Returns:
        dict: Contains:
            - 'daily_cost': Daily water expense
            - 'annual_cost': Annual water expense
            - 'comparison_to_average': Percentage difference from national average
            - 'potential_annual_savings': Possible savings through efficiency measures
    """
    def compute_water_metrics(daily_usage):
        # Default assumptions (USD)
        cost_per_liter = 0.005  # Average water rate
        avg_daily_usage = 142   # UK average from statistics
        
        # Basic calculations
        daily_cost = daily_usage * cost_per_liter
        annual_cost = daily_cost * 365
        
        # Comparison analysis
        usage_difference = daily_usage - avg_daily_usage
        comparison_percent = (usage_difference/avg_daily_usage) * 100
        
        # Savings potential calculation
        if usage_difference > 0:
            savings = usage_difference * cost_per_liter * 365
        else:
            savings = 0

        return {
            "daily_cost": round(daily_cost, 2),
            "annual_cost": round(annual_cost, 2),
            "comparison_to_average": f"{round(comparison_percent, 1)}% {'above' if usage_difference > 0 else 'below'} average",
            "potential_annual_savings": round(savings, 2)
        }

    return compute_water_metrics(daily_usage_liters)

handle_tool_error(state) -> dict:
    """
    Function to handle errors that occur during tool execution.
    
    Args:
        state (dict): The current state of the AI agent, which includes messages and tool call details.
    
    Returns:
        dict: A dictionary containing error messages for each tool that encountered an issue.
    """
    # Retrieve the error from the current state
    error = state.get("error")
    
    # Access the tool calls from the last message in the state's message history
    tool_calls = state["messages"][-1].tool_calls
    
    # Return a list of ToolMessages with error details, linked to each tool call ID
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",  # Format the error message for the user
                tool_call_id=tc["id"],  # Associate the error message with the corresponding tool call ID
            )
            for tc in tool_calls  # Iterate over each tool call to produce individual error messages
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    """
    Function to create a tool node with fallback error handling.
    
    Args:
        tools (list): A list of tools to be included in the node.
    
    Returns:
        dict: A tool node that uses fallback behavior in case of errors.
    """
    # Create a ToolNode with the provided tools and attach a fallback mechanism
    # If an error occurs, it will invoke the handle_tool_error function to manage the error
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)],  # Use a lambda function to wrap the error handler
        exception_key="error"  # Specify that this fallback is for handling errors

def create_hf_llm(model_name: str = "HuggingFaceH4/zephyr-7b-alpha"):
    """Initialize a Hugging Face LLM with specified model"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    
    return HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        model_kwargs={
            "temperature": 0,  # Matches original Bedrock config
            "max_length": 2048,
            "do_sample": True
        }
    )

# Initialize the LLM
llm = create_hf_llm()

primary_assistant_prompt = ChatPromptTemplate.from_messages([(
"system",
'''You are a helpful assistant providing information about global water usage and water management.
    Your tasks:
        - Present up-to-date worldwide water usage statistics, including major sector shares (agriculture, industry, domestic) and per-country or per-capita figures where relevant.
        - Offer practical, evidence-based tips for efficient water management and conservation that are applicable globally.

        If the user requests further details or clarification, politely ask for specifics.

        After gathering the necessary information or upon user request, display the relevant statistics and water management tips.
        ''',
    ),
    ("placeholder", "{messages}"),
])

builder = StateGraph(State)
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
builder.add_edge(START, "assistant")  # Start with the assistant
builder.add_conditional_edges("assistant", tools_condition)  # Move to tools after input
builder.add_edge("tools", "assistant")  # Return to assistant after tool execution
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)