#!/usr/bin/env python3
"""
Basic usage example for AgentLabs framework.

This example demonstrates how to:
1. Create and configure an agent
2. Execute a simple research task
3. Use tools for data collection and analysis
4. Create a research project with multiple tasks
"""

import asyncio
import json
from pathlib import Path

from agentlabs import (
    AgentFactory, AgentRole, LLMConfig,
    ProjectManager, ResearchTask, TaskPriority,
    WorkflowEngine, WorkflowStepType
)


async def basic_agent_example():
    """Example of creating and running a basic agent."""
    print("=== Basic Agent Example ===")
    
    # Create LLM configuration
    llm_config = LLMConfig(
        provider="openai",  # or "anthropic"
        model="gpt-3.5-turbo",  # or "gpt-4", "claude-3-sonnet"
        api_key="your-api-key-here",  # Set your API key
        temperature=0.7,
        max_tokens=1000
    )
    
    # Create a research agent
    agent = AgentFactory.create_agent(
        name="research_agent",
        role=AgentRole.RESEARCHER,
        description="Agent for conducting research on various topics",
        llm_config=llm_config
    )
    
    # Execute the agent
    print("Executing research agent...")
    result = await agent.execute(
        "Research the latest developments in artificial intelligence and machine learning"
    )
    
    if result.success:
        print("✅ Agent execution completed successfully!")
        print(f"Duration: {result.duration:.2f}s")
        print(f"Iterations: {result.iterations}")
        print(f"Tool calls: {len(result.tool_calls)}")
        print("\n📄 Output:")
        print(result.output)
    else:
        print("❌ Agent execution failed!")
        print(f"Error: {result.error}")


async def project_example():
    """Example of creating and running a research project."""
    print("\n=== Research Project Example ===")
    
    # Create LLM configuration
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        api_key="your-api-key-here",
        temperature=0.7,
        max_tokens=1000
    )
    
    # Create project manager
    project_manager = ProjectManager(llm_config)
    
    # Create a research project
    project = project_manager.create_project(
        name="AI Market Research",
        description="Comprehensive research on AI market trends and opportunities"
    )
    
    print(f"Created project: {project.name} (ID: {project.id})")
    
    # Add research tasks
    tasks = [
        {
            "name": "Market Analysis",
            "description": "Analyze current AI market trends",
            "role": AgentRole.RESEARCHER,
            "input_data": "Research current trends in artificial intelligence market, including key players, market size, and growth projections",
            "priority": TaskPriority.HIGH
        },
        {
            "name": "Technology Review",
            "description": "Review latest AI technologies",
            "role": AgentRole.ANALYST,
            "input_data": "Analyze the latest AI technologies including large language models, computer vision, and robotics",
            "priority": TaskPriority.MEDIUM
        },
        {
            "name": "Report Generation",
            "description": "Generate comprehensive research report",
            "role": AgentRole.WRITER,
            "input_data": "Create a comprehensive report summarizing the AI market research findings",
            "priority": TaskPriority.LOW
        }
    ]
    
    for task_data in tasks:
        task = project_manager.create_task(
            project_id=project.id,
            **task_data
        )
        print(f"Added task: {task.name} (ID: {task.id})")
    
    # Execute the project
    print(f"\nExecuting project with {len(project.tasks)} tasks...")
    results = await project_manager.execute_project(project.id, max_concurrent=2)
    
    print("✅ Project execution completed!")
    print(f"Total tasks: {results['total_tasks']}")
    print(f"Completed: {results['completed_tasks']}")
    print(f"Failed: {results['failed_tasks']}")
    
    # Show project summary
    summary = project_manager.get_project_summary(project.id)
    if summary:
        print(f"Total duration: {summary['total_duration']:.2f}s")
        print(f"Average duration: {summary['average_duration']:.2f}s")


async def workflow_example():
    """Example of creating and running a workflow."""
    print("\n=== Workflow Example ===")
    
    # Create LLM configuration
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        api_key="your-api-key-here",
        temperature=0.7,
        max_tokens=1000
    )
    
    # Create workflow engine
    workflow_engine = WorkflowEngine(llm_config)
    
    # Create a workflow
    workflow = workflow_engine.create_workflow(
        name="Data Collection Workflow",
        description="Automated workflow for collecting and analyzing data"
    )
    
    print(f"Created workflow: {workflow.name} (ID: {workflow.id})")
    
    # Add workflow steps
    steps = [
        {
            "name": "Data Collection",
            "step_type": WorkflowStepType.AGENT,
            "config": {
                "agent": {
                    "role": "researcher",
                    "input_template": "Collect data about {topic}",
                    "agent_config": {
                        "max_iterations": 5,
                        "temperature": 0.3
                    }
                }
            }
        },
        {
            "name": "Data Analysis",
            "step_type": WorkflowStepType.AGENT,
            "config": {
                "agent": {
                    "role": "analyst",
                    "input_template": "Analyze the collected data: {data_collection}",
                    "agent_config": {
                        "max_iterations": 3,
                        "temperature": 0.2
                    }
                }
            },
            "dependencies": ["Data Collection"]
        }
    ]
    
    for step_data in steps:
        step = workflow_engine.create_step(
            workflow_id=workflow.id,
            **step_data
        )
        print(f"Added step: {step.name} (ID: {step.id})")
    
    # Execute the workflow
    print(f"\nExecuting workflow with {len(workflow.steps)} steps...")
    result = await workflow_engine.execute_workflow(
        workflow.id,
        initial_context={"topic": "quantum computing developments"}
    )
    
    print("✅ Workflow execution completed!")
    print(f"Total steps: {result['total_steps']}")
    print(f"Executed steps: {result['executed_steps']}")
    print(f"Status: {result['status']}")


async def tools_example():
    """Example of using tools directly."""
    print("\n=== Tools Example ===")
    
    from agentlabs.core.tools import tool_registry
    
    # List available tools
    tools = tool_registry.list_tools()
    print(f"Available tools: {len(tools)}")
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")
    
    # Use web search tool
    web_search_tool = tool_registry.get_tool("web_search")
    if web_search_tool:
        print("\nTesting web search tool...")
        result = await web_search_tool.execute({
            "query": "artificial intelligence trends 2024",
            "num_results": 3,
            "search_engine": "google"
        })
        
        if result.success:
            print("✅ Web search completed!")
            print(f"Found {len(result.data)} results")
            for i, item in enumerate(result.data, 1):
                print(f"  {i}. {item['title']}")
                print(f"     {item['url']}")
        else:
            print(f"❌ Web search failed: {result.error}")
    
    # Use data analyzer tool
    data_analyzer_tool = tool_registry.get_tool("data_analyzer")
    if data_analyzer_tool:
        print("\nTesting data analyzer tool...")
        csv_data = "category,value,count\nAI,85,120\nML,92,95\nDL,78,60"
        result = await data_analyzer_tool.execute({
            "data": csv_data,
            "analysis_type": "summary",
            "columns": "",
            "output_format": "text"
        })
        
        if result.success:
            print("✅ Data analysis completed!")
            print(f"Data shape: {result.data['data_shape']}")
            print(f"Analysis type: {result.data['analysis_type']}")
        else:
            print(f"❌ Data analysis failed: {result.error}")


def save_results_example():
    """Example of saving and loading results."""
    print("\n=== Save/Load Example ===")
    
    # Example data to save
    results = {
        "project_name": "AI Research",
        "execution_time": "2024-01-15T10:30:00",
        "results": {
            "market_analysis": "AI market is growing rapidly...",
            "technology_review": "Latest developments include...",
            "recommendations": "Focus on machine learning..."
        }
    }
    
    # Save to file
    output_file = Path("research_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Load from file
    with open(output_file, 'r') as f:
        loaded_results = json.load(f)
    
    print("Loaded results:")
    print(f"Project: {loaded_results['project_name']}")
    print(f"Execution time: {loaded_results['execution_time']}")
    print(f"Number of results: {len(loaded_results['results'])}")


async def main():
    """Main function to run all examples."""
    print("AgentLabs Framework - Basic Usage Examples")
    print("=" * 50)
    
    # Note: You need to set your API key before running these examples
    print("⚠️  Note: Set your API key in the examples before running!")
    print("   You can set it in the LLMConfig or use environment variables.")
    print()
    
    try:
        # Run examples (comment out if you don't have API keys set)
        # await basic_agent_example()
        # await project_example()
        # await workflow_example()
        # await tools_example()
        
        # This example doesn't require API keys
        save_results_example()
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure you have set up your API keys and dependencies correctly.")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main()) 