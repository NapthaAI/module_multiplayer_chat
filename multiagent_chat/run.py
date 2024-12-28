import logging
import json
from naptha_sdk.agent import Agent
from naptha_sdk.environment import Environment
from naptha_sdk.schemas import OrchestratorRunInput
from multiagent_chat.schemas import InputSchema
from typing import Dict, List

logger = logging.getLogger(__name__)
def reverse_roles(messages: List[Dict[str, str]]):
    """Reverse user and assistant roles in messages."""
    for msg in messages[1:]:  # Skip the system message
        if msg["role"] == "user":
            msg["role"] = "assistant"
        elif msg["role"] == "assistant":
            msg["role"] = "user"
    return messages

class ChatEnvironment(Environment):
    """Chat-specific environment implementation"""
    pass

async def run(module_run: OrchestratorRunInput, *args, **kwargs):
    """Run the chat orchestration between two agents."""
    try:
        environment_deployment = module_run.deployment.environment_deployments[0]
        # Validate environment URL
        if not environment_deployment.node:
            raise ValueError("environment node is required")
        
        run_id = module_run.id

        # Initialize environment and catch potential initialization errors
        try:
            env = await ChatEnvironment.create(environment_deployment)
        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}")
            raise

        # Initialize message history
        messages = [
            {"role": "user", "content": module_run.inputs.prompt},
        ]
        
        # Store initial message
        try:
            await env.upsert_simulation(run_id, messages)
        except Exception as e:
            logger.error(f"Failed to store initial message: {e}")
            # Continue even if storage fails

        # Initialize agents
        agents = [
            Agent(module_run=module_run, agent_index=0, *args, **kwargs),
            Agent(module_run=module_run, agent_index=1, *args, **kwargs)
        ]

        # Run conversation rounds
        for round_num in range(10):
            for agent_num, agent in enumerate(agents):
                try:
                    # Get agent response
                    response = await agent.call_agent_func(
                        tool_name="chat", 
                        tool_input_data=messages
                    )
                    
                    # Parse and process messages
                    if response.results:
                        messages = json.loads(response.results[-1])
                        messages = reverse_roles(messages)
                        
                        # Update database after each agent interaction
                        logger.info(f"Updating database for round {round_num}, agent {agent_num}")
                        try:
                            await env.upsert_simulation(run_id, messages)
                        except Exception as e:
                            logger.error(f"Failed to update simulation: {e}")
                            # Continue even if storage fails
                    else:
                        logger.warning(f"No results from agent {agent_num} in round {round_num}")
                        
                except Exception as e:
                    logger.error(f"Error in round {round_num} with agent {agent_num}: {e}")
                    # Continue with next agent/round if one fails
        
        return messages

    except Exception as e:
        logger.error(f"Fatal error in run: {e}")
        raise

if __name__ == "__main__":
    import asyncio
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import load_agent_deployments, load_environment_deployments, load_orchestrator_deployments
    from naptha_sdk.schemas import OrchestratorRun
    import uuid

    try:
        naptha = Naptha()

        input_params = InputSchema(prompt="lets count up one number at a time. ill start. one.")
            
        # Load configurations
        agent_deployments = load_agent_deployments(
            "multiagent_chat/configs/agent_deployments.json", 
            load_persona_data=False, 
            load_persona_schema=False
        )
        orchestrator_deployments = load_orchestrator_deployments(
            "multiagent_chat/configs/orchestrator_deployments.json"
        )
        environment_deployments = load_environment_deployments(
            "multiagent_chat/configs/environment_deployments.json"
        )

        orchestrator_deployments[0].environment_deployments = environment_deployments
        orchestrator_deployments[0].agent_deployments = agent_deployments

        # Create orchestrator run instance
        module_run = OrchestratorRun(
            inputs=input_params,
            deployment=orchestrator_deployments[0],
            consumer_id=naptha.user.id,
        )
        module_run.id = str(uuid.uuid4())

        # Run the orchestration
        response = asyncio.run(run(module_run))
        print(response)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise