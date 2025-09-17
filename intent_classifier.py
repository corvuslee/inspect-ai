from inspect_ai import Task, eval, task
from inspect_ai.dataset import json_dataset
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import match
from inspect_ai.solver import (
    Generate,
    TaskState,
    generate,
    solver,
    system_message,
    use_tools,
)
from inspect_ai.tool import tool


@task
def intent_classifier():
    dataset = json_dataset("data/intent_classifier/dataset.json")
    system = "data/intent_classifier/system.txt"
    return Task(
        dataset=dataset,
        solver=[
            system_message(system),
            use_tools(classify_intent()),
            generate(tool_calls="single"),
            format_output(),
        ],
        scorer=match(),
    )


@tool
def classify_intent():
    async def execute(category: str):
        """
        Classify the user message into an intent category

        Args:
            category: the intent category:
                - informational: requests for information, facts, or explanations
                - transactional: requests to perform actions like purchases or bookings
                - account_management: requests related to user accounts, profiles, or settings
                - technical: requests for technical support or troubleshooting
                - customer_service: requests for customer service like feedback or complain
                - general: general inquiries that don't fit other categories
        """
        return str(category)

    return execute


@solver
def format_output():
    async def solve(state: TaskState, generate: Generate):
        model_output = ModelOutput.from_content(
            model=state.model.name, content=state.messages[-1].content
        )
        state.output = model_output
        return state

    return solve


if __name__ == "__main__":
    eval(
        intent_classifier(),
        model="openai-api/bedrock/openai.gpt-oss-20b-1:0",
        reasoning_effort="low"
    )