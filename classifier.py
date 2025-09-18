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
def binary_classifier():
    dataset = json_dataset("data/binary_classifier/dataset.json")
    system = "data/binary_classifier/system.txt"
    return Task(
        dataset=dataset,
        solver=[
            system_message(system),
            use_tools(json_output()),
            generate(tool_calls="single"),
            format_output(),
        ],
        scorer=match(),
    )


@tool
def json_output():
    async def execute(recommend: bool):
        """
        Format the output to the predefined JSON schema

        Args:
            recommend: whether or not you would recommend a cookery class
        """
        return str(recommend)

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
        binary_classifier(),
        model="openai-api/bedrock/openai.gpt-oss-20b-1:0",
        reasoning_effort="low"
    )
