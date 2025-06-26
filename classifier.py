import os

from inspect_ai import Task, eval, task
from inspect_ai.dataset import json_dataset
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import match
from inspect_ai.solver import (Generate, TaskState, generate, solver,
                               system_message, use_tools)
from inspect_ai.tool import ToolFunction, tool


@task
def binary_classifier():
    dataset = json_dataset("data/binary_classifier/dataset.json")
    system = "data/binary_classifier/system.txt"
    return Task(
        dataset=dataset,
        solver=[
            system_message(system),
            use_tools(
                json_output(),
                tool_choice=ToolFunction(name='json_output')
            ),
            generate(tool_calls="single"),
            format_output()
        ],
        scorer=match(),
    )


@tool
def json_output():
    async def execute(reasons: str, recommend: bool):
        """
        Format the output to the predefined JSON schema

        Args:
            reasons: the thinking pad for you to write down relevant thinking logic
            recommend: whether or not you would recommend a cookery class
        """
        return str(recommend)
    return execute


@solver
def format_output():
    async def solve(state: TaskState, generate: Generate):
        model_output = ModelOutput.from_content(
            model=state.model.name,
            content=state.messages[-1].content
        )
        state.output = model_output
        return state
    return solve
