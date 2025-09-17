import copy
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import ClassVar

from litellm import ModelResponse
from pydantic import BaseModel, Field

from core.events.action import (
    Action,
    AgentFinishAction,
    AgentThinkAction,
    BrowseInteractiveAction,
    BrowseURLAction,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    FileWriteAction,
    IPythonRunCellAction,
    MessageAction,
    NullAction,
    PaperReproductionAnalyzerAction,
    RepoAnalyzerAction,
    RepoCreateAction,
    RepoDebugAction,
    RepoEditAction,
    RepoPlanAction,
    RepoRunAction,
    RepoUpdateAction,
    RepoVerifyAction,
    RepoJudgeAction,
    PDFQueryAction,
    PaperRubricAction,
    ExperimentManagerAction,
    TaskGraphBuildAction,
    SnapshotAction,
    RollbackAction,
    CondensationAction,
)
from core.events.event import (EventSource, FileEditSource, FileReadSource,
                               ToolCallMetadata)
from core.events.observation import (AgentThinkObservation,
                                     BrowserOutputObservation,
                                     CmdOutputMetadata, CmdOutputObservation,
                                     ErrorObservation, FileEditObservation,
                                     FileReadObservation, NullObservation,
                                     Observation, TaskGraphBuildObservation, SnapshotObservation, RollbackObservation,
                                     RepoPlanObservation, RepoCreateObservation, RepoAnalyzerObservation, RepoUpdateObservation, RepoVerifyObservation, RepoRunObservation, PaperReproductionAnalyzerObservation, RepoDebugObservation, RepoEditObservation, PDFQueryObservation, IPythonRunCellObservation, RepoJudgeObservation, PaperRubricObservation)
from core.events.observation.image import ImageEntityExtractObservation, ImageEditJudgeObservation
from core.events.observation.repo import GoTEditObservation, QwenAPIObservation, AnyDoorEditObservation
from core.events.observation.experiment import ExperimentManagerObservation
from core.utils.types.exceptions import LLMMalformedActionError
from core.events.action.image import ImageEntityExtractAction, GoTEditAction, QwenAPIAction, ImageEditJudgeAction, AnyDoorEditAction

actions = (
    FileEditAction,
    FileWriteAction,
    CmdRunAction,
    FileReadAction,
    NullAction,
    AgentThinkAction,
    AgentFinishAction,
    MessageAction,
    BrowseURLAction,
    BrowseInteractiveAction,
    TaskGraphBuildAction,
    SnapshotAction,
    RollbackAction,
    RepoPlanAction,
    RepoCreateAction,
    RepoAnalyzerAction,
    RepoUpdateAction,
    RepoVerifyAction,
    RepoRunAction,
    PaperReproductionAnalyzerAction,
    RepoDebugAction,
    RepoEditAction,
    PDFQueryAction,
    IPythonRunCellAction,
    RepoJudgeAction,
    PaperRubricAction,
    CondensationAction,
    ExperimentManagerAction,
    ImageEntityExtractAction,
    GoTEditAction,
    QwenAPIAction,
    ImageEditJudgeAction,
    AnyDoorEditAction,
)

observations = (
    FileEditObservation,
    CmdOutputObservation,
    ErrorObservation,
    FileReadObservation,
    AgentThinkObservation,
    NullObservation,
    BrowserOutputObservation,
    TaskGraphBuildObservation,
    SnapshotObservation,
    RollbackObservation,
    RepoPlanObservation,
    RepoCreateObservation,
    RepoAnalyzerObservation,
    RepoUpdateObservation,
    RepoVerifyObservation,
    RepoRunObservation,
    PaperReproductionAnalyzerObservation,
    RepoDebugObservation,
    RepoEditObservation,
    PDFQueryObservation,
    IPythonRunCellObservation,
    RepoJudgeObservation,
    PaperRubricObservation,
    ImageEntityExtractObservation,
    ImageEditJudgeObservation,
    GoTEditObservation,
    QwenAPIObservation,
    AnyDoorEditObservation,
    ExperimentManagerObservation,
)


# TODO: move `content` into `extras`
TOP_KEYS = [
    "id",
    "timestamp",
    "source",
    "message",
    "cause",
    "action",
    "observation",
    "tool_call_metadata",
    "llm_metrics",
]
UNDERSCORE_KEYS = [
    "id",
    "timestamp",
    "source",
    "cause",
    "tool_call_metadata",
    "llm_metrics",
]
DELETE_FROM_TRAJECTORY_EXTRAS = {
    "screenshot",
    "dom_object",
    "axtree_object",
    "active_page_index",
    "last_browser_action",
    "last_browser_action_error",
    "focused_element_bid",
    "extra_element_properties",
}

ACTION_TYPE_TO_CLASS = {action_class.action: action_class for action_class in actions}  # type: ignore[attr-defined]


OBSERVATION_TYPE_TO_CLASS = {
    observation_class.observation: observation_class  # type: ignore[attr-defined]
    for observation_class in observations
}


def _update_cmd_output_metadata(
    metadata: dict | CmdOutputMetadata | None, **kwargs
) -> dict | CmdOutputMetadata:
    """Update the metadata of a CmdOutputObservation.

    If metadata is None, create a new CmdOutputMetadata instance.
    If metadata is a dict, update the dict.
    If metadata is a CmdOutputMetadata instance, update the instance.
    """
    if metadata is None:
        return CmdOutputMetadata(**kwargs)

    if isinstance(metadata, dict):
        metadata.update(**kwargs)
    elif isinstance(metadata, CmdOutputMetadata):
        for key, value in kwargs.items():
            setattr(metadata, key, value)
    return metadata


def handle_observation_deprecated_extras(extras: dict) -> dict:
    # These are deprecated in https://github.com/All-Hands-AI/OpenHands/pull/4881
    if "exit_code" in extras:
        extras["metadata"] = _update_cmd_output_metadata(
            extras.get("metadata", None), exit_code=extras.pop("exit_code")
        )
    if "command_id" in extras:
        extras["metadata"] = _update_cmd_output_metadata(
            extras.get("metadata", None), pid=extras.pop("command_id")
        )

    # formatted_output_and_error has been deprecated in https://github.com/All-Hands-AI/OpenHands/pull/6671
    if "formatted_output_and_error" in extras:
        extras.pop("formatted_output_and_error")
    return extras


def observation_from_dict(observation: dict) -> Observation:
    observation = observation.copy()
    if "observation" not in observation:
        raise KeyError(f"'observation' key is not found in {observation=}")
    observation_class = OBSERVATION_TYPE_TO_CLASS.get(observation["observation"])
    if observation_class is None:
        raise KeyError(
            f"'{observation['observation']=}' is not defined. Available observations: {OBSERVATION_TYPE_TO_CLASS.keys()}"
        )
    observation.pop("observation")
    observation.pop("message", None)
    content = observation.pop("content", "")
    extras = copy.deepcopy(observation.pop("extras", {}))

    extras = handle_observation_deprecated_extras(extras)

    # convert metadata to CmdOutputMetadata if it is a dict
    if observation_class is CmdOutputObservation:
        if "metadata" in extras and isinstance(extras["metadata"], dict):
            extras["metadata"] = CmdOutputMetadata(**extras["metadata"])
        elif "metadata" in extras and isinstance(extras["metadata"], CmdOutputMetadata):
            pass
        else:
            extras["metadata"] = CmdOutputMetadata()

    return observation_class(content=content, **extras)


def handle_action_deprecated_args(args: dict) -> dict:
    # keep_prompt has been deprecated in https://github.com/All-Hands-AI/OpenHands/pull/4881
    if "keep_prompt" in args:
        args.pop("keep_prompt")

    # Handle translated_ipython_code deprecation
    if "translated_ipython_code" in args:
        code = args.pop("translated_ipython_code")

        # Check if it's a file_editor call
        file_editor_pattern = r"print\(file_editor\(\*\*(.*?)\)\)"
        if code is not None and (match := re.match(file_editor_pattern, code)):
            try:
                # Extract and evaluate the dictionary string
                import ast

                file_args = ast.literal_eval(match.group(1))

                # Update args with the extracted file editor arguments
                args.update(file_args)
            except (ValueError, SyntaxError):
                # If parsing fails, just remove the translated_ipython_code
                pass

        if args.get("command") == "view":
            args.pop(
                "command"
            )  # "view" will be translated to FileReadAction which doesn't have a command argument

    return args


def action_from_dict(action: dict) -> Action:
    if not isinstance(action, dict):
        raise LLMMalformedActionError("action must be a dictionary")
    action = action.copy()
    if "action" not in action:
        raise LLMMalformedActionError(f"'action' key is not found in {action=}")
    if not isinstance(action["action"], str):
        raise LLMMalformedActionError(
            f"'{action['action']=}' is not defined. Available actions: {ACTION_TYPE_TO_CLASS.keys()}"
        )
    action_class = ACTION_TYPE_TO_CLASS.get(action["action"])
    if action_class is None:
        raise LLMMalformedActionError(
            f"'{action['action']=}' is not defined. Available actions: {ACTION_TYPE_TO_CLASS.keys()}"
        )
    args = action.get("args", {})
    # Remove timestamp from args if present
    timestamp = args.pop("timestamp", None)

    # compatibility for older event streams
    # is_confirmed has been renamed to confirmation_state
    is_confirmed = args.pop("is_confirmed", None)
    if is_confirmed is not None:
        args["confirmation_state"] = is_confirmed

    # images_urls has been renamed to image_urls
    if "images_urls" in args:
        args["image_urls"] = args.pop("images_urls")

    # handle deprecated args
    args = handle_action_deprecated_args(args)

    try:
        decoded_action = action_class(**args)
        if "timeout" in action:
            blocking = args.get("blocking", False)
            decoded_action.set_hard_timeout(action["timeout"], blocking=blocking)

        # Set timestamp if it was provided
        if timestamp:
            decoded_action._timestamp = timestamp

    except TypeError as e:
        raise LLMMalformedActionError(
            f"action={action} has the wrong arguments: {str(e)}"
        )
    return decoded_action


def event_from_dict(data) -> "Event":
    evt: Event
    if "action" in data:
        evt = action_from_dict(data)
    elif "observation" in data:
        evt = observation_from_dict(data)
    else:
        raise ValueError("Unknown event type: " + data)
    for key in UNDERSCORE_KEYS:
        if key in data:
            value = data[key]
            if key == "timestamp" and isinstance(value, datetime):
                value = value.isoformat()
            if key == "source":
                value = EventSource(value)
            if key == "tool_call_metadata":
                value = ToolCallMetadata(**value)

            setattr(evt, "_" + key, value)
    return evt


def _convert_pydantic_to_dict(obj: BaseModel | dict) -> dict:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return obj


def event_to_dict(event: "Event") -> dict:
    props = asdict(event)
    d = {}
    for key in TOP_KEYS:
        if hasattr(event, key) and getattr(event, key) is not None:
            d[key] = getattr(event, key)
        elif hasattr(event, f"_{key}") and getattr(event, f"_{key}") is not None:
            d[key] = getattr(event, f"_{key}")
        if key == "id" and d.get("id") == -1:
            d.pop("id", None)
        if key == "timestamp" and "timestamp" in d:
            if isinstance(d["timestamp"], datetime):
                d["timestamp"] = d["timestamp"].isoformat()
        if key == "source" and "source" in d:
            d["source"] = d["source"].value
        if key == "recall_type" and "recall_type" in d:
            d["recall_type"] = d["recall_type"].value
        if key == "tool_call_metadata" and "tool_call_metadata" in d:
            d["tool_call_metadata"] = d["tool_call_metadata"].model_dump()
        if key == "llm_metrics" and "llm_metrics" in d:
            d["llm_metrics"] = d["llm_metrics"].get()
        props.pop(key, None)
    if "security_risk" in props and props["security_risk"] is None:
        props.pop("security_risk")
    if "action" in d:
        d["args"] = props
        if event.timeout is not None:
            d["timeout"] = event.timeout
    elif "observation" in d:
        d["content"] = props.pop("content", "")

        # props is a dict whose values can include a complex object like an instance of a BaseModel subclass
        # such as CmdOutputMetadata
        # we serialize it along with the rest
        # we also handle the Enum conversion for RecallObservation
        d["extras"] = {
            k: (v.value if isinstance(v, Enum) else _convert_pydantic_to_dict(v))
            for k, v in props.items()
        }
        # Include success field for CmdOutputObservation
        if hasattr(event, "success"):
            d["success"] = event.success
    else:
        raise ValueError("Event must be either action or observation")
    return d


def event_to_trajectory(event: "Event") -> dict:
    d = event_to_dict(event)
    if "extras" in d:
        remove_fields(d["extras"], DELETE_FROM_TRAJECTORY_EXTRAS)
    return d


def remove_fields(obj, fields: set[str]):
    """Remove fields from an object.

    Parameters:
    - obj: The dictionary, or list of dictionaries to remove fields from
    - fields (set[str]): A set of field names to remove from the object
    """
    if isinstance(obj, dict):
        for field in fields:
            if field in obj:
                del obj[field]
        for _, value in obj.items():
            remove_fields(value, fields)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            remove_fields(item, fields)
    elif hasattr(obj, "__dataclass_fields__"):
        raise ValueError(
            "Object must not contain dataclass, consider converting to dict first"
        )
