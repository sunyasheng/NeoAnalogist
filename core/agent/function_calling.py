import json

from litellm import ChatCompletionToolParam, ModelResponse

from core.agent.tools.bash import create_cmd_run_tool
from core.agent.tools.browser import BrowserTool
from core.agent.tools.finish import FinishTool
from core.agent.tools.str_replace_editor import create_str_replace_editor_tool
from core.agent.tools.think import ThinkTool
from core.agent.tools.web_read import WebReadTool
from core.agent.tools.repo_plan import RepoPlanTool
from core.agent.tools.repo_create import RepoCreateTool
from core.agent.tools.repo_analyzer import RepoAnalyzerTool
from core.agent.tools.repo_update import RepoUpdateTool
from core.agent.tools.repo_verify import RepoVerifyTool
from core.agent.tools.repo_run import RepoRunTool
from core.agent.tools.paper_reproduction_analyzer import PaperReproductionAnalyzerTool
from core.agent.tools.task_graph import create_task_graph_tool
from core.agent.tools.repo_debug import RepoDebugTool
from core.agent.tools.repo_edit import RepoEditTool
from core.agent.tools.repo_judge import RepoJudgeTool
from core.agent.tools.pdf_query import PDFQueryTool
from core.agent.tools.ipython import IPythonTool
from core.agent.tools.paper_rubric import PaperRubricTool
from core.agent.tools.image_entity_extract import ImageEntityExtractTool
from core.agent.tools.experiment_manager import ExperimentManagerTool
from core.agent.tools.got_edit import GoTEditTool
from core.agent.tools.anydoor_edit import AnyDoorEditTool
from core.agent.tools.grounding_sam import GroundingSAMTool
from core.agent.tools.grounding_dino import GroundingDINOTool
from core.agent.tools.inpaint_remove import InpaintRemoveTool
from core.agent.tools.sdxl_inpaint import SDXLInpaintTool
from core.agent.tools.lama_remove import LAMARemoveTool
from core.agent.tools.qwen_api import QwenAPITool
from core.agent.tools.image_edit_judge import ImageEditJudgeTool
from core.events.action import (Action, AgentFinishAction, AgentThinkAction,
                                BrowseInteractiveAction, BrowseURLAction,
                                CmdRunAction, FileEditAction, FileReadAction,
                                MessageAction, TaskGraphBuildAction, RepoPlanAction, RepoCreateAction, RepoAnalyzerAction, RepoUpdateAction, RepoVerifyAction, RepoRunAction, PaperReproductionAnalyzerAction, RepoDebugAction, RepoEditAction, PDFQueryAction, IPythonRunCellAction, RepoJudgeAction, PaperRubricAction, ExperimentManagerAction)
from core.events.action.image import QwenAPIAction, ImageEditJudgeAction, AnyDoorEditAction, GroundingSAMAction, GroundingDINOAction
from core.events.event import FileEditSource, FileReadSource, ToolCallMetadata


class FunctionCallNotExistsError(Exception):
    """Exception raised when an LLM call a tool that is not registered."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class FunctionCallValidationError(Exception):
    """Exception raised when FunctionCallingConverter failed to validate a function call message.

    This typically happens when the LLM outputs unrecognized function call / parameter names / values.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


def combine_thought(action: Action, thought: str) -> Action:
    if not hasattr(action, "thought"):
        return action
    if thought and action.thought:
        action.thought = f"{thought}\n{action.thought}"
    elif thought:
        action.thought = thought
    return action


def response_to_actions(response: ModelResponse) -> list[Action]:
    actions: list[Action] = []
    assert len(response.choices) == 1, "Only one choice is supported for now"
    choice = response.choices[0]
    assistant_msg = choice.message
    if hasattr(assistant_msg, "tool_calls") and assistant_msg.tool_calls:
        # Check if there's assistant_msg.content. If so, add it to the thought
        thought = ""
        if isinstance(assistant_msg.content, str):
            thought = assistant_msg.content
        elif isinstance(assistant_msg.content, list):
            for msg in assistant_msg.content:
                if msg["type"] == "text":
                    thought += msg["text"]

        # Process each tool call to OpenHands action
        for i, tool_call in enumerate(assistant_msg.tool_calls):
            action: Action
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.decoder.JSONDecodeError as e:
                raise RuntimeError(
                    f"Failed to parse tool call arguments: {tool_call.function.arguments}"
                ) from e


            # ================================================
            # CmdRunTool (Bash)
            # ================================================
            if tool_call.function.name == create_cmd_run_tool()["function"]["name"]:
                if "command" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "command" in tool call {tool_call.function.name}'
                    )
                # convert is_input to boolean
                is_input = arguments.get("is_input", "false") == "true"
                action = CmdRunAction(command=arguments["command"], is_input=is_input)

            # ================================================
            # ImageEntityExtractTool
            # ================================================
            elif tool_call.function.name == ImageEntityExtractTool["function"]["name"]:
                if "image_path" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "image_path" in tool call {tool_call.function.name}'
                    )
                action = RepoRunAction  # placeholder to get type hints quiet
                from core.events.action.image import ImageEntityExtractAction
                action = ImageEntityExtractAction(
                    image_path=arguments["image_path"],
                    model=arguments.get("model", "gpt-4o"),
                )
                action.set_hard_timeout(arguments.get("timeout", 180), blocking=False)
            # ================================================
            # GoTEditTool
            # ================================================
            elif tool_call.function.name == GoTEditTool["function"]["name"]:
                from core.events.action.image import GoTEditAction
                mode = arguments.get("mode", "t2i")
                if mode not in ["t2i", "edit"]:
                    raise FunctionCallValidationError(
                        f'Invalid mode {mode} in tool call {tool_call.function.name}'
                    )
                action = GoTEditAction(
                    image_path=arguments.get("image_path", None),
                    prompt=arguments.get("prompt", ""),
                    mode=mode,
                    height=arguments.get("height", 1024),
                    width=arguments.get("width", 1024),
                    max_new_tokens=arguments.get("max_new_tokens", 1024),
                    num_inference_steps=arguments.get("num_inference_steps", 50),
                    guidance_scale=arguments.get("guidance_scale", 7.5),
                    image_guidance_scale=arguments.get("image_guidance_scale", 1.0),
                    cond_image_guidance_scale=arguments.get("cond_image_guidance_scale", 4.0),
                    output_path=arguments.get("output_path", ""),
                    thought=thought,
                )
                action.set_hard_timeout(arguments.get("timeout", 600), blocking=False)
            # ================================================
            # AnyDoorEditTool
            # ================================================
            elif tool_call.function.name == AnyDoorEditTool["function"]["name"]:
                action = AnyDoorEditAction(
                    ref_image_path=arguments.get("ref_image_path", ""),
                    target_image_path=arguments.get("target_image_path", ""),
                    target_mask_path=arguments.get("target_mask_path", ""),
                    ref_mask_path=arguments.get("ref_mask_path", None),
                    guidance_scale=arguments.get("guidance_scale", 5.0),
                    output_path=arguments.get("output_path", ""),
                    thought=thought,
                )
                action.set_hard_timeout(arguments.get("timeout", 600), blocking=False)
            # ================================================
            # ImageEditJudgeTool
            # ================================================
            elif tool_call.function.name == ImageEditJudgeTool["function"]["name"]:
                action = ImageEditJudgeAction(
                    original_path=arguments.get("original_path", ""),
                    edited_path=arguments.get("edited_path", ""),
                    instruction=arguments.get("instruction", ""),
                    thought=thought,
                )
                action.set_hard_timeout(arguments.get("timeout", 600), blocking=False)
            # ================================================
            # GroundingSAMTool
            # ================================================
            elif tool_call.function.name == GroundingSAMTool["function"]["name"]:
                action = GroundingSAMAction(
                    image_path=arguments.get("image_path", ""),
                    text_prompt=arguments.get("text_prompt", ""),
                    return_type="image",
                    output_dir=None,
                    output_path=arguments.get("output_path", None),
                    thought=thought,
                )
                action.set_hard_timeout(arguments.get("timeout", 600), blocking=False)
            # ================================================
            # GroundingDINOTool
            # ================================================
            elif tool_call.function.name == GroundingDINOTool["function"]["name"]:
                action = GroundingDINOAction(
                    image_path=arguments.get("image_path", ""),
                    text_prompt=arguments.get("text_prompt", ""),
                    box_threshold=arguments.get("box_threshold", 0.3),
                    text_threshold=arguments.get("text_threshold", 0.25),
                    return_type=arguments.get("return_type", "json"),
                    output_path=arguments.get("output_path", None),
                    thought=thought,
                )
                action.set_hard_timeout(arguments.get("timeout", 600), blocking=False)
            # ================================================
            # InpaintRemoveTool
            # ================================================
            elif tool_call.function.name == InpaintRemoveTool["function"]["name"]:
                from core.events.action.image import InpaintRemoveAction
                action = InpaintRemoveAction(
                    image_path=arguments.get("image_path", ""),
                    point_coords=None,
                    mask_path=arguments.get("mask_path", None),
                    dilate_kernel_size=arguments.get("dilate_kernel_size", 0),
                    return_type="image",
                    output_dir=arguments.get("output_dir", None),
                    output_path=arguments.get("output_path", None),
                    thought=thought,
                )
                action.set_hard_timeout(arguments.get("timeout", 600), blocking=False)
            # ================================================
            # SDXLInpaintTool
            # ================================================
            elif tool_call.function.name == SDXLInpaintTool["function"]["name"]:
                from core.events.action.image import SDXLInpaintAction
                action = SDXLInpaintAction(
                    image_path=arguments.get("image_path", ""),
                    mask_path=arguments.get("mask_path", ""),
                    prompt=arguments.get("prompt", ""),
                    guidance_scale=arguments.get("guidance_scale", 8.0),
                    num_inference_steps=arguments.get("num_inference_steps", 20),
                    strength=arguments.get("strength", 0.99),
                    use_smart_crop=arguments.get("use_smart_crop", False),
                    seed=arguments.get("seed", None),
                    output_path=arguments.get("output_path", None),
                )
                action.set_hard_timeout(arguments.get("timeout", 600), blocking=False)
            # ================================================
            # LAMARemoveTool
            # ================================================
            elif tool_call.function.name == LAMARemoveTool["function"]["name"]:
                from core.events.action.image import LAMARemoveAction
                action = LAMARemoveAction(
                    image_path=arguments.get("image_path", ""),
                    mask_path=arguments.get("mask_path", ""),
                    dilate_kernel_size=arguments.get("dilate_kernel_size", 0),
                    output_path=arguments.get("output_path", None),
                )
                action.set_hard_timeout(arguments.get("timeout", 600), blocking=False)
            # ================================================
            # QwenAPITool
            # ================================================
            elif tool_call.function.name == QwenAPITool["function"]["name"]:
                action = QwenAPIAction(
                    prompt=arguments.get("prompt", ""),
                    image_path=arguments.get("image_path", ""),
                    mode=arguments.get("mode", "generate"),
                    max_new_tokens=arguments.get("max_new_tokens", 128),
                    temperature=arguments.get("temperature", 0.7),
                    top_p=arguments.get("top_p", 0.9),
                    messages=arguments.get("messages", None),
                    thought=thought,
                )
                action.set_hard_timeout(arguments.get("timeout", 600), blocking=False)  # Increased timeout to 10 minutes for Qwen API
            # ================================================
            # IPythonTool (Jupyter)
            # ================================================
            elif tool_call.function.name == IPythonTool['function']['name']:
                if 'code' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "code" in tool call {tool_call.function.name}'
                    )
                code = arguments['code']
                cwd = arguments.get('cwd')
                if cwd:
                    code = f"%cd {cwd}\n" + code
                action = IPythonRunCellAction(code=code)
                
            # ================================================
            # AgentFinishAction
            # ================================================
            elif tool_call.function.name == FinishTool["function"]["name"]:
                action = AgentFinishAction(
                    final_thought=arguments.get("message", ""),
                    task_completed=arguments.get("task_completed", None),
                )

            elif (
                tool_call.function.name
                == create_str_replace_editor_tool()["function"]["name"]
            ):
                if "command" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "command" in tool call {tool_call.function.name}'
                    )
                if "path" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "path" in tool call {tool_call.function.name}'
                    )
                path = arguments["path"]
                command = arguments["command"]
                other_kwargs = {
                    k: v for k, v in arguments.items() if k not in ["command", "path"]
                }

                if command == "view":
                    action = FileReadAction(
                        path=path,
                        impl_source=FileReadSource.OH_ACI,
                        view_range=other_kwargs.get("view_range", None),
                    )
                else:
                    if "view_range" in other_kwargs:
                        # Remove view_range from other_kwargs since it is not needed for FileEditAction
                        other_kwargs.pop("view_range")
                    action = FileEditAction(
                        path=path,
                        command=command,
                        impl_source=FileEditSource.OH_ACI,
                        **other_kwargs,
                    )
            # ================================================
            # AgentThinkAction
            # ================================================
            elif tool_call.function.name == ThinkTool["function"]["name"]:
                action = AgentThinkAction(thought=arguments.get("thought", ""))

            # # ================================================
            # # BrowserTool
            # # ================================================
            elif tool_call.function.name == BrowserTool["function"]["name"]:
                if "code" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "code" in tool call {tool_call.function.name}'
                    )
                action = BrowseInteractiveAction(browser_actions=arguments["code"])
                action.set_hard_timeout(120)
            # ================================================
            # WebReadTool (simplified browsing)
            # ================================================
            elif tool_call.function.name == WebReadTool["function"]["name"]:
                if "url" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "url" in tool call {tool_call.function.name}'
                    )
                action = BrowseURLAction(url=arguments["url"])
                action.set_hard_timeout(120)
            # ================================================
            # TaskGraphTool
            # ================================================
            elif tool_call.function.name == create_task_graph_tool()["function"]["name"]:
                if "task_description" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "task_description" in tool call {tool_call.function.name}'
                    )
                action = TaskGraphBuildAction(task_description=arguments["task_description"])
            # ================================================
            # RepoPlanTool
            # ================================================
            elif tool_call.function.name == RepoPlanTool["function"]["name"]:
                if "paper_path" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "paper_path" in tool call {tool_call.function.name}'
                    )
                action = RepoPlanAction(paper_path=arguments["paper_path"])
            # ================================================
            # RepoCreateTool
            # ================================================
            elif tool_call.function.name == RepoCreateTool["function"]["name"]:
                if "paper_path" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "paper_path" in tool call {tool_call.function.name}'
                    )
                if "output_repo_dir" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "output_repo_dir" in tool call {tool_call.function.name}'
                    )
                action = RepoCreateAction(
                    paper_path=arguments["paper_path"],
                    output_dir="",  # No longer use output_dir, leave empty
                    output_repo_dir=arguments["output_repo_dir"]
                )
            # ================================================
            # RepoAnalyzerTool
            # ================================================
            elif tool_call.function.name == RepoAnalyzerTool["function"]["name"]:
                if "paper_path" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "paper_path" in tool call {tool_call.function.name}'
                    )
                if "codebase_path" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "codebase_path" in tool call {tool_call.function.name}'
                    )
                action = RepoAnalyzerAction(
                    paper_path=arguments["paper_path"],
                    codebase_path=arguments["codebase_path"],
                    output_dir=arguments.get("output_dir", "")
                )
            # ================================================
            # RepoUpdateTool
            # ================================================
            elif tool_call.function.name == RepoUpdateTool["function"]["name"]:
                if "repo_path" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "repo_path" in tool call {tool_call.function.name}'
                    )
                if "requirements" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "requirements" in tool call {tool_call.function.name}'
                    )
                action = RepoUpdateAction(
                    repo_path=arguments["repo_path"],
                    requirements=arguments["requirements"],
                    target_files=arguments.get("target_files", []),
                    context=arguments.get("context", ""),
                    apply_changes=arguments.get("apply_changes", False),
                    thought=thought,
                )
            # ================================================
            # RepoVerifyTool
            # ================================================
            elif tool_call.function.name == RepoVerifyTool["function"]["name"]:
                if "repo_path" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "repo_path" in tool call {tool_call.function.name}'
                    )
                action = RepoVerifyAction(
                    repo_path=arguments["repo_path"],
                    requirement=arguments.get("requirement", ""),
                    verification_level=arguments.get("verification_level", "comprehensive"),
                    thought=thought,
                )
            # ================================================
            # RepoRunTool
            # ================================================
            elif tool_call.function.name == RepoRunTool["function"]["name"]:
                if "repo_path" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "repo_path" in tool call {tool_call.function.name}'
                    )
                action = RepoRunAction(
                    repo_path=arguments["repo_path"],
                    timeout=arguments.get("timeout", 3600),
                    docker_image=arguments.get("docker_image", "pb-reproducer:latest"),
                    memory_limit=arguments.get("memory_limit", "4g"),
                    network_enabled=arguments.get("network_enabled", True),
                    gpu_enabled=arguments.get("gpu_enabled", True),
                    use_persistent_containers=arguments.get("use_persistent_containers", True),
                    thought=thought,
                )
            # ================================================
            # PaperReproductionAnalyzerTool
            # ================================================
            elif tool_call.function.name == PaperReproductionAnalyzerTool["function"]["name"]:
                if "paper_content" not in arguments and "paper_path" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "paper_content" or "paper_path" in tool call {tool_call.function.name}'
                    )
                action = PaperReproductionAnalyzerAction(
                    paper_content=arguments.get("paper_content", ""),
                    paper_path=arguments.get("paper_path", ""),
                    analysis_level=arguments.get("analysis_level", "detailed"),
                    thought=thought,
                )
            # ================================================
            # RepoDebugTool
            # ================================================
            elif tool_call.function.name == RepoDebugTool["function"]["name"]:
                if "repo_path" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "repo_path" in tool call {tool_call.function.name}'
                    )
                if "action_description" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "action_description" in tool call {tool_call.function.name}'
                    )
                action = RepoDebugAction(
                    repo_path=arguments["repo_path"],
                    action_description=arguments["action_description"],
                    thought=thought,
                )
            # ================================================
            # RepoEditTool
            # ================================================
            elif tool_call.function.name == RepoEditTool["function"]["name"]:
                if "repo_path" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "repo_path" in tool call {tool_call.function.name}'
                    )
                if "edit_description" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "edit_description" in tool call {tool_call.function.name}'
                    )
                action = RepoEditAction(
                    repo_path=arguments["repo_path"],
                    edit_description=arguments["edit_description"],
                    traceback=arguments.get("traceback", ""),
                    thought=thought,
                )
            # ================================================
            # PDFQueryTool
            # ================================================
            elif tool_call.function.name == PDFQueryTool["function"]["name"]:
                if "pdf_path" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "pdf_path" in tool call {tool_call.function.name}'
                    )
                if "query" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "query" in tool call {tool_call.function.name}'
                    )
                action = PDFQueryAction(
                    pdf_path=arguments["pdf_path"],
                    query=arguments["query"],
                    embedding_model=arguments.get("embedding_model", "openai"),
                    top_k=arguments.get("top_k", 5),
                    chunk_size=arguments.get("chunk_size", 1000),
                    chunk_overlap=arguments.get("chunk_overlap", 200),
                    cache_dir=arguments.get("cache_dir", ""),
                    thought=thought,
                )
            # ================================================
            # RepoJudgeTool
            # ================================================
            elif tool_call.function.name == RepoJudgeTool["function"]["name"]:
                if "repo_path" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "repo_path" in tool call {tool_call.function.name}'
                    )
                if "rubric_file_path" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "rubric_file_path" in tool call {tool_call.function.name}'
                    )
                action = RepoJudgeAction(
                    repo_path=arguments["repo_path"],
                    rubric_file_path=arguments["rubric_file_path"],
                    thought=thought,
                )
            # ================================================
            # PaperRubricTool
            # ================================================
            elif tool_call.function.name == PaperRubricTool["function"]["name"]:
                if "paper_path" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "paper_path" in tool call {tool_call.function.name}'
                    )
                if "output_dir" not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "output_dir" in tool call {tool_call.function.name}'
                    )
                action = PaperRubricAction(
                    paper_path=arguments["paper_path"],
                    output_dir=arguments["output_dir"]
                )
            # ================================================
            # ExperimentManagerTool
            # ================================================
            elif tool_call.function.name == ExperimentManagerTool["function"]["name"]:
                mode = arguments.get("mode")
                if not mode:
                    raise FunctionCallValidationError(
                        f'Missing required argument "mode" in tool call {tool_call.function.name}'
                    )
                action = ExperimentManagerAction(
                    mode=mode,
                    command=arguments.get("command", ""),
                    repo_path=arguments.get("repo_path", ""),
                    experiment_name=arguments.get("experiment_name", ""),
                    output_dir=arguments.get("output_dir", ""),
                    thought=thought,
                )
            else:
                raise FunctionCallNotExistsError(
                    f"Tool {tool_call.function.name} is not registered. (arguments: {arguments}). Please check the tool name and retry with an existing tool."
                )

            # We only add thought to the first action
            if i == 0:
                action = combine_thought(action, thought)
            # Add metadata for tool calling
            action.tool_call_metadata = ToolCallMetadata(
                tool_call_id=tool_call.id,
                function_name=tool_call.function.name,
                model_response=response,
                total_calls_in_response=len(assistant_msg.tool_calls),
            )
            actions.append(action)
    else:
        actions.append(
            MessageAction(
                content=str(assistant_msg.content) if assistant_msg.content else "",
                wait_for_response=True,
            )
        )

    assert len(actions) >= 1
    return actions


def get_tools(
    codeact_enable_browsing: bool = False,
    codeact_enable_llm_editor: bool = False,
    codeact_enable_jupyter: bool = False,
    # llm: LLM | None = None,
) -> list[ChatCompletionToolParam]:
    SIMPLIFIED_TOOL_DESCRIPTION_LLM_SUBSTRS = ["gpt-", "o3", "o1"]

    # use_simplified_tool_desc = True
    use_simplified_tool_desc = False
    # if llm is not None:
    #     use_simplified_tool_desc = any(
    #         model_substr in llm.config.model
    #         for model_substr in SIMPLIFIED_TOOL_DESCRIPTION_LLM_SUBSTRS
    #     )

    tools = [
        create_cmd_run_tool(use_simplified_description=use_simplified_tool_desc),
        ThinkTool,
        FinishTool,
        # GoTEditTool,
        # AnyDoorEditTool,
        GroundingSAMTool,
        # InpaintRemoveTool,
        SDXLInpaintTool,
        LAMARemoveTool,
        ImageEditJudgeTool,
        # IPythonTool,
        
        # QwenAPITool, # Not accurate
        # ImageEntityExtractTool,
        # PDFQueryTool,
        # PaperRubricTool,
        # RepoCreateTool,
        # RepoEditTool,
        # RepoJudgeTool,
    ]
    # import pdb; pdb.set_trace()

    # if codeact_enable_browsing:
    #     tools.append(WebReadTool)
    #     tools.append(BrowserTool)
    if codeact_enable_jupyter:
        # tools.append(IPythonTool)
        pass
    if codeact_enable_llm_editor:
        # tools.append(LLMBasedFileEditTool)
        pass
    else:
        # tools.append(
        #     create_str_replace_editor_tool(
        #         use_simplified_description=use_simplified_tool_desc
        #     )
        # )
        pass
    return tools
