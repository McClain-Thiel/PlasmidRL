from torchrl.envs.llm import ChatEnv
from tensordict import TensorDict


class StatelessChatEnv(ChatEnv):
    """ChatEnv variant that avoids accumulating conversation history in text mode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_text_prompt = None

    def _reset(self, tensordict: TensorDict, **kwargs):
        td = super()._reset(tensordict, **kwargs)
        if self.input_mode == "text":
            prompt = td.get(("text", "prompt"))
            self._base_text_prompt = prompt.clone() if prompt is not None else None
        return td

    def _step_text(self, tensordict: TensorDict):
        td = super()._step_text(tensordict)
        if self._base_text_prompt is not None:
            td.set(("text", "prompt"), self._base_text_prompt.clone())
            td.set(("text", "full"), self._base_text_prompt.clone())
        return td
