from torchrl.envs import Transform
from tensordict import TensorDict


class DefaultQueryOnReset(Transform):
    def __init__(self, default_query: list[str]):
        super().__init__()
        self.default_query = default_query

    # This is the hook TransformedEnv invokes before calling base_env._reset(...)
    def _reset_env_preprocess(self, tensordict: TensorDict | None) -> TensorDict:
        if tensordict is None or ("query" not in tensordict.keys(True)):
            b = (len(self.default_query),)
            tensordict = TensorDict({"query": self.default_query}, batch_size=b)
        return tensordict