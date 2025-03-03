from typing import List
class FewShotDataset:
    def __init__(self, train_inputs: List[str], train_outputs: List[str], test_input: str, test_output: str):
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.test_input = test_input
        self.test_output = test_output

    def __len__(self):
        """
        データセット内の「テスト例の数」を返すようにする。
        ここでは単に1例とみなして1を返す。
        必要に応じて、train_inputs の長さを返すなど、用途に合わせて変更してください。
        """
        return 1
