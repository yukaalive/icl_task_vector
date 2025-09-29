from core.data.datasets.few_shot_dataset import FewShotDataset
from typing import Optional, List, Dict


class FewShotFormat:
    def __init__(
        self,
        example_format: str = "example:{input}->{output}",
        # example_format: str = "input:{input}, output:{output}",
        example_separator: str = "\n",
        task_description: Optional[str] = None,
        test_example_format: Optional[str] = "example:{input}->",
        # test_example_format: Optional[str] = "input:{input}, output:",
        dataset_specific_prompts: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        self.example_format = example_format
        self.example_separator = example_separator
        self.task_description = task_description
        self.test_example_format = test_example_format
        # データセット固有のプロンプト設定を保存
        self.dataset_specific_prompts = dataset_specific_prompts or {}

    def get_dataset_specific_format(self, dataset_name: str, format_type: str) -> Optional[str]:
        """
        データセット固有のフォーマットを取得する
        Args:
            dataset_name: データセット名（例: "ja_en"）
            format_type: フォーマットタイプ（"task_description", "example_format", "test_example_format"）
        Returns:
            データセット固有のフォーマット文字列、または None
        """
        if dataset_name in self.dataset_specific_prompts:
            return self.dataset_specific_prompts[dataset_name].get(format_type)
        return None

    def format_train_example(self, inp: str, out: str, dataset_name: Optional[str] = None) -> str:
        """
        訓練例をフォーマットする
        Args:
            inp: 入力文字列
            out: 出力文字列
            dataset_name: データセット名（データセット固有のフォーマットを使用する場合）
        """
        # データセット固有のフォーマットがあれば使用
        example_format = self.example_format
        if dataset_name:
            specific_format = self.get_dataset_specific_format(dataset_name, "example_format")
            if specific_format:
                example_format = specific_format
        
        return example_format.format(input=inp, output=out)

    def format_test_example(self, inp: str, dataset_name: Optional[str] = None) -> str:
        """
        テスト例をフォーマットする
        Args:
            inp: 入力文字列
            dataset_name: データセット名（データセット固有のフォーマットを使用する場合）
        """
        # データセット固有のテストフォーマットがあれば使用
        test_format = self.test_example_format
        if dataset_name:
            specific_format = self.get_dataset_specific_format(dataset_name, "test_example_format")
            if specific_format:
                test_format = specific_format
        
        if test_format is None:
            return self.format_train_example(inp, "", dataset_name)
        else:
            return test_format.format(input=inp)

    def format_datasets(self, datasets: List[FewShotDataset], dataset_name: Optional[str] = None, **kwargs) -> List[str]:
        """
        複数のデータセットをフォーマットする
        Args:
            datasets: フォーマットするデータセットのリスト
            dataset_name: データセット名（データセット固有のプロンプトを使用する場合）
        """
        return [self.format_dataset(dataset, dataset_name=dataset_name, **kwargs) for dataset in datasets]

    def format_dataset(self, dataset: FewShotDataset, dataset_name: Optional[str] = None, include_train: bool = True, include_test: bool = True) -> str:
        """
        単一のデータセットをフォーマットする
        Args:
            dataset: フォーマットするデータセット
            dataset_name: データセット名（データセット固有のプロンプトを使用する場合）
            include_train: 訓練例を含めるかどうか
            include_test: テスト例を含めるかどうか
        """
        # データセット固有のタスク説明があれば使用
        task_description = self.task_description
        if dataset_name:
            specific_description = self.get_dataset_specific_format(dataset_name, "task_description")
            if specific_description:
                task_description = specific_description

        base_prompt = ""
        if task_description is not None:
            base_prompt += f"{task_description}{self.example_separator}"

        if len(dataset.train_inputs) > 0:
            train_examples = [
                self.format_train_example(x, y, dataset_name) for x, y in zip(dataset.train_inputs, dataset.train_outputs)
            ]
            train_examples_prompt = self.example_separator.join(train_examples)
            train_examples_prompt += self.example_separator
        else:
            train_examples_prompt = ""

        test_example_prompt = self.format_test_example(dataset.test_input, dataset_name)

        prompt = base_prompt
        if include_train:
            prompt += train_examples_prompt
        if include_test:
            prompt += test_example_prompt

        return prompt
