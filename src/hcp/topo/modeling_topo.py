import random

from pathlib import Path
from typing import List, Dict

from ..node.modeling_node import FileNode
from ..retriever import AutoRetriever

random.seed(42)


class RepoTopo():
    supported_filetypes: List[str] = ['.py', '.pyi']
    supported_strategies: List[str] = [
        'random_all',
        'preliminary',
        'hcp',
    ]

    def __init__(
            self,
            repo_name_or_path: str,
    ) -> None:

        self.repo_name_or_path = repo_name_or_path
        self.file_nodes = self.build_file_nodes(repo_name_or_path)
        self.num_files = len(self.file_nodes)

    def build_file_nodes(self, repo_name_or_path: str) -> List[FileNode]:
        repo = Path(repo_name_or_path)
        if not repo.exists() or not repo.is_dir():
            raise ValueError("Provided path does not exist or is not a directory")

        files = []
        for supported_filetype in self.supported_filetypes:
            available_files = list(repo.rglob(f'*{supported_filetype}'))
            files.extend(available_files)
        file_nodes = {
            f"{str(file)}": FileNode(
                repo_path=self.repo_name_or_path,
                file_path=str(file),
            ) for file in files
        }

        return file_nodes

    def get_hierarchical_cross_file_context(
            self,
            retriever: AutoRetriever,
            file_node: FileNode,
            top_k: List[int] = [5],
            top_p: List[float] = [0.1],
            d_level: int = 1,
    ) -> str:
        """
        This funtion is implemented for Our Hierarchical Context Pruning.
        """
        dependencies = self.get_dependencies(file_node, d_level)
        dependencies.reverse()

        other_files = []
        for node in self.file_nodes:
            if self.file_nodes[node] != file_node and self.file_nodes[node] not in dependencies:
                other_files.append(self.file_nodes[node])
        random.shuffle(other_files)

        # get query text from file with row and col

        query_text = file_node.get_query_text(1, 1)

        candidate_function_nodes = []
        for node in other_files:
            candidate_function_nodes.extend(node.functions)
            for cls in node.classes:
                candidate_function_nodes.extend(cls.class_method_nodes)

        assert len(candidate_function_nodes) > 0, "No candidate function nodes found"
        top_nodes = retriever.retrieve(query_text, candidate_function_nodes, top_k, top_p)
        total_context = {}
        for nodes_info in top_nodes:
            k = nodes_info['topk']
            p = nodes_info['topp']
            topk_nodes = nodes_info['topk_nodes']
            topp_nodes = nodes_info['topp_nodes']

            for node in topp_nodes:
                node.related = 'medium'

            for node in topk_nodes:
                node.related = 'high'

            context = {}
            for node in other_files:
                repo_path = node.repo_path
                file_path = node.file_path
                relative_path = str(Path(repo_path).name / Path(file_path).relative_to(repo_path))
                content = node.get_context()
                if relative_path not in context:
                    context[relative_path] = {'content': content, 'score': node.score}

            context = dict(sorted(context.items(), key=lambda x: x[1]['score']))
            for ctx in context:
                if context[ctx]['score'] == 0.0:
                    context[ctx]['content'] = ""
            context = {k: v['content'] for k, v in context.items()}

            for node in dependencies:
                repo_path = node.repo_path
                file_path = node.file_path
                relative_path = str(Path(repo_path).name / Path(file_path).relative_to(repo_path))
                content = node.get_concise_context()
                if relative_path not in context:
                    context[relative_path] = content

            for node in topp_nodes:
                node.related = None

            total_context[f"topp_{p}_topk_{k}"] = context

            current_file_content = self.get_current_context(file_node)
        current_file_path = str(
            Path(file_node.repo_path).name / Path(file_node.file_path).relative_to(file_node.repo_path))

        total_context["current_file"] = {current_file_path: current_file_content}

        return total_context

    def get_current_context(self, file_node: FileNode) -> str:
        lines = file_node.content.split('\n')
        return '\n'.join(lines) + '\n'

    def get_dependencies(self, file_node, d_level) -> List[FileNode]:

        def collect_dependencies(
                file_node: FileNode,
                d_level: int,
                dependencies: List[FileNode],
        ) -> List[FileNode]:

            assert d_level >= 0, "Dependency level must be a non-negative integer"

            if d_level == 0:
                return

            for dep_path in file_node.dependencies:
                if self.file_nodes[dep_path] not in dependencies:
                    dependencies.append(self.file_nodes[dep_path])
                collect_dependencies(self.file_nodes[dep_path], d_level - 1, dependencies)

        dependencies = []

        if d_level == -1:
            # including all repo files according to the dependency level
            # including dependency files within 3 levels
            collect_dependencies(
                file_node,
                3,  # 3 is a large enough number to get nessasary dependencies
                dependencies,
            )
            # then including all other files
            for node in self.file_nodes:
                if self.file_nodes[node] != file_node and self.file_nodes[node] not in dependencies:
                    dependencies.append(self.file_nodes[node])
        else:
            # only including dependency files within the specified level
            collect_dependencies(
                file_node,
                d_level,
                dependencies,
            )

        return dependencies
