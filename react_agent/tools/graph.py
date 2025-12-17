import os
import torch
from torch_geometric.data import HeteroData
import pickle


class EventsGraph:
    def __init__(self, data_dir: str):
        """
        Initialize the event graph and the embedding model to translate node index

        :param self:
        :param data_dir: Directory where graph data files are stored
        :type data_dir: str
        :param model_dir: Directory where embedding model files are stored
        :type model_dir: str
        """

        # Load graph data
        self.data_dir = data_dir
        print(f"Loading graph data from {data_dir}...")
        try:
            self.data = torch.load(
                os.path.join(data_dir, "graph_data_torch.pt"),
                map_location="cpu",
                weights_only=False,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not load graph data files from {data_dir}. Ensure graph_data_torch.pt exists."
            ) from e

        # Pre-fetch edge index for User->Logon->Computer
        if ("User", "Logon", "Computer") in self.data.edge_index_dict:
            self.user_logon_comp_edge_index = self.data[
                "User", "Logon", "Computer"
            ].edge_index
        else:
            print(
                "Warning: Edge type ('User', 'Logon', 'Computer') not found in graph data."
            )
            self.user_logon_comp_edge_index = torch.empty((2, 0), dtype=torch.long)


    # DON'T HAVE TO DO THIS ANYMORE, IT'S CONTAINED IN THE OUTPUT FILE.
    #     # Load model data
    #     # Entity index offset will be accessed with `embedding_model.start["<entity_name>"]`
    #     try:
    #         print(f"Loading model from {model_dir}...")
    #         self.embedding_model = torch.load(
    #             os.path.join(model_dir, "model.pt"),
    #             map_location="cpu",
    #             weights_only=False,
    #         )
    #     except FileNotFoundError as e:
    #         raise FileNotFoundError(
    #             f"Could not load model data files from {model_dir}. Ensure model.pt exists."
    #         ) from e

    #     # Load path embeddings:
    #     try:
    #         print(f"Loading embeddings from {model_dir}...")
    #         self.embeddings = pickle.load(
    #             open(os.path.join(model_dir, "path_embedding_tuple.pkl"))
    #         )
    #     except FileNotFoundError as e:
    #         raise FileNotFoundError(
    #             f"Could not load embedding file from {model_dir}. Ensure path_embedding_tuple.pkl exists."
    #         ) from e

    # # def get_original_index(self, index: int, entity_name: str):
    # #     return index - self.embedding_model.start[entity_name]
