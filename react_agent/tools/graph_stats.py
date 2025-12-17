import os
import torch
from torch_geometric.data import HeteroData
from tools.graph import EventsGraph
from langchain_core.tools import tool
from typing import Set, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class GraphTools:
    """Provides tools for graph"""

    def __init__(
        self,
        events_graph: EventsGraph,
        embeddings_dictionary: Dict,
        labels_dictionary: Dict,
        labels_origin_dictionary: Dict,
    ):
        """Initialize"""
        self.events_graph = events_graph
        self.embeddings_dictionary = embeddings_dictionary
        self.labels_dictionary = labels_dictionary
        self.labels_origin_dictionary = labels_origin_dictionary

    # @tool
    def get_computer_stats(
        self, computer_index: int, from_csv_output: bool = False
    ) -> dict:
        """asdfasdf"""
        # Convert metapath2vec index to original graph index
        computer_id = computer_index
        if from_csv_output:
            computer_id = self.get_graph_index(computer_index, "Computer")

        # Get edge index for User->Logon->Computer
        edge_index = self.events_graph.data["User", "Logon", "Computer"].edge_index

        # Find all users that logged into this computer
        mask = edge_index[1] == computer_id
        source_user_ids = edge_index[0][mask]

        unique_users, counts = torch.unique(source_user_ids, return_counts=True)

        # Sort by counts descending
        sorted_indices = torch.argsort(counts, descending=True)

        # Get top 10 (or fewer if less than 10)
        num_items = min(10, len(sorted_indices))
        top_indices = sorted_indices[:num_items]

        top_users = unique_users[top_indices]
        top_counts = counts[top_indices]

        top_logins_list = []
        for uid, count in zip(top_users.tolist(), top_counts.tolist()):
            top_logins_list.append({f"U{uid}": count})

        return {"no_of_unique": unique_users.numel(), "top_logins": top_logins_list}

    # @tool
    def get_user_stats(self, user_index: int, from_csv_output: bool = False) -> dict:
        """
        Get statistics for a User node regarding logins to Computers.

        Args:
            user_index (int): The metapath2vec index of the user.

        Returns:
            dict: A dictionary containing unique computer count and top 10 computers by login count.
        """
        # Convert metapath2vec index to original graph index
        user_id = user_index
        if from_csv_output:
            user_id = self.get_graph_index(user_index, "User")

        # Get edge index for User->Logon->Computer
        edge_index = self.events_graph.data["User", "Logon", "Computer"].edge_index

        # Find all computers this user logged into
        mask = edge_index[0] == user_id
        target_comp_ids = edge_index[1][mask]

        unique_computers, counts = torch.unique(target_comp_ids, return_counts=True)

        # Sort by counts descending
        sorted_indices = torch.argsort(counts, descending=True)

        # Get top 10 (or fewer if less than 10)
        num_items = min(10, len(sorted_indices))
        top_indices = sorted_indices[:num_items]

        top_computers = unique_computers[top_indices]
        top_counts = counts[top_indices]

        top_logins_list = []
        for cid, count in zip(top_computers.tolist(), top_counts.tolist()):
            top_logins_list.append({f"C{cid}": count})

        return {"no_of_unique": unique_computers.numel(), "top_logins": top_logins_list}

    def tuple_to_vector(t: Tuple):
        listified_tuple = list(t)
        return np.array(listified_tuple).reshape(1, -1)

    def cal_cosine_sim(x: Tuple, y: Tuple):
        x_vec = GraphTools.tuple_to_vector(x)
        y_vec = GraphTools.tuple_to_vector(y)
        similarity = cosine_similarity(x_vec, y_vec)
        return similarity[0][0]

    # @tool
    # def get_closest_paths_labels(
    #     self, comparative_paths: Dict[Tuple, int], target_path: Tuple, top_k: int = 5
    # ):
    #     # Implemeneted specifically for CUC metapath

    #     # Comparative paths are raws from csv files, so we can use it directly, but first, we'll need to process it to Dict[Tuple, int] elsewhere
    #     comparative_embeddings = {
    #         index_tuple: np.array(self.events_graph.embeddings[index_tuple]).reshape(
    #             1, -1
    #         )
    #         for index_tuple in comparative_paths
    #     }
    #     target_embedding = np.array(self.events_graph.embeddings[target_path]).reshape(
    #         1, -1
    #     )

    #     # Calculate cosine similarity
    #     comparative_embeddings_similarity = {
    #         index_tuple: cosine_similarity(
    #             target_embedding, comparative_embeddings[index_tuple]
    #         )[0][0]  # Extract scalar value from 2D array
    #         for index_tuple in comparative_embeddings
    #     }

    #     # Sort by similarity score (descending)
    #     sorted_paths = sorted(
    #         comparative_embeddings_similarity.items(), key=lambda x: x[1], reverse=True
    #     )

    #     # Get top 5 most similar paths
    #     top_embeds = sorted_paths[:top_k]

    #     # Format results with path, similarity score, and label
    #     results = []
    #     for path_tuple, similarity_score in top_embeds:
    #         results.append(
    #             {
    #                 "path": [path_tuple[0], path_tuple[1], path_tuple[2]],
    #                 "similarity": float(similarity_score),
    #                 "label": comparative_paths[path_tuple],
    #             }
    #         )

    #     return results

    # @tool
    def get_closest_labels(self, target_path: Tuple, top_k: int = 3):
        """asdfasd"""
        # Calculate rankings
        target_embedding = self.embeddings_dictionary[target_path]
        comparatives_embeddings = [
            (path, GraphTools.cal_cosine_sim(target_embedding, e))
            for path, e in self.embeddings_dictionary.items()
            if (path != target_path)
        ]
        # Sort by similarity score (the second element of the tuple) in descending order
        sorted_paths = sorted(
            comparatives_embeddings, key=lambda item: item[1], reverse=True
        )

        # Get top k most similar paths
        top_paths = sorted_paths[:top_k]

        # Format results with path, similarity score, and label
        results = []
        for path_tuple, similarity_score in top_paths:
            results.append(
                {
                    "path": list(path_tuple),
                    "similarity": float(similarity_score),
                    "label": self.labels_dictionary.get(path_tuple, "Unknown"),
                    "label origin": self.labels_origin_dictionary.get(
                        path_tuple, "Unknown"
                    ),
                }
            )
        return results
