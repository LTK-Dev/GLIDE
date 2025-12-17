"""Full evaluation script using LangGraph agent."""

from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import os
import sys
import pickle

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from core.langgraph_agent import PathClassifierAgent


def main():
    # Configuration
    pickle_path = (
        "/mnt/drive/hoangbaoan/repos/llm-ids/LMTracker/Phase2/data/seed5_df.pkl"
    )
    graph_data_dir = "/mnt/drive/hoangbaoan/repos/llm-ids/LMTracker/Phase2/data/graph_data_20251120161219"

    model_name = "gemini-2.5-flash"
    max_iterations = 15

    # Load pickled dataframe
    print(f"Loading data from {pickle_path}...")
    if not os.path.exists(pickle_path):
        print(f"Error: {pickle_path} not found.")
        return

    df = pickle.load(open(pickle_path, "rb"))
    print(f"Loaded {len(df)} paths from pickled dataframe")

    # Filter for autoencoder predictions with label_pred = 1 (candidates for LLM re-verification)
    candidates_mask = (df["label_pred"] == 1) & (df["from"] == "autoencoder prediction")
    candidates = df[candidates_mask].copy()
    print(
        f"Found {len(candidates)} candidates (autoencoder predictions with label_pred=1) out of {len(df)} total rows."
    )

    if len(candidates) == 0:
        print("No candidates found. Exiting.")
        return

    # Create embeddings and labels dictionaries from the dataframe
    try:
        embeddings_dict = {
            tuple([row["node_0"], row["node_1"], row["node_2"]]): row["embedding"]
            for _, row in df.iterrows()
        }

        labels_dict = {
            tuple([row["node_0"], row["node_1"], row["node_2"]]): row["label_pred"]
            for _, row in df.iterrows()
        }

        labels_origin_dict = {
            tuple([row["node_0"], row["node_1"], row["node_2"]]): row["from"]
            for _, row in df.iterrows()
        }

        print(f"Created dictionaries with {len(embeddings_dict)} path embeddings")
    except Exception as e:
        print(f"Error creating dictionaries: {e}")
        return

    # Initialize LangGraph Agent
    try:
        print(f"Initializing LangGraph agent with model {model_name}...")
        agent = PathClassifierAgent(
            graph_data_dir=graph_data_dir,
            embeddings_dict=embeddings_dict,
            labels_dict=labels_dict,
            labels_origin_dict=labels_origin_dict,
            model_name=model_name,
            max_iterations=max_iterations,
        )

        print("LangGraph agent initialized successfully!")

    except Exception as e:
        print(f"Error initializing agent: {e}")
        import traceback

        traceback.print_exc()
        return

    # Run classification
    llm_preds = []
    llm_responses = []
    print("\nRunning LangGraph agent classification...")
    print("The agent will use tools to investigate each path.\n")

    for idx, (index, row) in enumerate(
        tqdm(candidates.iterrows(), total=len(candidates))
    ):
        c1 = int(row["node_0"])
        u = int(row["node_1"])
        c2 = int(row["node_2"])

        print(f"\n{'=' * 80}")
        print(f"Processing path {idx + 1}/{len(candidates)}: C{c1} -> U{u} -> C{c2}")
        print(f"{'=' * 80}")

        try:
            classification, response_text = agent.classify_path(c1, u, c2)

            if classification == "MALICIOUS":
                llm_preds.append(1)
                print("✗ Classification: MALICIOUS")
            else:
                llm_preds.append(0)
                print("✓ Classification: BENIGN")

            llm_responses.append(response_text)

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            import traceback

            traceback.print_exc()
            llm_preds.append(0)
            llm_responses.append(f"Error: {str(e)}")

    # Update predictions - Initialize columns for all rows first
    if "final_pred" not in df.columns:
        df["final_pred"] = df["label_pred"]
    if "llm_reasoning" not in df.columns:
        df["llm_reasoning"] = None

    # Update only the candidates that were processed
    df.loc[candidates.index, "final_pred"] = llm_preds
    df.loc[candidates.index, "llm_reasoning"] = llm_responses

    # Calculate metrics only on autoencoder predictions
    autoencoder_mask = df["from"] == "autoencoder prediction"
    df_autoencoder = df[autoencoder_mask]

    if "label_true" not in df.columns:
        print("\nNo ground truth labels available. Saving predictions only.")
        output_path = "all_paths_predictions_with_langgraph.pkl"
        pickle.dump(df, open(output_path, "wb"))
        print(f"Results saved to {output_path}")
        return

    # Calculate F1 score on autoencoder predictions only
    y_true_autoencoder = df_autoencoder["label_true"]
    y_pred_autoencoder = df_autoencoder["final_pred"]

    f1_autoencoder = f1_score(y_true_autoencoder, y_pred_autoencoder)

    print(f"\n{'=' * 80}")
    print("FINAL RESULTS (on autoencoder predictions)")
    print(f"{'=' * 80}")
    print(f"Total paths: {len(df)}")
    print(f"Autoencoder predictions: {len(df_autoencoder)}")
    print(f"Paths verified by LLM: {len(candidates)}")
    print(f"\nF1 Score (autoencoder predictions): {f1_autoencoder:.4f}")
    print("\nClassification Report (autoencoder predictions):")
    print(classification_report(y_true_autoencoder, y_pred_autoencoder))
    print("\nConfusion Matrix (autoencoder predictions):")
    print(confusion_matrix(y_true_autoencoder, y_pred_autoencoder))

    # Save results as pickle
    output_path = "all_paths_predictions_with_langgraph_dec6.pkl"
    pickle.dump(df, open(output_path, "wb"))
    print(f"\nResults saved to {output_path}")
    df.to_csv(f"{output_path.split('.')[0]}.csv")


if __name__ == "__main__":
    main()
