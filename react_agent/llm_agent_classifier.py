"""
LLM Agent-based Path Classifier using LangGraph.

This classifier follows a multi-step workflow:
1. Get similar paths from labeled dataset
2. LLM evaluates based on similarity
3. If unpredictable, gather graph statistics
4. Make final classification
"""

import sys
import os
from typing import Annotated, Sequence, TypedDict, Literal
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.graph_stats import GraphTools
from graph_data.graph import EventsGraph

load_dotenv()


class AgentState(TypedDict):
    """State for the agent workflow."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    target_path: tuple  # (c1, u, c2)
    similar_paths: list  # List of similar paths with labels
    classification: str  # MALICIOUS or BENIGN
    confidence: str  # HIGH, MEDIUM, LOW
    needs_stats: bool  # Whether to gather additional stats


class LLMAgentClassifier:
    def __init__(
        self,
        graph_data_dir: str,
        model_data_dir: str,
        labeled_paths: dict,  # Dict[Tuple, int] - path tuple to label (0=benign, 1=malicious)
        model_name: str = "gemini-2.0-flash-exp",
        device: str = "cpu",
    ):
        """
        Initialize the LLM Agent Classifier.

        Args:
            graph_data_dir: Path to graph data directory
            model_data_dir: Path to model data directory (for embeddings)
            labeled_paths: Dictionary mapping path tuples to labels
            model_name: Gemini model name
            device: Device (unused for API models)
        """
        self.graph_data_dir = graph_data_dir
        self.model_data_dir = model_data_dir
        self.labeled_paths = labeled_paths

        # Load graph and tools
        print("Initializing EventsGraph...")
        self.events_graph = EventsGraph(
            data_dir=graph_data_dir,
            model_dir=model_data_dir,
            embeddings_dir=model_data_dir,
        )
        self.graph_tools = GraphTools(self.events_graph)

        # Initialize LLM
        print(f"Loading model {model_name}...")
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            max_output_tokens=8192,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            },
        )

        # Build the workflow graph
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow following the flowchart."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("get_similar_paths", self._get_similar_paths)
        workflow.add_node("llm_evaluation", self._llm_evaluation)
        workflow.add_node("get_stats", self._get_stats)
        workflow.add_node("predict", self._predict)

        # Define edges
        workflow.set_entry_point("get_similar_paths")
        workflow.add_edge("get_similar_paths", "llm_evaluation")
        workflow.add_conditional_edges(
            "llm_evaluation",
            self._is_predictable,
            {"predictable": "predict", "unpredictable": "get_stats"},
        )
        workflow.add_edge("get_stats", "llm_evaluation")
        workflow.add_edge("predict", END)

        return workflow

    def _get_similar_paths(self, state: AgentState) -> AgentState:
        """Step 1: Get similar paths from labeled dataset."""
        target_path = state["target_path"]
        print(f"\n[Get Similar Paths] Analyzing path: {target_path}")

        # Get top 5 most similar paths
        similar_paths = self.graph_tools.get_closest_paths_labels(
            comparative_paths=self.labeled_paths, target_path=target_path, top_k=5
        )

        print(f"[Get Similar Paths] Found {len(similar_paths)} similar paths")

        # Add message about similar paths
        similar_paths_msg = self._format_similar_paths(similar_paths, target_path)

        return {
            **state,
            "similar_paths": similar_paths,
            "messages": [HumanMessage(content=similar_paths_msg)],
            "needs_stats": False,
        }

    def _llm_evaluation(self, state: AgentState) -> AgentState:
        """Step 2: LLM evaluates the path based on available information."""
        print("\n[LLM Evaluation] Evaluating path...")

        # Prepare system message
        system_msg = SystemMessage(content=self._get_system_prompt())

        # Get LLM response
        messages = [system_msg] + list(state["messages"])
        response = self.llm.invoke(messages)

        print(f"[LLM Evaluation] Response: {response.content[:200]}...")

        # Parse response
        classification, confidence = self._parse_llm_response(response.content)

        return {
            **state,
            "messages": [response],
            "classification": classification,
            "confidence": confidence,
        }

    def _get_stats(self, state: AgentState) -> AgentState:
        """Step 3: Get additional graph statistics for unpredictable cases."""
        target_path = state["target_path"]
        c1_id, u_id, c2_id = target_path

        print("\n[Get Stats] Gathering additional statistics...")

        # Get stats for all entities in the path
        c1_stats = self.graph_tools.get_computer_stats(c1_id, from_csv_output=True)
        u_stats = self.graph_tools.get_user_stats(u_id, from_csv_output=True)
        c2_stats = self.graph_tools.get_computer_stats(c2_id, from_csv_output=True)

        # Format stats message
        stats_msg = self._format_stats_message(target_path, c1_stats, u_stats, c2_stats)

        print(f"[Get Stats] Stats gathered for C{c1_id}, U{u_id}, C{c2_id}")

        return {
            **state,
            "messages": [HumanMessage(content=stats_msg)],
            "needs_stats": False,
        }

    def _predict(self, state: AgentState) -> AgentState:
        """Step 4: Make final prediction."""
        classification = state["classification"]
        confidence = state["confidence"]

        print(
            f"\n[Predict] Final classification: {classification} (Confidence: {confidence})"
        )

        return state

    def _is_predictable(
        self, state: AgentState
    ) -> Literal["predictable", "unpredictable"]:
        """Decision node: Check if the path is predictable based on confidence."""
        confidence = state.get("confidence", "LOW")
        needs_stats = state.get("needs_stats", False)

        # If we already gathered stats, proceed to prediction
        if needs_stats:
            return "predictable"

        # Check confidence level
        if confidence == "HIGH":
            print("[Decision] HIGH confidence - proceeding to prediction")
            return "predictable"
        else:
            print(f"[Decision] {confidence} confidence - gathering more stats")
            state["needs_stats"] = True
            return "unpredictable"

    def _format_similar_paths(self, similar_paths: list, target_path: tuple) -> str:
        """Format similar paths information for the LLM."""
        c1, u, c2 = target_path

        msg = f"""Authentication Path to Analyze: C{c1} -> U{u} -> C{c2}

Top 5 Most Similar Paths from Labeled Dataset:
"""

        for i, path_info in enumerate(similar_paths, 1):
            path = path_info["path"]
            similarity = path_info["similarity"]
            label = "MALICIOUS" if path_info["label"] == 1 else "BENIGN"

            msg += f"\n{i}. Path: C{path[0]} -> U{path[1]} -> C{path[2]}"
            msg += f"\n   Similarity: {similarity:.4f}"
            msg += f"\n   Label: {label}"

        return msg

    def _format_stats_message(
        self, target_path: tuple, c1_stats: dict, u_stats: dict, c2_stats: dict
    ) -> str:
        """Format graph statistics for the LLM."""
        c1_id, u_id, c2_id = target_path

        def format_top_list(top_list):
            if not top_list:
                return "None"
            return ", ".join(
                [f"{k}: {v}" for item in top_list for k, v in item.items()]
            )

        msg = f"""Additional Graph Statistics:

Source Computer (C{c1_id}):
- Unique users who logged in: {c1_stats["no_of_unique"]}
- Top users: {format_top_list(c1_stats["top_logins"])}

User (U{u_id}):
- Unique computers accessed: {u_stats["no_of_unique"]}
- Top computers: {format_top_list(u_stats["top_logins"])}

Destination Computer (C{c2_id}):
- Unique users who logged in: {c2_stats["no_of_unique"]}
- Top users: {format_top_list(c2_stats["top_logins"])}

With this additional context, please re-evaluate your classification."""

        return msg

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return """You are a cybersecurity analyst specializing in detecting lateral movement attacks.

Your task: Analyze an authentication path (Computer -> User -> Computer) and determine if it represents MALICIOUS lateral movement or BENIGN normal activity.

You will be provided with:
1. The target path to analyze
2. Top 5 most similar paths from a labeled dataset with their similarity scores and labels
3. (Optional) Additional graph statistics if needed

ANALYSIS GUIDELINES:
- High similarity to MALICIOUS paths suggests the target is likely MALICIOUS
- High similarity to BENIGN paths suggests the target is likely BENIGN
- Mixed signals (similar to both types) require deeper analysis
- Consider the similarity scores - higher scores mean stronger evidence
- Graph statistics help identify unusual patterns (e.g., rare user-computer combinations)

OUTPUT FORMAT:
You MUST respond in this exact format:

Reasoning: <Your analysis in 3-10 sentences. Explain which similar paths are most relevant, what patterns you observe, and why you're confident or uncertain.>

Confidence: <HIGH, MEDIUM, or LOW>
- HIGH: Strong evidence from similar paths (similarity > 0.8) with consistent labels
- MEDIUM: Moderate evidence or some conflicting signals
- LOW: Weak similarity scores or highly mixed signals

Classification: <MALICIOUS or BENIGN>

CRITICAL RULES:
1. Always provide all three sections: Reasoning, Confidence, Classification
2. Confidence MUST be exactly one of: HIGH, MEDIUM, LOW
3. Classification MUST be exactly: MALICIOUS or BENIGN
4. Do NOT generate any text after the Classification line
5. Be concise but thorough in your reasoning"""

    def _parse_llm_response(self, response: str) -> tuple[str, str]:
        """Parse LLM response to extract classification and confidence."""
        lines = response.strip().split("\n")

        classification = "BENIGN"  # Default
        confidence = "LOW"  # Default

        for line in lines:
            line = line.strip()

            # Parse confidence
            if line.startswith("Confidence:"):
                conf_text = line.replace("Confidence:", "").strip().upper()
                if "HIGH" in conf_text:
                    confidence = "HIGH"
                elif "MEDIUM" in conf_text:
                    confidence = "MEDIUM"
                elif "LOW" in conf_text:
                    confidence = "LOW"

            # Parse classification
            elif line.startswith("Classification:"):
                class_text = line.replace("Classification:", "").strip().upper()
                if "MALICIOUS" in class_text:
                    classification = "MALICIOUS"
                elif "BENIGN" in class_text:
                    classification = "BENIGN"

        return classification, confidence

    def classify_path(
        self, c1_id: int, u_id: int, c2_id: int, verbose: bool = True
    ) -> dict:
        """
        Classify a path using the agent workflow.

        Args:
            c1_id: Source computer ID
            u_id: User ID
            c2_id: Destination computer ID
            verbose: Whether to print detailed information

        Returns:
            Dictionary with classification results including:
            - classification: MALICIOUS or BENIGN
            - confidence: HIGH, MEDIUM, or LOW
            - reasoning: LLM's reasoning
            - similar_paths: List of similar paths used
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Classifying Path: C{c1_id} -> U{u_id} -> C{c2_id}")
            print(f"{'=' * 60}")

        # Initialize state
        initial_state = {
            "messages": [],
            "target_path": (c1_id, u_id, c2_id),
            "similar_paths": [],
            "classification": "BENIGN",
            "confidence": "LOW",
            "needs_stats": False,
        }

        # Run the workflow
        final_state = self.app.invoke(initial_state)

        # Extract reasoning from AI messages
        reasoning = ""
        for msg in final_state["messages"]:
            if isinstance(msg, AIMessage):
                reasoning = msg.content
                break

        result = {
            "path": f"C{c1_id}->U{u_id}->C{c2_id}",
            "classification": final_state["classification"],
            "confidence": final_state["confidence"],
            "reasoning": reasoning,
            "similar_paths": final_state["similar_paths"],
        }

        if verbose:
            print(f"\n{'=' * 60}")
            print(
                f"RESULT: {result['classification']} (Confidence: {result['confidence']})"
            )
            print(f"{'=' * 60}\n")

        return result
