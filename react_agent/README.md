# Re-run instruction
## Prerequisite
Run the `hybrid_ensemble.ipynb` notebook from Kaggle, then download it's output.

## Running the experiment
1. First, convert torch vectors (in string format) in to tuples by running `convert_to_tuple.ipynb`
2. Then, run the `create_vector_store.ipynb` to re-scale the embedding vectors
3. Finally, get yourself a GEMINI_API_KEY, plug the key into .env file, then run `main.py`


