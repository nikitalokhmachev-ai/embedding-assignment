# Embedding Model Implementation for Company Interview Coding Assignment

#### Steps to Reproduce

- Place the `data.csv` file in the root directory of the repository.
- Open the `main.py` file and change the queries in the list if needed.
- Run `python main.py`.
- After completion, a file named `search_results.json` will be generated.

#### Key Notes

- The entire assignment took approximately 6 hours to complete.
- I utilized [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for this project.
- The model processes at an average speed of around 250 tokens per second.
- It reliably interprets single-column queries based on natural language input.

#### Potential Improvements

Given more time, I would consider several enhancements:

- Fine-tune a custom embedding model on an artificially generated dataset using tools like ChatGPT or Claude, rather than relying solely on a pre-trained model.
- Utilize a small open-source language model, such as [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B), fine-tuning it for semantic parsing. For instance, a query like `apparel product with a distance shorter than 200 and a high priority` could be parsed into structured elements:
  ```json
  [
  	{ "column": "Product_Category", "value": "Apparel" },
  	{ "column": "Delivery_distance", "operator": "<", "value": 200 },
  	{ "column": "Priority", "value": "High" }
  ]
  ```
  Quantized versions of the Llama 3.2 1B model have shown the ability to process over 350 tokens per second during the prefill phase and 40 tokens per second in the decoding phase, even on mobile devices.
